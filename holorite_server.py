"""holorite_server.py — a tiny HTTP server the companion talks to for
Holorite models. Uses NodePagedEmbedding so the embedding matrix never
needs to fit on the GPU.

Run:    py D:\\Holorites\\holorite_server.py
Listens on:  http://127.0.0.1:41511

The companion sends:   POST /chat   {"manifest": "...\\manifest.json",
                                    "text": "user prompt",
                                    "max_new_tokens": 200,
                                    "system": ""}
Server returns:   {"text": "model reply", "stats": {...}}
"""
from __future__ import annotations
import json, os, sys, threading, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torus_lattice import NodePagedEmbedding, NODES_TOTAL, CELLS, SLOTS

HOST, PORT = "127.0.0.1", 41511
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Snapshot the device's clean free memory ONCE at startup, when no model is
# loaded. `torch.cuda.mem_get_info()` lies during HF `from_pretrained` because
# the load transient briefly pins weights on the GPU and the caching allocator
# holds those reservations even after `low_cpu_mem_usage=True` moves them to
# CPU. Reading free memory mid-load showed us 321 MiB on a 4 GiB card — way
# under the real budget. Capturing it now gives us a stable reference.
if DEVICE == "cuda":
    # ensure CUDA is initialized before mem_get_info — otherwise it can
    # report a stale low value before the first tensor lands on the device.
    _ = torch.empty(1, device="cuda")
    del _
    torch.cuda.synchronize()
    _CLEAN_FREE, _DEV_TOTAL = torch.cuda.mem_get_info()
    print(f"[holorite] clean GPU snapshot: free={_CLEAN_FREE/1_048_576:.0f} MiB of {_DEV_TOTAL/1_048_576:.0f} MiB", flush=True)
else:
    _CLEAN_FREE, _DEV_TOTAL = 1 << 40, 1 << 40

# Share the clean snapshot with body_pager so its own working-set decision
# uses the truth instead of the lying-mid-load mem_get_info value.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import body_pager as _body_pager_module
_body_pager_module.set_clean_free_hint(_CLEAN_FREE)

# very small in-process cache so picking the same Holorite twice doesn't
# re-load the whole model each chat. {manifest_path: (tok, model, paged_emb)}
_loaded_lock = threading.Lock()
_loaded = {}
_current_path: str | None = None
# Most-recent /chat stats, exposed via /stats for the lattice live view.
_last_chat_stats: dict | None = None

def fmt_mb(b): return f"{b/1_048_576:.1f} MiB"

def _load_streamed(man: dict, manifest_path: str):
    """`runtime: "streamer"` path — body weights live on disk as a chunked
    asset tree. Loads only the architecture + the embedding torus; each
    transformer layer is streamed per-forward via HoloriteStreamer.

    Required manifest fields:
      runtime: "streamer"
      asset_root: absolute path to the dir containing tree.json
      model_id: HF id for the skeleton (we use the architecture only)
      embeddings_torus: path to the torus sidecar (built by holoritify.py)
    """
    from streaming_engine import HoloriteStreamer, stream_body
    print(f"[holorite] streaming runtime — asset tree {man['asset_root']}", flush=True)
    tok = AutoTokenizer.from_pretrained(man["model_id"])
    skel = AutoModelForCausalLM.from_pretrained(
        man["model_id"], torch_dtype=DTYPE, low_cpu_mem_usage=True,
    ).eval()
    # The skeleton's BODY weights will be supplied by the streamer; we still
    # let the model load them once on CPU so the shape/structure is preserved.
    streamer = HoloriteStreamer(man["asset_root"], compute_device=DEVICE,
                                working_set=int(man.get("working_set") or 8),
                                prefetch_fanout=int(man.get("prefetch_fanout") or 8),
                                fp_dtype=DTYPE).open()
    n_streamed = stream_body(skel, streamer)
    print(f"[holorite] streamed {n_streamed} body layers from disk "
          f"(ws={streamer.working_set}, fanout={streamer.prefetch_fanout})", flush=True)
    # Paged embedding (same path as fp16 Holorites)
    emb_orig = skel.get_input_embeddings()
    full_W = emb_orig.weight.detach().to("cpu").clone()
    paged = NodePagedEmbedding(full_W, pad_token_id=man.get("pad_token_id"),
                               cpu_device="cpu", compute_device=DEVICE,
                               max_cached_nodes=NODES_TOTAL,
                               helical_prefetch=True, prefetch_fanout=8)
    skel.set_input_embeddings(paged)
    # tiny tensors stay on GPU
    for name, p in skel.named_parameters():
        if "model.norm" in name or "lm_head" in name or "model.embed_norm" in name:
            p.data = p.data.to(DEVICE)
    for name, b in skel.named_buffers():
        if "embed_tokens" in name: continue
        try: b.data = b.data.to(DEVICE)
        except Exception: pass
    full_emb_bytes = full_W.numel() * full_W.element_size()
    return tok, skel, paged, full_emb_bytes


def load_holorite(manifest_path: str):
    global _current_path
    with _loaded_lock:
        if manifest_path in _loaded: return _loaded[manifest_path]
        # evict the previous one — 4 GB GPU can't hold two LLMs.
        if _current_path is not None and _current_path != manifest_path:
            prev = _loaded.pop(_current_path, None)
            if prev is not None:
                # walk every resident-on-GPU layer back to CPU before dropping
                # the reference, otherwise the GPU storage lingers in the
                # caching allocator and starves the next model. The 7B OOM
                # we saw was the 1.5B's working-set never getting released.
                try:
                    prev_model = prev[1]
                    for sub in prev_model.modules():
                        pager = getattr(sub, "pager", None)
                        if pager is not None and hasattr(pager, "evict_all"):
                            pager.evict_all(); break
                except Exception: pass
                # explicitly drop the tuple's references so refcount hits 0
                del prev_model
            del prev
            import gc; gc.collect()
            if DEVICE == "cuda":
                # double empty_cache + ipc_collect to actually release reservations.
                # Without this the auto-int8 trigger fires on the *next* load
                # because it reads a stale low free-memory number.
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        with open(manifest_path, encoding="utf-8") as f: man = json.load(f)
        # Branch on runtime — "streamer" loads via the asset tree, everything
        # else uses the in-RAM CPU-master body_pager path.
        if man.get("runtime") == "streamer":
            _loaded[manifest_path] = _load_streamed(man, manifest_path)
            _current_path = manifest_path
            return _loaded[manifest_path]
        print(f"[holorite] loading {man['name']}  vocab={man['vocab_size']} hidden={man['hidden_dim']}", flush=True)
        tok = AutoTokenizer.from_pretrained(man["model_id"])

        # Step 1 — load the whole model on CPU (the master copy). Nothing
        # touches the GPU yet. This means even a 14 GiB body just sits in
        # system RAM; the body_pager will stream layers to the GPU one at a
        # time during forward passes. "Flowing water through a window."
        model = AutoModelForCausalLM.from_pretrained(
            man["model_id"], torch_dtype=DTYPE, low_cpu_mem_usage=True,
        ).eval()

        # Step 2 — wrap every transformer block so its weights only land on
        # the GPU during that block's own forward call, then come straight off.
        # With the side-stream prefetch enabled, the next layer's copy is
        # queued while the current layer is computing, so PCIe traffic hides
        # under compute (the brief's spiral δ applied at the layer axis).
        from body_pager import paged_body, _iter_body_layers, _layer_cpu_bytes
        # Spontaneous working set: let the system use what it needs.
        #   - small models (whole body fits): keep all layers resident, no paging churn
        #   - 7B-class (body > VRAM): auto-size the rolling window to what fits and
        #     auto-engage int8 master so PCIe transfer drops 4×
        # Prefetch fanout is still 8 (the HoloStream window on the body axis),
        # because prefetching is *free* when next-needed is well predicted.
        body_layers = _iter_body_layers(model)
        avg_layer_fp16 = sum(_layer_cpu_bytes(l) for l in body_layers) / max(len(body_layers), 1)
        n_layers = len(body_layers)
        # Use the *clean* GPU snapshot taken at server startup, not the live
        # value — `mem_get_info` reports 321 MiB mid-HF-load even when the
        # actual usable budget is ~3.5 GiB (caching-allocator stickiness from
        # the transient that `low_cpu_mem_usage` placed on the GPU and then
        # supposedly moved off). The clean snapshot is the truth.
        free_now = _CLEAN_FREE if DEVICE == "cuda" else (1 << 40)
        body_budget = max(0, free_now - 250 * 1_048_576)   # 250 MiB reserve — user's other apps already eat most of the 4 GiB
        # Explicit overrides (env or manifest) — env wins, manifest second.
        # Manifest keys: quant: "fp16"/"int8"/"int4", or legacy int8_body bool.
        env_quant = os.environ.get("HOLORITE_QUANT", "").lower()
        if env_quant in ("fp16", "int8", "int4"):
            quant = env_quant
        elif man.get("quant") in ("fp16", "int8", "int4"):
            quant = man["quant"]
        elif man.get("int8_body") is True: quant = "int8"
        elif man.get("int8_body") is False: quant = "fp16"
        else:
            # auto: pick the lightest quant that lets the whole body live in
            # `body_budget`. Aim for "fits with no paging churn" — that's how
            # we get back to GPU-compute-speed instead of being PCIe-bound.
            body_fp16 = n_layers * avg_layer_fp16
            if body_fp16 <= body_budget:
                quant = "fp16"
            elif body_fp16 / 4 <= body_budget:
                quant = "int8"
            else:
                # the 7B-on-4GiB lever: int4 cuts body 8× → 14 GiB → 1.75 GiB
                # → fits even with the user's other apps holding most of VRAM.
                quant = "int4"
        int8_master = (quant == "int8")   # legacy alias used elsewhere
        # working_set=None → auto-size in paged_body, with round-up to whole body
        # when ≥85% would fit (so small models hold everything, no churn).
        # reserve_mb=250 mirrors the budget calc above — user's GPU is shared
        # with other apps so we keep the reserve tight.
        print(f"[holorite-debug] free_now={free_now/1_048_576:.0f} MiB · "
              f"body_fp16_total={n_layers*avg_layer_fp16/1_048_576:.0f} MiB · "
              f"body_budget={body_budget/1_048_576:.0f} MiB · "
              f"quant={quant}", flush=True)
        n_wrapped, ws = paged_body(model, compute_device=DEVICE, prefetch=True,
                                   working_set=None, prefetch_fanout=8, reserve_mb=250,
                                   quant=quant)
        if ws >= n_wrapped:
            mode = f"all {n_wrapped} layers fit (no eviction)"
        else:
            mode = f"working set {ws}/{n_wrapped} layers (LRU streams the rest)"
        q_desc = {"fp16": "fp16 body master (byte-exact)",
                  "int8": "int8 body master (4× less PCIe, ~byte-exact)",
                  "int4": "int4 body master (8× less PCIe — the 4 GiB / 7B lever)"}[quant]
        print(f"[holorite] body-paged: {mode}  ·  {q_desc}", flush=True)

        # Step 3 — paged embedding on the input side.
        emb_orig = model.get_input_embeddings()
        full_W = emb_orig.weight.detach().to("cpu").clone()
        paged = NodePagedEmbedding(full_W, pad_token_id=man.get("pad_token_id"),
                                   cpu_device="cpu", compute_device=DEVICE,
                                   max_cached_nodes=NODES_TOTAL,
                                   helical_prefetch=True,
                                   prefetch_fanout=8)
        model.set_input_embeddings(paged)

        # Step 4 — the tiny, always-needed pieces (final layer norm, LM head)
        # stay on the GPU so we don't pay PCIe for them every single token.
        for name, p in model.named_parameters():
            if "model.norm" in name or "lm_head" in name or "model.embed_norm" in name:
                p.data = p.data.to(DEVICE)
            elif "embed_tokens" in name:
                continue   # paged embedding owns this
            # everything else is body (covered by paged_body) — leave on CPU
        for name, b in model.named_buffers():
            if "embed_tokens" in name: continue
            try: b.data = b.data.to(DEVICE)
            except Exception: pass
        _loaded[manifest_path] = (tok, model, paged, full_W.numel() * full_W.element_size())
        _current_path = manifest_path
        if DEVICE == "cuda":
            free, total = torch.cuda.mem_get_info()
            print(f"[holorite] {man['name']} loaded. VRAM used={fmt_mb(total-free)} free={fmt_mb(free)}/{fmt_mb(total)}", flush=True)
        return _loaded[manifest_path]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass  # quiet

    def _send_json(self, code: int, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path == "/healthz":
            return self._send_json(200, {"ok": True, "device": DEVICE,
                                         "loaded": list(_loaded.keys())})
        if self.path == "/stats":
            # Live snapshot for the companion's 13-torus lattice view.
            # We prefer the per-forward stats from the currently-loaded paged
            # embedding (updated EVERY forward, so the visualizer pulses
            # during generation, not only after the chat completes). Falls
            # back to the last-finished chat's full stats when the model is
            # idle.
            live = None
            try:
                if _current_path and _current_path in _loaded:
                    _, _, paged, full_emb_bytes = _loaded[_current_path]
                    s = getattr(paged, "last_stats", None)
                    if s is not None:
                        node_bytes_per = SLOTS * paged.embedding_dim * (
                            paged.torus.element_size() if hasattr(paged.torus, "element_size") else 2)
                        on_gpu = s.used_units * node_bytes_per
                        live = {
                            "nodes_used": s.used_units,
                            "nodes_total": s.total_units,
                            "fraction": round(s.fraction, 4),
                            "active_cells": s.active_cells or [],
                            "embed_on_gpu_mb": round(on_gpu / 1_048_576, 1),
                            "embed_full_mb": round(full_emb_bytes / 1_048_576, 1),
                            "saved_mb": round((full_emb_bytes - on_gpu) / 1_048_576, 1),
                            "tok_per_s": (_last_chat_stats or {}).get("tok_per_s"),
                            "tokens": (_last_chat_stats or {}).get("tokens"),
                            "live": True,
                        }
            except Exception: pass
            payload = {"ok": True, "device": DEVICE,
                       "loaded": list(_loaded.keys()),
                       "active_holorite": _current_path or "",
                       "last_stats": live or _last_chat_stats}
            return self._send_json(200, payload)
        return self._send_json(404, {"error": "not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            req = json.loads(raw or b"{}")
        except Exception:
            return self._send_json(400, {"error": "bad json"})
        # GGUF Holorites run inference via node-llama-cpp on the companion
        # side; they just /announce themselves so the lattice HUD knows which
        # model is active. No body inference happens here for those.
        if self.path == "/announce":
            global _current_path, _last_chat_stats
            manifest = req.get("manifest") or ""
            if manifest and os.path.exists(manifest):
                _current_path = manifest
                # also stash a barebones last_stats so the visualizer HUD
                # has something to render until a chat populates real numbers
                if _last_chat_stats is None:
                    _last_chat_stats = {"announced": True, "active_cells": []}
            return self._send_json(200, {"ok": True, "active": _current_path})
        if self.path != "/chat":
            return self._send_json(404, {"error": "not found"})
        manifest = req.get("manifest") or ""
        text = req.get("text") or ""
        max_new = int(req.get("max_new_tokens") or 160)
        system = req.get("system") or ""
        if not manifest or not os.path.exists(manifest):
            return self._send_json(400, {"error": f"manifest not found: {manifest}"})
        if not text:
            return self._send_json(400, {"error": "text is required"})
        try:
            tok, model, paged, full_emb_bytes = load_holorite(manifest)
        except Exception as e:
            traceback.print_exc()
            return self._send_json(500, {"error": f"load failed: {e}"})
        # build messages
        msgs = []
        if system: msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": text})
        try:
            res = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            # transformers 5.x returns BatchEncoding; older returns a tensor
            if hasattr(res, "input_ids"):
                ids = res.input_ids
            elif isinstance(res, torch.Tensor):
                ids = res
            elif isinstance(res, dict) and "input_ids" in res:
                ids = res["input_ids"]
            else:
                ids = torch.as_tensor(res)
        except Exception:
            ids = tok(text, return_tensors="pt").input_ids
        ids = ids.to(DEVICE)
        import time
        # Collect every "end" token the tokenizer knows about. Qwen2.5 has
        # several (<|im_end|>, <|endoftext|>, eos) and missing any of them
        # is what produced the doom-loops ("ContentLoaded ContentLoaded…",
        # " turnovers turnovers…", Chinese repeats). Multiple eos_token_id
        # values together let `generate` stop on any of them.
        eos_ids = set()
        if getattr(tok, "eos_token_id", None) is not None:
            eos_ids.add(int(tok.eos_token_id))
        for tok_name in ("<|im_end|>", "<|endoftext|>", "<|end|>", "</s>", "<|eot_id|>"):
            try:
                tid = tok.convert_tokens_to_ids(tok_name)
                if isinstance(tid, int) and tid is not None and tid >= 0 and tid != tok.unk_token_id:
                    eos_ids.add(tid)
            except Exception: pass
        eos_list = sorted(eos_ids) or [tok.eos_token_id]
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                ids,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=0.7, top_p=0.9, top_k=50,
                repetition_penalty=1.15,   # the missing piece — kills the loops
                no_repeat_ngram_size=4,    # extra belt: forbid any 4-gram from repeating
                eos_token_id=eos_list,
                pad_token_id=tok.eos_token_id,
            )
        dt = time.perf_counter() - t0
        new = out[0, ids.shape[1]:]
        reply = tok.decode(new, skip_special_tokens=True)
        s = paged.last_stats
        stats = {"tokens": int(new.numel()), "seconds": round(dt, 2),
                 "tok_per_s": round(new.numel() / max(dt, 1e-6), 2)}
        if s:
            node_bytes_per = SLOTS * paged.embedding_dim * (
                paged.torus.element_size() if hasattr(paged.torus, "element_size") else 2)
            on_gpu = s.used_units * node_bytes_per
            stats.update({"nodes_used": s.used_units, "nodes_total": s.total_units,
                          "fraction": round(s.fraction, 4),
                          "embed_on_gpu_mb": round(on_gpu / 1_048_576, 1),
                          "embed_full_mb": round(full_emb_bytes / 1_048_576, 1),
                          "saved_mb": round((full_emb_bytes - on_gpu) / 1_048_576, 1),
                          # active (ring, node) cells for the 13th-torus visualizer
                          "active_cells": s.active_cells or []})
        # _last_chat_stats was already declared global at the top of the
        # /announce branch above; re-declaring it here is a SyntaxError in
        # Python 3.12 ("name '_last_chat_stats' is used prior to global
        # declaration"). One declaration covers the whole do_POST scope.
        _last_chat_stats = stats   # noqa: F823 — global already declared above
        return self._send_json(200, {"text": reply, "stats": stats})


def main():
    print(f"[holorite] server starting at http://{HOST}:{PORT}  device={DEVICE}", flush=True)
    print(f"[holorite] POST /chat  body: {{ 'manifest': '...\\\\manifest.json', 'text': 'hi' }}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()

if __name__ == "__main__":
    main()
