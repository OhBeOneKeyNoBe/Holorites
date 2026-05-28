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

# very small in-process cache so picking the same Holorite twice doesn't
# re-load the whole model each chat. {manifest_path: (tok, model, paged_emb)}
_loaded_lock = threading.Lock()
_loaded = {}
_current_path: str | None = None

def fmt_mb(b): return f"{b/1_048_576:.1f} MiB"

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
            import gc; gc.collect()
            if DEVICE == "cuda": torch.cuda.empty_cache()
        with open(manifest_path, encoding="utf-8") as f: man = json.load(f)
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
        from body_pager import paged_body
        n_wrapped, ws = paged_body(model, compute_device=DEVICE, prefetch=True,
                                   prefetch_fanout=8, reserve_mb=700)
        if ws >= n_wrapped:
            mode = f"all {n_wrapped} layers fit (no eviction)"
        else:
            mode = f"working set {ws}/{n_wrapped} layers (LRU streams the rest)"
        print(f"[holorite] body-paged: {mode}", flush=True)

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
        return self._send_json(404, {"error": "not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            req = json.loads(raw or b"{}")
        except Exception:
            return self._send_json(400, {"error": "bad json"})
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
                          "saved_mb": round((full_emb_bytes - on_gpu) / 1_048_576, 1)})
        return self._send_json(200, {"text": reply, "stats": stats})


def main():
    print(f"[holorite] server starting at http://{HOST}:{PORT}  device={DEVICE}", flush=True)
    print(f"[holorite] POST /chat  body: {{ 'manifest': '...\\\\manifest.json', 'text': 'hi' }}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()

if __name__ == "__main__":
    main()
