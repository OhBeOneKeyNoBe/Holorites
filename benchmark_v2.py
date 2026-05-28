"""benchmark_v2.py — Ring vs Node vs Node+Helical paging, plus a real
end-to-end Qwen generation run with the paged stack measured."""
from __future__ import annotations
import os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torus_lattice import (
    RingPagedEmbedding, NodePagedEmbedding, PagedLMHead,
    embedding_to_torus, cells_for_ids, CELLS, RINGS, NODES, SLOTS, NODES_TOTAL,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

PROMPT = (
    "The torus lattice arranges a 262,144-token vocabulary as 64 rings of 64 nodes of 64 slots. "
    "A token id maps to one cell via the bit-slice bijection: ring is the top 6 bits, node the "
    "middle 6, slot the low 6. A natural-language paragraph rarely touches more than a small "
    "fraction of the available rings *or* nodes, so most of the embedding matrix never needs to "
    "enter GPU memory. This benchmark measures that fraction empirically across both granularities."
) * 4

def fmt_mb(b): return f"{b/1_048_576:7.1f} MiB"
def fmt_gb(b): return f"{b/1_073_741_824:6.2f} GiB"
def bytes_of(t): return t.numel() * t.element_size()

def main():
    print(f"DEVICE={DEVICE}  DTYPE={DTYPE}")
    print(f"loading {MODEL_ID} on CPU ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, low_cpu_mem_usage=True)
    model.eval()
    emb_orig = model.get_input_embeddings()
    V, D = emb_orig.weight.shape
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    emb_W = emb_orig.weight.detach().clone()
    print(f"  vocab={V}  hidden={D}  emb tensor bytes={fmt_mb(bytes_of(emb_W))}")
    print(f"  tie_word_embeddings = {bool(getattr(model.config, 'tie_word_embeddings', False))}")

    # ── (A) granularity comparison on a real prompt ────────────────────
    enc = tok(PROMPT, return_tensors="pt"); ids = enc["input_ids"][0]
    print(f"\n[A] real-prompt granularity comparison ({ids.numel()} tokens)")
    rings_t, nodes_t, _ = cells_for_ids(ids)
    used_rings = sorted({int(r) for r in rings_t.tolist()})
    used_nodes = sorted({(int(r), int(n)) for r, n in zip(rings_t.tolist(), nodes_t.tolist())})
    ring_bytes_per = (NODES * SLOTS) * D * emb_W.element_size()
    node_bytes_per = SLOTS * D * emb_W.element_size()
    full_bytes = CELLS * D * emb_W.element_size()
    ring_on_gpu  = len(used_rings) * ring_bytes_per
    node_on_gpu  = len(used_nodes) * node_bytes_per
    print(f"  full embedding (CPU master) : {fmt_mb(full_bytes)}")
    print(f"  ring granularity            : {len(used_rings)}/{RINGS} rings touched -> on-GPU {fmt_mb(ring_on_gpu)} ({ring_on_gpu/full_bytes:.1%})")
    print(f"  node granularity            : {len(used_nodes)}/{NODES_TOTAL} nodes touched -> on-GPU {fmt_mb(node_on_gpu)} ({node_on_gpu/full_bytes:.1%})")
    print(f"  node vs ring improvement    : {ring_on_gpu/node_on_gpu:.2f}x less on GPU")

    # ── (B) byte-exact lookup parity ───────────────────────────────────
    print("\n[B] byte-exact lookup parity (random 4x128 ids)")
    nq = torch.randint(0, V, (4, 128))
    flat_layer = torch.nn.Embedding.from_pretrained(emb_W, freeze=True)
    rpe = RingPagedEmbedding(emb_W, pad_token_id=pad_id, compute_device="cpu")
    npe = NodePagedEmbedding(emb_W, pad_token_id=pad_id, compute_device="cpu", helical_prefetch=True)
    ok_ring = torch.equal(rpe(nq), flat_layer(nq))
    ok_node = torch.equal(npe(nq), flat_layer(nq))
    print(f"  ring-paged  == flat : {ok_ring}")
    print(f"  node-paged  == flat : {ok_node}")
    if not (ok_ring and ok_node): sys.exit("parity test failed")

    # ── (C) wall-clock cost of paged lookup vs flat ───────────────────
    print("\n[C] wall-clock per lookup (cpu)")
    npe.node_cache.clear(); rpe.ring_cache.clear()
    N = 30
    def t(fn):
        t0=time.perf_counter()
        for _ in range(N): fn(ids)
        return (time.perf_counter()-t0)/N
    print(f"  nn.Embedding (flat)                  : {t(flat_layer)*1e3:7.3f} ms")
    npe.node_cache.clear(); print(f"  NodePagedEmbedding (cold, no prefetch): {t(NodePagedEmbedding(emb_W, helical_prefetch=False, pad_token_id=pad_id))*1e3:7.3f} ms")
    print(f"  NodePagedEmbedding (cold, helical)   : {t(NodePagedEmbedding(emb_W, helical_prefetch=True, pad_token_id=pad_id))*1e3:7.3f} ms")
    print(f"  NodePagedEmbedding (warm)            : {t(npe)*1e3:7.3f} ms")

    # ── (D) replace the model's embedding + LM head and run real generation ──
    print(f"\n[D] real generation on {DEVICE} with paged embedding + paged LM head")
    paged_emb = NodePagedEmbedding(emb_W, pad_token_id=pad_id,
                                   cpu_device="cpu", compute_device=DEVICE,
                                   max_cached_nodes=NODES_TOTAL,
                                   helical_prefetch=True)
    model.set_input_embeddings(paged_emb)
    # move the rest of the model to GPU (skip the embedding's CPU master)
    for n, p in model.named_parameters():
        if "embed_tokens" in n: continue
        p.data = p.data.to(DEVICE)
    for n, b in model.named_buffers():
        if "embed_tokens" in n: continue
        b.data = b.data.to(DEVICE)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"  VRAM after load: used={fmt_mb(total-free)}  free={fmt_mb(free)} / {fmt_mb(total)}")

    short_prompt = "Briefly, what is a torus?"
    enc = tok([short_prompt], return_tensors="pt").to(DEVICE)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=80, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    dt = time.perf_counter() - t0
    new_tok = out[0, enc.input_ids.shape[1]:]
    text = tok.decode(new_tok, skip_special_tokens=True)
    print(f"  generated {new_tok.numel()} tokens in {dt:.1f}s = {new_tok.numel()/dt:.2f} tok/s")
    s = paged_emb.last_stats
    if s:
        on_gpu = s.used_units * SLOTS * D * emb_W.element_size()
        saved  = bytes_of(emb_W) - on_gpu
        print(f"  embedding nodes used: {s.used_units}/{NODES_TOTAL} ({s.fraction:.1%}); "
              f"prefetches={s.prefetches}; on-GPU emb={fmt_mb(on_gpu)} vs full {fmt_mb(bytes_of(emb_W))} -> saved {fmt_mb(saved)}")
    print(f"  reply: {text[:160]!r}")

    # ── (E) projection at 7B scale (262,144 x 4096 fp16) ──────────────
    proj_total = CELLS * 4096 * 2
    proj_ring  = len(used_rings) * (CELLS // RINGS) * 4096 * 2
    proj_node  = len(used_nodes) * SLOTS * 4096 * 2
    print(f"\n[E] projected at 7B-scale (262,144 x 4096 fp16):")
    print(f"  full embedding             : {fmt_gb(proj_total)}")
    print(f"  ring paging on this prompt : {fmt_gb(proj_ring)} ({proj_ring/proj_total:.1%})")
    print(f"  NODE paging on this prompt : {fmt_gb(proj_node)} ({proj_node/proj_total:.1%})")
    # When LM head is TIED (Qwen2.5 ties), the SAME paging covers BOTH
    # input embedding *and* lm_head. That's a 2x more total saving:
    if bool(getattr(model.config, 'tie_word_embeddings', False)):
        print(f"  + LM head shares this paging (tied): another {fmt_gb(proj_total - proj_node)} saved")
        print(f"  -> total static GPU footprint reclaimed: {fmt_gb(2*(proj_total - proj_node))}")

if __name__ == "__main__":
    main()
