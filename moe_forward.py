"""moe_forward.py — minimal end-to-end Qwen3-MoE forward through the streamer.

This is the proof that the streamer infrastructure powers actual computation,
not just admit experts. Loads the non-expert weights one-shot from the GGUF
mmap (attention/norms/embed/lm_head), then per layer per token:

  1. RMS-norm the hidden state
  2. Compute router gate logits (linear matmul with the ffn_gate_inp.weight)
  3. Top-k selection (k=8 for Qwen3-MoE)
  4. streamer.route(top_experts) — single batched admit (DeepGEMM pattern)
  5. For each selected expert: gate_proj × silu × up_proj → down_proj
  6. Weighted sum of expert outputs by router softmax probabilities
  7. Residual + return

The forward measures wall time per layer so we can show the
streamer-driven path actually works end-to-end and what's hot.

For the prototype we implement ONE LAYER end-to-end (the MoE block), with
the attention path simplified to a single matmul + RMS-norm. A full Qwen3
forward would extend this with proper GQA + RoPE + KV caching — the
streamer infrastructure doesn't change; only more wiring code does.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_streamer import (MoEAssetTree, ExpertStreamer, CacheTiers,
                           RoutingHistogram)
import moe_kernels as mk
from gguf import GGMLQuantizationType


# ─── one-shot dequant for non-expert tensors (attention, norms, gates) ───

def _dequant_full_tensor(tensor, device):
    """Read a ReaderTensor's bytes from the mmap and dequant the whole
    thing on GPU. Used for tensors loaded once at model init: attention
    projections, layer norms, the router gate weight, embedding table, etc.
    """
    n_elem = 1
    for d in tensor.shape: n_elem *= int(d)
    blob = bytes(np.asarray(tensor.data).tobytes())
    return mk.dequant_gpu(tensor.tensor_type, blob, n_elem, device)


# ─── RMS norm (the only "weight + math" detail we need beyond the streamer) ──

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """LlamaRMSNorm / Qwen2RMSNorm. weight has shape (hidden,)."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return (x.to(orig_dtype) * weight)


# ─── the MoE layer forward (the streamer is the engine here) ─────────────

def moe_layer_forward(hidden: torch.Tensor,
                       layer_idx: int,
                       tree: MoEAssetTree,
                       streamer: ExpertStreamer,
                       ffn_norm_weight: torch.Tensor,
                       router_gate_weight: torch.Tensor,
                       *,
                       k: int = 8,
                       histogram: RoutingHistogram | None = None,
                       trace: dict | None = None) -> torch.Tensor:
    """One MoE layer's forward through the streamer.

    Inputs:
      hidden               (T, hidden)  — pre-MoE hidden state
      ffn_norm_weight      (hidden,)    — RMS-norm weights (one-shot loaded)
      router_gate_weight   (n_routed, hidden) — gate linear weight
                                          (note: GGUF stores it transposed
                                           as (hidden, n_routed) so we
                                           expect the caller to have done
                                           the transpose)
    Outputs:
      out (T, hidden) — MoE block output (NOT including the residual; the
                         caller adds skip connection)
    """
    T, hidden_dim = hidden.shape
    rms_norm_eps = 1e-6
    device = hidden.device

    # pre-FFN RMS norm
    t0 = time.perf_counter()
    x = rms_norm(hidden, ffn_norm_weight, rms_norm_eps)
    if trace is not None: trace["rmsnorm"] = (time.perf_counter() - t0) * 1000

    # router gate: (T, hidden) @ (hidden, n_routed) -> (T, n_routed)
    t0 = time.perf_counter()
    gate_logits = (x.to(torch.float32) @ router_gate_weight.to(torch.float32).T).to(torch.float16)
    # top-k experts per token
    topk_vals, topk_idx = torch.topk(gate_logits, k, dim=-1)
    # softmax over the top-k slots only (Qwen3-MoE behavior)
    topk_probs = torch.softmax(topk_vals.to(torch.float32), dim=-1).to(torch.float16)
    if trace is not None: trace["router"] = (time.perf_counter() - t0) * 1000

    # Which experts are needed across ALL tokens in this layer's batch?
    unique_eids = sorted(set(topk_idx.flatten().tolist()))
    if trace is not None: trace["n_unique_experts"] = len(unique_eids)

    # ── streamer batched admit (DeepEP / DeepGEMM pattern) ──
    t0 = time.perf_counter()
    expert_weights = streamer.route(unique_eids)
    if trace is not None: trace["admit"] = (time.perf_counter() - t0) * 1000

    # Update histogram for anticipatory prefetch on the NEXT layer/token
    if histogram is not None:
        histogram.update(layer_idx, unique_eids)

    # Compute each expert's contribution. Naive loop here; a full
    # implementation would use grouped GEMM (DeepGEMM `m_grouped_fp8_gemm_*`)
    # to fuse the per-expert matmuls.
    t0 = time.perf_counter()
    out = torch.zeros_like(hidden)
    for k_slot in range(k):
        for tok_i in range(T):
            eid = int(topk_idx[tok_i, k_slot].item())
            p   = topk_probs[tok_i, k_slot]
            w = expert_weights[eid]
            # GGUF stores weights as (output_dim, input_dim) for Linear; PyTorch
            # uses input @ W.T. The per-expert dequanted shape from our streamer
            # is (ffn_dim, hidden) for gate/up and (hidden, ffn_dim) for down.
            gate_w = w["ffn_gate.weight"]                          # (ffn_dim, hidden)
            up_w   = w["ffn_up.weight"]                            # (ffn_dim, hidden)
            down_w = w["ffn_down.weight"]                          # (hidden, ffn_dim)
            xi = x[tok_i].unsqueeze(0)                             # (1, hidden)
            # Streamer hands back (output, input) layouts; PyTorch matmul
            # consumes x @ W.T to produce (1, output_dim).
            gate_out = torch.nn.functional.silu(xi @ gate_w.T)     # (1, ffn_dim)
            up_out   = xi @ up_w.T                                  # (1, ffn_dim)
            ffn_out  = (gate_out * up_out) @ down_w.T              # (1, hidden)
            out[tok_i] = out[tok_i] + (p * ffn_out.squeeze(0))
    if trace is not None: trace["compute"] = (time.perf_counter() - t0) * 1000
    return out


# ─── CLI: run one full MoE layer end-to-end and report timing ────────────

def run_one_layer_demo(gguf_path: str, layer_idx: int = 0, T: int = 4):
    """Demonstrate: load layer 0's non-expert weights one-shot, then run
    a MoE forward through the streamer on T synthetic tokens. Reports
    wall time for each phase.
    """
    print(f"[moe-forward] indexing {gguf_path}")
    tree = MoEAssetTree(gguf_path)
    print(f"  arch={tree.arch}, layers={tree.n_layers}, "
          f"n_routed={tree.n_routed}, k={tree.experts_per_tok}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device={device}")

    # One-shot dequant of the non-expert layer-0 weights.
    layer = tree.layers[layer_idx]
    print(f"\n[init] dequanting non-expert weights for layer {layer_idx}…")
    t0 = time.perf_counter()
    ffn_norm = _dequant_full_tensor(layer.other_tensors["ffn_norm.weight"], device).to(torch.float16)
    router_gate = _dequant_full_tensor(layer.other_tensors["ffn_gate_inp.weight"], device)
    n_routed = tree.n_routed
    router_gate = router_gate.view(n_routed, -1).to(torch.float16)
    t_init = (time.perf_counter() - t0) * 1000
    print(f"  ffn_norm: shape={tuple(ffn_norm.shape)}, dtype={ffn_norm.dtype}")
    print(f"  router_gate: shape={tuple(router_gate.shape)}, dtype={router_gate.dtype}")
    print(f"  one-shot init: {t_init:.1f} ms")

    # Streamer + routing histogram. The tier-0 budget is the biggest
    # dial: too small forces cache thrashing (rotating experts through
    # GPU for each token); too large eats VRAM that attention/KV needs.
    # 64 routed experts fits comfortably on a 4 GiB card (~2 MiB each
    # at fp16 for Qwen3-Coder).
    hist = RoutingHistogram(n_layers=tree.n_layers, n_experts=n_routed)
    streamer = ExpertStreamer(tree, layer=layer_idx,
                              tiers=CacheTiers(hot=64, warm=128),
                              compute_device=device, histogram=hist)
    print(f"  streamer ready: tier 0 capacity={64}, tier 1={128}")

    # Build a synthetic hidden state (T tokens × hidden_dim)
    hidden_dim = tree.hidden_dim
    torch.manual_seed(0)
    hidden = torch.randn(T, hidden_dim, dtype=torch.float16, device=device)
    print(f"\n[forward] running MoE layer on {T} synthetic tokens...")

    # Warm up CUDA kernels (the first dequant is slow due to JIT compilation
    # of the elementwise tensor ops in moe_kernels.dequant_q4_K_gpu). After
    # this, every subsequent dequant runs at the GPU's steady-state rate.
    print("  warmup pass...")
    _ = streamer.route([0, 1, 2, 3])

    # Cold pass — all experts miss cache, hit NVMe + dequant + matmul
    trace_cold = {}
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    out_cold = moe_layer_forward(hidden, layer_idx, tree, streamer,
                                  ffn_norm, router_gate,
                                  k=tree.experts_per_tok,
                                  histogram=hist, trace=trace_cold)
    torch.cuda.synchronize() if device == "cuda" else None
    t_cold = (time.perf_counter() - t0) * 1000
    print(f"\n  COLD pass: {t_cold:.0f} ms")
    for k, v in trace_cold.items(): print(f"    {k:20s}: {v}")

    # Warm pass — re-route same tokens, all experts now tier-0 resident
    trace_warm = {}
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    out_warm = moe_layer_forward(hidden, layer_idx, tree, streamer,
                                  ffn_norm, router_gate,
                                  k=tree.experts_per_tok,
                                  histogram=hist, trace=trace_warm)
    torch.cuda.synchronize() if device == "cuda" else None
    t_warm = (time.perf_counter() - t0) * 1000
    print(f"\n  WARM pass: {t_warm:.0f} ms (cache reuse)")
    for k, v in trace_warm.items(): print(f"    {k:20s}: {v}")

    print(f"\n  streamer stats: {streamer.stats()}")

    # Project per-token full-forward time
    n_layers = tree.n_layers
    cold_per_token = t_cold * n_layers
    warm_per_token = t_warm * n_layers
    print(f"\n  projection: T={T}, n_layers={n_layers}")
    print(f"    full-forward COLD: {cold_per_token/1000:.1f} s / T tokens "
          f"({cold_per_token/T/1000:.1f} s/token, {T*1000/cold_per_token:.2f} tok/s)")
    print(f"    full-forward WARM: {warm_per_token/1000:.1f} s / T tokens "
          f"({warm_per_token/T/1000:.1f} s/token, {T*1000/warm_per_token:.2f} tok/s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:  py moe_forward.py <moe.gguf> [layer_idx=0] [T=4]")
        sys.exit(1)
    layer_idx = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
    T = int(sys.argv[3]) if len(sys.argv) >= 4 else 4
    run_one_layer_demo(sys.argv[1], layer_idx, T)
