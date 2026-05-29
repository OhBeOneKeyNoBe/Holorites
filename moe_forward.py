"""moe_forward.py — Qwen3-MoE forward through the streamer.

Powers actual computation, not just admit experts. Loads non-expert weights
one-shot from the GGUF mmap (attention/norms/embed/lm_head), then per layer
per token:

  1. Pre-attention RMS norm + GQA attention (32 query heads, 4 KV heads,
     128 head dim, RoPE θ = 10,000,000) with KV cache.
  2. Residual.
  3. Pre-FFN RMS norm.
  4. Router gate matmul + top-k=8 + softmax.
  5. streamer.route(unique_experts) — batched admit.
  6. Per-token per-expert: gate_proj × silu × up_proj → down_proj.
  7. Weighted sum by router probs + residual.

Multi-layer sequential generation supported via KVCache + generate_tokens.
"""
from __future__ import annotations
import os, sys, time, math
from pathlib import Path
from dataclasses import dataclass
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


# ─── RoPE (rotary position embeddings, Qwen3 style: θ = 10,000,000) ──────

def build_rope_freqs(head_dim: int, max_pos: int, theta: float, device, dtype=torch.float32):
    """Return cos/sin tables of shape (max_pos, head_dim) for RoPE."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    t = torch.arange(max_pos, device=device, dtype=dtype)
    angles = torch.outer(t, freqs)                # (max_pos, half)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               positions: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to x. x shape: (T, n_heads, head_dim).
    cos/sin: (max_pos, head_dim). positions: (T,)."""
    c = cos[positions]    # (T, head_dim)
    s = sin[positions]
    c = c.unsqueeze(1)    # (T, 1, head_dim)
    s = s.unsqueeze(1)
    # split into halves; rotate
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * c + rotated * s).to(x.dtype)


# ─── GQA attention (Qwen3 layout) ──────────────────────────────────────────

def qwen3_attention(hidden: torch.Tensor,
                     attn_q_w: torch.Tensor, attn_k_w: torch.Tensor,
                     attn_v_w: torch.Tensor, attn_o_w: torch.Tensor,
                     q_norm: torch.Tensor, k_norm: torch.Tensor,
                     rope_cos: torch.Tensor, rope_sin: torch.Tensor,
                     positions: torch.Tensor,
                     kv_cache: dict,
                     layer_idx: int,
                     n_heads: int, n_kv_heads: int, head_dim: int,
                     causal: bool = True) -> torch.Tensor:
    """One GQA attention layer's forward (Qwen3 layout).

    Shapes:
      hidden:    (T, hidden_dim)
      attn_q_w:  (n_heads*head_dim, hidden_dim)   — Linear weight, output × input
      attn_k_w:  (n_kv_heads*head_dim, hidden_dim)
      attn_v_w:  (n_kv_heads*head_dim, hidden_dim)
      attn_o_w:  (hidden_dim, n_heads*head_dim)
      q_norm:    (head_dim,) — per-head Q RMS norm weight (Qwen3 specific)
      k_norm:    (head_dim,) — per-head K RMS norm weight
      kv_cache:  dict keyed by layer_idx → {"k": Tensor, "v": Tensor} appended each step
      positions: (T,) absolute token positions

    Returns attention output (T, hidden_dim) BEFORE the residual.
    """
    T = hidden.shape[0]
    hidden_dim = hidden.shape[-1]
    # project
    q = (hidden @ attn_q_w.T).view(T, n_heads, head_dim)
    k = (hidden @ attn_k_w.T).view(T, n_kv_heads, head_dim)
    v = (hidden @ attn_v_w.T).view(T, n_kv_heads, head_dim)
    # per-head Q/K RMS norm (Qwen3 detail)
    q = rms_norm(q, q_norm)
    k = rms_norm(k, k_norm)
    # RoPE on Q and K
    q = apply_rope(q, rope_cos, rope_sin, positions)
    k = apply_rope(k, rope_cos, rope_sin, positions)
    # append to KV cache
    if layer_idx not in kv_cache:
        kv_cache[layer_idx] = {"k": k, "v": v}
    else:
        kv_cache[layer_idx]["k"] = torch.cat([kv_cache[layer_idx]["k"], k], dim=0)
        kv_cache[layer_idx]["v"] = torch.cat([kv_cache[layer_idx]["v"], v], dim=0)
    k_full = kv_cache[layer_idx]["k"]    # (T_cache, n_kv_heads, head_dim)
    v_full = kv_cache[layer_idx]["v"]
    # GQA: expand KV heads to match Q heads via repeat
    if n_kv_heads < n_heads:
        rep = n_heads // n_kv_heads
        k_full = k_full.repeat_interleave(rep, dim=1)
        v_full = v_full.repeat_interleave(rep, dim=1)
    # attention: (T, n_heads, head_dim) @ (T_cache, n_heads, head_dim) -> (n_heads, T, T_cache)
    q_p = q.permute(1, 0, 2)    # (n_heads, T, head_dim)
    k_p = k_full.permute(1, 0, 2)  # (n_heads, T_cache, head_dim)
    v_p = v_full.permute(1, 0, 2)
    scores = q_p @ k_p.transpose(-2, -1) / math.sqrt(head_dim)
    # causal mask for the prefill case (T > 1); single-step inference has T=1
    if causal and T > 1:
        T_cache = k_full.shape[0]
        # positions of Q tokens (relative to the cache) are the last T positions
        mask = torch.full((T, T_cache), float("-inf"), device=scores.device, dtype=scores.dtype)
        q_start = T_cache - T
        for i in range(T):
            mask[i, : q_start + i + 1] = 0
        scores = scores + mask
    weights = torch.softmax(scores.to(torch.float32), dim=-1).to(scores.dtype)
    out = weights @ v_p   # (n_heads, T, head_dim)
    out = out.permute(1, 0, 2).contiguous().view(T, n_heads * head_dim)
    return (out @ attn_o_w.T).to(hidden.dtype)


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

    # Grouped expert GEMM (the DeepGEMM `m_grouped_fp8_gemm_*` pattern):
    # for each token, the k routed experts are stacked into a (k, ffn_dim,
    # hidden) batch and the per-token activations are batched-matmul'd
    # against them in three single calls instead of 3·k·T Python-loop
    # matmuls. Each torch.bmm is a single CUDA launch — far less overhead
    # than the per-iteration .T + silu + dict-lookup chain.
    t0 = time.perf_counter()
    out = torch.zeros_like(hidden)
    # T=1 fast path — the common case for decode-time generation. For
    # prefill (T>1) we fall through to the per-token version below; this
    # keeps the math simple while still capturing 90%+ of inference time.
    if T == 1:
        # For this single token, k experts route to it
        active = [int(topk_idx[0, j].item()) for j in range(k)]
        gate_ws = torch.stack([expert_weights[e]["ffn_gate.weight"] for e in active])
        up_ws   = torch.stack([expert_weights[e]["ffn_up.weight"]   for e in active])
        down_ws = torch.stack([expert_weights[e]["ffn_down.weight"] for e in active])
        x_batch = x.expand(k, 1, -1)                                # (k, 1, hidden)
        # Three batched matmuls — one CUDA kernel each, vs 24 separate launches
        gate_b = torch.bmm(x_batch, gate_ws.transpose(-2, -1))      # (k, 1, ffn_dim)
        up_b   = torch.bmm(x_batch, up_ws.transpose(-2, -1))        # (k, 1, ffn_dim)
        hidden_b = torch.nn.functional.silu(gate_b) * up_b          # (k, 1, ffn_dim)
        expert_out_b = torch.bmm(hidden_b, down_ws.transpose(-2, -1))  # (k, 1, hidden)
        # Weighted sum over k experts: probs (k,) × outputs (k, hidden)
        weights = topk_probs[0].unsqueeze(-1)                       # (k, 1)
        out[0] = (expert_out_b.squeeze(1) * weights).sum(dim=0)
    else:
        # Prefill / multi-token path — fall back to per-token loop. Could be
        # batched too, but routing creates a sparse pattern that's annoying
        # to express in dense matmuls; left for follow-up optimization.
        for tok_i in range(T):
            active = [int(topk_idx[tok_i, j].item()) for j in range(k)]
            gate_ws = torch.stack([expert_weights[e]["ffn_gate.weight"] for e in active])
            up_ws   = torch.stack([expert_weights[e]["ffn_up.weight"]   for e in active])
            down_ws = torch.stack([expert_weights[e]["ffn_down.weight"] for e in active])
            xi = x[tok_i:tok_i+1].expand(k, 1, -1)
            gate_b = torch.bmm(xi, gate_ws.transpose(-2, -1))
            up_b   = torch.bmm(xi, up_ws.transpose(-2, -1))
            hidden_b = torch.nn.functional.silu(gate_b) * up_b
            expert_out_b = torch.bmm(hidden_b, down_ws.transpose(-2, -1))
            weights = topk_probs[tok_i].unsqueeze(-1)
            out[tok_i] = (expert_out_b.squeeze(1) * weights).sum(dim=0)
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


@dataclass
class LoadedQwen3MoE:
    """Holds the dequant'd-once non-expert weights for ALL layers + globals."""
    tree: MoEAssetTree
    device: torch.device
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    n_routed: int
    k: int
    rms_eps: float
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    # per-layer (in lists)
    attn_q_w:    list
    attn_k_w:    list
    attn_v_w:    list
    attn_o_w:    list
    q_norm:      list
    k_norm:      list
    attn_norm_w: list
    ffn_norm_w:  list
    router_gate: list
    # globals
    embed_w:     torch.Tensor
    output_norm: torch.Tensor
    output_w:    torch.Tensor


def load_qwen3_moe(gguf_path: str, device: str = "cuda") -> LoadedQwen3MoE:
    """One-shot dequant of every non-expert weight. Held in GPU memory.
    For Qwen3-Coder-30B: 48 layers × (Q/K/V/O + 2 norms + router) ≈ 5 GiB
    fp16 on GPU once at load. Expert FFNs stream from disk per token."""
    print(f"[load] indexing {gguf_path}")
    tree = MoEAssetTree(gguf_path)
    print(f"  arch={tree.arch}, layers={tree.n_layers}, "
          f"n_routed={tree.n_routed}, k={tree.experts_per_tok}")
    # Qwen3 specific constants (from blk.0 metadata)
    n_heads = 32; n_kv_heads = 4; head_dim = 128; rms_eps = 1e-6
    rope_theta = 1e7; max_pos = 16384
    rope_cos, rope_sin = build_rope_freqs(head_dim, max_pos, rope_theta,
                                           torch.device(device), torch.float32)
    rope_cos = rope_cos.to(torch.float16); rope_sin = rope_sin.to(torch.float16)

    def load_w(t): return _dequant_full_tensor(t, device).to(torch.float16)

    attn_q_w, attn_k_w, attn_v_w, attn_o_w = [], [], [], []
    q_norm, k_norm, attn_norm_w, ffn_norm_w, router_gate = [], [], [], [], []
    print(f"[load] dequanting {tree.n_layers} layers' non-expert weights…")
    for li in sorted(tree.layers.keys()):
        layer = tree.layers[li]
        T = layer.other_tensors
        attn_q_w.append(load_w(T["attn_q.weight"]).view(n_heads * head_dim, -1).contiguous())
        attn_k_w.append(load_w(T["attn_k.weight"]).view(n_kv_heads * head_dim, -1).contiguous())
        attn_v_w.append(load_w(T["attn_v.weight"]).view(n_kv_heads * head_dim, -1).contiguous())
        attn_o_w.append(load_w(T["attn_output.weight"]).view(-1, n_heads * head_dim).contiguous())
        q_norm.append(load_w(T["attn_q_norm.weight"]))
        k_norm.append(load_w(T["attn_k_norm.weight"]))
        attn_norm_w.append(load_w(T["attn_norm.weight"]))
        ffn_norm_w.append(load_w(T["ffn_norm.weight"]))
        router_gate.append(load_w(T["ffn_gate_inp.weight"]).view(tree.n_routed, -1).contiguous())
        if (li + 1) % 8 == 0:
            print(f"  layers 0..{li} loaded")
    # globals: embedding (Q4_K, 128256×7168), output (LM head, Q6_K), output_norm
    print(f"[load] embedding + LM head + output norm…")
    embed_w = load_w(tree.globals["token_embd.weight"]).view(tree.vocab_size, tree.hidden_dim).contiguous()
    output_w = load_w(tree.globals["output.weight"]).view(tree.vocab_size, tree.hidden_dim).contiguous()
    output_norm = load_w(tree.globals["output_norm.weight"])
    print(f"  embedding: {tuple(embed_w.shape)}, output: {tuple(output_w.shape)}")
    return LoadedQwen3MoE(
        tree=tree, device=torch.device(device),
        hidden_dim=tree.hidden_dim, n_layers=tree.n_layers,
        n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
        n_routed=tree.n_routed, k=tree.experts_per_tok, rms_eps=rms_eps,
        rope_cos=rope_cos, rope_sin=rope_sin,
        attn_q_w=attn_q_w, attn_k_w=attn_k_w, attn_v_w=attn_v_w, attn_o_w=attn_o_w,
        q_norm=q_norm, k_norm=k_norm,
        attn_norm_w=attn_norm_w, ffn_norm_w=ffn_norm_w, router_gate=router_gate,
        embed_w=embed_w, output_norm=output_norm, output_w=output_w,
    )


def forward_one_token(model: LoadedQwen3MoE, token_id: int,
                       streamers: dict, kv_cache: dict, position: int
                       ) -> torch.Tensor:
    """Full Qwen3-MoE forward for ONE token at the given position.
    Returns the logits over vocab (vocab_size,)."""
    x = model.embed_w[token_id].view(1, model.hidden_dim).contiguous()
    positions = torch.tensor([position], device=model.device)
    for li in range(model.n_layers):
        # pre-attention norm + attention
        x_norm = rms_norm(x, model.attn_norm_w[li], model.rms_eps)
        attn_out = qwen3_attention(
            x_norm, model.attn_q_w[li], model.attn_k_w[li],
            model.attn_v_w[li], model.attn_o_w[li],
            model.q_norm[li], model.k_norm[li],
            model.rope_cos, model.rope_sin, positions, kv_cache, li,
            model.n_heads, model.n_kv_heads, model.head_dim, causal=False)
        x = x + attn_out
        # pre-FFN norm + MoE block
        s = streamers.get(li)
        if s is None:
            from moe_streamer import ExpertStreamer, CacheTiers
            s = ExpertStreamer(model.tree, layer=li, tiers=CacheTiers(hot=64, warm=128),
                                compute_device=model.device)
            streamers[li] = s
        moe_out = moe_layer_forward(
            x, li, model.tree, s, model.ffn_norm_w[li],
            model.router_gate[li], k=model.k)
        x = x + moe_out
    x = rms_norm(x, model.output_norm, model.rms_eps)
    return (x @ model.output_w.T).view(-1)


def generate_tokens(model: LoadedQwen3MoE, prompt_ids: list, n_new: int
                     ) -> tuple[list, float]:
    """Greedy generation of n_new tokens. Returns (token_ids, wall_seconds)."""
    streamers, kv_cache = {}, {}
    output = list(prompt_ids)
    print(f"[gen] prefilling {len(prompt_ids)} prompt tokens…")
    t0 = time.perf_counter()
    for pos, tid in enumerate(prompt_ids):
        _ = forward_one_token(model, tid, streamers, kv_cache, pos)
    t_prefill = time.perf_counter() - t0
    print(f"  prefill done in {t_prefill:.1f}s")
    print(f"[gen] generating {n_new} tokens…")
    t0 = time.perf_counter()
    for step in range(n_new):
        last = output[-1]
        logits = forward_one_token(model, last, streamers, kv_cache, len(output))
        next_id = int(logits.argmax().item())
        output.append(next_id)
        if (step + 1) % 5 == 0:
            dt = time.perf_counter() - t0
            print(f"  {step+1} tokens in {dt:.1f}s ({(step+1)/dt:.2f} tok/s)")
    return output, time.perf_counter() - t0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:  py moe_forward.py <moe.gguf> [--single-layer | --generate N=10]")
        sys.exit(1)
    if "--single-layer" in sys.argv or len(sys.argv) < 3:
        layer_idx = 0; T = 4
        run_one_layer_demo(sys.argv[1], layer_idx, T)
    else:
        n_new = int(sys.argv[2]) if sys.argv[2].isdigit() else 10
        model = load_qwen3_moe(sys.argv[1])
        # use BOS token = 151643 for Qwen3 + greedy generate
        prompt = [151643]
        out, dt = generate_tokens(model, prompt, n_new)
        print(f"\n[gen] DONE in {dt:.1f}s ({n_new/dt:.2f} tok/s)")
        print(f"  generated token ids: {out[1:]}")
