"""nan_probe.py — find where the streaming forward first produces NaN.

Runs one full forward pass on a single token with NaN checks inserted
after every consequential step:

    embedding              -> x_emb
    layer L pre-attention norm        -> x_attn_norm
    layer L attention out             -> attn_out
    layer L residual after attention  -> x_post_attn
    layer L pre-MoE norm + gate       -> gate_logits
    layer L MoE forward out           -> moe_out
    layer L residual after MoE        -> x_post_moe
    final norm + LM head              -> logits

For each step we report:
    fraction of NaN entries, fraction of inf entries,
    min/max of finite entries, mean abs.

The FIRST step where NaN appears is the culprit. Usually:
    - NaN at layer-0 attn  -> Q4_K dequant produced inf/NaN
    - NaN after softmax    -> attention all-masked row
    - NaN at RMS norm      -> zero-variance input
    - NaN at gate          -> matmul with NaN router_gate
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vram_budget import apply_expandable_segments
apply_expandable_segments()

import math
import torch
from moe_forward import (load_qwen3_moe, rms_norm, qwen3_attention,
                          moe_layer_forward, _trace_set_token)


def _stats(t: torch.Tensor) -> str:
    if t is None:
        return "(none)"
    t = t.detach().float()
    n_total = t.numel()
    n_nan = int(torch.isnan(t).sum().item())
    n_inf = int(torch.isinf(t).sum().item())
    finite = t[torch.isfinite(t)]
    if finite.numel() == 0:
        return f"NaN={n_nan}/{n_total} inf={n_inf}/{n_total} (no finite values!)"
    return (f"NaN={n_nan}/{n_total} inf={n_inf}/{n_total} "
            f"min={float(finite.min()):+.3e} max={float(finite.max()):+.3e} "
            f"mean|x|={float(finite.abs().mean()):.3e}")


def probe(gguf_path: str, token_id: int = 9707, position: int = 0,
          stop_at_first_nan: bool = True, max_layer: int = -1,
          prefill_prompt: list = None):
    """
    If `prefill_prompt` is set, prefill those tokens silently, then
    instrument the token at the END (mirroring heart_shape.py's scenario
    where the bug appears at the first generated token after prefill).
    """
    print(f"[nan-probe] gguf:  {gguf_path}")
    print(f"[nan-probe] token: {token_id}, position: {position}")
    if prefill_prompt:
        print(f"[nan-probe] prefill prompt: {prefill_prompt}")
    model = load_qwen3_moe(gguf_path)
    print(f"[nan-probe] model loaded; n_layers={model.n_layers}")

    streamers, kv_cache = {}, {}
    from moe_streamer import ExpertStreamer, CacheTiers
    from vram_budget import plan_budget, OUTER_PROFILES
    arch_nonexp, mb_each, _ = OUTER_PROFILES["qwen3moe_coder_q4_k_m"]
    plan = plan_budget(outer_nonexpert_mb=arch_nonexp, heart_resident_mb=0,
                       expert_mb_each=mb_each, n_layers=model.n_layers)
    tiers = CacheTiers(hot=plan.hot, warm=plan.warm)

    # Optional silent prefill so the probe sees the same KV-cache state
    # as heart_shape.py's first generated token.
    if prefill_prompt:
        from moe_forward import forward_one_token
        for pos, tid in enumerate(prefill_prompt):
            _ = forward_one_token(model, tid, streamers, kv_cache, pos, heart=None)
        position = len(prefill_prompt)
        print(f"[nan-probe] after prefill: KV cache has {len(kv_cache)} layers, "
              f"about to probe token {token_id} at position {position}")

    _trace_set_token(token_id, position)
    x = model.embed_w[token_id].view(1, model.hidden_dim).contiguous()
    print(f"\n[embed]     {_stats(x)}")
    positions = torch.tensor([position], device=model.device)

    first_nan_step = None
    n_layers = model.n_layers if max_layer < 0 else min(model.n_layers, max_layer + 1)

    for li in range(n_layers):
        x_norm = rms_norm(x, model.attn_norm_w[li], model.rms_eps)
        print(f"[L{li:02d} attn_norm] {_stats(x_norm)}")
        if torch.isnan(x_norm).any() and first_nan_step is None:
            first_nan_step = f"L{li} attn_norm"
            if stop_at_first_nan: break

        attn_out = qwen3_attention(
            x_norm, model.attn_q_w[li], model.attn_k_w[li],
            model.attn_v_w[li], model.attn_o_w[li],
            model.q_norm[li], model.k_norm[li],
            model.rope_cos, model.rope_sin, positions, kv_cache, li,
            model.n_heads, model.n_kv_heads, model.head_dim, causal=False)
        print(f"[L{li:02d} attn_out]  {_stats(attn_out)}")
        if torch.isnan(attn_out).any() and first_nan_step is None:
            first_nan_step = f"L{li} attn_out"
            if stop_at_first_nan: break

        x = x + attn_out
        print(f"[L{li:02d} x_post_attn] {_stats(x)}")

        # MoE block
        s = streamers.get(li)
        if s is None:
            s = ExpertStreamer(model.tree, layer=li, tiers=tiers,
                                compute_device=model.device,
                                chunks_index=model.chunks_index)
            streamers[li] = s
        moe_out = moe_layer_forward(
            x, li, model.tree, s, model.ffn_norm_w[li],
            model.router_gate[li], k=model.k)
        print(f"[L{li:02d} moe_out]   {_stats(moe_out)}")
        if torch.isnan(moe_out).any() and first_nan_step is None:
            first_nan_step = f"L{li} moe_out"
            if stop_at_first_nan: break

        x = x + moe_out
        if torch.isnan(x).any() and first_nan_step is None:
            first_nan_step = f"L{li} x_post_moe"
            if stop_at_first_nan: break

    if first_nan_step is None and max_layer < 0:
        x_final = rms_norm(x, model.output_norm, model.rms_eps)
        print(f"\n[final_norm] {_stats(x_final)}")
        logits = (x_final @ model.output_w.T).view(-1)
        print(f"[logits]    {_stats(logits)}")
        argmax = int(logits.argmax().item())
        print(f"[argmax]    token {argmax}")
        if torch.isnan(logits).any():
            first_nan_step = "final logits"

    if first_nan_step:
        print(f"\n[nan-probe] FIRST NaN at: {first_nan_step}")
    else:
        print(f"\n[nan-probe] No NaN found in {n_layers} layers + final")


if __name__ == "__main__":
    gguf = sys.argv[1]
    tok = int(sys.argv[2]) if len(sys.argv) > 2 else 9707
    max_layer = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    # Use --prefill <a,b,c,d,e> to silently prefill before probing
    prefill = None
    for i, a in enumerate(sys.argv):
        if a == "--prefill" and i + 1 < len(sys.argv):
            prefill = [int(x) for x in sys.argv[i+1].split(",")]
    probe(gguf, token_id=tok, max_layer=max_layer, prefill_prompt=prefill)
