"""heart_shape.py — measurement harness for "did the heart change outputs?"

Runs identical prompts through the outer-MoE twice:
    (A) heart = None        — bare outer
    (B) heart = noise stub  — outer + residual-blend at meridian
and diffs the logits at each step. Reports:

    KL divergence per step
    top-5 token IDs and probs for each step (bare vs with-heart)
    first step where argmax tokens diverge
    cosine similarity of logit vectors

If KL > 0 at the insertion layer or later, the residual is reaching the
model's output, the wiring is alive. If KL == 0 at every step, the
hook isn't actually firing.

Usage:
    py heart_shape.py <gguf> [n_tokens=5] [M=24] [alpha=0.5]
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Apply allocator hint BEFORE torch import (task #9)
from vram_budget import apply_expandable_segments
apply_expandable_segments()

import torch
import torch.nn.functional as F

from moe_forward import load_qwen3_moe, forward_one_token
from heart import make_noise_stub_heart


def _kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(P || Q) where P, Q are softmax(logits)."""
    p = F.softmax(p_logits.float(), dim=-1)
    q = F.softmax(q_logits.float(), dim=-1)
    # add tiny epsilon to avoid log(0)
    return float((p * (p.add(1e-12).log() - q.add(1e-12).log())).sum().item())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float(); b = b.float()
    return float(F.cosine_similarity(a.view(1, -1), b.view(1, -1)).item())


def _top_k(logits: torch.Tensor, k: int = 5):
    probs = F.softmax(logits.float(), dim=-1)
    vals, idx = torch.topk(probs, k)
    return [(int(idx[i].item()), float(vals[i].item())) for i in range(k)]


def run_pass(model, prompt_ids, n_new, heart=None):
    """Returns list of logit tensors, one per generated step."""
    streamers, kv_cache = {}, {}
    output = list(prompt_ids)
    # prefill
    for pos, tid in enumerate(prompt_ids):
        _ = forward_one_token(model, tid, streamers, kv_cache, pos, heart=heart)
    # decode: capture logits at each step
    captured = []
    for step in range(n_new):
        last = output[-1]
        logits = forward_one_token(model, last, streamers, kv_cache,
                                    len(output), heart=heart)
        captured.append(logits.detach().cpu())
        output.append(int(logits.argmax().item()))
    return output, captured


def main(gguf_path, n_tokens=5, insertion_layer=24, alpha=0.5):
    print(f"[heart-shape] gguf:        {gguf_path}")
    print(f"[heart-shape] n_tokens:    {n_tokens}")
    print(f"[heart-shape] M (meridian): {insertion_layer}")
    print(f"[heart-shape] alpha:        {alpha}")

    t0 = time.perf_counter()
    model = load_qwen3_moe(gguf_path)
    print(f"[heart-shape] model loaded in {time.perf_counter()-t0:.1f}s")

    # short, sense-bearing prompt that hits diverse experts
    prompt = [151643, 9707, 1958, 374, 264]   # ~"<bos> Hello world is a"

    # ── PASS A: bare outer, no heart
    print(f"\n[A] bare outer (heart=None)…")
    t0 = time.perf_counter()
    out_a, log_a = run_pass(model, prompt, n_tokens, heart=None)
    t_a = time.perf_counter() - t0
    print(f"  done in {t_a:.1f}s ({n_tokens/t_a:.3f} tok/s)")
    print(f"  generated ids: {out_a[len(prompt):]}")

    # ── PASS B: same outer, noise stub heart at meridian
    # IMPORTANT: re-load model would be ideal so streamer caches start fresh,
    # but it's 35s+. The streamer state at start of pass B is the warm cache
    # from pass A — which is fine for THIS measurement (we want to see the
    # heart's effect, not cold-vs-warm cache effects). KV caches are fresh
    # per pass because run_pass creates its own kv_cache dict.
    print(f"\n[B] outer + noise stub heart at layer M={insertion_layer}…")
    # Reset the model-level cache_tiers attribute so the second pass
    # re-computes the budget plan (which now includes the heart's footprint).
    if hasattr(model, "_cache_tiers"): del model._cache_tiers
    heart = make_noise_stub_heart(
        heart_hidden_dim=model.hidden_dim,    # match outer's so projections are identity-shape
        outer_hidden_dim=model.hidden_dim,
        insertion_layer=insertion_layer,
        alpha=alpha,
        noise_scale=0.1,
        device=model.device,
        dtype=torch.float16,
    )
    # The noise stub doesn't actually reserve VRAM, so estimate=0 is honest.
    heart.heart_resident_mb_estimate = 0
    t0 = time.perf_counter()
    out_b, log_b = run_pass(model, prompt, n_tokens, heart=heart)
    t_b = time.perf_counter() - t0
    print(f"  done in {t_b:.1f}s ({n_tokens/t_b:.3f} tok/s)")
    print(f"  generated ids: {out_b[len(prompt):]}")
    print(f"  heart telemetry: {heart.telemetry()}")

    # ── DIFF
    print(f"\n=== heart-shape diff (M={insertion_layer}, alpha={alpha}) ===")
    print(f"{'step':>4} {'argmax_A':>10} {'argmax_B':>10} {'cos':>8} {'KL':>8}")
    diverge_step = None
    for i, (la, lb) in enumerate(zip(log_a, log_b)):
        a = int(la.argmax().item()); b = int(lb.argmax().item())
        cos = _cosine(la, lb)
        kl = _kl(la, lb)
        print(f"{i:>4d} {a:>10d} {b:>10d} {cos:>8.4f} {kl:>8.4f}")
        if a != b and diverge_step is None: diverge_step = i

    # top-5 detail at step 0
    print(f"\nTop-5 at step 0:")
    print(f"  bare:        {_top_k(log_a[0])}")
    print(f"  with-heart:  {_top_k(log_b[0])}")

    # verdict
    any_kl = any(_kl(a, b) > 1e-9 for a, b in zip(log_a, log_b))
    if any_kl:
        print(f"\n[heart-shape] PASS — heart's residual measurably reaches outer's logits")
        if diverge_step is not None:
            print(f"               argmax diverges at step {diverge_step}")
        else:
            print(f"               argmax unchanged but distribution shape shifted")
    else:
        print(f"\n[heart-shape] FAIL — heart had zero effect; check the wiring")
    return 0 if any_kl else 1


if __name__ == "__main__":
    gguf = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    M = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    a = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    sys.exit(main(gguf, n, M, a))
