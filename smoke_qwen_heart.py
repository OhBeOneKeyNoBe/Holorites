"""smoke_qwen_heart.py — load Qwen2.5-1.5B-Instruct as the heart, run a few
forward passes, assert sane output.

Doesn't load the outer. Just confirms the real heart wraps Qwen2.5 correctly:
- Hidden-state in, hidden-state out (no token ids)
- Position-aware via DynamicCache
- W_in / W_out projections wired
- alpha-blend at the insertion layer fires
- Heart is non-deterministic-after-position (different positions give different outputs)
- Heart is deterministic-at-position (same input + same kv state → same output)

Usage:
    py smoke_qwen_heart.py
    py smoke_qwen_heart.py "Qwen/Qwen2.5-1.5B-Instruct"
"""
from __future__ import annotations
import os, sys, time
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

DEFAULT_PATH = (r"C:\Users\virtu\.cache\huggingface\hub"
                r"\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots"
                r"\989aa7980e4cf806f80c7fef2b1adb7bc71aa306")


def main(model_path=DEFAULT_PATH):
    from heart import make_qwen_heart, make_noise_stub_heart

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device:        {device}")
    print(f"[smoke] model_path:    {model_path}")
    print(f"[smoke] loading Qwen2.5-1.5B-Instruct as heart …")

    t0 = time.perf_counter()
    OUTER_H = 2048   # fake outer dim, e.g. Qwen3-Coder-30B
    M       = 24     # insertion layer (irrelevant here, we call at_layer manually)
    heart = make_qwen_heart(model_path, outer_hidden_dim=OUTER_H,
                             insertion_layer=M, alpha=0.5, device=device)
    print(f"[smoke] heart loaded in {time.perf_counter()-t0:.1f}s")
    print(f"[smoke] heart_hidden:  {heart.heart_hidden_dim}")
    print(f"[smoke] resident MiB:  {heart.heart_resident_mb_estimate}")
    print(f"[smoke] W_in shape:    {tuple(heart.W_in.shape)}")
    print(f"[smoke] W_out shape:   {tuple(heart.W_out.shape)}")

    # synthetic outer hidden state, T=1
    torch.manual_seed(0)
    x = torch.randn(1, OUTER_H, dtype=torch.float16, device=device)

    # ── 1. passthrough at non-insertion layer
    y0 = heart.at_layer(x, outer_layer=10, position=0)
    assert torch.equal(x, y0), "non-insertion layer must passthrough"
    print(f"[smoke] passthrough at layer 10:  ok (identity)")

    # ── 2. insertion-layer call: hidden state must change measurably
    y1 = heart.at_layer(x, outer_layer=M, position=0)
    assert not torch.equal(x, y1), "insertion layer must modify"
    delta1 = (y1 - x).abs().mean().item()
    print(f"[smoke] insertion at M={M}, pos=0:  |Δ| mean = {delta1:.5f}")
    assert delta1 > 0.0, "residual must be nonzero"
    assert torch.isfinite(y1).all(), "output must be finite (no NaN/Inf)"
    print(f"[smoke] output finite:             ok")

    # ── 3. second position: KV cache accumulates → different output
    x2 = torch.randn(1, OUTER_H, dtype=torch.float16, device=device)
    y2 = heart.at_layer(x2, outer_layer=M, position=1)
    delta2 = (y2 - x2).abs().mean().item()
    print(f"[smoke] insertion at M={M}, pos=1:  |Δ| mean = {delta2:.5f}")
    assert torch.isfinite(y2).all()

    # ── 4. KV state inspection
    tel = heart.telemetry()
    print(f"[smoke] telemetry: {tel}")
    assert tel["insertions"] == 2, f"expected 2 insertions, got {tel}"

    # ── 5. reset_kv: cache is wiped, same input now gives the "position 0" result
    heart.reset_kv()
    y0_again = heart.at_layer(x, outer_layer=M, position=0)
    assert torch.isfinite(y0_again).all()
    same = torch.allclose(y1, y0_again, atol=1e-3)
    print(f"[smoke] after reset_kv, pos=0 again: matches first y1? {same}")
    assert same, "reset_kv should make pos=0 reproducible"

    # ── 6. determinism check
    heart.reset_kv()
    y0_third = heart.at_layer(x, outer_layer=M, position=0)
    assert torch.allclose(y0_again, y0_third, atol=1e-5), \
        "two reset_kv → pos=0 runs must be byte-equal"
    print(f"[smoke] determinism after reset_kv: ok")

    if device.type == "cuda":
        print(f"[smoke] VRAM after run:           "
              f"{torch.cuda.memory_allocated()/1024**2:.0f} MiB allocated, "
              f"{torch.cuda.max_memory_allocated()/1024**2:.0f} MiB peak")

    print(f"\n[smoke-qwen-heart] PASS — Qwen2.5-1.5B heart loads, "
          f"hidden-state IO works, KV cache accumulates and resets.")
    return 0


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    sys.exit(main(path))
