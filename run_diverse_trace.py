"""run_diverse_trace.py — multi-prompt trace generator for richer ring layout.

The original run_trace.py uses ONE prompt and exercises a narrow expert
subset. The first ring layout that came out of it was structurally
correct but semantically degenerate (62 rings got 1 expert, 1 ring got
62) because most experts had too little signal for the affinity scorer.

This runner cycles through a battery of short diverse prompts so the
trace covers a wider routing distribution. For a model with 128 experts
and top-k=8, the probability of any expert being picked on a given token
is 8/128 = 6.25%; to see each expert ~50 times we need roughly 800
tokens spread across diverse contexts.

Strategy:
    N_PROMPTS prompts, each generating N_PER tokens with fresh KV.
    Total trace rows = N_PROMPTS * N_PER * n_layers.
    For N_PROMPTS=20, N_PER=10, n_layers=48: 9600 rows
    (vs the earlier 698, ~14x more).

Usage:
    set HOLORITE_TRACE=trace_diverse.jsonl
    py run_diverse_trace.py <gguf> [n_prompts=20] [n_per=10]
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Apply allocator hint BEFORE moe_forward imports torch (task #9)
from vram_budget import apply_expandable_segments
apply_expandable_segments()
import moe_forward as mf


# Battery of short, diverse seed prompts. Each is tokenized fresh per
# generation so the KV cache starts empty (otherwise OOM accumulates).
DIVERSE_PROMPTS = [
    [151643, 9707, 1958],                  # "Hello world"
    [151643, 791, 4885, 311, 2581],        # "The answer to"
    [151643, 3957, 1455, 2873],            # "While she walked"
    [151643, 11871, 374, 264, 21390],      # "Python is a language"
    [151643, 39584, 25351, 4869, 369],     # "Recipes provide ingredients for"
    [151643, 16107, 374, 539],             # "Music is not"
    [151643, 12978, 2225, 1144, 67943],    # "Multiple ways combined"
    [151643, 7361, 369, 13340, 304],       # "Notes for reference in"
    [151643, 791, 18435, 11865, 369],      # "The compiler handles for"
    [151643, 38, 56391, 3958, 304],        # "Cards generated in"
    [151643, 1734, 50617, 2585, 4528],     # "When children study cells"
    [151643, 3957, 3823, 25, 264],         # "Suppose tomorrow: a"
    [151643, 41504, 2851, 11, 87018],      # "Maybe just, possibly"
    [151643, 16, 489, 220, 17, 284],       # "1 + 2 ="
    [151643, 18540, 374, 5552, 25],        # "Light is described:"
    [151643, 1986, 374, 1101, 264],        # "This is just a"
    [151643, 96270, 18815, 304, 9876],     # "Quantum cycles in motion"
    [151643, 16, 13, 18840, 11, 38149],    # "1.) First, second"
    [151643, 3957, 3823, 76090, 11],       # "Suppose tomorrow brings,"
    [151643, 4438, 311, 1518, 902, 5764],  # "How to obtain no input"
]


def main(gguf, n_prompts=20, n_per=10):
    trace_path = os.environ.get("HOLORITE_TRACE", "(unset)")
    print(f"[diverse-trace] gguf:        {gguf}")
    print(f"[diverse-trace] n_prompts:   {n_prompts}")
    print(f"[diverse-trace] n_per:       {n_per}")
    print(f"[diverse-trace] trace_path:  {trace_path}")
    n_prompts = min(n_prompts, len(DIVERSE_PROMPTS))

    t0 = time.perf_counter()
    model = mf.load_qwen3_moe(gguf)
    print(f"[diverse-trace] model loaded in {time.perf_counter()-t0:.1f}s")

    grand_t0 = time.perf_counter()
    total_tokens = 0
    for i in range(n_prompts):
        prompt = DIVERSE_PROMPTS[i]
        print(f"\n[{i+1}/{n_prompts}] prompt ids={prompt}")
        try:
            t0 = time.perf_counter()
            out, dt = mf.generate_tokens(model, prompt, n_per)
            total_tokens += n_per
            print(f"  generated {n_per} tokens in {dt:.1f}s "
                  f"({n_per/dt:.3f} tok/s); next ids: {out[len(prompt):]}")
        except Exception as e:
            print(f"  FAILED on prompt {i}: {type(e).__name__}: {e}")
            print(f"  continuing to next prompt (KV cache will reset)")
            # Force-clear any residue. The forward_one_token uses per-call
            # streamers/kv_cache dicts so subsequent calls start fresh.
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_dt = time.perf_counter() - grand_t0
    print(f"\n[diverse-trace] DONE: {total_tokens} tokens in {total_dt:.1f}s "
          f"({total_tokens/max(0.001, total_dt):.3f} tok/s)")

    # Report trace stats
    if os.path.exists(trace_path) and trace_path != "(unset)":
        n_rows = sum(1 for _ in open(trace_path))
        print(f"[diverse-trace] trace file: {n_rows} rows")


if __name__ == "__main__":
    gguf = sys.argv[1]
    n_prompts = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    n_per = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    main(gguf, n_prompts, n_per)
