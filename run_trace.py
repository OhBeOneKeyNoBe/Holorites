"""run_trace.py — small sanity-probe trace run.

Loads Qwen3-Coder-30B through moe_forward, generates a handful of
tokens with HOLORITE_TRACE set, and exits. Used to:
  1. Confirm the trace emitter (Hook A) actually writes JSONL rows.
  2. Produce a minimal trace the planner can consume end-to-end so
     we can verify the whole pipeline before committing to a long
     overnight diverse-prompt run.

Usage:
  set HOLORITE_TRACE=trace.jsonl
  py run_trace.py <gguf> [n_tokens=10]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Apply allocator hint BEFORE moe_forward imports torch (task #9)
from vram_budget import apply_expandable_segments
apply_expandable_segments()
import moe_forward as mf

if __name__ == "__main__":
    gguf = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    trace_path = os.environ.get("HOLORITE_TRACE", "(unset)")
    print(f"[trace-run] gguf:  {gguf}")
    print(f"[trace-run] N:     {n}")
    print(f"[trace-run] trace: {trace_path}")
    t0 = time.perf_counter()
    model = mf.load_qwen3_moe(gguf)
    t_load = time.perf_counter() - t0
    print(f"[trace-run] model loaded in {t_load:.1f}s")
    # Use a short, varied prompt instead of just BOS so the gate
    # exercises a wider expert set even at small token counts.
    # Qwen3 BOS = 151643. Add 4 diverse seed tokens to spread routing.
    prompt = [151643, 9707, 1958, 374, 264]   # ~"Hello world is a"
    out, dt = mf.generate_tokens(model, prompt, n)
    print(f"[trace-run] DONE: {n} tokens in {dt:.1f}s ({n/dt:.3f} tok/s)")
    print(f"[trace-run] generated ids: {out[len(prompt):]}")
