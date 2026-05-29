"""moe_chat.py — end-to-end chat via the streamer-driven MoE forward.

Putting it all together:

  1. Load Qwen3-MoE non-expert weights one-shot via moe_forward.load_qwen3_moe
  2. Load Qwen3 tokenizer via HF transformers AutoTokenizer
  3. Tokenize the user prompt with the model's chat template
  4. Per-token: forward through streamer (with anticipatory prefetch on the
     side stream) + sample (temperature / top_p / top_k or greedy)
  5. Decode generated token ids back to text
  6. Stream tokens to stdout as they're produced

The streamer's `anticipate()` call now fires after each layer's route(),
predicting the next-likely experts via HoloStream walk from the
recently-routed set. This hides the next layer's expert admit under
the current layer's matmul.

Usage:
    py moe_chat.py <model.gguf> <hf_model_id_for_tokenizer> "prompt text" [n_new=50]

Example:
    py moe_chat.py \\
        "D:/.../Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf" \\
        "Qwen/Qwen3-Coder-30B-A3B-Instruct" \\
        "Write a python function that..." \\
        50
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_forward import (load_qwen3_moe, forward_one_token,
                          moe_layer_forward, rms_norm)
from moe_streamer import ExpertStreamer, CacheTiers, RoutingHistogram


# ─── sampler ──────────────────────────────────────────────────────────────

def sample_logits(logits: torch.Tensor, *, temperature: float = 0.7,
                  top_p: float = 0.9, top_k: int = 40) -> int:
    """Standard nucleus + top-k + temperature sampler.
    If temperature == 0 → greedy (argmax)."""
    if temperature <= 0:
        return int(logits.argmax().item())
    logits = logits.to(torch.float32) / max(1e-6, temperature)
    # top-k
    if top_k > 0:
        kth_value = torch.topk(logits, top_k).values[-1]
        logits[logits < kth_value] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    # top-p (nucleus)
    if 0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff_mask = cumulative > top_p
        # shift right by 1 to always keep the most-probable token
        cutoff_mask = torch.roll(cutoff_mask, 1, dims=-1)
        cutoff_mask[0] = False
        kill = sorted_idx[cutoff_mask]
        probs[kill] = 0
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1).item())


# ─── streamer-driven forward with anticipatory prefetch ──────────────────

def forward_with_prefetch(model, token_id, streamers, histograms, kv_cache,
                           position):
    """Same as moe_forward.forward_one_token but fires anticipate() on each
    layer's streamer after the route call. The histogram tracks recent
    activations; anticipate uses it + geometric walk to prefetch the next
    likely experts on the side stream during compute."""
    x = model.embed_w[token_id].view(1, model.hidden_dim).contiguous()
    positions = torch.tensor([position], device=model.device)
    for li in range(model.n_layers):
        # attention
        x_norm = rms_norm(x, model.attn_norm_w[li], model.rms_eps)
        from moe_forward import qwen3_attention
        attn_out = qwen3_attention(
            x_norm, model.attn_q_w[li], model.attn_k_w[li],
            model.attn_v_w[li], model.attn_o_w[li],
            model.q_norm[li], model.k_norm[li],
            model.rope_cos, model.rope_sin, positions, kv_cache, li,
            model.n_heads, model.n_kv_heads, model.head_dim, causal=False)
        x = x + attn_out
        # MoE block
        s = streamers.get(li)
        h = histograms.get(li)
        if s is None:
            h = RoutingHistogram(n_layers=model.n_layers, n_experts=model.n_routed)
            histograms[li] = h
            # tier-0 = 16 experts per layer × 48 layers ≈ 1.5 GiB on top of
            # the ~1 GiB of one-shot non-expert weights + KV cache. This
            # leaves headroom under the 4 GiB GTX 1650 budget.
            s = ExpertStreamer(model.tree, layer=li,
                                tiers=CacheTiers(hot=16, warm=32),
                                compute_device=model.device, histogram=h)
            streamers[li] = s
        # the recent eids are tracked in s.trajectory's last entries.
        # Trajectory tuple is (layer, ring, node, slot); reconstruct eid =
        # (ring << 12) | (node << 6) | slot to match the original integer.
        moe_out = moe_layer_forward(
            x, li, model.tree, s, model.ffn_norm_w[li],
            model.router_gate[li], k=model.k, histogram=h)
        recent = [(c[1] << 12) | (c[2] << 6) | c[3] for c in s.trajectory[-model.k:]]
        s.anticipate(fanout=4, recent_eids=recent, use_geometric=True)
        # Cap trajectory list growth so memory doesn't drift over a long chat
        if len(s.trajectory) > 256:
            s.trajectory[:] = s.trajectory[-128:]
        x = x + moe_out
    x = rms_norm(x, model.output_norm, model.rms_eps)
    return (x @ model.output_w.T).view(-1)


def generate(model, prompt_ids: list, n_new: int = 50,
             temperature: float = 0.7, top_p: float = 0.9, top_k: int = 40,
             eos_token_ids: list = None, on_token=None):
    """Stream generation with sampler + anticipatory prefetch.
    Calls `on_token(token_id, decoded_text)` per generated token if given."""
    streamers, histograms, kv_cache = {}, {}, {}
    output = list(prompt_ids)
    t_prefill_start = time.perf_counter()
    print(f"[gen] prefilling {len(prompt_ids)} prompt tokens …", flush=True)
    for pos, tid in enumerate(prompt_ids):
        _ = forward_with_prefetch(model, tid, streamers, histograms, kv_cache, pos)
    t_prefill = time.perf_counter() - t_prefill_start
    print(f"[gen] prefill done in {t_prefill:.1f}s "
          f"({len(prompt_ids)/t_prefill:.2f} tok/s)", flush=True)
    print(f"[gen] generating up to {n_new} new tokens …", flush=True)
    t_gen_start = time.perf_counter()
    eos_set = set(eos_token_ids or [])
    for step in range(n_new):
        last = output[-1]
        logits = forward_with_prefetch(model, last, streamers, histograms,
                                         kv_cache, len(output))
        next_id = sample_logits(logits, temperature=temperature,
                                 top_p=top_p, top_k=top_k)
        output.append(next_id)
        if on_token: on_token(next_id, step)
        if next_id in eos_set:
            print(f"\n[gen] EOS at step {step+1}", flush=True)
            break
        if (step + 1) % 5 == 0:
            dt = time.perf_counter() - t_gen_start
            print(f"  {step+1} tokens in {dt:.1f}s ({(step+1)/dt:.2f} tok/s)", flush=True)
    t_gen = time.perf_counter() - t_gen_start
    return output, t_prefill, t_gen


# ─── CLI ──────────────────────────────────────────────────────────────────

def _usage():
    print("Usage:")
    print("  py moe_chat.py <model.gguf> <tokenizer_id> [prompt='Hello'] [n_new=20]")
    print("                                              [--greedy] [--temp=0.7]")
    print()
    print("Example:")
    print("  py moe_chat.py qwen3-coder.gguf Qwen/Qwen3-Coder-30B-A3B-Instruct 'def quick' 30")


if __name__ == "__main__":
    if len(sys.argv) < 3: _usage(); sys.exit(1)
    gguf = sys.argv[1]
    tokenizer_id = sys.argv[2]
    prompt = sys.argv[3] if len(sys.argv) >= 4 else "Hello, "
    n_new = int(sys.argv[4]) if len(sys.argv) >= 5 and sys.argv[4].isdigit() else 20
    temperature = 0.0 if "--greedy" in sys.argv else 0.7
    for a in sys.argv:
        if a.startswith("--temp="): temperature = float(a.split("=")[1])

    print(f"[chat] loading tokenizer: {tokenizer_id}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    eos_ids = []
    if tok.eos_token_id is not None: eos_ids.append(tok.eos_token_id)
    for n in ("<|im_end|>", "<|endoftext|>"):
        try:
            t = tok.convert_tokens_to_ids(n)
            if isinstance(t, int) and t >= 0: eos_ids.append(t)
        except Exception: pass
    print(f"[chat] prompt: {prompt!r}")
    print(f"[chat] prompt_ids ({len(prompt_ids)}): {prompt_ids[:10]}...")
    print(f"[chat] eos_token_ids: {eos_ids}")

    print(f"[chat] loading model: {gguf}")
    model = load_qwen3_moe(gguf)
    print(f"[chat] sampling: temperature={temperature}, top_p=0.9, top_k=40")
    print(f"[chat] === generation ===")

    # streaming print of each token as it's generated
    buf = []
    def on_token(tid, step):
        buf.append(tid)
        try:
            text = tok.decode(buf, skip_special_tokens=True)
            print(text[len(prev[0]):], end="", flush=True)
            prev[0] = text
        except Exception: pass
    prev = [""]

    out, t_prefill, t_gen = generate(model, prompt_ids, n_new=n_new,
                                       temperature=temperature, top_p=0.9, top_k=40,
                                       eos_token_ids=eos_ids, on_token=on_token)
    print()
    print(f"\n[chat] === stats ===")
    print(f"  prefill: {t_prefill:.1f}s for {len(prompt_ids)} tokens "
          f"({len(prompt_ids)/t_prefill:.2f} tok/s)")
    print(f"  gen:     {t_gen:.1f}s for {len(out) - len(prompt_ids)} tokens "
          f"({(len(out)-len(prompt_ids))/t_gen:.2f} tok/s)")
    print(f"  full decoded text:")
    print(f"  {tok.decode(out, skip_special_tokens=True)}")
