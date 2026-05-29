# Holorite — STATUS

**Date:** 2026-05-29
**Head commit:** `56d5fbd` (v0.9.0)
**Repo:** https://github.com/OhBeOneKeyNoBe/Holorites

---

## v0.9.0 shipped

Everything in the v0.9.0 release notes is real, in the repo, and tested:

| Piece | State |
|---|---|
| Hook A: routing-trace emitter (`HOLORITE_TRACE=...`) | ✅ shipped, verified — produces real JSONL routing decisions |
| Hook B: ring-layout chunkifier (`--ring-layout`) | ✅ shipped — wrote 16.35 GiB sidecar in ring-major order |
| Hook C: strand walker + chunks_index ingestion | ✅ shipped — `verify_strand.py` passes |
| VRAM budget plan (`vram_budget.plan_budget`) | ✅ shipped — auto-detects 4095 MiB, sizes tier0/warm honestly |
| Stream-fence NaN fix in `route()` | ✅ shipped — logits now clean at position > 0 after prefill |
| NestedHeart scaffolding (heart.py) | ✅ shipped — self-test passes |
| Heart-shape measurement harness | ✅ shipped — wiring verified, KL/cos blocked by NaN until fix |

## What's verified end-to-end

- **20-token generation on Qwen3-Coder-30B Q4_K_M** at 0.17 tok/s with no OOM, no NaN. argmax = real tokens.
- **Strand-subscription streamer:** 64/64 rings populated from layout, all 128 experts mapped to rings, 8/8 strand walk hits land in populated rings.
- **Diverse-trace pipeline:** 11056-row trace captured across 20 prompts × 10 generated tokens.
- **Discord posted** on v0.9.0 release to #updates, #achievements, #zioniel-ai-prs (3 channels, HTTP 204 each).

## Open work items

### 1. V3 download (blocker for nesting test against real DeepSeek)

- **Not yet found locally on D:** at last check. The `dl_priority.sh` queue may need a re-kick.
- Holorite supports DeepSeek-V3 / R1 / V4-Pro architectures in `moe_streamer.py` (61-layer / 80-layer envelopes already in `OUTER_PROFILES`).
- Action: when V3 GGUF lands, run `expert_chunkifier --ring-layout` to build the V3 sidecar (~280 GiB extra disk).

### 2. Real Qwen2.5-1.5B at the heart (replace noise stub)

- `NestedHeart` is built and tested with noise-stub heart.
- A real Qwen2.5-1.5B heart needs: HF transformers loading, hidden-state-input wrapper (the heart consumes the outer's hidden state, not token ids), KV cache management.
- This is a 3-4 hour build job. Worth doing before V3 lands so we have a working nesting demo on Qwen3-Coder-as-outer first.

### 3. Semantic ring-layout planner needs redesign

**Honest finding (re-plan run 2026-05-29):** the 11056-row diverse trace produced the same degenerate distribution as the 698-row baseline:

```
Layer 0 ring sizes:  {1: 62 rings,  62: 1 ring,  5: 1 ring}
Layer 24:            {1: 62 rings,  62: 1 ring,  4: 1 ring}
Layer 47:            {1: 62 rings,  61: 1 ring,  5: 1 ring}
```

That's 62 rings with 1 expert + 1 ring with 60+ leftovers + 1 small bucket. The cosine-affinity scorer's signal-to-noise ratio is too low — per-expert token-id fingerprints are sparse against the 151k-token vocab, and most cells are 0.

**Fix candidates** (deferred):
- **Co-activation graph clustering** — two experts that fire together for the same token are semantically related. Hexagrams become cluster centroids.
- **Embedding-space fingerprints** — project each expert's gate-activation pattern to a low-dim space; cluster there.
- **Bigger trace** — 100k+ tokens. ~5 hours of GPU at current speed.

The pipeline works end-to-end **with the structurally-valid layout we have**; the chunkifier sidecar in ring-major order gives byte-locality regardless of semantic strength. Semantic placement is the next-level win.

### 4. CUDA expandable_segments isn't supported on Windows

Confirmed by `PyTorch warning: expandable_segments not supported on this platform`. The smaller per-layer tier sizes from `plan_budget()` alone are doing all the OOM-mitigation work on Windows. The helper still helps on Linux/cloud — keep it.

### 5. Seven-lever path to 100+ tok/s

Status of each lever from the original plan:

| Lever | State |
|---|---|
| 1. Pinned warm tier (host RAM) | ✅ done in `_admit_to_host_pinned` |
| 2. Three CUDA streams (compute/admit/prefetch) | ✅ done in ExpertStreamer.__init__ |
| 3. Grouped expert GEMM (batched bmm) | ✅ done in moe_forward.py for T=1 fast path |
| 4. Q3_K / Q2_K expert quant decoders | ✅ done in moe_kernels.py |
| 5. Pre-laid expert chunk layout | ✅ done in expert_chunkifier.py |
| 6. Speculative residency / demote-on-evict | ✅ done in ExpertStreamer.route |
| 7. Token batching through same experts | ⚠ partial — T=1 fast path; T>1 still per-token |

**Speculative decoding** (lever 8 — beyond the original 7): not yet started. Path to 200 tok/s. Requires a small draft model + verification loop.

**Fused dequant+GEMM kernel** (lever 9): not yet started. DeepGEMM-style port. Path to 500 tok/s aggregate.

## What's testable right now (without V3)

1. **NestedHeart on Qwen3-Coder-30B as the outer** — wire `make_noise_stub_heart` into `forward_with_prefetch`, run on a prompt, observe heart-shape KL diff (should be non-NaN now that the stream-fence fix is in).
2. **Strand-subscription prefetch hit rate measurement** — instrument `anticipate()` to log which strand-walked rings ended up being chosen by the gate vs not. Validates the videogram-receiver claim.
3. **Longer diverse trace + per-layer KL comparison of layouts** — could quantify how degenerate the current layout is vs. random.

## Sister project: Incarnate-Rust

Holorite serves Zion'iel's voice through the Incarnate-Rust bridge when:

1. Holorite has a coherent local model serving on `/v1/chat/completions` (host the user's eventual trained Zion'iel weights).
2. Incarnate-Rust bridge config has `backends.holorite.enabled: true` with the right `base_url`.

Until then: Incarnate-Rust uses Claude via Anthropic API as the bridge backend. Architecturally, swapping in Holorite is a config change once Holorite serves coherent output. The current Qwen3-Coder-30B output is **structurally clean (non-NaN logits)** but the trained Zion'iel weights aren't built yet — that's `Zioniel-Trainer` territory.
