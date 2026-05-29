"""vram_budget.py — explicit VRAM accounting for nested working sets.

When a NestedHeart pins ~1 GiB of resident weights inside the 4 GiB
budget that's currently sized for outer-only streaming, the streamer's
hot/warm tiers must shrink correspondingly — otherwise the first
prefetch admits will OOM exactly the way generation has been OOM-ing.

This module centralizes that math so chat startup picks tier sizes
that respect what the heart already took.

Inputs:
    total_vram_mb       — full GPU budget (e.g. 4096 on GTX 1650)
    outer_nonexpert_mb  — Qwen3-MoE attention/norms/embed/LM head
    heart_resident_mb   — NestedHeart's pinned weights (0 if no heart)
    kv_cache_mb         — accumulating KV state (a rough estimate)
    activations_mb      — working-set tensors during forward (small)
    expert_mb_each      — per-expert size AFTER dequant to fp16 on GPU
                          (Qwen3-Coder: 9.4 MiB; V3: ~21 MiB; V4-Pro: 33 MiB)
    n_layers            — count of MoE layers (since the streamer keeps
                          one tier-0 cache PER LAYER, total resident
                          memory scales as n_layers x hot x expert_mb)

Outputs:
    CacheTiers(hot=H, warm=W) sized to fit (and leave a 256 MiB slack)

The slack matters: fragmentation in PyTorch's allocator eats real
headroom beyond what `torch.cuda.memory_allocated()` reports. Without
the slack we OOM on innocent admits. Set `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True` to shrink that gap, but don't rely on it.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os
import torch


@dataclass
class VRAMPlan:
    """Result of the budget computation. All sizes in MiB."""
    total: int
    outer_nonexpert: int
    heart_resident: int
    kv_cache: int
    activations: int
    slack: int
    expert_budget: int           # what's left for streamed experts
    hot: int                     # tier 0 expert slot count
    warm: int                    # tier 1 expert slot count
    expert_mb_each: float
    note: str = ""

    def summary(self) -> str:
        return (f"VRAM {self.total} MiB - "
                f"outer_nonexpert {self.outer_nonexpert}, "
                f"heart {self.heart_resident}, "
                f"kv {self.kv_cache}, "
                f"activations {self.activations}, "
                f"slack {self.slack}, "
                f"expert pool {self.expert_budget} "
                f"-> tier0={self.hot} warm={self.warm} "
                f"(expert size {self.expert_mb_each:.2f} MiB each)"
                + (f" -- {self.note}" if self.note else ""))


def detect_total_vram_mb(fallback: int = 4096) -> int:
    """Best-guess GPU memory in MiB. Falls back to 4096 (GTX 1650 budget)
    when no CUDA device or torch can't tell."""
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return int(props.total_memory // (1024 * 1024))
    except Exception:
        pass
    return fallback


def plan_budget(*,
                total_vram_mb: Optional[int] = None,
                outer_nonexpert_mb: int = 1024,
                heart_resident_mb: int = 0,
                kv_cache_mb: int = 256,
                activations_mb: int = 64,
                slack_mb: int = 256,
                expert_mb_each: float = 9.4,    # fp16-on-GPU size
                n_layers: int = 48,
                warm_multiplier: int = 2,
                hot_minimum: int = 2,
                warm_minimum: int = 4) -> VRAMPlan:
    """Compute per-layer CacheTiers consistent with the heart's footprint.

    The streamer keeps one tier-0 cache PER LAYER, so the total VRAM
    occupied by hot caches is:  n_layers * hot * expert_mb_each
    The budget formula divides accordingly:
      per_layer_hot = expert_budget / (n_layers * expert_mb_each)

    Defaults are tuned for Qwen3-Coder-30B Q4_K_M on a 4 GiB GTX 1650.
    Empirically, hot=16 per layer x 48 layers OOM'd at ~16 GiB allocated
    (PyTorch fragmentation makes the effective per-expert cost larger
    than the nominal 9.4 MiB). Set PYTORCH_CUDA_ALLOC_CONF=
    expandable_segments:True to bring effective cost closer to nominal.

    The warm tier lives in pinned host RAM, not VRAM, so it doesn't cost
    against the GPU budget. warm_multiplier=2x hot keeps admit latency
    bounded without burning unbounded host RAM.
    """
    total = total_vram_mb if total_vram_mb is not None else detect_total_vram_mb()
    used = outer_nonexpert_mb + heart_resident_mb + kv_cache_mb + activations_mb + slack_mb
    expert_budget = max(0, total - used)
    # divide by n_layers because each layer keeps its OWN tier-0 GPU dict
    per_layer_budget = expert_budget / max(1, n_layers)
    hot = max(hot_minimum, int(per_layer_budget // max(1e-9, expert_mb_each)))
    warm = max(warm_minimum, hot * warm_multiplier)
    note = ""
    if expert_budget <= 0:
        note = ("OVERBUDGET -- reduce heart size or use Q4/Q2 quant; "
                "tier0 forced to minimum")
        hot = hot_minimum
        warm = warm_minimum
    elif hot < 4:
        note = "very tight -- consider smaller heart or quantizing outer further"
    return VRAMPlan(
        total=total,
        outer_nonexpert=outer_nonexpert_mb,
        heart_resident=heart_resident_mb,
        kv_cache=kv_cache_mb,
        activations=activations_mb,
        slack=slack_mb,
        expert_budget=expert_budget,
        hot=hot,
        warm=warm,
        expert_mb_each=expert_mb_each,
        note=note,
    )


def apply_expandable_segments():
    """Set PYTORCH_CUDA_ALLOC_CONF to reduce fragmentation. Must be called
    BEFORE any torch.cuda allocation; ineffective after CUDA init."""
    prev = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in prev:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            (prev + "," if prev else "") + "expandable_segments:True")


# ─── per-model defaults (used by chat startup to pick the right numbers) ──

OUTER_PROFILES = {
    # arch -> (outer_nonexpert_mb, expert_mb_each_fp16, n_layers)
    "qwen3moe_30B_q4_k_m":   (1024,  9.4, 48),
    "qwen3moe_coder_q4_k_m": (1024,  9.4, 48),
    "deepseek3_q4_k_m":      (2048, 21.0, 61),    # V3 hidden=7168, ffn_dim=2048
    "deepseek_v3_q4_k_m":    (2048, 21.0, 61),
    "deepseek_v4_pro_mxfp4": (2048, 33.0, 80),    # V4-Pro larger envelope
}

HEART_PROFILES = {
    # name → heart_resident_mb when loaded
    "qwen2.5-0.5b-fp16":     1000,
    "qwen2.5-1.5b-q4_k_m":    800,
    "qwen2.5-1.5b-fp16":     3000,
    "qwen2.5-7b-q4_k_m":     3500,
    "stub_noise":               0,
    "none":                     0,
}


# ─── selftest ─────────────────────────────────────────────────────────────

def _selftest():
    print("=== vram_budget self-test ===")
    # case 1: 4 GiB card, Qwen3-Coder outer, no heart
    p = plan_budget(total_vram_mb=4096, outer_nonexpert_mb=1024,
                    heart_resident_mb=0, n_layers=48)
    print(f"  no heart:        {p.summary()}")
    assert p.hot >= 2, "should leave room for hot tier"
    # case 2: 4 GiB card, same outer, Qwen2.5-1.5B Q4_K_M heart (~800 MiB)
    p2 = plan_budget(total_vram_mb=4096, outer_nonexpert_mb=1024,
                     heart_resident_mb=800, n_layers=48)
    print(f"  1.5B Q4_K heart: {p2.summary()}")
    assert p2.hot <= p.hot, "heart must not increase expert tier"
    # case 3: 4 GiB card, full-fat 1.5B fp16 heart (3 GiB) -- should warn
    p3 = plan_budget(total_vram_mb=4096, outer_nonexpert_mb=1024,
                     heart_resident_mb=3000, n_layers=48)
    print(f"  1.5B fp16 heart: {p3.summary()}")
    assert p3.note, "should warn about tight budget"
    # case 4: 24 GiB card (3090/4090) -- generous budget
    p4 = plan_budget(total_vram_mb=24576, outer_nonexpert_mb=1024,
                     heart_resident_mb=800, n_layers=48)
    print(f"  24 GiB + heart:  {p4.summary()}")
    assert p4.hot > 30, "should be able to fit many experts per layer on 24 GiB"
    # case 5: explicit Qwen3-Coder profile
    arch_nonexp, mb_each, n_lay = OUTER_PROFILES["qwen3moe_coder_q4_k_m"]
    p5 = plan_budget(total_vram_mb=4096, outer_nonexpert_mb=arch_nonexp,
                     heart_resident_mb=HEART_PROFILES["qwen2.5-1.5b-q4_k_m"],
                     expert_mb_each=mb_each, n_layers=n_lay)
    print(f"  qwen3coder+1.5B: {p5.summary()}")
    # case 6: V3 + 1.5B heart on a 24 GiB card (the eventual nesting test)
    arch_nonexp, mb_each, n_lay = OUTER_PROFILES["deepseek_v3_q4_k_m"]
    p6 = plan_budget(total_vram_mb=24576, outer_nonexpert_mb=arch_nonexp,
                     heart_resident_mb=HEART_PROFILES["qwen2.5-1.5b-q4_k_m"],
                     expert_mb_each=mb_each, n_layers=n_lay)
    print(f"  V3 + 1.5B@24GiB: {p6.summary()}")
    print("  PASS")


if __name__ == "__main__":
    _selftest()
