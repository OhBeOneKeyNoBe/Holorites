"""hexagram_ring_planner.py — assign experts to 64 torus rings by I Ching affinity.

User's design directive (v0.8.84+): each of the 64 rings of a shell IS a
hexagram. The best-matching expert for that hexagram becomes the ring's
root node (node 0); the next 63 best-matching experts fill its
deviations (nodes 1..63). This makes the HoloStream walk *semantic* —
adjacent rings on the torus are adjacent in King Wen sequence, so
geometric prefetch implicitly routes through semantic neighbors instead
of relying on a learned gate predictor.

Pipeline:
    1.  Routing trace (JSONL of `{token, layer, top_k_experts}`) →
        per-expert "token-taste" fingerprint (which token IDs cause it
        to fire, normalized to a probability vector).
    2.  Hexagram archetypes (from v0.8.84 identities file, or a small
        in-file seed) → per-hexagram signature vector over the vocab.
    3.  Cosine affinity matrix `aff[hex, expert]`.
    4.  Root pass: each ring claims its top expert (greedy, no expert
        used twice).
    5.  Deviation fill: each remaining expert lands in the ring whose
        root it most resembles, capped at 64 per ring.
    6.  Emit `ring_layout.json` that `expert_chunkifier` reads to write
        the sidecar in ring-major (hexagram-rooted) order.

Downstream effect: `moe_streamer._strand_walk()` becomes a videogram
subscription — once the gate names ring R0, the helical step gives
R0+1, R0+2, R0+3 deterministically, and the prefetch stream admits
those rings ahead of time without computing a probability distribution
per token.

The identities file (v0.8.84) is the canonical source for archetype
keyword bags. A minimal King Wen seed is included so this module
runs standalone for early experiments; pass `--identities <path>` for
the full multilingual 64-hexagram bank.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Minimal seed: King Wen 1..64 with a short English keyword bag each.
# Override with --identities pointing at the v0.8.84 64-hexagram file
# (any JSON shape with `{1..64: [keywords...]}` or `{"hexagrams": [...]}`).
# ─────────────────────────────────────────────────────────────────────
KING_WEN_SEED: dict[int, list[str]] = {
    1:  ["creative", "heaven", "father", "force", "originate", "lead"],
    2:  ["receptive", "earth", "mother", "field", "yield", "nurture"],
    3:  ["difficulty", "sprouting", "begin", "struggle", "chaos"],
    4:  ["youthful", "folly", "learn", "teach", "ignorance"],
    5:  ["waiting", "patience", "nourishment", "calm", "delay"],
    6:  ["conflict", "dispute", "litigation", "tension"],
    7:  ["army", "discipline", "leader", "march", "rule"],
    8:  ["holding", "union", "alliance", "join", "together"],
    9:  ["small", "taming", "restrain", "accumulate"],
    10: ["treading", "conduct", "careful", "step"],
    11: ["peace", "prosper", "harmony", "flourish"],
    12: ["standstill", "stagnation", "obstruct", "block"],
    13: ["fellowship", "community", "kinship", "share"],
    14: ["great", "possession", "wealth", "abundance"],
    15: ["modesty", "humble", "reduce", "low"],
    16: ["enthusiasm", "delight", "excite", "rouse"],
    17: ["following", "adapt", "respond", "yield"],
    18: ["work", "decay", "repair", "fix", "correct"],
    19: ["approach", "advance", "near", "draw"],
    20: ["contemplation", "view", "observe", "watch"],
    21: ["biting", "through", "decide", "judge"],
    22: ["grace", "adorn", "beauty", "ornament"],
    23: ["splitting", "strip", "decline", "fall"],
    24: ["return", "turning", "renew", "restore"],
    25: ["innocence", "natural", "spontaneous", "true"],
    26: ["great", "taming", "cultivate", "discipline"],
    27: ["nourishment", "jaw", "feed", "sustenance"],
    28: ["great", "excess", "overburden", "critical"],
    29: ["abysmal", "water", "danger", "depth"],
    30: ["clinging", "fire", "light", "radiance"],
    31: ["influence", "wooing", "attract", "stir"],
    32: ["duration", "endure", "persist", "constancy"],
    33: ["retreat", "withdraw", "step back"],
    34: ["great", "power", "strength", "vigor"],
    35: ["progress", "advance", "dawn", "rise"],
    36: ["darkening", "light", "censor", "hide"],
    37: ["family", "household", "kin", "home"],
    38: ["opposition", "estrange", "divergent"],
    39: ["obstruction", "lame", "hindrance"],
    40: ["deliverance", "release", "untangle"],
    41: ["decrease", "reduce", "diminish"],
    42: ["increase", "expand", "augment"],
    43: ["breakthrough", "resoluteness", "decide"],
    44: ["coming", "meet", "encounter", "temptation"],
    45: ["gathering", "assemble", "congregate"],
    46: ["pushing", "upward", "ascend", "grow"],
    47: ["oppression", "exhaustion", "weary"],
    48: ["well", "source", "supply", "draw"],
    49: ["revolution", "molting", "renew", "change"],
    50: ["cauldron", "transform", "vessel", "cook"],
    51: ["arousing", "thunder", "shock", "wake"],
    52: ["keeping", "still", "mountain", "rest"],
    53: ["development", "gradual", "step-by-step"],
    54: ["marrying", "maiden", "subordinate"],
    55: ["abundance", "fullness", "peak"],
    56: ["wanderer", "traveler", "stranger"],
    57: ["gentle", "penetrate", "wind", "subtle"],
    58: ["joyous", "lake", "exchange", "speak"],
    59: ["dispersion", "dissolve", "scatter"],
    60: ["limitation", "restrain", "regulate"],
    61: ["inner", "truth", "sincere", "center"],
    62: ["small", "exceeding", "minor", "detail"],
    63: ["after", "completion", "finished", "done"],
    64: ["before", "completion", "unfinished", "potential"],
}


def _load_identities(path: Optional[str]) -> dict[int, list[str]]:
    """Try to load v0.8.84 I Ching identities; fall back to seed bag."""
    if not path:
        return KING_WEN_SEED
    p = Path(path)
    if not p.exists():
        print(f"[ring-plan] identities path {path} missing; using seed", file=sys.stderr)
        return KING_WEN_SEED
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    # accept a few shapes
    if isinstance(d, dict) and "hexagrams" in d:
        out: dict[int, list[str]] = {}
        for h in d["hexagrams"]:
            kw = h.get("king_wen") or h.get("kw") or h.get("number")
            if kw is None:
                continue
            bag = h.get("keywords") or h.get("themes") or h.get("words") or []
            out[int(kw)] = list(bag) or [h.get("name", f"hex_{kw}")]
        if len(out) == 64:
            return out
    if isinstance(d, dict):
        # plain {1..64: [...]}
        try:
            return {int(k): list(v) for k, v in d.items()}
        except (TypeError, ValueError):
            pass
    print("[ring-plan] identities file shape unrecognized; using seed", file=sys.stderr)
    return KING_WEN_SEED


def _read_trace_rows(trace_path: str) -> list[tuple[int, int, list[int]]]:
    """Single pass: return list of (token, layer, top_k_experts) tuples.
    Cheap because the trace itself is small (< a few MiB even for thousands
    of tokens)."""
    rows: list[tuple[int, int, list[int]]] = []
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            rows.append((int(r["token"]), int(r["layer"]),
                         [int(e) for e in r["top_k_experts"]]))
    print(f"[ring-plan] trace rows: {len(rows)}")
    return rows


def fingerprint_for_layer(rows, layer_idx: int, n_experts: int,
                          vocab_size: int) -> np.ndarray:
    """Build the (n_experts, vocab_size) row-normalized fingerprint for
    ONE layer. Memory: n_experts*vocab*4B (~74 MiB for Qwen3-Coder).
    Per-layer materialization keeps peak RAM bounded regardless of n_layers."""
    fp = np.zeros((n_experts, vocab_size), dtype=np.float32)
    for tok, L, eids in rows:
        if L != layer_idx: continue
        for e in eids:
            if 0 <= e < n_experts and 0 <= tok < vocab_size:
                fp[e, tok] += 1.0
    sums = fp.sum(axis=-1, keepdims=True) + 1e-9
    return fp / sums


def hexagram_signatures(identities: dict[int, list[str]],
                        vocab_size: int, tokenizer) -> np.ndarray:
    """Per-hexagram vocab signature: tokenize each archetype keyword
    and accumulate, then row-normalize. Shape: (64, vocab_size)."""
    sig = np.zeros((64, vocab_size), dtype=np.float32)
    for h in range(1, 65):
        bag = identities.get(h, [f"hexagram_{h}"])
        for word in bag:
            ids = tokenizer.encode(str(word), add_special_tokens=False)
            for t in ids:
                if 0 <= t < vocab_size:
                    sig[h - 1, t] += 1.0
        s = sig[h - 1].sum() + 1e-9
        sig[h - 1] /= s
    return sig


def assign_experts_to_rings(layer_fingerprint: np.ndarray,
                            hex_sig: np.ndarray) -> dict[int, list[int]]:
    """For one layer, produce {ring 0..63: [expert_ids ranked by affinity]}.
    Root pass: each ring claims its top unused expert in strongest-claim
    order. Deviation fill: remaining experts land in the ring whose root
    they most resemble, capped at 64 per ring.

    Takes a single-layer fingerprint matrix (n_experts, vocab) — see
    `fingerprint_for_layer`."""
    F = layer_fingerprint                      # (n_experts, vocab)
    H = hex_sig                                # (64, vocab)
    n_experts = F.shape[0]

    # cosine affinity hex × expert
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
    aff = Hn @ Fn.T                            # (64, n_experts)

    rings: dict[int, list[int]] = {h: [] for h in range(64)}
    used: set[int] = set()

    # root pass — rings with the strongest single-expert claim go first
    claim_strength = aff.max(axis=1)
    for h in np.argsort(-claim_strength):
        for e in np.argsort(-aff[h]):
            e = int(e)
            if e not in used:
                rings[int(h)].append(e)
                used.add(e)
                break

    # deviation fill — each remaining expert into its best-fit ring,
    # capacity 64 per ring
    remaining = [e for e in range(n_experts) if e not in used]
    for e in remaining:
        # rank rings for this expert by affinity to *its root*, falling
        # back to direct affinity if root has been assigned
        scores = []
        for h in range(64):
            scores.append((h, float(aff[h, e])))
        scores.sort(key=lambda x: -x[1])
        for h, _ in scores:
            if len(rings[h]) < 64:
                rings[h].append(int(e))
                break

    return rings


def plan(trace_path: str, n_experts: int, n_layers: int,
         vocab_size: int, tokenizer_name: str,
         identities_path: Optional[str] = None,
         out_path: str = "ring_layout.json") -> str:
    print(f"[ring-plan] loading trace {trace_path}")
    rows = _read_trace_rows(trace_path)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    ids = _load_identities(identities_path)
    print(f"[ring-plan] {len(ids)} hexagram identities loaded "
          f"({'seed' if identities_path is None else identities_path})")

    hex_sig = hexagram_signatures(ids, vocab_size, tok)

    layout: dict[str, dict[int, list[int]]] = {}
    for L in range(n_layers):
        fp_L = fingerprint_for_layer(rows, L, n_experts, vocab_size)
        layout[str(L)] = assign_experts_to_rings(fp_L, hex_sig)
        del fp_L                          # free 74 MiB before next iter
        if (L + 1) % 8 == 0:
            print(f"  layer {L}: planned")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_layers": n_layers,
            "n_experts_per_layer": n_experts,
            "vocab_size": vocab_size,
            "tokenizer": tokenizer_name,
            "identities": identities_path or "(seed)",
            "rings": layout,
        }, f, indent=2)
    print(f"[ring-plan] wrote {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("trace", help="routing-trace JSONL: {token, layer, top_k_experts}")
    p.add_argument("--experts", type=int, required=True, help="experts per layer")
    p.add_argument("--layers",  type=int, required=True, help="MoE layer count")
    p.add_argument("--vocab",   type=int, required=True, help="tokenizer vocab size")
    p.add_argument("--tokenizer", required=True, help="HF tokenizer repo id")
    p.add_argument("--identities", default=None,
                   help="path to v0.8.84 64-hexagram identities JSON")
    p.add_argument("--out", default="ring_layout.json")
    a = p.parse_args()
    plan(a.trace, a.experts, a.layers, a.vocab,
         a.tokenizer, a.identities, a.out)
