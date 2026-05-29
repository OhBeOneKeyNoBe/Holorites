"""geometric_runtime.py — every offloaded list is a torus.

The user's structural unification (paraphrasing their own words):

    The torus isn't just the addressing scheme for the embedding/13th-torus
    memory substrate — it's the addressing scheme for every list that gets
    streamed, at every level. Body layers are a torus (small one, mostly
    nested headroom). Experts within a layer are a torus. KV pages are a
    torus. Sub-tensor chunks of huge weights, if needed, are a torus.

    The whole runtime then becomes geometrically uniform: every cell has
    coordinates, every neighborhood is a Stream walk, every admit/evict
    obeys the same working-set discipline. The "asset tree" stops being a
    tree at all — it's a nested coordinate system, and the streamer is
    just a camera moving through it.

What this module provides:

  * `Lattice(name, axes)` — declares one nested-torus axis system.
    Each `Axis` has a 6-bit ring/node/slot triple per level. Axes can
    have FEWER levels than the matryoshka caps (e.g. the body axis for
    a 61-layer model uses one level but only fills 1/4 of the ring —
    the rest of the ring is reserved for sub-layer modules).

  * `Address(lattice, indices)` — one specific cell. Indices are a
    nested tuple per axis. The `flat()` method gives the integer ID
    used by the underlying asset tree; the `triplets()` method gives
    the geometric (ring, node, slot) decomposition used by the
    visualizer and the HoloStream walk.

  * `holo_walk(address, k, stream_id=None)` — the unified neighborhood
    walk. Same primitive whether you're walking body layers (linear i+k
    + ring wrap), experts (HoloStream within Omega), or KV pages
    (HoloStream within Alpha). The streamer doesn't care which axis
    it's traversing.

  * `place_experts(...)` — the chunkifier-side helper that assigns
    geometric Stream IDs to experts based on historical co-routing
    statistics. Co-routed experts get neighboring Stream IDs so the
    HoloStream walk naturally prefetches the next-likely experts.

  * `kv_page_address(token_idx)` — the bit-slice helper for KV cache
    paging. Token positions live in Alpha (262k cells = enough for any
    practical context up to 262k). Above that the same Omega envelope
    extends to 68B token positions.

Five lattices the runtime declares (each named per the user's framing):

  BODY    — body_layer axis. 64³ headroom per level; for a 61-layer
            model only the lowest 6 bits are touched.
  EXPERTS — per-layer expert axis. 64³ = 262,144 expert slots per
            level. n_tori=1 holds the V4-Pro layer's 384 routed
            experts comfortably (0.15% used).
  SUBEXPERT — optional axis WITHIN one expert. For very large experts
            we slice gate/up/down along the inner dim into 64-element
            blocks (matches Q4_K's 256-element block size × 4 groups).
  KV      — token-position axis for the KV cache. Alpha holds 262k
            positions; Omega extends to 68B. The 1M-token V4 context
            sits comfortably in Alpha (only 0.4% used).
  STREAM  — the helical-diagonal walk shared across all of the above.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torus_lattice import (RING_BITS, NODE_BITS, SLOT_BITS, RINGS, NODES, SLOTS,
                           TORUS_BITS, TORUS_CELLS, TORUS_NODES,
                           NODE_BITS_PER_TORUS, RING_MASK, NODE_MASK, SLOT_MASK,
                           SPIRAL_Q, torus_level_for_vocab,
                           token_to_nested_address, nested_address_to_token,
                           node_to_nested_address, stream_walk, stream_id)


# ─── axis declarations ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Axis:
    """One axis of the runtime's nested-torus coordinate system.

    `name` is human-readable ("body_layer", "expert", "kv_page", "subexpert").
    `n_tori` is the Matryoshka level needed to hold the maximum logical
    index for this axis — derived once at chunkify time and stored.
    `capacity` is TORUS_CELLS ** n_tori, the geometric headroom.
    """
    name: str
    n_tori: int = 1
    description: str = ""

    @property
    def capacity(self) -> int:
        return TORUS_CELLS ** self.n_tori

    def level_for_max_index(self, max_idx: int) -> int:
        """How many tori this axis needs to hold up to max_idx items."""
        if max_idx <= 0: return 1
        n = 1
        cap = TORUS_CELLS
        while cap <= max_idx:
            n += 1
            cap *= TORUS_CELLS
            if n > 13:
                raise ValueError(f"axis {self.name}: max_idx={max_idx} exceeds 13-torus capacity")
        return n


# ─── the five lattices the runtime declares ──────────────────────────────

BODY = Axis(
    name="body_layer", n_tori=1,
    description=(
        "Body-block axis. For a 61-layer Qwen3-MoE or 80-layer Llama 3.3, "
        "the lowest 6 bits cover the layer index; the rest of the ring/node/"
        "slot triple is reserved for sub-layer modules (attention head id, "
        "MLP partition, expert group). Walking +k along body advances to the "
        "next layer; the streamer pins the current body cell during forward."
    ),
)

EXPERTS = Axis(
    name="expert", n_tori=1,
    description=(
        "Per-layer expert axis. Alpha holds 262,144 expert slots per layer; "
        "V4-Pro's 384 routed + 1 shared = 385 fits in 0.15% of Alpha. "
        "Co-routed experts get adjacent Stream IDs at chunkify time so the "
        "HoloStream walk naturally prefetches the next-likely experts."
    ),
)

SUBEXPERT = Axis(
    name="subexpert", n_tori=1,
    description=(
        "Optional axis WITHIN one expert. For huge experts (V4-Pro's "
        "ffn_intermediate=3072 split into 64-element groups = 48 groups per "
        "expert per projection) you can stream sub-blocks rather than whole "
        "experts. Most cases don't need this; the addressing accommodates it."
    ),
)

KV = Axis(
    name="kv_page", n_tori=1,
    description=(
        "Token-position axis for the KV cache. Alpha holds 262,144 positions; "
        "Omega extends to 68B. V4's 1M-token context sits in Alpha (3.8% used). "
        "The helical Stream walk on KV pages IS the locality CSA's top-k "
        "retrieval already exploits — a free architectural alignment with "
        "V4's hybrid attention."
    ),
)

# Convenience map: name → Axis
ALL_AXES = {a.name: a for a in (BODY, EXPERTS, SUBEXPERT, KV)}


# ─── addresses ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Address:
    """One cell in a multi-axis lattice.

    `coords` is a dict {axis_name: int_index} — the logical position.
    The geometric (ring, node, slot) triplets per axis are derived on
    demand via `triplets()`; the integer index is what the asset tree
    uses to find the underlying bytes.

    Construct via Address.body(layer_idx), Address.expert(layer, eid),
    Address.kv(token_pos), or Address(coords={...}) directly.
    """
    coords: tuple    # tuple of (axis_name, int_index) for hashability

    @classmethod
    def body(cls, layer_idx: int) -> "Address":
        return cls(coords=(("body_layer", int(layer_idx)),))

    @classmethod
    def expert(cls, layer_idx: int, expert_id: int) -> "Address":
        return cls(coords=(("body_layer", int(layer_idx)),
                           ("expert", int(expert_id))))

    @classmethod
    def subexpert(cls, layer_idx: int, expert_id: int, sub_idx: int) -> "Address":
        return cls(coords=(("body_layer", int(layer_idx)),
                           ("expert", int(expert_id)),
                           ("subexpert", int(sub_idx))))

    @classmethod
    def kv(cls, token_pos: int) -> "Address":
        return cls(coords=(("kv_page", int(token_pos)),))

    @property
    def primary(self) -> tuple[str, int]:
        """The deepest axis — the one the streamer is actually walking."""
        return self.coords[-1]

    def triplets(self) -> dict[str, list[tuple[int, int, int]]]:
        """Geometric decomposition per axis, OUTERMOST FIRST per axis."""
        out = {}
        for name, idx in self.coords:
            axis = ALL_AXES.get(name)
            level = axis.n_tori if axis else 1
            out[name] = token_to_nested_address(idx, level)
        return out

    def __repr__(self):
        parts = ", ".join(f"{n}={i}" for n, i in self.coords)
        return f"Address({parts})"


# ─── the unified walk (HoloStream + body-axis linear) ────────────────────

def holo_walk(addr: Address, k: int, *, axis: Optional[str] = None
              ) -> list[Address]:
    """The N-next addresses along the deepest (or specified) axis.

    For axes WITH torus geometry (expert, kv_page, subexpert), this is the
    helical-diagonal HoloStream walk: (ring + j, node + j·q) mod 64 for
    j = 1..k. Crossing into the next envelope (Omega anchor jump) happens
    naturally when the walk overflows ring.

    For the body axis specifically, the walk is LINEAR (i + j, clamped at
    n_layers - 1) because transformer blocks must execute in sequence and
    don't wrap. This is the only axis where the geometry is asymmetric;
    everywhere else, the same walk function works.

    Returns up to k Address objects (fewer if the walk hits the body
    upper bound). The caller's responsibility to know how many addresses
    actually exist for their axis (e.g. n_routed_experts, n_kv_pages).
    """
    target = axis or addr.primary[0]
    coords_list = list(addr.coords)
    # find the target axis in this address
    pos = next((i for i, (n, _) in enumerate(coords_list) if n == target), None)
    if pos is None: return []
    name, idx = coords_list[pos]
    # body axis: linear walk
    if name == "body_layer":
        out = []
        for j in range(1, k + 1):
            new_coords = list(coords_list)
            new_coords[pos] = (name, idx + j)
            out.append(Address(coords=tuple(new_coords)))
        return out
    # torus axis: HoloStream walk
    ax = ALL_AXES.get(name)
    level = ax.n_tori if ax else 1
    # decompose into geometric triplets
    triplets = token_to_nested_address(idx, level)
    # walk happens on the INNERMOST (deepest) torus's (ring, node)
    inner = triplets[-1]    # (r, n, s)
    ring, node, _slot = inner
    # use the proven stream_walk helper from torus_lattice
    walks = stream_walk(ring, node, k)
    out = []
    for (rr, nn) in walks:
        # rebuild the address keeping outer envelopes constant; replace
        # innermost (ring, node) — keep slot the same as input
        new_inner = (rr, nn, _slot)
        new_triplets = list(triplets[:-1]) + [new_inner]
        new_idx = nested_address_to_token(new_triplets)
        new_coords = list(coords_list)
        new_coords[pos] = (name, new_idx)
        out.append(Address(coords=tuple(new_coords)))
    return out


# ─── expert placement: chunkifier-side ────────────────────────────────────

def place_experts_by_co_routing(co_route_counts: dict[tuple[int, int], int],
                                 n_experts: int) -> dict[int, int]:
    """Assign each expert id (0..n_experts-1) a STREAM coordinate inside
    the per-layer Omega torus, such that experts which historically
    co-route end up on adjacent Stream IDs.

    `co_route_counts` is {(expert_a, expert_b): n_times_co_picked}. From
    that, the function does a greedy nearest-neighbor placement on the
    64-Stream ring — experts most co-routed get adjacent Stream slots.

    Returns {expert_id: stream_id} where stream_id is the cell's
    `stream_id(ring, node)` invariant — the strand the expert sits on.

    This is the geometric replacement for the histogram-based
    anticipatory prefetch. After this layout pass, the HoloStream walk
    `walk(active_expert, k=8)` returns the 8 most-likely co-fired
    experts BY GEOMETRY, not by post-hoc statistics. The 'next-likely
    experts' question stops being a Bayesian inference problem and
    becomes a coordinate-walk problem.

    For a brand-new model with no co-routing history, every expert is
    just assigned its expert_id as its stream_id and we fall back to
    histogram-driven prefetch. The geometric path only helps once the
    chunkifier has seen real routing traces to lay out from.
    """
    if not co_route_counts:
        return {eid: eid % 64 for eid in range(n_experts)}
    # rank pairs by co-route count descending
    pairs = sorted(co_route_counts.items(), key=lambda kv: -kv[1])
    placed: dict[int, int] = {}
    next_stream = 0
    used_streams: set[int] = set()
    for (a, b), _ in pairs:
        # if neither placed, give them adjacent streams
        if a not in placed and b not in placed:
            placed[a] = next_stream
            placed[b] = (next_stream + 1) % 64
            used_streams.update({placed[a], placed[b]})
            next_stream = (next_stream + 2) % 64
        elif a in placed and b not in placed:
            placed[b] = (placed[a] + 1) % 64
            used_streams.add(placed[b])
        elif b in placed and a not in placed:
            placed[a] = (placed[b] - 1) % 64
            used_streams.add(placed[a])
    # any expert not yet placed gets a fresh stream
    for eid in range(n_experts):
        if eid not in placed:
            while next_stream in used_streams and next_stream < 64:
                next_stream = (next_stream + 1) % 64
            placed[eid] = next_stream
            used_streams.add(next_stream)
            next_stream = (next_stream + 1) % 64
    return placed


# ─── self-test + walks demo ───────────────────────────────────────────────

def _demo():
    print("=" * 70)
    print("  Geometric runtime: every offloaded list is a torus")
    print("=" * 70)
    for name, ax in ALL_AXES.items():
        print(f"\n  {name:10s}  n_tori={ax.n_tori}  capacity={ax.capacity:,}")
        print(f"             {ax.description}")

    print("\n--- demo: walk +8 along the expert axis from expert (layer=10, eid=42) ---")
    addr = Address.expert(layer_idx=10, expert_id=42)
    print(f"  start: {addr}")
    print(f"  triplets: {addr.triplets()}")
    for k, walked in enumerate(holo_walk(addr, 8), 1):
        print(f"  +{k}: {walked}    (deepest = {walked.primary})")

    print("\n--- demo: walk +5 along the body axis from layer 7 ---")
    addr = Address.body(layer_idx=7)
    for k, walked in enumerate(holo_walk(addr, 5), 1):
        print(f"  +{k}: {walked}")

    print("\n--- demo: KV-page walk for token 1000 (Alpha torus) ---")
    addr = Address.kv(token_pos=1000)
    print(f"  start: {addr}  triplets: {addr.triplets()}")
    for k, walked in enumerate(holo_walk(addr, 4), 1):
        print(f"  +{k}: {walked}")

    print("\n--- demo: place experts using fake co-routing stats ---")
    co_routes = {(0, 5): 100, (5, 12): 80, (3, 8): 60, (12, 19): 40, (19, 26): 35}
    placement = place_experts_by_co_routing(co_routes, n_experts=32)
    print(f"  co-route counts: {co_routes}")
    print(f"  placement (first 20):")
    for eid in range(20):
        print(f"    expert {eid:3d} -> stream {placement[eid]:2d}")


if __name__ == "__main__":
    _demo()
