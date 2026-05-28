"""64x64x64 torus lattice — Holorite substrate.

    Active nodes are stationary windows. Streams are the helical diagonals
    that flow through them. Prefetch walks the Stream, not the grid axes.
    The lattice is an apple — its poles meet at the heart, where every
    Stream crosses, so any windowed location eventually observes the whole
    code regardless of the lattice's total size.

Invariants the runtime MUST preserve:
    1. All neighborhood walks use modular arithmetic on every torus axis.
       No clamping. The torus has no edges; the apple's poles meet at the heart.
    2. A prefetch step is ONE diagonal step, not two flat steps.
       (ring + k, node + k·q) mod 64 — both indices advance together, every step.

Four-axis prefetch table (the only axes a runtime should walk):

    ┌────────┬─────────────┬───────────────────────────────────────┬─────────────────┐
    │  Axis  │ Index space │      Neighborhood (k ∈ 1…fanout)      │      Wrap?      │
    ├────────┼─────────────┼───────────────────────────────────────┼─────────────────┤
    │ Body   │ 0 … L−1     │ i + k                                 │ clamp at L−1    │
    │ Ring   │ 0 … 63      │ only via Stream walk (not standalone) │ mod 64          │
    │ Node   │ 0 … 63      │ only via Stream walk (not standalone) │ mod 64          │
    │ Stream │ 0 … 63      │ (ring + k, node + k·q) mod 64         │ mod 64 (closed) │
    └────────┴─────────────┴───────────────────────────────────────┴─────────────────┘

The bit-slice bijection: token id  ->  (ring, node, slot)
    ring = (idx >> 12) & 0x3F   # top 6 bits   — 0..63
    node = (idx >>  6) & 0x3F   # middle 6 bits — 0..63
    slot =  idx        & 0x3F   # low 6 bits    — 0..63

This module is the in-RAM substrate for Holorites:

    RingPagedEmbedding(nn.Module)
        - paging unit: a ring (64*64 = 4,096 ids), 64 cache entries max.
        - kept for benchmarking comparison (the 'before' picture).
    NodePagedEmbedding(nn.Module)
        - paging unit: a single node (64 ids), 4,096 entries max.
        - typical natural-language prompt hits a small subset, ~5x finer than rings.
        - optional helical prefetch: when (ring,node) is loaded we async pull
          (ring, (node+SPIRAL_Q)&0x3F) and ((ring+1)&0x3F, node) so the next
          steps of the sequence don't stall — the brief's spiral δ used at runtime.
    PagedLMHead(nn.Module)
        - drop-in for the LM output projection (vocab x hidden).
        - tied mode (Qwen): reuses the embedding torus directly.
        - untied mode: streams logits ring-by-ring (memory peak = one ring's matmul).

KV-cache paging is noted at the bottom as the next prize (it's dynamic
activations, needs per-layer offload — not a static-weight swap).
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
import torch
import torch.nn as nn

RING_BITS = NODE_BITS = SLOT_BITS = SHELL_BITS = 6
RINGS = 1 << RING_BITS    # 64
NODES = 1 << NODE_BITS    # 64
SLOTS = 1 << SLOT_BITS    # 64
SHELLS = 1 << SHELL_BITS  # 64 — the SHELL axis nests one whole 64³ torus inside another
CELLS = RINGS * NODES * SLOTS                  # 262,144   — the 13th-torus capacity
NESTED_CELLS = SHELLS * RINGS * NODES * SLOTS  # 16,777,216 — the 12th-torus capacity
RING_MASK = RINGS - 1
NODE_MASK = NODES - 1
SLOT_MASK = SLOTS - 1
SHELL_MASK = SHELLS - 1
NODES_TOTAL = RINGS * NODES               # 4,096 distinct (ring, node) addresses (inner)
NODES_TOTAL_NESTED = SHELLS * RINGS * NODES   # 262,144 nodes when the shell axis is active

# Nesting decision: a torus with `level = 4` axes (shell, ring, node, slot)
# wraps the `level = 3` inner one. The 13 nested tori from the visualizer's
# blueprint are conceptually levels 3..15 — each step out adds a 6-bit axis
# and multiplies capacity by 64. For paging substrate we only need 3 and 4:
#     level 3 (inner)  → vocab ≤ 262,144      (Qwen2.5, Mistral, Gemma-3/4)
#     level 4 (nested) → vocab ≤ 16,777,216   (zion'iel-v350/e1 sacred-vocab,
#                                              any future >262k vocab model)
def torus_level_for_vocab(vocab: int) -> int:
    """Pick the smallest nesting level whose capacity holds `vocab`."""
    if vocab <= CELLS: return 3
    if vocab <= NESTED_CELLS: return 4
    # Future-proof: levels 5+ exist conceptually (64⁵ = 1,073,741,824). Not
    # materialized in storage; just bit-slice further.
    raise ValueError(f"vocab {vocab} > level-4 nested torus capacity {NESTED_CELLS}")

# helical spiral from the 3D build brief: q coprime with RINGS=64 → a single
# strand threads all 64*64 cells. q=1 is the gentlest barber-pole.
SPIRAL_Q = 1


# ─── bit-slice bijection ──────────────────────────────────────────────────

def token_to_cell(idx: int) -> tuple[int, int, int]:
    return (idx >> (NODE_BITS + SLOT_BITS)) & RING_MASK, \
           (idx >> SLOT_BITS) & NODE_MASK, \
           idx & SLOT_MASK

def cell_to_token(ring: int, node: int, slot: int) -> int:
    return ((ring & RING_MASK) << (NODE_BITS + SLOT_BITS)) | \
           ((node & NODE_MASK) << SLOT_BITS) | (slot & SLOT_MASK)


# ─── nested (4-axis) bit-slice ────────────────────────────────────────────
# When vocab exceeds 64³, the next torus envelops the inner one: an extra
# 6-bit SHELL axis pushes capacity to 64⁴ = 16,777,216. Tokens 0..262,143
# live in shell=0 (the inner 13th torus); tokens 262,144..16,777,215 live
# in shells 1..63 (the enveloping 12th torus).

def token_to_cell4(idx: int) -> tuple[int, int, int, int]:
    return (((idx >> (RING_BITS + NODE_BITS + SLOT_BITS)) & SHELL_MASK),
            ((idx >> (NODE_BITS + SLOT_BITS)) & RING_MASK),
            ((idx >> SLOT_BITS) & NODE_MASK),
            (idx & SLOT_MASK))


def cell4_to_token(shell: int, ring: int, node: int, slot: int) -> int:
    return (((shell & SHELL_MASK) << (RING_BITS + NODE_BITS + SLOT_BITS)) |
            ((ring  & RING_MASK)  << (NODE_BITS + SLOT_BITS)) |
            ((node  & NODE_MASK)  << SLOT_BITS) |
            (slot   & SLOT_MASK))


def token_to_address(idx: int, level: int):
    """Generic bit-slice — returns a (ring, node, slot) or (shell, ring, node, slot)
    tuple depending on nesting level. Both decompose `idx` losslessly."""
    return token_to_cell4(idx) if level >= 4 else token_to_cell(idx)


# ─── HoloStreams: helical diagonals through the (ring, node) torus ────────
#
# The torus has 64 closed helical strands. With q = SPIRAL_Q coprime to 64,
# the strand starting at (r, n) and stepping by (+1, +q) returns to (r, n)
# after exactly 64 steps. So there are exactly 64 distinct strands and every
# (ring, node) cell lies on exactly one of them.
#
# A cell's "stream id" is the strand label that uniquely identifies which of
# the 64 strands it belongs to. With q=1 the strand label is just (n - r) mod
# 64; with general q the invariant of the strand is (n - q·r) mod 64. We
# expose this so the runtime can talk about Streams as first-class objects.

def stream_id(ring: int, node: int) -> int:
    """The HoloStream label of cell (ring, node).

    Two cells lie on the same Stream iff they share this label. There are
    exactly 64 Streams, each closed (poles meet at the heart).
    """
    return (node - SPIRAL_Q * ring) & NODE_MASK


def stream_walk(ring: int, node: int, fanout: int) -> list[tuple[int, int]]:
    """The next `fanout` cells along the HoloStream starting at (ring, node).

    Single diagonal step: (ring + k, node + k·q) mod 64 — BOTH indices
    advance together every step (the slant the ring twist encodes).
    Returns a fresh list of (ring, node) pairs, k = 1 … fanout.
    """
    return [(((ring + k) & RING_MASK), ((node + SPIRAL_Q * k) & NODE_MASK))
            for k in range(1, max(0, int(fanout)) + 1)]


def stream_window(ring: int, node: int, behind: int, ahead: int) -> list[tuple[int, int]]:
    """A window of cells centered on (ring, node) along its HoloStream.

    Useful for stream-coherent admission: when admitting (r, n), the runtime
    can also admit a small neighborhood of its own Stream (a few cells behind,
    a few ahead) so cache hits cluster along the active strand instead of
    scattering across the grid.
    """
    out = []
    for k in range(-int(behind), int(ahead) + 1):
        if k == 0: continue
        out.append((((ring + k) & RING_MASK), ((node + SPIRAL_Q * k) & NODE_MASK)))
    return out


def cells_for_ids(ids: torch.Tensor):
    """Inner-torus bit-slice for batches of token ids: (ring, node, slot)."""
    ids = ids.to(torch.long)
    return ((ids >> (NODE_BITS + SLOT_BITS)) & RING_MASK,
            (ids >> SLOT_BITS) & NODE_MASK,
            ids & SLOT_MASK)


def cells_for_ids4(ids: torch.Tensor):
    """Nested-torus bit-slice: (shell, ring, node, slot)."""
    ids = ids.to(torch.long)
    return (((ids >> (RING_BITS + NODE_BITS + SLOT_BITS)) & SHELL_MASK),
            ((ids >> (NODE_BITS + SLOT_BITS)) & RING_MASK),
            ((ids >> SLOT_BITS) & NODE_MASK),
            (ids & SLOT_MASK))


def node_index_and_slot_for_ids(ids: torch.Tensor):
    """Layout-agnostic addressing: just (flat node index, slot). Works for
    every nesting level because token `idx` always lives in node `idx >> 6`,
    slot `idx & 0x3F`. The runtime uses this when paging; the (shell, ring,
    node-in-ring) decomposition is purely for the visualizer."""
    ids = ids.to(torch.long)
    return (ids >> SLOT_BITS, ids & SLOT_MASK)


# ─── flat ↔ torus reshape ─────────────────────────────────────────────────

def embedding_to_torus(weight: torch.Tensor, pad_token_id: int | None = None) -> torch.Tensor:
    """Inner 64³ torus only — for vocabularies that fit in 262,144 cells."""
    V, D = weight.shape
    if V > CELLS:
        raise ValueError(f"vocab {V} > lattice capacity {CELLS} — use embedding_to_nested_torus")
    if V < CELLS:
        pad_row = weight[pad_token_id] if pad_token_id is not None else None
        full = weight.new_zeros((CELLS, D))
        full[:V] = weight
        if pad_row is not None: full[V:] = pad_row
        weight = full
    return weight.view(RINGS, NODES, SLOTS, D).contiguous()

def torus_to_embedding(torus: torch.Tensor) -> torch.Tensor:
    if torus.dim() == 5:    # nested (shell, ring, node, slot, D)
        S, R, N, Sl, D = torus.shape
        assert (S, R, N, Sl) == (SHELLS, RINGS, NODES, SLOTS)
        return torus.view(S * R * N * Sl, D)
    R, N, S, D = torus.shape
    assert (R, N, S) == (RINGS, NODES, SLOTS)
    return torus.view(R * N * S, D)


def embedding_to_nested_torus(weight: torch.Tensor, pad_token_id: int | None = None
                              ) -> tuple[torch.Tensor, int]:
    """Pick the smallest nested torus that holds `weight` and reshape into it.

    Returns (torus_tensor, level) — level 3 = (R, N, S, D), level 4 = (Shell,
    R, N, S, D). Sparse-only padding: only the cells the vocab actually fills
    are populated; the rest of the chosen torus stays zero (or `pad_token_id`).
    The 4-level case materializes the full 16,777,216-cell tensor on disk
    ONLY if you call it — for typical hidden sizes this is 30 GiB at level 4.
    Callers that just need to address the cells should keep the embedding
    flat and use `token_to_address(idx, level=4)`; this function is for when
    you actually want the geometric tensor.
    """
    V, D = weight.shape
    level = torus_level_for_vocab(V)
    total = CELLS if level == 3 else NESTED_CELLS
    pad_row = weight[pad_token_id] if pad_token_id is not None else None
    if V < total:
        full = weight.new_zeros((total, D))
        full[:V] = weight
        if pad_row is not None: full[V:] = pad_row
        weight = full
    if level == 3:
        return weight.view(RINGS, NODES, SLOTS, D).contiguous(), 3
    return weight.view(SHELLS, RINGS, NODES, SLOTS, D).contiguous(), 4


# ── flat node-chunked layout (the memory-efficient nested-torus storage) ──
# For vocab in the millions (level 4+), materializing the full nested torus
# is wasteful when only the actual vocab rows are filled. The flat layout
# stores (n_nodes, 64, D) where n_nodes = ceil(V / 64). Each node holds 64
# consecutive token rows — the paging unit. Addressing uses bit-slicing on
# the token id; the layout is the same regardless of nesting level.

def embedding_to_node_chunks(weight: torch.Tensor) -> torch.Tensor:
    """Flat node-chunked layout: (n_nodes, 64, D). Pads to a multiple of 64.

    Memory cost = ceil(V/64) * 64 * D — only ~0.04% overhead vs the raw
    embedding for typical vocabularies. Works for any nesting level because
    the bit-slice still puts token `idx` at chunk `(idx >> 6)`, slot
    `(idx & 0x3F)`.
    """
    V, D = weight.shape
    pad = (-V) % SLOTS
    if pad:
        weight = torch.cat([weight, weight.new_zeros(pad, D)], dim=0)
    n_nodes = weight.shape[0] // SLOTS
    return weight.view(n_nodes, SLOTS, D).contiguous()


def node_chunks_to_torus_address(node_idx: int, level: int) -> tuple:
    """Decompose a node index into the torus axes for the given level.
    level 3: (ring, node_in_ring)         — 4,096 nodes
    level 4: (shell, ring, node_in_ring)  — 262,144 nodes
    """
    if level >= 4:
        return ((node_idx >> (RING_BITS + NODE_BITS)) & SHELL_MASK,
                (node_idx >> NODE_BITS) & RING_MASK,
                node_idx & NODE_MASK)
    return ((node_idx >> NODE_BITS) & RING_MASK, node_idx & NODE_MASK)


# ─── paging stats ─────────────────────────────────────────────────────────

@dataclass
class PageStats:
    granularity: str = "node"
    total_units: int = NODES_TOTAL
    used_units: int = 0
    cache_size: int = 0
    pages_in: int = 0
    pages_out: int = 0
    prefetches: int = 0
    tokens_in_pass: int = 0
    # The (ring, node) cells touched in the last forward — the 13th-torus
    # visualizer in the companion lights these up so you can see which
    # HoloStream strands the active prompt is flowing through.
    active_cells: list = None        # list of [ring, node]
    @property
    def fraction(self) -> float:
        return self.used_units / self.total_units if self.total_units else 0.0


# ─── RingPagedEmbedding (kept for comparison) ─────────────────────────────

class RingPagedEmbedding(nn.Module):
    def __init__(self, weight: torch.Tensor, *, pad_token_id=None,
                 cpu_device="cpu", compute_device="cpu", max_cached_rings=RINGS):
        super().__init__()
        torus = embedding_to_torus(weight.detach(), pad_token_id=pad_token_id)
        self.register_buffer("torus", torus.to(cpu_device), persistent=True)
        self.embedding_dim = int(torus.shape[-1])
        self.compute_device = torch.device(compute_device)
        self.max_cached_rings = max_cached_rings
        self.ring_cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        self.last_stats: PageStats | None = None

    @property
    def num_embeddings(self): return CELLS
    @property
    def weight(self): return torus_to_embedding(self.torus)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        rings_t, nodes_t, slots_t = cells_for_ids(ids)
        used = sorted({int(r) for r in rings_t.flatten().tolist()})
        s = PageStats(granularity="ring", total_units=RINGS)
        for r in used:
            if r not in self.ring_cache:
                self.ring_cache[r] = self.torus[r].to(self.compute_device, non_blocking=True)
                s.pages_in += 1
            else:
                self.ring_cache.move_to_end(r)
        while len(self.ring_cache) > self.max_cached_rings:
            self.ring_cache.popitem(last=False); s.pages_out += 1
        out = self.torus.new_empty((int(ids.numel()), self.embedding_dim),
                                   device=self.compute_device)
        flat_r = rings_t.flatten().tolist()
        flat_n = nodes_t.flatten().tolist()
        flat_s = slots_t.flatten().tolist()
        for i, (r, n, sl) in enumerate(zip(flat_r, flat_n, flat_s)):
            out[i] = self.ring_cache[r][n, sl]
        s.used_units = len(used); s.cache_size = len(self.ring_cache); s.tokens_in_pass = int(ids.numel())
        self.last_stats = s
        return out.view(*ids.shape, self.embedding_dim)


# ─── NodePagedEmbedding — the upgrade ─────────────────────────────────────

class NodePagedEmbedding(nn.Module):
    """Drop-in for nn.Embedding(V, D); paging unit is a single NODE (64 ids).

    With helical prefetch the spiral δ from the 3D brief is *used at runtime*
    (not just stored). When node (r, n) is paged in we also queue async copies
    of (r, (n+SPIRAL_Q)&63) and ((r+1)&63, n) — the helical and the next-ring
    neighbours — so the next forward pass usually finds them already on device.
    """
    def __init__(self, weight: torch.Tensor, *, pad_token_id=None,
                 cpu_device="cpu", compute_device="cpu",
                 max_cached_nodes: int = NODES_TOTAL,
                 helical_prefetch: bool = True,
                 prefetch_fanout: int = 8,
                 vocab_size: int | None = None):
        super().__init__()
        # `weight` here can already be node-chunked (n_nodes, 64, D) from
        # a Holorite manifest's saved torus, or a flat (V, D) HF embedding.
        # Either way we normalize to the flat node-chunked layout so the
        # paging logic is shape-agnostic to nesting level. Nesting level is
        # inferred from how many nodes ended up in the chunked tensor.
        w = weight.detach()
        if w.dim() == 4:
            # legacy inner shape (R, N, S, D) — squash R,N into n_nodes
            R, N, S, D = w.shape
            w = w.view(R * N, S, D)
        elif w.dim() == 5:
            # nested shape (Shell, R, N, S, D) — squash Shell,R,N into n_nodes
            Sh, R, N, S, D = w.shape
            w = w.view(Sh * R * N, S, D)
        elif w.dim() == 3:
            pass    # already (n_nodes, 64, D)
        elif w.dim() == 2:
            w = embedding_to_node_chunks(w)
        else:
            raise ValueError(f"unexpected embedding tensor rank {w.dim()}")
        self._vocab = int(vocab_size) if vocab_size is not None else w.shape[0] * w.shape[1]
        self.n_nodes = int(w.shape[0])
        # nesting level decides how (ring, node) decompose into addressing axes
        if self.n_nodes <= NODES_TOTAL:
            self._level = 3
        elif self.n_nodes <= NODES_TOTAL_NESTED:
            self._level = 4
        else:
            raise ValueError(f"n_nodes {self.n_nodes} exceeds level-4 capacity")
        self.register_buffer("torus", w.to(cpu_device).contiguous(), persistent=True)
        self.embedding_dim = int(w.shape[-1])
        self.compute_device = torch.device(compute_device)
        self.max_cached_nodes = int(max_cached_nodes)
        self.helical_prefetch = bool(helical_prefetch)
        self.prefetch_fanout = max(0, int(prefetch_fanout))
        self.node_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
        self.last_stats: PageStats | None = None

    @property
    def num_embeddings(self): return CELLS
    @property
    def weight(self): return torus_to_embedding(self.torus)

    def _page_in_node(self, node_idx: int, stats: PageStats) -> None:
        """Layout-agnostic page-in. The torus is (n_nodes, 64, D) flat; we
        slice the chunk for `node_idx` and admit it. Works the same whether
        the model is using the inner 64³ torus or the nested 64⁴ one."""
        if node_idx in self.node_cache:
            self.node_cache.move_to_end(node_idx); return
        self.node_cache[node_idx] = self.torus[node_idx].to(self.compute_device, non_blocking=True)
        stats.pages_in += 1
        while len(self.node_cache) > self.max_cached_nodes:
            self.node_cache.popitem(last=False); stats.pages_out += 1

    def _prefetch_holostream(self, node_idx: int, stats: PageStats) -> None:
        """Walk the HoloStream from the active node. For level-3 (inner)
        models the walk runs in (ring, node-in-ring) space — the canonical
        helical diagonal. For level-4 (nested) we walk inside the active
        shell first; the runtime visualizer can paint the shell boundary
        when k overflows back to shell+1."""
        if not self.helical_prefetch or self.prefetch_fanout <= 0: return
        if self._level == 3:
            r = (node_idx >> NODE_BITS) & RING_MASK
            n = node_idx & NODE_MASK
            for (rr, nn) in stream_walk(r, n, self.prefetch_fanout):
                cand = (rr << NODE_BITS) | nn
                if cand not in self.node_cache and len(self.node_cache) < self.max_cached_nodes:
                    self.node_cache[cand] = self.torus[cand].to(
                        self.compute_device, non_blocking=True)
                    stats.prefetches += 1
        else:
            # nested: walk the helix WITHIN the current shell; the same
            # spiral δ applies and a single strand still threads 64×64 nodes.
            sh = (node_idx >> (RING_BITS + NODE_BITS)) & SHELL_MASK
            r  = (node_idx >> NODE_BITS) & RING_MASK
            n  = node_idx & NODE_MASK
            for (rr, nn) in stream_walk(r, n, self.prefetch_fanout):
                cand = (sh << (RING_BITS + NODE_BITS)) | (rr << NODE_BITS) | nn
                if cand >= self.n_nodes: continue
                if cand not in self.node_cache and len(self.node_cache) < self.max_cached_nodes:
                    self.node_cache[cand] = self.torus[cand].to(
                        self.compute_device, non_blocking=True)
                    stats.prefetches += 1

    # back-compat alias used by older callers
    def _page_in(self, ring: int, node: int, stats: PageStats) -> None:
        self._page_in_node((ring << NODE_BITS) | node, stats)
    def _prefetch(self, ring: int, node: int, stats: PageStats) -> None:
        self._prefetch_holostream((ring << NODE_BITS) | node, stats)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # flat node index = idx >> 6 ; slot = idx & 0x3F.  Level-agnostic.
        nodes_t, slots_t = node_index_and_slot_for_ids(ids)
        flat_n = nodes_t.flatten().tolist()
        flat_s = slots_t.flatten().tolist()
        unique = sorted({int(n) for n in flat_n})
        stats = PageStats(granularity="node",
                          total_units=(NODES_TOTAL_NESTED if self._level == 4 else NODES_TOTAL))
        for n in unique:
            self._page_in_node(n, stats)
            self._prefetch_holostream(n, stats)
        out = self.torus.new_empty((len(flat_n), self.embedding_dim),
                                   device=self.compute_device)
        for i, (n, s) in enumerate(zip(flat_n, flat_s)):
            out[i] = self.node_cache[n][s]
        stats.used_units = len(unique)
        stats.cache_size = len(self.node_cache)
        stats.tokens_in_pass = int(ids.numel())
        # Active cells for the visualizer — decompose each touched node into
        # the level-appropriate axes so the HUD paints the correct shell/ring/node.
        cells = []
        for n in unique[:64]:
            if self._level == 4:
                sh = (n >> (RING_BITS + NODE_BITS)) & SHELL_MASK
                r  = (n >> NODE_BITS) & RING_MASK
                nn = n & NODE_MASK
                cells.append([int(sh), int(r), int(nn)])
            else:
                r  = (n >> NODE_BITS) & RING_MASK
                nn = n & NODE_MASK
                cells.append([int(r), int(nn)])
        stats.active_cells = cells
        self.last_stats = stats
        return out.view(*ids.shape, self.embedding_dim)


# ─── PagedLMHead — output projection ──────────────────────────────────────

class PagedLMHead(nn.Module):
    """Drop-in for the LM head (Linear(hidden, vocab) without bias).

    a) TIED mode: tied_with = a NodePagedEmbedding. We compute
       logits = hidden @ tied.torus_view.T  on the compute device, paging the
       embedding into a single contiguous tensor for the matmul. This is free
       when the model ties weights (Qwen2.5 does) — paging the embedding
       *is* paging the LM head.
    b) UNTIED mode: holds its own torus. Computes logits one ring at a time
       (4096 vocab rows per matmul). Memory peak per step = one ring × D.
    """
    def __init__(self, *, tied_with: NodePagedEmbedding | None = None,
                 weight: torch.Tensor | None = None, pad_token_id=None,
                 cpu_device="cpu", compute_device="cpu"):
        super().__init__()
        if tied_with is None and weight is None:
            raise ValueError("PagedLMHead needs `tied_with` or `weight`")
        self.tied = tied_with
        self.compute_device = torch.device(compute_device)
        if tied_with is None:
            torus = embedding_to_torus(weight.detach(), pad_token_id=pad_token_id)
            self.register_buffer("torus", torus.to(cpu_device), persistent=True)
            self.embedding_dim = int(torus.shape[-1])
        else:
            self.embedding_dim = tied_with.embedding_dim

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.tied is not None:
            full = self.tied.torus.view(CELLS, self.embedding_dim).to(
                self.compute_device, non_blocking=True)
            return torch.matmul(hidden, full.T)
        out_shape = hidden.shape[:-1] + (CELLS,)
        out = hidden.new_empty(out_shape, device=self.compute_device)
        for r in range(RINGS):
            chunk = self.torus[r].view(NODES * SLOTS, self.embedding_dim).to(
                self.compute_device, non_blocking=True)
            out[..., r * NODES * SLOTS:(r + 1) * NODES * SLOTS] = \
                torch.matmul(hidden, chunk.T)
        return out


# ─── KV cache paging — stub + plan ────────────────────────────────────────
#
# Status: NOT IMPLEMENTED in this pass.
#
# Why: the KV cache is dynamic *activations*, not static weights. At each
# attention call, the model needs *every position's* K and V to compute scores.
# Naive eviction of any past position breaks the context.
#
# What WOULD work:
#   - per-layer KV offload (active layer's KV on GPU during its attention;
#     page out as soon as that layer's attention completes; page in layer L+1
#     before its attention).
#   - sliding-window / sparse attention (Mistral, Phi) — paged-evict positions
#     outside the window for free.
#   - GQA models share K/V across heads → roughly Q-fold smaller KV.
#
# Both belong in a separate kv_pager.py integrated at the model's attention
# layer (deeper hook than set_input_embeddings).

class _KVCachePagedStub:
    pass


# ─── self-test ────────────────────────────────────────────────────────────

def _selftest():
    for idx in (0, 1, 63, 64, 4095, 4096, 262_143):
        r, n, s = token_to_cell(idx)
        assert cell_to_token(r, n, s) == idx
    W = torch.randn(150_000, 8)
    npe = NodePagedEmbedding(W)
    flat = nn.Embedding.from_pretrained(W, freeze=True)
    sample = torch.tensor([0, 1, 64, 4095, 100_000, 149_999])
    assert torch.equal(npe(sample), flat(sample)), "NodePagedEmbedding diverged"
    s = npe.last_stats
    print(f"selftest OK  used {s.used_units}/{NODES_TOTAL} nodes "
          f"({s.fraction:.1%}) on 6-token sample; prefetches={s.prefetches}")

if __name__ == "__main__":
    _selftest()
