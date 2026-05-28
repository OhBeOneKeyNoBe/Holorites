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

RING_BITS = NODE_BITS = SLOT_BITS = 6
RINGS = 1 << RING_BITS    # 64
NODES = 1 << NODE_BITS    # 64
SLOTS = 1 << SLOT_BITS    # 64
CELLS = RINGS * NODES * SLOTS    # 262,144
RING_MASK = RINGS - 1
NODE_MASK = NODES - 1
SLOT_MASK = SLOTS - 1
NODES_TOTAL = RINGS * NODES    # 4,096 distinct (ring, node) addresses

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
    ids = ids.to(torch.long)
    return ((ids >> (NODE_BITS + SLOT_BITS)) & RING_MASK,
            (ids >> SLOT_BITS) & NODE_MASK,
            ids & SLOT_MASK)


# ─── flat ↔ torus reshape ─────────────────────────────────────────────────

def embedding_to_torus(weight: torch.Tensor, pad_token_id: int | None = None) -> torch.Tensor:
    V, D = weight.shape
    if V > CELLS:
        raise ValueError(f"vocab {V} > lattice capacity {CELLS}")
    if V < CELLS:
        pad_row = weight[pad_token_id] if pad_token_id is not None else None
        full = weight.new_zeros((CELLS, D))
        full[:V] = weight
        if pad_row is not None: full[V:] = pad_row
        weight = full
    return weight.view(RINGS, NODES, SLOTS, D).contiguous()

def torus_to_embedding(torus: torch.Tensor) -> torch.Tensor:
    R, N, S, D = torus.shape
    assert (R, N, S) == (RINGS, NODES, SLOTS)
    return torus.view(R * N * S, D)


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
                 prefetch_fanout: int = 8):
        super().__init__()
        torus = embedding_to_torus(weight.detach(), pad_token_id=pad_token_id)
        self.register_buffer("torus", torus.to(cpu_device), persistent=True)
        self.embedding_dim = int(torus.shape[-1])
        self.compute_device = torch.device(compute_device)
        self.max_cached_nodes = int(max_cached_nodes)
        self.helical_prefetch = bool(helical_prefetch)
        # how many nodes-per-access to pull async on each page-in. 8 keeps a
        # working set of helically-adjacent nodes warm so consecutive
        # generation steps don't stall waiting for the next node to copy.
        self.prefetch_fanout = max(0, int(prefetch_fanout))
        self.node_cache: "OrderedDict[tuple[int, int], torch.Tensor]" = OrderedDict()
        self.last_stats: PageStats | None = None

    @property
    def num_embeddings(self): return CELLS
    @property
    def weight(self): return torus_to_embedding(self.torus)

    def _page_in(self, ring: int, node: int, stats: PageStats) -> None:
        key = (ring, node)
        if key in self.node_cache:
            self.node_cache.move_to_end(key); return
        self.node_cache[key] = self.torus[ring, node].to(self.compute_device, non_blocking=True)
        stats.pages_in += 1
        while len(self.node_cache) > self.max_cached_nodes:
            self.node_cache.popitem(last=False); stats.pages_out += 1

    def _prefetch(self, ring: int, node: int, stats: PageStats) -> None:
        """Walk the HoloStream — the helical diagonal — from the active cell.

        One step along the strand changes ring AND node together (the slant
        the ring twist encodes). For active cell (r, n) the next `fanout`
        cells along its Stream are
            (r + k, n + k·q) mod 64    for k = 1 … fanout
        which is exactly `stream_walk(r, n, fanout)`. Stream-coherent
        admission: cache hits cluster along the active strand instead of
        scattering across the grid; eviction naturally drops cold strands
        first because their cells arrived together along the same diagonal.
        """
        if not self.helical_prefetch or self.prefetch_fanout <= 0: return
        for cell_rn in stream_walk(ring, node, self.prefetch_fanout):
            if cell_rn not in self.node_cache and len(self.node_cache) < self.max_cached_nodes:
                self.node_cache[cell_rn] = self.torus[cell_rn[0], cell_rn[1]].to(
                    self.compute_device, non_blocking=True)
                stats.prefetches += 1

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        rings_t, nodes_t, slots_t = cells_for_ids(ids)
        flat_r = rings_t.flatten().tolist()
        flat_n = nodes_t.flatten().tolist()
        flat_s = slots_t.flatten().tolist()
        unique = list({(r, n) for r, n in zip(flat_r, flat_n)})
        stats = PageStats(granularity="node", total_units=NODES_TOTAL)
        for r, n in unique:
            self._page_in(r, n, stats)
            self._prefetch(r, n, stats)
        out = self.torus.new_empty((len(flat_r), self.embedding_dim),
                                   device=self.compute_device)
        for i, (r, n, s) in enumerate(zip(flat_r, flat_n, flat_s)):
            out[i] = self.node_cache[(r, n)][s]
        stats.used_units = len(unique)
        stats.cache_size = len(self.node_cache)
        stats.tokens_in_pass = int(ids.numel())
        # Active (ring, node) cells for the visualizer. Capped at 64 to keep
        # the /stats payload small (each forward typically only hits a handful
        # anyway; this is an upper bound on what the lattice HUD will paint).
        stats.active_cells = [[int(r), int(n)] for r, n in unique[:64]]
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
