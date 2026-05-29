"""moe_streamer.py — expert-router-aware HoloriteStreamer for MoE asset trees.

Designed for trillion-parameter Mixture-of-Experts models (V3, R1, V4-Pro,
V4-Flash) where most parameters are expert FFNs and only a sparse subset
activates per token. The streamer treats experts as the asset granularity
and reads them on demand from a memory-mapped GGUF, never loading the whole
model.

Architectural lineage (acknowledging what we learned from DeepSeek's
infrastructure stack and the broader topology literature):

  * Three-tier cache (MoETuner): GPU-pinned (hot) / host-pinned (warm) /
    NVMe mmap (cold). Promotes/demotes by observed routing frequency.

  * Grouped admit (DeepGEMM `m_grouped_fp8_gemm_nt_masked`): when the
    router selects K experts for the current token, the streamer admits
    them as ONE batched operation, not K serial round-trips. The downstream
    grouped GEMM concatenates tokens along M and computes them as a
    single matmul.

  * Handle caching (DeepEP `EPHandle`): the `(expert_id → tensor offset)`
    map is built once at model load and reused across all forward passes,
    avoiding the GGUF index re-parse per token.

  * Anticipatory prefetch (DeepSeek's training-time anticipatory routing):
    the streamer tracks the per-expert hit count over a sliding window of
    the last N tokens. The top-fanout experts most likely to be picked
    next (by historical frequency) are prefetched on the side stream while
    the current token's compute runs — same overlap principle as DualPipe's
    mutual computation-communication hiding, applied to NVMe→GPU.

  * CSA + HCA hybrid (V4 attention): NOT applied at the expert axis (that's
    a Lattice question for tokens, not for experts). The expert axis has
    its own router-driven walk, which is itself a kind of "sparse top-k
    selection within a dense pool" — directly analogous to CSA's "select
    top-k from a compressed pool" pattern.

What the streamer doesn't do (and shouldn't):

  * It doesn't dequantize. The tensors come off NVMe in their GGUF block
    format (Q4_K, Q6_K, MXFP4, etc.). A downstream kernel (DeepGEMM-style
    or our int4_dequantize equivalent) handles that on the GPU side after
    admit.

  * It doesn't route. The router lives in the model (per layer); the
    streamer is told what experts are needed each forward call.

  * It doesn't replace the body pager. The non-expert tensors (attention
    projections, norms, embedding, head) go through the existing layer-
    granular path. This module is the expert-axis extension on top.
"""
from __future__ import annotations
import os, sys, time, threading, queue, json
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import torch
import gguf
from gguf import GGUFReader, ReaderTensor, GGMLQuantizationType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torus_lattice import (RING_BITS, NODE_BITS, SLOT_BITS,
                            RING_MASK, NODE_MASK, SLOT_MASK,
                            SPIRAL_Q)


# ─── strand walker (videogram receiver) ───────────────────────────────────

class StrandWalker:
    """Deterministic ring sequence for the prefetch stream.

    User's "videogram" insight: with the hexagram-keyed ring layout in
    place, prefetch stops being a prediction problem. Once the gate
    names a starting ring, the helical step (SPIRAL_Q=1, coprime with
    64) gives the next ring positions in a fixed strand — the cache
    just *receives* those positions like a video lookahead buffer.

    The walk is one-dimensional per shell: ring → ring+1 → ring+2 →
    … mod 64. (Node and slot positions are NOT walked — we admit
    the ring's root node for each predicted ring; if the gate picks
    a deviation node it lives in the same ring, so the strand still
    landed in the right neighborhood.)
    """
    __slots__ = ("step",)

    def __init__(self, step: int = SPIRAL_Q):
        # step coprime with 64 → walking k steps gives k DISTINCT rings
        self.step = int(step) if (int(step) & 1) else 1

    def walk(self, start_ring: int, k: int) -> list[int]:
        """k ring positions starting one step past `start_ring`."""
        r = int(start_ring) & RING_MASK
        out = []
        for i in range(1, k + 1):
            out.append((r + i * self.step) & RING_MASK)
        return out

    def walk_from_many(self, start_rings: Iterable[int], k_each: int
                       ) -> list[int]:
        """Walk from each starting ring; deduplicate while preserving
        the strand order. The earliest-seen position wins."""
        seen: set[int] = set()
        out: list[int] = []
        for sr in start_rings:
            for r in self.walk(sr, k_each):
                if r in seen: continue
                seen.add(r)
                out.append(r)
        return out


def load_chunks_index(chunks_path: str) -> Optional[dict]:
    """Read the sidecar's <chunks>.json. Returns None if missing — the
    streamer falls back to its non-layout path in that case."""
    if not chunks_path: return None
    p = Path(str(chunks_path) + ".json")
    if not p.exists(): return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


class _NoCtx:
    """No-op context manager used when the streamer is running on CPU (no
    CUDA streams). Lets the `with stream(): …` blocks compile uniformly."""
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ─── tier classification ──────────────────────────────────────────────────

@dataclass
class CacheTiers:
    """How many experts live in each tier.

    Hot   = always resident on GPU. Includes the shared expert plus the
            top-K most-frequently-routed routed experts by recent history.
    Warm  = pinned host memory; admit to GPU is a single PCIe copy.
    Cold  = on-disk mmap; admit pulls bytes through page cache to host
            pinned, then PCIe to GPU.

    Sizing rule of thumb (V4-Pro at hidden=7168, moe_intermediate=3072, int4):
      one expert (gate+up+down) ≈ 3 * 7168 * 3072 / 2 bytes = ~33 MiB
      GPU 4 GiB - body norms/embed (1 GiB) = ~3 GiB free
      hot = floor(3 GiB / 33 MiB) = ~90 experts can live on GPU
      Per-layer top-90 picks from 384 routed = covers ~80%+ of routes
    """
    hot: int = 32     # GPU-resident expert slots
    warm: int = 128   # pinned host RAM expert slots
    # cold = unlimited (whole asset tree on NVMe)


# ─── expert asset bundle ──────────────────────────────────────────────────

@dataclass
class ExpertAsset:
    """One expert (FFN) as an asset bundle.

    For DeepSeek-V3/V4 layouts an expert is gate_proj + up_proj + down_proj
    (3 tensors). For Qwen3-MoE or Mixtral it's similar. The streamer doesn't
    care about the model class — it groups by `expert_id` and stores the
    constituent ReaderTensors so admit pulls all three in one swoop.
    """
    layer: int
    expert_id: int                 # the routed-expert index (0..n_routed-1)
    tensors: dict[str, ReaderTensor] = field(default_factory=dict)
    is_shared: bool = False        # the always-resident shared expert(s)

    @property
    def n_bytes(self) -> int:
        return sum(t.n_bytes for t in self.tensors.values())

    def names(self) -> list[str]:
        return sorted(self.tensors.keys())


# ─── MoE asset tree (DeepSeek-V3 / V4 / Qwen3-MoE / Mixtral layouts) ──────

# Common expert-naming conventions in GGUF for major MoE families:
#   DeepSeek-V3/V4:  blk.{L}.ffn_gate_exps.weight   (packed all experts)
#                    blk.{L}.ffn_down_exps.weight
#                    blk.{L}.ffn_up_exps.weight
#                    blk.{L}.ffn_gate_shexp.weight  (shared expert)
#                    blk.{L}.ffn_down_shexp.weight
#                    blk.{L}.ffn_up_shexp.weight
#   Mixtral:         blk.{L}.ffn_gate.{E}.weight    (per-expert tensors)
#   Qwen3-MoE:       blk.{L}.ffn_gate_exps.weight   (packed)
#
# The streamer needs to recognize BOTH packed and per-expert layouts.

@dataclass
class MoELayerLayout:
    """Per-layer expert layout discovered from the GGUF tensor names."""
    layer: int
    n_routed: int = 0
    n_shared: int = 0
    is_packed: bool = False    # True for DeepSeek/Qwen layouts where all
                               # routed experts are in one giant tensor
    expert_tensors: dict[str, ReaderTensor] = field(default_factory=dict)
    shared_tensors: dict[str, ReaderTensor] = field(default_factory=dict)
    # non-expert per-layer tensors (attn_q/k/v/o, norms, gate weight)
    other_tensors: dict[str, ReaderTensor] = field(default_factory=dict)


class MoEAssetTree:
    """Mmap-backed asset tree for any MoE GGUF.

    Builds the `(layer, expert_id) → ExpertAsset` index once at construction;
    the GGUFReader's mmap window holds the file. RAM cost is the mmap working
    set (typically < 1 GiB), not the model size.

    For the V4-Pro (1.6T, ~6,000 body tensors expected), the index itself
    is small (a few MiB of Python objects) and constant in cost regardless
    of model size — exactly the videogame-engine asset-tree pattern.
    """
    def __init__(self, gguf_path: str):
        self.path = str(Path(gguf_path).resolve())
        self.reader = GGUFReader(self.path)
        # discover architecture
        kv = {f.name: f for f in self.reader.fields.values()}
        def kvget(*names):
            for n in names:
                if n in kv:
                    parts = kv[n].parts
                    if not parts: return None
                    if kv[n].data:
                        try: return int(parts[kv[n].data[0]])
                        except Exception: pass
                    try:
                        v = parts[-1]
                        return int(v[0]) if hasattr(v, "__len__") else int(v)
                    except Exception: return None
            return None
        def kvstr(*names):
            for n in names:
                if n in kv:
                    parts = kv[n].parts
                    if not parts: return None
                    if kv[n].data:
                        try: return bytes(parts[kv[n].data[0]]).decode("utf-8", "replace")
                        except Exception: pass
                    try: return str(parts[-1])
                    except Exception: return None
            return None
        self.arch = kvstr("general.architecture") or "unknown"
        # vocab — try metadata first, fall back to the embedding tensor's
        # GGUF shape (last axis), since some Qwen3 dumps don't write the
        # vocab_size metadata key.
        embed_tensor = None
        for t in self.reader.tensors:
            if t.name in ("token_embd.weight", "tok_embeddings.weight", "wte.weight"):
                embed_tensor = t; break
        vocab_from_embed = int(embed_tensor.shape[-1]) if embed_tensor is not None else 0
        self.n_layers = (kvget(f"{self.arch}.block_count",
                               f"{self.arch}.n_layer") or 0)
        self.n_routed = (kvget(f"{self.arch}.expert_count",
                               f"{self.arch}.n_expert",
                               f"{self.arch}.expert_used_count") or 0)
        self.n_shared = (kvget(f"{self.arch}.expert_shared_count",
                               f"{self.arch}.n_expert_shared") or 0)
        self.experts_per_tok = (kvget(f"{self.arch}.expert_used_count",
                                       f"{self.arch}.n_expert_per_token") or 0)
        self.hidden_dim   = kvget(f"{self.arch}.embedding_length",
                                  f"{self.arch}.hidden_size") or 0
        self.vocab_size   = (kvget(f"{self.arch}.vocab_size") or vocab_from_embed or 0)
        self.moe_intermediate = kvget(f"{self.arch}.expert_feed_forward_length",
                                       f"{self.arch}.moe_intermediate_size") or 0
        # index tensors by layer + role
        self.layers: dict[int, MoELayerLayout] = {}
        self.globals: dict[str, ReaderTensor] = {}
        import re
        BLK_RX = re.compile(r"^blk\.(\d+)\.(.+)$")
        for t in self.reader.tensors:
            m = BLK_RX.match(t.name)
            if not m: self.globals[t.name] = t; continue
            li, sub = int(m.group(1)), m.group(2)
            layer = self.layers.setdefault(li, MoELayerLayout(layer=li))
            # role classification
            if "exps" in sub or "_exp" in sub:
                # packed all-routed-experts layout (DeepSeek/Qwen)
                layer.is_packed = True
                layer.expert_tensors[sub] = t
            elif "shexp" in sub:
                # shared expert (DeepSeek)
                layer.shared_tensors[sub] = t
                layer.n_shared = max(layer.n_shared, 1)
            elif re.match(r"ffn_(gate|up|down)\.(\d+)\.weight", sub):
                # per-expert layout (Mixtral) — group by expert id
                em = re.match(r"ffn_(gate|up|down)\.(\d+)\.weight", sub)
                proj, eid = em.group(1), int(em.group(2))
                layer.expert_tensors.setdefault(f"e{eid}_{proj}", t)
                layer.n_routed = max(layer.n_routed, eid + 1)
            else:
                layer.other_tensors[sub] = t
        # backfill n_routed if header didn't say
        if not self.n_routed:
            self.n_routed = max((L.n_routed for L in self.layers.values()), default=0)

    def total_bytes(self) -> int:
        return sum(t.n_bytes for t in self.reader.tensors)

    def expert_assets(self, layer_idx: int) -> dict[int, ExpertAsset]:
        """Returns the (expert_id → ExpertAsset) map for the given layer.

        For packed layouts, slicing into the giant `*_exps` tensor is the
        admit primitive — we don't pre-split here; admit_experts() does it
        on demand by computing the byte offset within the packed tensor."""
        layer = self.layers[layer_idx]
        out: dict[int, ExpertAsset] = {}
        # shared expert(s): always-resident
        if layer.shared_tensors:
            out[-1] = ExpertAsset(layer=layer_idx, expert_id=-1,
                                  tensors=dict(layer.shared_tensors),
                                  is_shared=True)
        if layer.is_packed:
            # represent each routed-expert id with a virtual asset; the actual
            # bytes are sliced from the packed tensors on admit.
            for eid in range(self.n_routed):
                out[eid] = ExpertAsset(layer=layer_idx, expert_id=eid,
                                       tensors=dict(layer.expert_tensors),
                                       is_shared=False)
        else:
            # per-expert tensors already split
            grouped: dict[int, dict[str, ReaderTensor]] = defaultdict(dict)
            for name, t in layer.expert_tensors.items():
                # name like "e7_gate"
                eid_part, proj = name.split("_", 1)
                eid = int(eid_part[1:])
                grouped[eid][proj] = t
            for eid, ts in grouped.items():
                out[eid] = ExpertAsset(layer=layer_idx, expert_id=eid,
                                       tensors=ts, is_shared=False)
        return out

    def summary(self) -> dict:
        return {
            "path": self.path, "arch": self.arch,
            "vocab_size": self.vocab_size, "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers, "n_routed_experts": self.n_routed,
            "n_shared_experts": self.n_shared,
            "experts_per_tok": self.experts_per_tok,
            "moe_intermediate": self.moe_intermediate,
            "asset_tree_bytes_on_disk": self.total_bytes(),
        }


# ─── routing frequency tracker (anticipatory prefetch driver) ─────────────

class RoutingHistogram:
    """Sliding-window per-layer per-expert hit counter.

    The streamer asks `top_predicted(layer, k)` to decide which experts to
    speculatively prefetch on the side stream during current-token compute.
    DeepSeek's anticipatory-routing trick (training-time) decoupled backbone
    + router updates; we use the same word for the inference analogue —
    decoupling 'which expert is needed now' from 'which expert will likely
    be needed soon based on history'.

    Decay is exponential per-token: hit count h_t = α·h_{t-1} + 1[picked].
    With α=0.97, a hit's influence halves every 23 tokens — short enough
    that the tracker adapts to topic shifts, long enough that within-topic
    expert hot-paths stay warm.
    """
    def __init__(self, n_layers: int, n_experts: int, decay: float = 0.97):
        self.h = np.zeros((n_layers, n_experts), dtype=np.float32)
        self.decay = float(decay)
        self.tokens_seen = 0

    def update(self, layer: int, expert_ids: Iterable[int]):
        # decay first, then add 1 to the picked experts
        self.h[layer] *= self.decay
        for e in expert_ids:
            if 0 <= e < self.h.shape[1]:
                self.h[layer, e] += 1.0
        self.tokens_seen += 1

    def top_predicted(self, layer: int, k: int) -> list[int]:
        """Top-k experts by current hit frequency for the given layer."""
        if k <= 0 or self.h.shape[1] == 0: return []
        # argsort descending; argpartition is O(n) but argsort is fine here
        idx = np.argsort(-self.h[layer])[:k]
        return idx.tolist()


# ─── three-tier cache + streamer ──────────────────────────────────────────

class ExpertStreamer:
    """Three-tier router-aware expert cache + admit pipeline.

    Lifecycle:
      __init__(tree, layer, tiers)         — wire to one layer's asset map
      route(expert_ids)                    — tell the streamer the router's
                                              selection for the current token;
                                              admits any missing experts on
                                              the main stream, returns a
                                              dict of {eid: gpu_tensor_handle}
      anticipate(k)                        — speculatively prefetch the k
                                              experts most likely to be
                                              routed next (from the histogram)
                                              on the side stream
      evict_to_budget()                    — LRU sweep tier 0 + tier 1
      stats()                              — telemetry for /stats

    The 'gpu_tensor_handle' is opaque: for the prototype it's the dequantized
    fp16 tensor; for production it'd be a (packed_int4_bytes, scales) pair
    handed to a grouped GEMM kernel.
    """
    def __init__(self, tree: MoEAssetTree, layer: int, tiers: CacheTiers,
                 compute_device: torch.device | str,
                 fp_dtype: torch.dtype = torch.float16,
                 histogram: Optional[RoutingHistogram] = None,
                 trajectory: Optional[list] = None,
                 chunks_index: Optional[dict] = None):
        self.tree = tree
        self.layer = layer
        self.tiers = tiers
        self.compute_device = torch.device(compute_device)
        self.fp_dtype = fp_dtype
        # discover the layer's expert map
        self.assets = tree.expert_assets(layer)
        # Optional: chunks-sidecar index from expert_chunkifier (used to
        # resolve eid → ring + node and to drive the strand walker). The
        # index is loaded once per layer and is tiny (a few KiB/layer).
        self.ring_of_eid: dict[int, int] = {}    # eid → ring (0..63)
        self.node_of_eid: dict[int, int] = {}    # eid → node within ring
        self.ring_to_eids: dict[int, list[int]] = {}   # ring → [eid by node]
        if chunks_index is not None:
            li_info = chunks_index.get("layers", {}).get(str(layer), {})
            ring_layout = li_info.get("ring_layout")
            if ring_layout:
                for r_str, eids in ring_layout.items():
                    r = int(r_str)
                    self.ring_to_eids[r] = [int(e) for e in eids]
                    for node_i, e in enumerate(eids):
                        self.ring_of_eid[int(e)] = r
                        self.node_of_eid[int(e)] = node_i
        self.strand = StrandWalker(step=SPIRAL_Q)
        self._recent_rings: list[int] = []      # rings touched on the last route call
        # LRU caches (key = expert_id)
        self.gpu: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self.host: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self.pinned_eids: set[int] = set()   # never evicted (shared + hot promotions)
        self.histogram = histogram
        self.trajectory = trajectory if trajectory is not None else []
        # Three-stream pipeline (lever 6 of the 100 tok/s plan):
        #   compute_stream   — the default stream where matmuls run
        #   admit_stream     — foreground admit triggered by route() misses
        #   prefetch_stream  — speculative anticipatory prefetch from the
        #                      histogram/HoloStream walk
        # With these distinct, the anticipatory load for the NEXT layer's
        # likely experts can start while the CURRENT layer's cache-miss
        # admit is still copying — overlapping NVMe→host→GPU phases the
        # DualPipe way. Hides ~95% of transfer latency at steady state.
        self.admit_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream() if self.compute_device.type == "cuda" else None)
        self.prefetch_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream() if self.compute_device.type == "cuda" else None)
        # admit the shared expert into GPU immediately and pin it
        if -1 in self.assets:
            self.gpu[-1] = self._admit_to_gpu(self.assets[-1])
            self.pinned_eids.add(-1)
        # telemetry
        self.t_pages_in = 0; self.t_pages_out = 0; self.t_prefetches = 0
        self.t_route_calls = 0; self.t_anticipate_hits = 0

    # ── admit (the unit of work — analogous to DeepEP `dispatch`) ────────

    def _expert_slab_bytes(self, packed_t: ReaderTensor) -> int:
        """How many bytes one expert's slab is inside a packed `*_exps` tensor.

        For packed layouts the first axis of the tensor is the expert axis
        (n_experts × ffn_dim × hidden), and the bytes are laid out
        per-expert contiguously. Each expert's slab is always block-aligned
        in the underlying ggml quant (Q4_K = 256-element blocks of 144 B,
        Q6_K = 256-element blocks of 210 B, Q8_0 = 32-element blocks of 34 B,
        F16 = 2 B per element) because the GGUF writer pads each row to a
        block boundary. So integer division of n_bytes by n_experts gives
        the per-expert slab size exactly.
        """
        n_experts = int(packed_t.shape[-1])    # GGUF stores axes in reverse
        return packed_t.n_bytes // n_experts

    def _slice_expert_from_packed(self, packed_t: ReaderTensor, eid: int
                                  ) -> tuple[np.ndarray, list[int]]:
        """Memmap-slice the bytes for one expert out of a packed tensor.

        Returns (bytes_view, per_expert_shape). The bytes_view is still a
        numpy memmap (no copy into RAM); only the .tobytes() / np.array()
        call when shipping to GPU realizes a real allocation. This is the
        16-64× NVMe-traffic reduction that makes MoE sparsity actually
        translate to sparse disk reads.
        """
        slab = self._expert_slab_bytes(packed_t)
        start = eid * slab
        end = start + slab
        # packed_t.data is a numpy memmap view of the GGUF; slice in bytes
        raw = np.asarray(packed_t.data)
        flat = raw.view(np.uint8).reshape(-1)
        byte_view = flat[start:end]
        # GGUF stores tensor axes in REVERSE order (fastest-varying first).
        # The per-expert shape in standard (PyTorch) order is shape[:-1]
        # reversed: for Qwen3 `ffn_gate_exps [2048, 768, 128]` (= GGUF order
        # `[hidden, ffn_dim, n_experts]`), per-expert standard shape is
        # `[ffn_dim, hidden] = [768, 2048]` — the canonical
        # `(output_dim, input_dim)` layout that PyTorch Linear expects.
        per_expert_shape = [int(d) for d in reversed(packed_t.shape[:-1])]
        return byte_view, per_expert_shape

    def _dequant_expert_slab(self, packed_t: ReaderTensor, eid: int
                             ) -> torch.Tensor:
        """Slice + dequant for ONE expert out of a packed tensor. Returns
        an fp16 tensor of shape (ffn_dim, hidden) — the per-expert weight
        matrix ready to feed a matmul.

        Uses GPU dequant when compute_device is cuda (10-30× faster than
        numpy after warmup). Falls back to numpy for CPU compute.
        """
        byte_view, shape = self._slice_expert_from_packed(packed_t, eid)
        n = 1
        for d in shape: n *= d
        if self.compute_device.type == "cuda":
            import moe_kernels as mk
            t = mk.dequant_gpu(packed_t.tensor_type, bytes(byte_view), n,
                               self.compute_device)
            return t.view(*shape).contiguous()
        # CPU fallback — numpy
        import gguf_holoritify as gh
        gt = packed_t.tensor_type
        blob = bytes(byte_view)
        if gt == GGMLQuantizationType.Q4_K:
            QK_K = 256
            return gh._dequant_q4_K(blob, ((n + QK_K-1)//QK_K)*QK_K)[:n].view(*shape).to(torch.float16).contiguous()
        if gt == GGMLQuantizationType.Q6_K:
            QK_K = 256
            return gh._dequant_q6_K(blob, ((n + QK_K-1)//QK_K)*QK_K)[:n].view(*shape).to(torch.float16).contiguous()
        if gt == GGMLQuantizationType.Q8_0:
            return gh._dequant_q8_0(blob, ((n + 31)//32)*32)[:n].view(*shape).to(torch.float16).contiguous()
        if gt == GGMLQuantizationType.Q4_0:
            QK = 32
            return gh._dequant_q4_0(blob, ((n + QK-1)//QK)*QK)[:n].view(*shape).to(torch.float16).contiguous()
        raise NotImplementedError(f"expert-slab dequant for {gt.name} not wired")

    def _admit_to_gpu(self, asset: ExpertAsset) -> dict[str, torch.Tensor]:
        """Pull the asset's bytes through to the GPU (synchronous).

        For PACKED layouts (DeepSeek/Qwen3-MoE), this slices the per-expert
        byte range out of each `*_exps` tensor — so only ~1/n_experts of
        the packed bytes get touched. That's the actual MoE-sparsity win:
        for 8-of-128 routing on Qwen3-Coder, 16× less NVMe traffic per
        token; for V4-Pro's 6-of-385, 64×.

        For PER-EXPERT layouts (Mixtral), each tensor is already that
        expert's slab — no slicing needed, just read.

        For SHARED expert tensors (DeepSeek), same — read the whole tensor.
        """
        out: dict[str, torch.Tensor] = {}
        eid = asset.expert_id
        for name, t in asset.tensors.items():
            # Detect packed vs unpacked: packed if tensor has n_experts axis
            # and the asset is a routed (non-shared) expert.
            is_packed_routed = (not asset.is_shared
                                and "_exps" in name
                                and int(t.shape[-1]) > 1
                                and eid >= 0)
            if is_packed_routed:
                # canonicalize the sub-name to "ffn_gate" / "ffn_up" / "ffn_down"
                # (drop the "_exps" suffix so the consumer can index by role)
                canonical = name.replace("_exps", "")
                ten = self._dequant_expert_slab(t, eid).to(
                    self.compute_device, non_blocking=True)
                out[canonical] = ten
            else:
                # whole-tensor path: shared expert, per-expert (Mixtral) layout,
                # or fp16/bf16 small tensors. Always dequant the full bytes.
                arr = np.array(t.data)
                ten = torch.from_numpy(arr).to(self.compute_device, non_blocking=True)
                out[name] = ten
        self.t_pages_in += 1
        return out

    def _admit_to_host_pinned(self, asset: ExpertAsset) -> dict[str, torch.Tensor]:
        """Stage into pinned host memory (tier 1, lever 1 of the 100 tok/s plan).

        Pinned host memory enables async DMA over PCIe. Promotion tier 1 → 0
        is then a single `to(device, non_blocking=True)` call that the GPU
        can issue while the previous matmul is still finishing. Without
        pinning, the host buffer is pageable and the H2D transfer has to
        synchronize through the CPU — 5-8× slower per miss.

        For packed `*_exps` tensors, we slice per-expert here too so the
        host pinned slab is just the one expert's 884 KiB-ish data, not
        the whole 108 MiB packed tensor.
        """
        out: dict[str, torch.Tensor] = {}
        eid = asset.expert_id
        for name, t in asset.tensors.items():
            is_packed_routed = (not asset.is_shared
                                and "_exps" in name
                                and int(t.shape[-1]) > 1
                                and eid >= 0)
            if is_packed_routed:
                # slice per-expert into pinned bytes first (small allocation)
                byte_view, shape = self._slice_expert_from_packed(t, eid)
                arr = np.array(byte_view)
                ten = torch.from_numpy(arr).pin_memory()
                canonical = name.replace("_exps", "")
                out[canonical] = (ten, shape, t.tensor_type)   # tuple — deferred dequant on GPU side
            else:
                arr = np.array(t.data)
                ten = torch.from_numpy(arr).pin_memory()
                out[name] = ten
        return out

    # ── route: the router told us what to fetch this token ──────────────

    def route(self, expert_ids: list[int]) -> dict[int, dict[str, torch.Tensor]]:
        """Main hot-path. Returns {expert_id → gpu_param_dict} for the
        router's selection. Admits any missing experts inline.

        Mirrors DeepGEMM's grouped-GEMM admit semantics: a batched admit
        (single PCIe pass, ideally) is the unit, not per-expert serial pulls.
        """
        self.t_route_calls += 1
        out: dict[int, dict[str, torch.Tensor]] = {}
        missing_from_gpu: list[int] = []
        for eid in expert_ids:
            if eid in self.gpu:
                out[eid] = self.gpu[eid]
                self.gpu.move_to_end(eid)   # mark MRU
            else:
                missing_from_gpu.append(eid)
        # admit the missing ones, promoting from host pinned if present
        # Issue all missing admits on the dedicated admit_stream so the
        # compute_stream isn't blocked by H2D copies (lever 6).
        ctx = (torch.cuda.stream(self.admit_stream)
               if self.admit_stream is not None else _NoCtx())
        with ctx:
            for eid in missing_from_gpu:
                if eid in self.host:
                    # warm path: pinned host → GPU async, single non-blocking copy
                    host_entry = self.host[eid]
                    gpu_dict = {}
                    for n, val in host_entry.items():
                        if isinstance(val, tuple):
                            # deferred dequant: (pinned_bytes, shape, tensor_type)
                            pinned, shape, tt = val
                            import moe_kernels as mk
                            n_elem = 1
                            for d in shape: n_elem *= d
                            gpu_dict[n] = mk.dequant_gpu(tt, pinned.numpy(),
                                                          n_elem, self.compute_device
                                                          ).view(*shape).contiguous()
                        else:
                            gpu_dict[n] = val.to(self.compute_device, non_blocking=True)
                    self.host.pop(eid)
                else:
                    # cold path: NVMe → host → GPU
                    gpu_dict = self._admit_to_gpu(self.assets[eid])
                # tier 0 admit, evict LRU non-pinned if at capacity. Demote
                # evictions to tier 1 (pinned host) so they're cheap to bring
                # back if the router picks them again on the next token —
                # this is "speculative residency" (lever 7).
                while len(self.gpu) >= self.tiers.hot:
                    victim = None
                    for k in self.gpu:
                        if k in self.pinned_eids: continue
                        victim = k; break
                    if victim is None: break
                    self.gpu.pop(victim)
                    self.t_pages_out += 1
                    # demote: if warm tier has room, stage there
                    if (victim not in self.host
                        and len(self.host) < self.tiers.warm
                        and victim in self.assets):
                        try:
                            self.host[victim] = self._admit_to_host_pinned(
                                self.assets[victim])
                        except Exception: pass
                self.gpu[eid] = gpu_dict
                out[eid] = gpu_dict
        # CROSS-STREAM FENCE (NaN fix, task #12).
        # The admits above ran on self.admit_stream; the caller is about
        # to consume out[*] tensors with bmm on the default compute
        # stream. Without an explicit fence, the matmul reads garbage
        # from memory that admit_stream hasn't finished writing yet --
        # producing NaN downstream. This wait_stream call is a zero-CPU
        # GPU-side barrier: compute waits for admit_stream's pending
        # work without blocking the host.
        if self.admit_stream is not None:
            try:
                torch.cuda.current_stream().wait_stream(self.admit_stream)
            except Exception:
                pass
        # update the routing histogram so anticipate() learns
        if self.histogram is not None:
            self.histogram.update(self.layer, expert_ids)
        # record this token's trajectory through the expert lattice. Each
        # admitted expert at this layer adds a cell to the ray. The
        # vertical_axis module reads these cells back as cos_alignment —
        # the "verticality" of the answer along its trajectory.
        rings_this_call: list[int] = []
        for eid in expert_ids:
            # Use the LAYOUT-derived ring assignment when available
            # (semantic ring from the hexagram planner). Fall back to
            # the bit-slice of the raw eid only when no layout is loaded.
            if eid in self.ring_of_eid:
                ring = self.ring_of_eid[eid]
                node = self.node_of_eid.get(eid, 0)
                slot = eid & SLOT_MASK
            else:
                ring = (eid >> (NODE_BITS + SLOT_BITS)) & RING_MASK
                node = (eid >> SLOT_BITS) & NODE_MASK
                slot = eid & SLOT_MASK
            self.trajectory.append((self.layer, ring, node, slot))
            rings_this_call.append(ring)
        # keep the last few rings so anticipate() can walk strands from them
        self._recent_rings = rings_this_call
        return out

    # ── anticipate: speculative prefetch on the side stream ─────────────

    def anticipate(self, fanout: int = 8, *, recent_eids: list = None,
                   use_geometric: bool = True):
        """Prefetch the top-`fanout` likely-next experts on the side stream.

        Two prediction strategies (use_geometric chooses):

          GEOMETRIC (default) — uses geometric_runtime.holo_walk to
            predict the next experts along the HoloStream from the most
            recently routed ones. If the chunkifier laid experts out so
            co-routed pairs sit on adjacent Stream IDs, walking the
            HoloStream from the active expert IS predicting the next
            most-likely expert by geometry, not by post-hoc statistics.

          HISTOGRAM — falls back to the routing-frequency tracker. Top-k
            most-frequently-routed experts get prefetched. Useful as a
            baseline + when the chunkifier hasn't yet laid out co-routing
            information.

        Both paths fire on the side CUDA stream so the PCIe traffic
        hides under the current token's compute (DualPipe's mutual-
        computation-communication hiding pattern).
        """
        if self.prefetch_stream is None: return
        predicted: list[int] = []
        # ── STRAND path (videogram receiver) ──
        # If the chunkifier has installed a ring layout, this is the
        # only path that matters: walk the strand from the rings the
        # gate just touched, take each predicted ring's root node, and
        # admit. No probability distribution computed; the geometry IS
        # the route. Falls back to the legacy paths only when no layout
        # is loaded (eg. first-run sidecars without ring metadata).
        if self.ring_to_eids and self._recent_rings:
            # walk ~ceil(fanout / starts) rings from each recently-touched ring
            starts = self._recent_rings[:4]
            per_start = max(1, fanout // max(1, len(starts)))
            future_rings = self.strand.walk_from_many(starts, per_start + 1)
            seen = set(self.gpu.keys()) | self.pinned_eids
            for r in future_rings:
                eids_on_r = self.ring_to_eids.get(r, [])
                if not eids_on_r: continue
                root_eid = eids_on_r[0]    # ring's root node (hexagram root)
                if root_eid in seen: continue
                if root_eid not in self.assets: continue
                predicted.append(root_eid)
                seen.add(root_eid)
                if len(predicted) >= fanout: break
        if predicted:
            pass    # strand produced enough — skip legacy paths
        elif use_geometric and recent_eids:
            try:
                from geometric_runtime import Address, holo_walk
                # walk the HoloStream from each recently-routed expert
                seen = set(recent_eids) | set(self.gpu.keys()) | self.pinned_eids
                for eid in recent_eids[:4]:    # take 4 active experts as walk starting points
                    addr = Address.expert(layer_idx=self.layer, expert_id=eid)
                    walked = holo_walk(addr, fanout)
                    for w_addr in walked:
                        w_eid = w_addr.primary[1]
                        if w_eid in seen: continue
                        if w_eid < 0 or w_eid not in self.assets: continue
                        predicted.append(w_eid)
                        seen.add(w_eid)
                        if len(predicted) >= fanout: break
                    if len(predicted) >= fanout: break
            except Exception: pass
        if not predicted and self.histogram is not None:
            # histogram fallback
            top = self.histogram.top_predicted(self.layer, fanout * 2)
            for eid in top:
                if eid in self.gpu or eid in self.pinned_eids: continue
                predicted.append(eid)
                if len(predicted) >= fanout: break
        for eid in predicted:
            if eid in self.gpu or eid in self.pinned_eids: continue
            if eid not in self.assets: continue
            try:
                with torch.cuda.stream(self.prefetch_stream):
                    gpu_dict = self._admit_to_gpu(self.assets[eid])
                if len(self.gpu) < self.tiers.hot:
                    self.gpu[eid] = gpu_dict
                    self.t_prefetches += 1
            except Exception: pass

    # ── promote / demote based on routing frequency (MoETuner-style) ────

    def rebalance(self):
        """Periodically swap the GPU resident set toward the histogram's
        current top-K. Cheap LRU is the default; this is the smarter
        version that uses long-term frequency.

        Call this every few hundred tokens, not every token.
        """
        if self.histogram is None: return
        top = set(self.histogram.top_predicted(self.layer, self.tiers.hot - 1))
        top.add(-1)   # always-pinned shared expert
        # demote anything currently on GPU not in top
        for eid in list(self.gpu.keys()):
            if eid in self.pinned_eids: continue
            if eid not in top:
                self.gpu.pop(eid)
                self.t_pages_out += 1
        # promote missing top to GPU (cold or warm path)
        for eid in top:
            if eid in self.gpu or eid == -1: continue
            if eid not in self.assets: continue
            self.gpu[eid] = self._admit_to_gpu(self.assets[eid])

    # ── telemetry ───────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {"layer": self.layer,
                "tier0_gpu_resident": list(self.gpu.keys()),
                "tier1_host_resident": list(self.host.keys()),
                "pages_in": self.t_pages_in,
                "pages_out": self.t_pages_out,
                "prefetches": self.t_prefetches,
                "route_calls": self.t_route_calls,
                "anticipate_hits": self.t_anticipate_hits}


# ─── CLI: inspect a MoE GGUF for streamability ────────────────────────────

def _summarize(path: str):
    print(f"[moe-streamer] indexing {path}")
    t = MoEAssetTree(path)
    s = t.summary()
    for k, v in s.items():
        if isinstance(v, int) and v > 1024**3:
            print(f"  {k:30s}: {v:,}  ({v/1024**3:.2f} GiB)")
        else:
            print(f"  {k:30s}: {v}")
    # peek at layer 0's expert layout
    if 0 in t.layers:
        L = t.layers[0]
        print(f"  layer 0 layout: packed={L.is_packed}, n_routed={L.n_routed}, "
              f"n_shared={L.n_shared}")
        print(f"  layer 0 expert tensor names: {sorted(L.expert_tensors.keys())[:6]}…")
        print(f"  layer 0 other (non-expert) tensors: {sorted(L.other_tensors.keys())[:8]}…")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:  py moe_streamer.py <model.gguf>")
        sys.exit(1)
    _summarize(sys.argv[1])
