"""body_pager.py — extend Holorite paging to the transformer body.

The embedding + LM head paging in torus_lattice.py covers the two big lookup
tables. The transformer body — every layer's attention projections, MLP
weights, norms — was still fully resident on GPU, which is why a 7B model
wouldn't fit on a 4 GiB card and why even the small ones run with the GPU
saturated (slow).

This module adds the same discipline to the body:

    PagedTransformerLayer(layer, compute_device, pager, ...)
        Wraps a single transformer block. Its parameters live on the CPU
        master copy. On forward(), the weights are copied to the compute
        device just before the call, the original forward runs, then —
        only when GPU budget is tight — the layer is evicted.

    BodyPager(working_set, prefetch_fanout, compute_device)
        Tracks which wrapped layers currently hold GPU storage. Keeps up
        to `working_set` layers resident at once on an LRU policy; admits
        the active layer + the next few prefetched ones first. When budget
        is reached, evicts the least-recently-used layer.

    paged_body(model, compute_device, working_set=None, prefetch_fanout=8)
        Walks the model, finds the transformer layers (the .layers
        ModuleList common to Llama / Qwen / Mistral / Phi-family), wraps
        each one, sizes the working set automatically to the GPU budget,
        and arranges helical prefetch.

Byte-exact, no quality loss — same contract as the embedding pager.

Why a working set?
    The previous version evicted every layer at the end of its own forward.
    For a 32-layer 7B body (~14 GiB) that meant streaming the entire body
    across PCIe every single token — ~14 GB / token = the 0.5 tok/s we saw.
    With a working-set budget:
      - 0.5B body (~250 MiB):   fits whole → instant, no eviction at all.
      - 1.5B body (~1.4 GiB):   fits whole on 4 GB → no eviction.
      - 7B  body (~14 GiB):     only the working-set fraction streams; the
                                hot tail of recent layers stays resident.
"""
from __future__ import annotations
from typing import Optional
from collections import OrderedDict
import torch
import torch.nn as nn


def _iter_body_layers(model: nn.Module):
    """Return the ModuleList of transformer blocks for common HF model layouts."""
    # Llama / Qwen2 / Mistral / Phi family: model.model.layers
    m = getattr(model, "model", None)
    if m is not None and hasattr(m, "layers") and isinstance(m.layers, (nn.ModuleList,)):
        return m.layers
    # GPT-2 family: model.transformer.h
    t = getattr(model, "transformer", None)
    if t is not None and hasattr(t, "h") and isinstance(t.h, (nn.ModuleList,)):
        return t.h
    raise RuntimeError(f"Couldn't find a body ModuleList on {type(model).__name__}")


def _layer_cpu_bytes(layer: nn.Module) -> int:
    return sum(p.data.numel() * p.data.element_size() for p in layer.parameters(recurse=True))


class BodyPager:
    """LRU-budgeted GPU resident set for the wrapped transformer layers.

    Working set ≥ 2 is enforced — one slot for the layer currently executing
    (pinned, never evicted during its own forward) plus at least one slot
    for the head of the HoloStream prefetch. With working_set==1 the active
    layer would be evicted by its own prefetch, which produced the
    'cuda:0 and cpu' RuntimeError we saw on the 7B run.
    """
    def __init__(self, working_set: int, compute_device: torch.device,
                 prefetch_fanout: int = 8):
        self._resident: "OrderedDict[PagedTransformerLayer, None]" = OrderedDict()
        self.working_set = max(2, int(working_set))   # min 2: active + 1 prefetch
        self.compute_device = compute_device
        self.prefetch_fanout = max(0, int(prefetch_fanout))
        # the layer currently inside its own forward; never evict this one
        self._pinned: "PagedTransformerLayer | None" = None

    def pin(self, layer: "PagedTransformerLayer | None"):
        self._pinned = layer

    def touch(self, layer: "PagedTransformerLayer"):
        if layer in self._resident:
            self._resident.move_to_end(layer)

    def admit(self, layer: "PagedTransformerLayer", stream: Optional[torch.cuda.Stream] = None):
        """Make `layer` resident on the GPU; evict LRU non-pinned layers if over budget."""
        if layer in self._resident:
            self._resident.move_to_end(layer)
            return
        # evict LRU layers until there's room — but never evict the pinned
        # active layer. Scan from the oldest end forward, skipping the pin.
        while len(self._resident) >= self.working_set:
            evicted = None
            for cand in list(self._resident.keys()):
                if cand is self._pinned: continue
                evicted = cand
                break
            if evicted is None: break    # only the pinned layer is resident; can't admit
            del self._resident[evicted]
            evicted._to_cpu_internal()
        layer._to_gpu_internal(stream=stream)
        self._resident[layer] = None

    def evict_all(self):
        self._pinned = None
        while self._resident:
            evicted, _ = self._resident.popitem(last=False)
            evicted._to_cpu_internal()


def _int8_quantize(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row int8 quantization (a.k.a. weight-only int8).

    Big linear weights in transformer layers are shape (out, in). Computing
    a per-row scale (max abs / 127) gives 4× less PCIe traffic than fp16
    with ~no quality loss on most models. Tiny tensors (norms, biases,
    scalars) are kept in fp16 — quantizing them buys nothing and can hurt.

    Returns (int8_tensor, fp16_scales). The forward path multiplies the
    incoming activations by `scales` after the int8→fp16 dequant, so the
    end-to-end math is identical to the fp16 weight up to int8 rounding.
    """
    if t.dim() < 2 or t.numel() < 2048:
        # too small to be worth quantizing — keep as-is, signal with scale=None
        return t.detach().contiguous(), None  # type: ignore[return-value]
    flat = t.detach().to(torch.float32).contiguous()
    # per-output-row scale
    max_per_row = flat.abs().amax(dim=tuple(range(1, flat.dim())), keepdim=True)
    scale = (max_per_row / 127.0).clamp(min=1e-8)
    q = (flat / scale).round().clamp(-127, 127).to(torch.int8)
    # store scale as fp16 so dequant on GPU is cheap
    return q.contiguous(), scale.squeeze(-1).to(torch.float16).contiguous()


def _int8_dequantize(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype,
                    device: torch.device) -> torch.Tensor:
    """Reconstruct fp16 weight from int8 master + per-row scale on the GPU."""
    if scale is None:
        return q.to(device=device, dtype=dtype, non_blocking=True)
    # dequant: (out, ...) * (out, 1, 1, ...)
    q_dev = q.to(device=device, dtype=dtype, non_blocking=True)
    s_dev = scale.to(device=device, dtype=dtype, non_blocking=True).view(-1, *([1] * (q.dim() - 1)))
    return q_dev * s_dev


# ── int4 (the real lever for 7B chat speed) ───────────────────────────────
#
# 4 bits per weight = 8× less PCIe than fp16, 2× less than int8. The whole
# 7B body lands in ~1.75 GiB instead of 14 GiB; it can fit on the GPU
# entirely (no paging at all) on a 4 GiB card with room to spare.
#
# Encoding: pack two signed-int4 values per uint8 byte. Range [-7, +7] (one
# code reserved to keep symmetric range, common in GGUF Q4_0). Per-output-
# row fp16 scale (max abs / 7).

INT4_GROUP = 64   # GGUF Q4_0-style: one fp16 scale per 64 input elements

def _int4_quantize(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group int4 quantization, packed two-per-byte, groups of 64 along
    the inner dim — GGUF Q4_0 layout. Returns ((packed_uint8, meta), scales).
    Tiny tensors stay full-precision."""
    if t.dim() < 2 or t.numel() < 2048:
        return t.detach().contiguous(), None  # type: ignore[return-value]
    orig_shape = list(t.shape)
    # Flatten everything except the OUT dim → (out, in)
    out = orig_shape[0]
    flat = t.detach().to(torch.float32).contiguous().view(out, -1)
    in_dim = flat.shape[1]
    # pad the inner dim to a multiple of the group size
    pad = (-in_dim) % INT4_GROUP
    if pad: flat = torch.cat([flat, torch.zeros(out, pad, dtype=torch.float32)], dim=1)
    padded = flat.shape[1]
    n_groups = padded // INT4_GROUP
    # Per-group max → per-group scale.  shape: (out, n_groups, GROUP)
    g = flat.view(out, n_groups, INT4_GROUP)
    max_per_g = g.abs().amax(dim=-1, keepdim=True)
    scale = (max_per_g / 7.0).clamp(min=1e-8)
    q = (g / scale).round().clamp(-7, 7).to(torch.int8)   # (out, n_groups, GROUP)
    # Pack two int4 → one uint8 along the group axis (GROUP must be even).
    q_pack = q.view(out, n_groups, INT4_GROUP // 2, 2)
    lo = (q_pack[..., 0].to(torch.int16) & 0x0F).to(torch.uint8)
    hi = (q_pack[..., 1].to(torch.int16) & 0x0F).to(torch.uint8)
    packed = (lo | (hi << 4)).contiguous().view(out, padded // 2)
    meta = torch.tensor([len(orig_shape)] + orig_shape + [int(pad), n_groups, INT4_GROUP],
                        dtype=torch.int32)
    # scale shape: (out, n_groups)
    return (packed, meta), scale.squeeze(-1).to(torch.float16).contiguous()


def _int4_dequantize(packed_meta: tuple, scale: torch.Tensor, dtype: torch.dtype,
                     device: torch.device) -> torch.Tensor:
    """Reconstruct an fp16 weight from per-group int4 master on the GPU."""
    if scale is None:
        packed = packed_meta if not isinstance(packed_meta, tuple) else packed_meta[0]
        return packed.to(device=device, dtype=dtype, non_blocking=True)
    packed, meta = packed_meta
    ndim = int(meta[0])
    orig_shape = [int(x) for x in meta[1:1 + ndim]]
    pad = int(meta[-3]); n_groups = int(meta[-2]); group = int(meta[-1])
    out = orig_shape[0]
    p_dev = packed.to(device=device, non_blocking=True)     # (out, padded/2) uint8
    lo = (p_dev & 0x0F).to(torch.int8)
    hi = ((p_dev >> 4) & 0x0F).to(torch.int8)
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    # interleave into (out, n_groups, group)
    inter = torch.empty(out, n_groups * group, dtype=torch.int8, device=device)
    inter[:, 0::2] = lo
    inter[:, 1::2] = hi
    g = inter.view(out, n_groups, group).to(dtype)
    s_dev = scale.to(device=device, dtype=dtype, non_blocking=True).view(out, n_groups, 1)
    deq = (g * s_dev).view(out, n_groups * group)
    if pad: deq = deq[:, : n_groups * group - pad]
    return deq.view(*orig_shape)


class PagedTransformerLayer(nn.Module):
    """Wraps a transformer block; pages its weights CPU↔GPU under a BodyPager budget.

    Holds the wrapped layer's parameters on the CPU master copy. On
    forward(), asks the BodyPager to admit this layer (which may evict an
    LRU sibling). With helical prefetch enabled, the next `fanout` layers
    are admitted asynchronously on a side stream during this layer's compute,
    so their PCIe transfer hides under our compute (the brief's spiral δ
    applied at the layer axis instead of the token axis).

    `quant`: "fp16" (byte-exact), "int8" (4× less PCIe), or "int4" (8× less
    — the lever that lets a 7B body fit entirely on a 4 GiB GPU at ~10 tok/s
    instead of being PCIe-throttled). Norms / biases stay fp16 in all modes
    so the residual stream is exact.
    """
    def __init__(self, layer: nn.Module, *, compute_device: torch.device,
                 pager: BodyPager, prefetch_stream: Optional[torch.cuda.Stream] = None,
                 quant: str = "fp16", int8_master: bool = False):
        super().__init__()
        self.layer = layer
        self.compute_device = compute_device
        self.pager = pager
        self.prefetch_stream = prefetch_stream
        # back-compat: int8_master=True (legacy) → quant="int8"
        if int8_master and quant == "fp16": quant = "int8"
        if quant not in ("fp16", "int8", "int4"):
            raise ValueError(f"quant must be fp16/int8/int4, got {quant!r}")
        self.quant = quant
        # the linked list of layer-index → next layer; set up by paged_body()
        self.index: int = -1
        self.siblings: list["PagedTransformerLayer"] = []   # all wrapped layers, in order
        # CPU master copies. value = (master_tensor_or_packed, fp16_scales_or_None).
        # For fp16: (cpu_tensor, None). int8: (int8_tensor, scale). int4:
        # ((packed_uint8, meta_int32), scale).
        self._cpu_master: dict[str, tuple] = {}
        self._fp_dtype: torch.dtype = torch.float16
        for name, p in layer.named_parameters(recurse=True):
            cpu = p.data.detach().to("cpu").contiguous()
            self._fp_dtype = cpu.dtype
            p.requires_grad_(False)
            if self.quant == "int8":
                q, scale = _int8_quantize(cpu)
                self._cpu_master[name] = (q, scale)
                p.data = torch.empty(0, dtype=cpu.dtype)
            elif self.quant == "int4":
                packed, scale = _int4_quantize(cpu)
                self._cpu_master[name] = (packed, scale)
                p.data = torch.empty(0, dtype=cpu.dtype)
            else:
                self._cpu_master[name] = (cpu, None)
                p.data = cpu
        # non-parameter buffers (rotary inv_freq, etc) are small; keep them on GPU
        for name, b in layer.named_buffers(recurse=True):
            try: b.data = b.data.to(compute_device)
            except Exception: pass
        self._on_gpu: bool = False
        self._param_map = {name: param for name, param in layer.named_parameters(recurse=True)}

    def _admit_one(self, name: str, cpu_t, scale):
        """Move one parameter's CPU master onto the compute device, dequantizing
        if needed. Used by _to_gpu_internal both with and without a side stream."""
        if scale is None:
            # fp16 path — straight copy
            self._param_map[name].data = cpu_t.to(self.compute_device,
                dtype=self._fp_dtype, non_blocking=True)
        elif self.quant == "int4":
            self._param_map[name].data = _int4_dequantize(cpu_t, scale,
                self._fp_dtype, self.compute_device)
        else:   # int8
            self._param_map[name].data = _int8_dequantize(cpu_t, scale,
                self._fp_dtype, self.compute_device)

    # internal — called only from BodyPager
    def _to_gpu_internal(self, stream: Optional[torch.cuda.Stream] = None):
        if self._on_gpu: return
        if stream is not None:
            with torch.cuda.stream(stream):
                for name, (cpu_t, scale) in self._cpu_master.items():
                    self._admit_one(name, cpu_t, scale)
        else:
            for name, (cpu_t, scale) in self._cpu_master.items():
                self._admit_one(name, cpu_t, scale)
        self._on_gpu = True

    def _to_cpu_internal(self):
        if not self._on_gpu: return
        for name, (cpu_t, scale) in self._cpu_master.items():
            if scale is None:
                self._param_map[name].data = cpu_t
            else:
                # int8/int4 master — .data must stay floating-point (param
                # invariant); the real bytes live in _cpu_master. Admit
                # dequantizes them onto the GPU when needed.
                self._param_map[name].data = torch.empty(0, dtype=self._fp_dtype)
        self._on_gpu = False

    def forward(self, *args, **kwargs):
        # admit + pin THIS layer for the whole forward — the pin prevents
        # the next prefetches from evicting our own weights mid-compute
        # (which was the 7B's 'cuda:0 and cpu' RuntimeError on input_layernorm).
        self.pager.admit(self)
        self.pager.pin(self)
        # then walk the HoloStream of the body axis: this is the layer-chain
        # analogue of the embedding torus's helical δ. For each step k =
        # 1…fanout, queue the next layer on the side stream so its PCIe
        # copy hides under our compute. The body has no ring-twist (it's
        # a 1-D chain), so "next k along the strand" is simply index + k.
        if self.prefetch_stream is not None and self.pager.prefetch_fanout > 0 and self.siblings:
            for off in range(1, self.pager.prefetch_fanout + 1):
                j = self.index + off
                if j >= len(self.siblings): break
                nxt = self.siblings[j]
                if not nxt._on_gpu:
                    try: self.pager.admit(nxt, stream=self.prefetch_stream)
                    except Exception: pass
        self.pager.touch(self)
        try:
            return self.layer(*args, **kwargs)
        finally:
            # release the pin — next layer's forward will pin itself
            self.pager.pin(None)


_CLEAN_FREE_HINT: Optional[int] = None
def set_clean_free_hint(bytes_free: int):
    """Tell the body pager what the *real* GPU free memory was before any HF
    load happened. `mem_get_info()` lies mid-load (caching allocator holds
    transient reservations). The server calls this once at boot."""
    global _CLEAN_FREE_HINT
    _CLEAN_FREE_HINT = int(bytes_free)


def _decide_working_set_from_avg(layers, compute_device: torch.device,
                                 explicit: Optional[int], avg_layer: float,
                                 reserve_mb: int = 700,
                                 round_up_threshold: float = 0.85) -> int:
    """Same as _decide_working_set but accepts an explicit avg-layer-bytes value
    (so int8 paths can advertise the smaller effective footprint)."""
    if explicit is not None: return int(explicit)
    if compute_device.type != "cuda":
        return len(layers)
    if _CLEAN_FREE_HINT is not None:
        free = _CLEAN_FREE_HINT
    else:
        free, _ = torch.cuda.mem_get_info()
    if avg_layer <= 0: return len(layers)
    budget = free - reserve_mb * 1_048_576
    if budget <= 0: return 2
    fit = int(budget // avg_layer)
    fit = max(2, min(len(layers), fit))
    if len(layers) > 0 and fit / len(layers) >= round_up_threshold:
        fit = len(layers)
    return fit


def _decide_working_set(layers, compute_device: torch.device,
                        explicit: Optional[int],
                        reserve_mb: int = 700,
                        round_up_threshold: float = 0.85) -> int:
    """Pick how many layers to hold on the GPU at once.

    If the user gave an explicit value, use that. Otherwise: take the GPU's
    currently-free memory, subtract a `reserve_mb` headroom for activations /
    KV cache / autograd workspace, divide by the average layer size, clamp to
    [1, num_layers]. On CPU device, hold all layers (no budget).

    `round_up_threshold`: if we land on a working set that covers ≥ this
    fraction of layers (e.g. 25/28 = 0.89 with the default 0.85), round up
    to *all* layers. Paging the last few layers per token is a much bigger
    speed loss than the small extra VRAM pressure of just keeping them
    resident — a few hundred MB more is fine, but 1 tok/s vs 8 tok/s is not.
    """
    if explicit is not None: return int(explicit)
    if compute_device.type != "cuda":
        return len(layers)
    free, total = torch.cuda.mem_get_info()
    avg_layer = sum(_layer_cpu_bytes(l) for l in layers) / max(len(layers), 1)
    if avg_layer <= 0: return len(layers)
    budget = free - reserve_mb * 1_048_576
    if budget <= 0: return 2
    fit = int(budget // avg_layer)
    fit = max(2, min(len(layers), fit))   # min 2 to keep the pin invariant
    # round up if we're within striking distance of the whole body
    if len(layers) > 0 and fit / len(layers) >= round_up_threshold:
        fit = len(layers)
    return fit


def paged_body(model: nn.Module, *, compute_device: torch.device | str,
               prefetch: bool = True, working_set: Optional[int] = None,
               prefetch_fanout: int = 8, reserve_mb: int = 700,
               quant: str = "fp16",
               int8_master: bool = False) -> tuple[int, int]:
    """Wrap every transformer layer in the body for CPU↔GPU paging with a
    working-set budget.

    `quant`: "fp16" (byte-exact), "int8" (~4× less PCIe), "int4" (~8× less
    + the whole 7B body fits on a 4 GiB GPU at full speed).

    Returns (num_layers_wrapped, working_set_chosen). The first paged layer's
    weights are loaded eagerly so the first forward isn't a cold start.
    """
    if int8_master and quant == "fp16": quant = "int8"   # back-compat
    if quant not in ("fp16", "int8", "int4"):
        raise ValueError(f"quant must be fp16/int8/int4, got {quant!r}")
    compute_device = torch.device(compute_device)
    layers = _iter_body_layers(model)
    # Quantization cuts the per-layer footprint, so the working-set decision
    # should be made against the *quantized* size.
    divisor = {"fp16": 1, "int8": 4, "int4": 8}[quant]
    total_bytes = sum(_layer_cpu_bytes(l) for l in layers)
    eff_layer_bytes = total_bytes // divisor
    avg_layer_bytes = eff_layer_bytes / max(len(layers), 1)
    ws = _decide_working_set_from_avg(layers, compute_device, working_set,
                                      avg_layer=avg_layer_bytes, reserve_mb=reserve_mb)
    stream = torch.cuda.Stream() if (prefetch and compute_device.type == "cuda") else None
    pager = BodyPager(working_set=ws, compute_device=compute_device,
                      prefetch_fanout=prefetch_fanout if prefetch else 0)
    wrapped: list[PagedTransformerLayer] = []
    for layer in layers:
        pl = PagedTransformerLayer(layer, compute_device=compute_device,
                                   pager=pager, prefetch_stream=stream,
                                   quant=quant)
        wrapped.append(pl)
    # set indices + sibling linkage for helical prefetch
    for i, pl in enumerate(wrapped):
        pl.index = i
        pl.siblings = wrapped
    # install the wrapped layers in place
    for i in range(len(layers)):
        layers[i] = wrapped[i]
    # warm the working set so the first forward isn't a cold-start cascade
    if wrapped:
        for i in range(min(ws, len(wrapped))):
            pager.admit(wrapped[i])
    return len(wrapped), ws
