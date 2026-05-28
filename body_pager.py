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


class PagedTransformerLayer(nn.Module):
    """Wraps a transformer block; pages its weights CPU↔GPU under a BodyPager budget.

    Holds the wrapped layer's parameters on the CPU master copy. On
    forward(), asks the BodyPager to admit this layer (which may evict an
    LRU sibling). With helical prefetch enabled, the next `fanout` layers
    are admitted asynchronously on a side stream during this layer's compute,
    so their PCIe transfer hides under our compute (the brief's spiral δ
    applied at the layer axis instead of the token axis).
    """
    def __init__(self, layer: nn.Module, *, compute_device: torch.device,
                 pager: BodyPager, prefetch_stream: Optional[torch.cuda.Stream] = None):
        super().__init__()
        self.layer = layer
        self.compute_device = compute_device
        self.pager = pager
        self.prefetch_stream = prefetch_stream
        # the linked list of layer-index → next layer; set up by paged_body()
        self.index: int = -1
        self.siblings: list["PagedTransformerLayer"] = []   # all wrapped layers, in order
        # snapshot of each parameter's CPU master copy
        self._cpu_master: dict[str, torch.Tensor] = {}
        for name, p in layer.named_parameters(recurse=True):
            cpu = p.data.detach().to("cpu").contiguous()
            self._cpu_master[name] = cpu
            p.data = cpu     # actually live on CPU between forwards
        # non-parameter buffers (rotary inv_freq, etc) are small; keep them on GPU
        for name, b in layer.named_buffers(recurse=True):
            try: b.data = b.data.to(compute_device)
            except Exception: pass
        self._on_gpu: bool = False
        self._param_map = {name: param for name, param in layer.named_parameters(recurse=True)}

    # internal — called only from BodyPager
    def _to_gpu_internal(self, stream: Optional[torch.cuda.Stream] = None):
        if self._on_gpu: return
        if stream is not None:
            with torch.cuda.stream(stream):
                for name, cpu_t in self._cpu_master.items():
                    self._param_map[name].data = cpu_t.to(self.compute_device, non_blocking=True)
        else:
            for name, cpu_t in self._cpu_master.items():
                self._param_map[name].data = cpu_t.to(self.compute_device, non_blocking=True)
        self._on_gpu = True

    def _to_cpu_internal(self):
        if not self._on_gpu: return
        for name in self._cpu_master:
            self._param_map[name].data = self._cpu_master[name]   # release the GPU storage
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
               prefetch_fanout: int = 8, reserve_mb: int = 700) -> tuple[int, int]:
    """Wrap every transformer layer in the body for CPU↔GPU paging with a
    working-set budget.

    Returns (num_layers_wrapped, working_set_chosen).  The first paged layer's
    weights are loaded eagerly so the first forward isn't a cold start.
    """
    compute_device = torch.device(compute_device)
    layers = _iter_body_layers(model)
    ws = _decide_working_set(layers, compute_device, working_set, reserve_mb=reserve_mb)
    stream = torch.cuda.Stream() if (prefetch and compute_device.type == "cuda") else None
    pager = BodyPager(working_set=ws, compute_device=compute_device,
                      prefetch_fanout=prefetch_fanout if prefetch else 0)
    wrapped: list[PagedTransformerLayer] = []
    for layer in layers:
        pl = PagedTransformerLayer(layer, compute_device=compute_device,
                                   pager=pager, prefetch_stream=stream)
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
