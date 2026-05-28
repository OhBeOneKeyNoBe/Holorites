"""streaming_engine.py — the asset-tree storage layer for Holorites.

Up to now the body pager has held the master copy of every weight in CPU
RAM. That works for a 7B in 32 GiB RAM but not for the long-term goal
("models larger than RAM stream from NVMe like an open-world game's scene
graph"). This module makes the storage layer match the geometry:

    chunkify_model(model_path, out_root) → writes each transformer-body
        weight to its own memory-mapped file on disk under
            out_root/body/L{i:02d}/{module}.npy
        plus a `tree.json` index. Each (.npy) file is plain numpy v1.0
        with a 128-byte header — mmap-friendly, zero-copy on read.

    HoloriteStreamer(asset_root, compute_device, working_set, fanout)
        Owns the mmap'd asset tree + GPU resident-set. Methods:
            .open()                    — mmap every chunk, no GPU traffic yet
            .advance(ChunkKey)         — admits the locus + queues the next
                                         `fanout` along the velocity vector
            .bind(layer_idx) → params  — returns a dict of name → CUDA tensor
                                         for the active layer (already
                                         admitted by advance(); pin during
                                         this layer's forward)
            .evict_to_budget()         — runs LRU eviction down to working_set
            .stats()                   — telemetry for /stats

The skeleton-model path (running an empty HF architecture and binding
weights per layer) is wired into holorite_server.py as `runtime: "streamer"`
in the manifest. Existing fp16/int8/int4 Holorites keep working through
body_pager; this is the cleaner long-term path for huge models.

Velocity vectors (per the geometry brief):
    body axis   → linear i + k for k = 1..fanout  (clamp at L-1)
    stream axis → (ring + k, node + k·q) mod 64 — but Streams live on the
                  embedding torus, not the body, so the body chunks only
                  use the linear walk.

Compression is independent of layout: each .npy can hold fp16, int8, or
int4-packed data; the streamer just respects the dtype header. To layer
this onto int4: chunkify with `quant="int4"` and the .npy holds packed
uint8 plus a sidecar `{module}.scales.npy`. Same dequant kernel as
body_pager._int4_dequantize.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
import json, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from body_pager import _iter_body_layers, _int4_quantize, _int4_dequantize, _int8_quantize, _int8_dequantize


@dataclass(frozen=True)
class ChunkKey:
    """Coordinates of one weight chunk in the asset tree.

    For the body, `axis="body"` + `layer` + `module` (the param name).
    For the embedding/LM-head, `axis="emb"` + `ring` + `node` (a 64×64
    HoloStream cell). The Streamer only stores body chunks here; embed
    paging still uses torus_lattice.NodePagedEmbedding which already
    works correctly.
    """
    axis: str
    layer: int = -1
    module: str = ""


# ─── chunkifier ───────────────────────────────────────────────────────────

def _save_chunk(path: str, tensor: torch.Tensor, quant: str) -> dict:
    """Persist one weight as a numpy mmap. Returns metadata dict for the index."""
    path = str(path)
    if quant == "fp16":
        np.save(path, tensor.detach().to(torch.float16).cpu().numpy(), allow_pickle=False)
        return {"path": Path(path).name + ".npy" if not path.endswith(".npy") else Path(path).name,
                "dtype": "fp16", "shape": list(tensor.shape)}
    if quant == "int8":
        q, scale = _int8_quantize(tensor)
        if scale is None:    # tiny tensor, kept fp16
            np.save(path, q.detach().to(torch.float16).cpu().numpy(), allow_pickle=False)
            return {"path": Path(path).stem + ".npy", "dtype": "fp16", "shape": list(tensor.shape)}
        np.save(path, q.detach().cpu().numpy(), allow_pickle=False)
        scale_path = path[:-4] + ".scales.npy" if path.endswith(".npy") else path + ".scales.npy"
        np.save(scale_path, scale.detach().to(torch.float16).cpu().numpy(), allow_pickle=False)
        return {"path": Path(path).stem + ".npy", "dtype": "int8",
                "scales": Path(scale_path).name, "shape": list(tensor.shape)}
    if quant == "int4":
        packed_meta, scale = _int4_quantize(tensor)
        if scale is None:
            np.save(path, packed_meta.detach().to(torch.float16).cpu().numpy(), allow_pickle=False)
            return {"path": Path(path).stem + ".npy", "dtype": "fp16", "shape": list(tensor.shape)}
        packed, meta = packed_meta
        np.save(path, packed.detach().cpu().numpy(), allow_pickle=False)
        scale_path = path[:-4] + ".scales.npy" if path.endswith(".npy") else path + ".scales.npy"
        np.save(scale_path, scale.detach().to(torch.float16).cpu().numpy(), allow_pickle=False)
        meta_path  = path[:-4] + ".meta.npy"   if path.endswith(".npy") else path + ".meta.npy"
        np.save(meta_path, meta.detach().cpu().numpy(), allow_pickle=False)
        return {"path": Path(path).stem + ".npy", "dtype": "int4",
                "scales": Path(scale_path).name, "meta": Path(meta_path).name,
                "shape": list(tensor.shape)}
    raise ValueError(f"unknown quant {quant!r}")


def chunkify_model(model, out_root: str, quant: str = "fp16") -> str:
    """Walk a HF causal-LM, dump every body layer's params into the asset tree.

    Returns the path to tree.json. The streamer reads tree.json to mmap chunks
    on demand. `model` should already be loaded on CPU (low_cpu_mem_usage=True).
    """
    root = Path(out_root); root.mkdir(parents=True, exist_ok=True)
    body_root = root / "body"; body_root.mkdir(exist_ok=True)
    layers = _iter_body_layers(model)
    n = len(layers)
    tree = {"version": 1, "quant": quant, "n_layers": n, "layers": []}
    for i, layer in enumerate(layers):
        ldir = body_root / f"L{i:02d}"; ldir.mkdir(exist_ok=True)
        params = []
        for name, p in layer.named_parameters(recurse=True):
            chunk = str(ldir / f"{name.replace('.', '__')}.npy")
            meta = _save_chunk(chunk, p.data, quant)
            meta["name"] = name
            params.append(meta)
        tree["layers"].append({"index": i, "dir": f"body/L{i:02d}", "params": params})
    idx = root / "tree.json"
    with open(idx, "w", encoding="utf-8") as f: json.dump(tree, f, indent=2)
    return str(idx)


# ─── streamer ─────────────────────────────────────────────────────────────

class _MmapChunk:
    """Lazy-load wrapper around one weight on disk. Mmaps on first touch."""
    __slots__ = ("path", "scales_path", "meta_path", "dtype", "shape",
                 "_mm", "_scales_mm", "_meta_mm")
    def __init__(self, path: str, dtype: str, shape: list[int],
                 scales_path: Optional[str] = None,
                 meta_path: Optional[str] = None):
        self.path = path
        self.scales_path = scales_path
        self.meta_path = meta_path
        self.dtype = dtype
        self.shape = shape
        self._mm = None; self._scales_mm = None; self._meta_mm = None
    def mmap(self):
        if self._mm is None:
            self._mm = np.load(self.path, mmap_mode="r")
            if self.scales_path:
                self._scales_mm = np.load(self.scales_path, mmap_mode="r")
            if self.meta_path:
                self._meta_mm = np.load(self.meta_path, mmap_mode="r")
        return self
    def to_gpu(self, device: torch.device, dtype: torch.dtype,
               stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        self.mmap()
        if self.dtype == "fp16":
            arr = self._mm
            t = torch.from_numpy(np.ascontiguousarray(arr))
            if stream is not None:
                with torch.cuda.stream(stream):
                    return t.to(device=device, dtype=dtype, non_blocking=True)
            return t.to(device=device, dtype=dtype, non_blocking=True)
        if self.dtype == "int8":
            q = torch.from_numpy(np.ascontiguousarray(self._mm))
            sc = torch.from_numpy(np.ascontiguousarray(self._scales_mm))
            if stream is not None:
                with torch.cuda.stream(stream):
                    return _int8_dequantize(q, sc, dtype, device)
            return _int8_dequantize(q, sc, dtype, device)
        if self.dtype == "int4":
            packed = torch.from_numpy(np.ascontiguousarray(self._mm))
            sc = torch.from_numpy(np.ascontiguousarray(self._scales_mm))
            meta = torch.from_numpy(np.ascontiguousarray(self._meta_mm))
            if stream is not None:
                with torch.cuda.stream(stream):
                    return _int4_dequantize((packed, meta), sc, dtype, device)
            return _int4_dequantize((packed, meta), sc, dtype, device)
        raise ValueError(f"unknown chunk dtype {self.dtype}")


class HoloriteStreamer:
    """Velocity-driven LRU streamer over the on-disk asset tree.

    Active layers are stationary windows; `advance(locus)` walks the velocity
    vector and queues the next `fanout` chunks. The 7B body's resident
    footprint stays constant at `working_set × avg_chunk_size`, regardless
    of total model size — same way scene streaming holds frame budget while
    the player walks a 50 GiB world.
    """
    def __init__(self, asset_root: str, *, compute_device: torch.device | str,
                 working_set: int = 8, prefetch_fanout: int = 8,
                 fp_dtype: torch.dtype = torch.float16):
        self.asset_root = Path(asset_root)
        with open(self.asset_root / "tree.json", encoding="utf-8") as f:
            self.tree = json.load(f)
        self.compute_device = torch.device(compute_device)
        self.fp_dtype = fp_dtype
        self.working_set = max(2, int(working_set))
        self.prefetch_fanout = max(0, int(prefetch_fanout))
        # build the chunk index keyed by (layer_idx, param_name) → _MmapChunk
        self.chunks: dict[tuple[int, str], _MmapChunk] = {}
        for layer in self.tree["layers"]:
            for p in layer["params"]:
                pdir = self.asset_root / layer["dir"]
                self.chunks[(layer["index"], p["name"])] = _MmapChunk(
                    str(pdir / p["path"]),
                    dtype=p["dtype"],
                    shape=p["shape"],
                    scales_path=str(pdir / p["scales"]) if p.get("scales") else None,
                    meta_path  =str(pdir / p["meta"])   if p.get("meta")   else None,
                )
        # LRU of layer indices currently resident on GPU
        self.resident: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self.pinned_layer: Optional[int] = None
        self.stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream() if self.compute_device.type == "cuda" else None)
        self.n_layers = int(self.tree["n_layers"])
        self.t_pages_in = 0
        self.t_pages_out = 0
        self.t_prefetches = 0

    def open(self):
        for c in self.chunks.values():
            c.mmap()
        return self

    def _admit_layer(self, layer_idx: int, use_prefetch_stream: bool = False) -> dict[str, torch.Tensor]:
        if layer_idx in self.resident:
            self.resident.move_to_end(layer_idx); return self.resident[layer_idx]
        # Evict LRU until budget is open (never evict the pinned layer)
        while len(self.resident) >= self.working_set:
            evicted = None
            for k in list(self.resident.keys()):
                if k == self.pinned_layer: continue
                evicted = k; break
            if evicted is None: break
            del self.resident[evicted]
            self.t_pages_out += 1
        params: dict[str, torch.Tensor] = {}
        stream = self.stream if use_prefetch_stream else None
        for (li, name), chunk in self.chunks.items():
            if li != layer_idx: continue
            params[name] = chunk.to_gpu(self.compute_device, self.fp_dtype, stream=stream)
        self.resident[layer_idx] = params
        self.t_pages_in += 1
        return params

    def advance(self, locus: ChunkKey):
        """Move the active window. For axis='body', admits locus.layer +
        queues prefetch of locus.layer+1..+fanout along the body chain."""
        if locus.axis != "body": return    # embedding axis is handled elsewhere
        self.pinned_layer = locus.layer
        self._admit_layer(locus.layer, use_prefetch_stream=False)
        for k in range(1, self.prefetch_fanout + 1):
            j = locus.layer + k
            if j >= self.n_layers: break
            if j not in self.resident:
                self._admit_layer(j, use_prefetch_stream=True)
                self.t_prefetches += 1

    def bind(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Return the live param dict for `layer_idx` (admitting if missing)."""
        if layer_idx not in self.resident:
            self._admit_layer(layer_idx)
        self.pinned_layer = layer_idx
        return self.resident[layer_idx]

    def stats(self) -> dict:
        return {"resident": len(self.resident), "working_set": self.working_set,
                "n_layers": self.n_layers, "pages_in": self.t_pages_in,
                "pages_out": self.t_pages_out, "prefetches": self.t_prefetches,
                "active_layer": self.pinned_layer}


# ─── skeleton model: drive a HF body via the streamer ─────────────────────

class _StreamedLayer(nn.Module):
    """Wraps a transformer block so its weights are bound from the streamer
    each forward instead of being parameters at all. The wrapped layer is
    still a normal HF module; we just monkey-patch `.data` on each forward."""
    def __init__(self, layer: nn.Module, layer_idx: int, streamer: HoloriteStreamer):
        super().__init__()
        self.layer = layer
        self.idx = layer_idx
        self.streamer = streamer
        # snapshot parameter names so we know what to bind each forward
        self._param_map = {n: p for n, p in layer.named_parameters(recurse=True)}
        for p in self._param_map.values():
            p.requires_grad_(False)
            p.data = torch.empty(0, dtype=p.dtype)
        # buffers (rotary inv_freq etc.) — move to compute device once
        for _, b in layer.named_buffers(recurse=True):
            try: b.data = b.data.to(streamer.compute_device)
            except Exception: pass

    def forward(self, *args, **kwargs):
        self.streamer.advance(ChunkKey(axis="body", layer=self.idx))
        params = self.streamer.bind(self.idx)
        for name, t in params.items():
            if name in self._param_map:
                self._param_map[name].data = t
        return self.layer(*args, **kwargs)


def stream_body(model, streamer: HoloriteStreamer):
    """Replace every transformer block with a _StreamedLayer. Returns count."""
    layers = _iter_body_layers(model)
    for i, layer in enumerate(layers):
        layers[i] = _StreamedLayer(layer, i, streamer)
    return len(layers)


# ─── CLI ──────────────────────────────────────────────────────────────────
def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Build a streaming asset tree from a HF model.")
    ap.add_argument("model_id", help="HF model id, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", required=True, help="output dir for the asset tree")
    ap.add_argument("--quant", default="fp16", choices=("fp16", "int8", "int4"))
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM
    print(f"[streamer] loading {args.model_id}…")
    m = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    idx = chunkify_model(m, args.out, quant=args.quant)
    print(f"[streamer] wrote {idx}")


if __name__ == "__main__":
    _main()
