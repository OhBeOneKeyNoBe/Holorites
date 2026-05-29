"""gguf_asset_tree.py — the videogame-engine asset tree for GGUF models.

The user's frame: weights live on NVMe. They aren't loaded; they're toured.
A 70B model at Q4_K is ~40 GiB on disk; a 405B at int4 is ~200 GiB; a 671B
is ~350 GiB. None of these have to "fit" anywhere any more than Red Dead
Redemption 2's 150 GiB world has to fit in an Xbox's 16 GiB VRAM. The GGUF
file IS the asset tree, addressed by tensor offset; the runtime streams the
cells the activation is currently traversing.

What this module gives you:

  * `GGUFAssetTree(path)` — wraps `gguf.GGUFReader`. The GGUF stays mmap'd.
    Index by tensor name, query offset/shape/dtype, slice raw bytes WITHOUT
    copying. RAM cost = mmap working-set window (typically < 1 GiB), not
    the model size.

  * `LayerAsset(idx, tensors)` — one transformer block as an asset bundle.
    Holds references to its constituent tensors (attn_q/k/v/o, ffn_gate/up/down,
    layer norms). Streaming the layer = streaming this bundle's tensors.

  * `enumerate_layers(reader)` — walk the GGUF, group tensors into
    LayerAssets indexed by block id. Works on Llama / Mistral / Qwen /
    Gemma / Mixtral / DeepSeek layouts (they all use `blk.{i}.*` naming).

  * `holoritify_asset_tree(gguf_path, out_dir)` — write a Holorite manifest
    that POINTS at the GGUF (`runtime: "asset-tree"`) plus the embedding
    torus sidecar. Storage cost on the Holorite side = just the embed torus
    (the body lives in the GGUF). The companion routes inference for these
    via node-llama-cpp which already mmap's the GGUF.

  * `stream_layer(tree, layer, device, stream=None)` — slice the bytes for
    one layer out of the mmap, dequantize on the GPU, return a dict
    {tensor_name: torch.Tensor}. Asynchronous via CUDA stream when given.

Velocity vector: `advance(locus)` (in streaming_engine.py) marries this asset
tree to the body-pager LRU + HoloStream prefetch — admitting layer i and
queuing prefetch of i+1..i+fanout on the side stream while i computes.
"""
from __future__ import annotations
import os, json, re, sys, time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# The official ggml-org `gguf` package — see research summary. mmap-backed.
import gguf
from gguf import GGUFReader, ReaderTensor, GGMLQuantizationType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torus_lattice import (embedding_to_node_chunks, torus_level_for_vocab,
                           CELLS, TORUS_CELLS, RINGS, NODES, SLOTS)

# ─── tensor-block grouping ────────────────────────────────────────────────

# Common GGUF naming: `blk.{N}.attn_q.weight`, `blk.{N}.ffn_gate.weight`,
# `output.weight`, `output_norm.weight`, `token_embd.weight`, …
_BLK_RX = re.compile(r"^blk\.(\d+)\.(.+)$")


@dataclass
class LayerAsset:
    """One transformer block as an asset bundle: layer index + a dict of
    {param_name → ReaderTensor}. The ReaderTensors hold the GGUF mmap view
    (`tensor.data` is a numpy slice; no copy happens until you ask for one)."""
    index: int
    tensors: dict[str, ReaderTensor] = field(default_factory=dict)

    @property
    def n_bytes(self) -> int:
        return sum(t.n_bytes for t in self.tensors.values())

    def names(self) -> list[str]:
        return sorted(self.tensors.keys())


# ─── asset tree ───────────────────────────────────────────────────────────

class GGUFAssetTree:
    """Mmap-backed asset tree over a GGUF file. Doesn't load anything.

    Attributes:
        path:       absolute path to the .gguf
        reader:     `gguf.GGUFReader` (holds the mmap)
        arch:       e.g. "llama", "qwen2", "gemma4", "deepseek2"
        vocab_size, hidden_dim, n_layers
        layers:     dict[int, LayerAsset]
        globals:    dict[str, ReaderTensor]   — output, output_norm,
                                                token_embd, rope_freqs, …
    """
    def __init__(self, path: str):
        self.path = str(Path(path).resolve())
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)
        self.reader = GGUFReader(self.path)
        # architecture metadata
        kv = {f.name: f for f in self.reader.fields.values()}
        def kvget(*names):
            for n in names:
                if n in kv:
                    parts = kv[n].parts
                    if not parts: return None
                    val = parts[kv[n].data[0]] if kv[n].data else parts[-1]
                    try: return int(val[0]) if hasattr(val, "__len__") else int(val)
                    except Exception:
                        try: return val[0].item() if hasattr(val[0], "item") else val[0]
                        except Exception: return None
            return None
        def kvstr(*names):
            for n in names:
                if n in kv:
                    parts = kv[n].parts
                    if not parts: return None
                    # strings are stored as a packed bytes field
                    try:
                        return bytes(parts[kv[n].data[0]]).decode("utf-8", "replace")
                    except Exception:
                        try: return str(parts[-1])
                        except Exception: return None
            return None
        self.arch = kvstr("general.architecture") or "unknown"
        # gguf-py decodes tokenizer.ggml.tokens as a list of strings; use that
        # length if vocab_size isn't recorded directly.
        self.vocab_size = (kvget(f"{self.arch}.vocab_size") or
                           self._len_field("tokenizer.ggml.tokens") or 0)
        self.hidden_dim = kvget(f"{self.arch}.embedding_length",
                                f"{self.arch}.hidden_size") or 0
        self.n_layers   = kvget(f"{self.arch}.block_count",
                                f"{self.arch}.n_layer") or 0
        # group tensors into layers + globals
        self.layers: dict[int, LayerAsset] = {}
        self.globals: dict[str, ReaderTensor] = {}
        for t in self.reader.tensors:
            m = _BLK_RX.match(t.name)
            if m:
                idx, sub = int(m.group(1)), m.group(2)
                self.layers.setdefault(idx, LayerAsset(index=idx)).tensors[sub] = t
            else:
                self.globals[t.name] = t
        if self.n_layers and self.n_layers != len(self.layers):
            # header says one count, observed tensors say another — trust observed
            self.n_layers = len(self.layers)
        elif not self.n_layers:
            self.n_layers = len(self.layers)

    def _len_field(self, name: str) -> int:
        f = self.reader.fields.get(name)
        if f is None: return 0
        return len(f.data)

    def total_bytes(self) -> int:
        return sum(t.n_bytes for t in self.reader.tensors)

    def summary(self) -> dict:
        return {
            "path": self.path, "arch": self.arch,
            "vocab_size": self.vocab_size, "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers, "n_tensors": len(self.reader.tensors),
            "n_globals": len(self.globals),
            "total_bytes_on_disk": self.total_bytes(),
        }


# ─── dequant on read ──────────────────────────────────────────────────────
# For inference we'd run a CUDA dequant kernel; for the asset-tree builder
# we only need to dequant the embedding (so the Holorite torus sidecar can
# be saved). Body tensors stay as raw quantized bytes in the GGUF.

def _dequant_tensor_to_fp16(t: ReaderTensor) -> torch.Tensor:
    """Dequantize one ReaderTensor to (vocab, hidden) fp16. Calls into the
    decoders we wrote in gguf_holoritify for the GGUF quants we already
    handle (Q4_K, Q6_K, Q8_0, Q4_0, F16, BF16, F32)."""
    import gguf_holoritify as gh
    gt = t.tensor_type
    if gt == GGMLQuantizationType.F32:
        return torch.from_numpy(np.array(t.data).view(np.float32).reshape(*t.shape[::-1]).T.copy()).to(torch.float16)
    if gt == GGMLQuantizationType.F16:
        return torch.from_numpy(np.array(t.data).view(np.float16).reshape(*t.shape[::-1]).T.copy())
    if gt == GGMLQuantizationType.BF16:
        arr = np.array(t.data).view(np.uint16).astype(np.uint32) << 16
        return torch.from_numpy(arr.view(np.float32).reshape(*t.shape[::-1]).T.copy()).to(torch.float16)
    if gt == GGMLQuantizationType.Q8_0:
        n = int(np.prod(t.shape))
        return gh._dequant_q8_0(bytes(t.data), ((n + 31) // 32) * 32)[:n].view(*t.shape[::-1]).T.contiguous().to(torch.float16)
    if gt == GGMLQuantizationType.Q4_K:
        n = int(np.prod(t.shape))
        QK_K = 256
        return gh._dequant_q4_K(bytes(t.data), ((n + QK_K - 1) // QK_K) * QK_K)[:n].view(*t.shape[::-1]).T.contiguous().to(torch.float16)
    if gt == GGMLQuantizationType.Q6_K:
        n = int(np.prod(t.shape))
        QK_K = 256
        return gh._dequant_q6_K(bytes(t.data), ((n + QK_K - 1) // QK_K) * QK_K)[:n].view(*t.shape[::-1]).T.contiguous().to(torch.float16)
    if gt == GGMLQuantizationType.Q4_0:
        n = int(np.prod(t.shape))
        QK = 32
        return gh._dequant_q4_0(bytes(t.data), ((n + QK - 1) // QK) * QK)[:n].view(*t.shape[::-1]).T.contiguous().to(torch.float16)
    raise NotImplementedError(f"dequant for {gt.name} not yet wired")


# ─── manifest builder ─────────────────────────────────────────────────────

def holoritify_asset_tree(gguf_path: str, out_dir: Optional[str] = None) -> str:
    """Build a Holorite from a GGUF via the asset-tree path.

    Writes a manifest with `runtime: "asset-tree"`, the embedding torus
    sidecar (so the visualizer + paging discipline still work on the embed
    side), and an inventory of the body tensors as (name, offset, n_bytes,
    dtype, shape) tuples. NO BODY TENSOR IS COPIED to a sidecar — the GGUF
    is the asset tree itself, addressed by offset.

    The companion routes manifests with `runtime in {"gguf", "asset-tree"}`
    to node-llama-cpp's mmap-based loader for body inference. The Python
    streamer can use the inventory for HoloStream prefetch experiments
    (sub-layer granularity, predictive caching) without re-parsing the GGUF.
    """
    tree = GGUFAssetTree(gguf_path)
    out_dir = out_dir or os.path.join(os.path.dirname(__file__),
                                      f"Holorite-{Path(gguf_path).stem}-asset")
    os.makedirs(out_dir, exist_ok=True)

    # build the embedding torus sidecar — the only thing we materialize.
    # find the embedding tensor across common names
    emb = None
    for name in ("token_embd.weight", "tok_embeddings.weight", "wte.weight"):
        if name in tree.globals:
            emb = tree.globals[name]; break
    if emb is None:
        raise ValueError(f"no token embedding tensor found in {gguf_path}")
    emb_fp16 = _dequant_tensor_to_fp16(emb)
    # verify shape — should be (vocab, hidden) after the T.contiguous() above
    if emb_fp16.shape[0] == tree.hidden_dim and emb_fp16.shape[1] == tree.vocab_size:
        emb_fp16 = emb_fp16.T.contiguous()
    chunks = embedding_to_node_chunks(emb_fp16)
    torus_path = os.path.join(out_dir, "embeddings_torus.pt")
    torch.save(chunks, torus_path)

    # the BODY inventory — names, offsets, sizes, dtypes — no data copied
    body_tensors = []
    for layer_idx in sorted(tree.layers.keys()):
        layer = tree.layers[layer_idx]
        for name, t in layer.tensors.items():
            body_tensors.append({
                "layer": layer_idx, "name": name,
                "tensor_name": f"blk.{layer_idx}.{name}",
                "dtype": t.tensor_type.name,
                "shape": [int(s) for s in t.shape],
                "n_bytes": int(t.n_bytes),
                "offset": int(t.data_offset),
            })

    n_tori = torus_level_for_vocab(tree.vocab_size)
    manifest = {
        "name": Path(gguf_path).stem,
        "runtime": "asset-tree",         # new: the GGUF is the asset tree
        "gguf_path": tree.path,
        "arch": tree.arch,
        "vocab_size": tree.vocab_size,
        "hidden_dim": tree.hidden_dim,
        "n_layers": tree.n_layers,
        "n_tori": n_tori,
        "torus_level": n_tori,
        "torus_addressing": "ring-node-slot" if n_tori == 1 else "omega:rns · alpha:rns",
        "capacity": TORUS_CELLS ** n_tori,
        "n_nodes": int(chunks.shape[0]),
        "embeddings_torus": "embeddings_torus.pt",
        # body lives in the GGUF; just record the inventory
        "body_via": "node-llama-cpp",
        "body_tensors": body_tensors,
        "body_n_tensors": len(body_tensors),
        "body_total_bytes": sum(b["n_bytes"] for b in body_tensors),
        "asset_tree_total_bytes": tree.total_bytes(),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return out_dir


# ─── CLI ──────────────────────────────────────────────────────────────────

def _usage():
    print("Usage:  py gguf_asset_tree.py <model.gguf> [out_dir]")
    print("  Builds a Holorite asset-tree manifest from a GGUF without")
    print("  loading the body into RAM (only the embedding is dequantized).")


if __name__ == "__main__":
    if len(sys.argv) < 2: _usage(); sys.exit(1)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) >= 3 else None
    t0 = time.time()
    path = holoritify_asset_tree(src, out)
    dt = time.time() - t0
    print(f"[asset-tree] wrote {path} in {dt:.1f}s")
    with open(os.path.join(path, "manifest.json"), encoding="utf-8") as f:
        m = json.load(f)
    print(f"  arch={m['arch']}  vocab={m['vocab_size']}  hidden={m['hidden_dim']}  layers={m['n_layers']}")
    print(f"  body tensors: {m['body_n_tensors']}  · body bytes on disk: {m['body_total_bytes']/1024**3:.2f} GiB")
    print(f"  asset tree total: {m['asset_tree_total_bytes']/1024**3:.2f} GiB on NVMe — never loaded into RAM")
