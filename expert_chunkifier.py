"""expert_chunkifier.py — re-lay MoE experts on disk for sequential reads.

The user's lever 5: pre-build the chunk layout so each expert is one
contiguous read. Random small reads kill NVMe throughput; one 50-100 MiB
sequential read per expert is the sweet spot.

GGUF natively stores experts in PACKED `*_exps` tensors — all 128 (or 256
or 384) experts' gate weights in one tensor, all up weights in another,
all down weights in a third. Per-expert reads cross three separate file
regions; per-expert WRITES from those reads are six separate seeks (3
tensors × 2 endpoints each).

This module produces a sibling `.chunks` file alongside each GGUF where
the experts are laid out PER-EXPERT contiguously:

  expert 0: [gate bytes][up bytes][down bytes]
  expert 1: [gate bytes][up bytes][down bytes]
  ...
  expert N: [gate bytes][up bytes][down bytes]

One `pread(file, expert_offset, expert_size)` brings in the entire expert's
weight slab as one sequential read. For Qwen3-Coder Q4_K_M, each expert's
slab is ~2.6 MiB; for V4-Pro MXFP4 it's ~33 MiB. Either size is squarely
in NVMe sweet-spot territory.

API:
    chunkify_moe_experts(gguf_path) → writes <gguf>.chunks + index .json
    load_chunked_expert(chunks_path, layer, expert_id) → bytes for that expert

The streamer's `_admit_to_gpu` can preferentially use the `.chunks` file
when present (one read, three slab offsets known from the index) and
fall back to the per-tensor-slice path when only the raw GGUF exists.

Note: this doesn't replace the GGUF — it adds a sidecar. The GGUF stays
canonical (node-llama-cpp continues to use it), and the .chunks file is
a streaming-optimized layout the Python streamer can opt into.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from typing import Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader


def chunkify_moe_experts(gguf_path: str, out_path: Optional[str] = None
                         ) -> str:
    """Write a per-expert-contiguous sidecar. Returns the .chunks path.

    Total disk cost ≈ total bytes of all `*_exps` tensors (the existing
    GGUF stays in place; this duplicates the expert bytes in a different
    layout). For Qwen3-Coder Q4_K_M: ~17 GiB extra. For DeepSeek V3 Q4_K_M:
    ~280 GiB extra. The trade-off: ~3-5× faster per-token streaming.

    For NVMe-tight setups, this can be deferred — the standard per-tensor
    slice path still works; it just doesn't get the sequential-read win.
    """
    gguf_path = str(Path(gguf_path).resolve())
    out_path = out_path or (gguf_path + ".chunks")
    index_path = out_path + ".json"
    r = GGUFReader(gguf_path)

    # discover layers + their packed expert tensors
    import re
    BLK_RX = re.compile(r"^blk\.(\d+)\.(.+)$")
    layers: dict[int, dict] = {}
    for t in r.tensors:
        m = BLK_RX.match(t.name)
        if not m: continue
        li, sub = int(m.group(1)), m.group(2)
        if sub not in ("ffn_gate_exps.weight", "ffn_up_exps.weight",
                       "ffn_down_exps.weight"):
            continue
        layers.setdefault(li, {})[sub] = t
    if not layers:
        raise ValueError(f"no MoE expert tensors found in {gguf_path}")

    # discover n_experts from the first expert tensor shape
    sample = next(iter(layers.values()))
    sample_t = next(iter(sample.values()))
    n_experts = int(sample_t.shape[-1])

    print(f"[chunkify] {gguf_path}")
    print(f"  layers with experts: {len(layers)}")
    print(f"  experts per layer: {n_experts}")

    # write contiguous-per-expert sidecar
    index = {
        "gguf_path": gguf_path,
        "n_layers": len(layers),
        "n_experts": n_experts,
        "layers": {},
    }
    out_f = open(out_path, "wb")
    cursor = 0
    for li in sorted(layers.keys()):
        L = layers[li]
        gate = L.get("ffn_gate_exps.weight")
        up   = L.get("ffn_up_exps.weight")
        down = L.get("ffn_down_exps.weight")
        if not (gate and up and down):
            print(f"  L{li}: missing some packed expert tensor — skipping")
            continue
        gate_per = gate.n_bytes // n_experts
        up_per   = up.n_bytes // n_experts
        down_per = down.n_bytes // n_experts
        layer_index = {
            "layer": li,
            "n_experts": n_experts,
            "gate_bytes_per_expert": gate_per,
            "up_bytes_per_expert":   up_per,
            "down_bytes_per_expert": down_per,
            "gate_shape": [int(d) for d in reversed(gate.shape[:-1])],
            "up_shape":   [int(d) for d in reversed(up.shape[:-1])],
            "down_shape": [int(d) for d in reversed(down.shape[:-1])],
            "gate_dtype": gate.tensor_type.name,
            "up_dtype":   up.tensor_type.name,
            "down_dtype": down.tensor_type.name,
            "experts": [],
        }
        gate_raw = np.asarray(gate.data).view(np.uint8).reshape(-1)
        up_raw   = np.asarray(up.data).view(np.uint8).reshape(-1)
        down_raw = np.asarray(down.data).view(np.uint8).reshape(-1)
        for eid in range(n_experts):
            gate_bytes = gate_raw[eid * gate_per : (eid + 1) * gate_per]
            up_bytes   = up_raw[eid * up_per     : (eid + 1) * up_per]
            down_bytes = down_raw[eid * down_per : (eid + 1) * down_per]
            expert_offset = cursor
            out_f.write(gate_bytes.tobytes())
            out_f.write(up_bytes.tobytes())
            out_f.write(down_bytes.tobytes())
            expert_size = gate_per + up_per + down_per
            layer_index["experts"].append({
                "expert_id": eid,
                "offset": expert_offset,
                "size": expert_size,
                "gate_offset": expert_offset,
                "up_offset":   expert_offset + gate_per,
                "down_offset": expert_offset + gate_per + up_per,
            })
            cursor += expert_size
        index["layers"][str(li)] = layer_index
        if (li + 1) % 8 == 0:
            print(f"  layers 0..{li} written; sidecar size {cursor/1024**3:.2f} GiB")
    out_f.close()
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"[chunkify] wrote {out_path} ({cursor/1024**3:.2f} GiB)")
    print(f"[chunkify] wrote {index_path}")
    return out_path


def load_chunked_expert(chunks_path: str, layer: int, expert_id: int
                        ) -> dict:
    """Read one expert's contiguous slab via a single pread.
    Returns {gate_bytes, up_bytes, down_bytes, gate_shape, ...}."""
    index_path = chunks_path + ".json"
    with open(index_path, encoding="utf-8") as f: idx = json.load(f)
    layer_info = idx["layers"][str(layer)]
    e = layer_info["experts"][expert_id]
    with open(chunks_path, "rb") as f:
        f.seek(e["offset"])
        full = f.read(e["size"])
    gate_per = layer_info["gate_bytes_per_expert"]
    up_per   = layer_info["up_bytes_per_expert"]
    down_per = layer_info["down_bytes_per_expert"]
    return {
        "gate_bytes": full[:gate_per],
        "up_bytes":   full[gate_per:gate_per + up_per],
        "down_bytes": full[gate_per + up_per:],
        "gate_shape": layer_info["gate_shape"],
        "up_shape":   layer_info["up_shape"],
        "down_shape": layer_info["down_shape"],
        "gate_dtype": layer_info["gate_dtype"],
        "up_dtype":   layer_info["up_dtype"],
        "down_dtype": layer_info["down_dtype"],
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:  py expert_chunkifier.py <moe.gguf> [out_chunks_path]")
        sys.exit(1)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) >= 3 else None
    chunkify_moe_experts(src, out)
