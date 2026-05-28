"""gguf_holoritify.py — extract a Holorite from a .gguf file.

What this does
--------------
A GGUF file already carries quantized weights (Q4_K_M, Q5_K, etc.) and the
companion's node-llama-cpp loader runs them natively at 10–30 tok/s on the
same 4 GiB card that struggles with fp16 HF inference. The Holorite torus
discipline still has something to add on top of GGUF: the embedding table
(`token_embd.weight`) is dense and uncompressed and benefits directly from
HoloStream paging.

So `gguf_holoritify.py`:

    py gguf_holoritify.py path/to/model.gguf  [out_dir]

reads the file's metadata + tensor index, extracts ONLY the embedding
tensor, builds the 64×64×64 torus from it, and writes a Holorite directory
with:

    Holorite-<basename>/
        manifest.json          { runtime: "gguf", gguf_path, vocab, hidden, … }
        embeddings_torus.pt    the (64,64,64,D) tensor sidecar

The companion's main process treats this directory like any other Holorite:
the model can be picked from the accordion list, and the visualizer's
13th-torus paint follows whichever cells the active prompt is touching.

Body inference still goes through node-llama-cpp (fastest path on this
hardware). The Holorite paging contributes the embedding-side stats —
which is what the visualizer was always for: showing the geometry, not
replacing the runtime.

GGUF format quick reference (so this file is self-contained):
    Magic     "GGUF"  4 bytes
    Version   uint32   3
    n_tensors uint64
    n_kv      uint64
    [kv]      key (string) + type (uint32) + value
    [tensor index]  name + dims + type + offset
    [padding to alignment]
    [tensor data blob, each tensor at offset+base]

We only need to read the metadata and the embedding tensor; everything
else can be skipped.
"""
from __future__ import annotations
import json, os, struct, sys
from pathlib import Path
import torch

# Lazy-import torus helpers so this file stays usable as a CLI even without
# the HF stack installed (it doesn't need transformers).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torus_lattice import embedding_to_torus, CELLS

GGUF_MAGIC = b"GGUF"

# GGUF value-type IDs (the subset we need to read)
GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

# GGUF tensor dtype IDs we know how to decode for the embedding tensor.
# Token embeddings in practice are always F32, F16, BF16, or Q8_0/Q4_K-like.
# We only NEED F32/F16/BF16 to build the torus; for quantized embeddings
# we dequantize on read.
GGML_TYPE_F32   = 0
GGML_TYPE_F16   = 1
GGML_TYPE_Q4_0  = 2
GGML_TYPE_Q4_1  = 3
GGML_TYPE_Q5_0  = 6
GGML_TYPE_Q5_1  = 7
GGML_TYPE_Q8_0  = 8
GGML_TYPE_Q2_K  = 10
GGML_TYPE_Q3_K  = 11
GGML_TYPE_Q4_K  = 12
GGML_TYPE_Q5_K  = 13
GGML_TYPE_Q6_K  = 14
GGML_TYPE_Q8_K  = 15
GGML_TYPE_BF16  = 30

_FP_TYPES = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}


def _read_struct(f, fmt: str):
    n = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(n))


def _read_str(f) -> str:
    (n,) = _read_struct(f, "<Q")
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f, vtype: int):
    if vtype == GGUF_TYPE_UINT8:   return _read_struct(f, "<B")[0]
    if vtype == GGUF_TYPE_INT8:    return _read_struct(f, "<b")[0]
    if vtype == GGUF_TYPE_UINT16:  return _read_struct(f, "<H")[0]
    if vtype == GGUF_TYPE_INT16:   return _read_struct(f, "<h")[0]
    if vtype == GGUF_TYPE_UINT32:  return _read_struct(f, "<I")[0]
    if vtype == GGUF_TYPE_INT32:   return _read_struct(f, "<i")[0]
    if vtype == GGUF_TYPE_FLOAT32: return _read_struct(f, "<f")[0]
    if vtype == GGUF_TYPE_BOOL:    return bool(_read_struct(f, "<B")[0])
    if vtype == GGUF_TYPE_STRING:  return _read_str(f)
    if vtype == GGUF_TYPE_UINT64:  return _read_struct(f, "<Q")[0]
    if vtype == GGUF_TYPE_INT64:   return _read_struct(f, "<q")[0]
    if vtype == GGUF_TYPE_FLOAT64: return _read_struct(f, "<d")[0]
    if vtype == GGUF_TYPE_ARRAY:
        (et,) = _read_struct(f, "<I")
        (n,)  = _read_struct(f, "<Q")
        return [_read_value(f, et) for _ in range(n)]
    raise ValueError(f"unsupported GGUF value type {vtype}")


def read_gguf_header(path: str):
    """Return (kv_dict, tensor_records, tensor_data_offset).

    tensor_records: list of (name, dims, dtype_id, offset_within_data).
    """
    f = open(path, "rb")
    magic = f.read(4)
    if magic != GGUF_MAGIC:
        raise ValueError(f"{path}: not a GGUF file (magic {magic!r})")
    (ver,)        = _read_struct(f, "<I")
    (n_tensors,)  = _read_struct(f, "<Q")
    (n_kv,)       = _read_struct(f, "<Q")
    kv = {}
    for _ in range(n_kv):
        k = _read_str(f)
        (vt,) = _read_struct(f, "<I")
        kv[k] = _read_value(f, vt)
    tensors = []
    for _ in range(n_tensors):
        name = _read_str(f)
        (ndim,) = _read_struct(f, "<I")
        dims = [_read_struct(f, "<Q")[0] for _ in range(ndim)]
        (dtype_id,) = _read_struct(f, "<I")
        (offset,) = _read_struct(f, "<Q")
        tensors.append((name, dims, dtype_id, offset))
    # GGUF aligns the tensor data blob; default alignment 32, override via kv.
    align = int(kv.get("general.alignment", 32))
    pos = f.tell()
    pad = (-pos) % align
    data_off = pos + pad
    f.close()
    return ver, kv, tensors, data_off


def _read_fp_tensor(path: str, dims: list[int], dtype_id: int, off: int) -> torch.Tensor:
    """Read a single F32/F16/BF16 tensor from the GGUF data blob."""
    n = 1
    for d in dims: n *= d
    if dtype_id == GGML_TYPE_F32:
        torch_dtype, item = torch.float32, 4
    elif dtype_id == GGML_TYPE_F16:
        torch_dtype, item = torch.float16, 2
    elif dtype_id == GGML_TYPE_BF16:
        torch_dtype, item = torch.bfloat16, 2
    else:
        raise ValueError(f"unsupported floating dtype {dtype_id} for the embedding tensor")
    with open(path, "rb") as f:
        f.seek(off)
        buf = f.read(n * item)
    t = torch.frombuffer(bytearray(buf), dtype=torch_dtype).clone()
    # GGUF stores tensors in row-major reverse — dims[0] is the FAST axis.
    # For an embedding, dims == [hidden, vocab], so reshape and transpose so
    # we end up with (vocab, hidden) which matches torch / HF convention.
    t = t.view(*reversed(dims))
    return t.contiguous()


def _dequant_q8_0(blob: bytes, n_elem: int) -> torch.Tensor:
    """Decode Q8_0: blocks of 32 int8 + one fp16 scale. Used for fallback."""
    block = 32
    n_blocks = n_elem // block
    bs = 2 + block   # 2 fp16 + 32 int8
    out = torch.empty(n_elem, dtype=torch.float32)
    arr = bytearray(blob)
    p = 0
    for i in range(n_blocks):
        scale_bytes = bytes(arr[p:p+2]); p += 2
        scale = torch.frombuffer(bytearray(scale_bytes), dtype=torch.float16).item()
        ints = torch.frombuffer(bytearray(arr[p:p+block]), dtype=torch.int8); p += block
        out[i*block:(i+1)*block] = ints.float() * scale
    return out


def holoritify_gguf(gguf_path: str, out_dir: str | None = None) -> str:
    """Build a Holorite directory from a GGUF model. Returns the dir path."""
    gguf_path = str(Path(gguf_path).resolve())
    if not os.path.isfile(gguf_path):
        raise FileNotFoundError(gguf_path)
    ver, kv, tensors, data_off = read_gguf_header(gguf_path)

    arch = kv.get("general.architecture", "unknown")
    vocab_val = kv.get(f"{arch}.vocab_size")
    if isinstance(vocab_val, int):
        vocab = vocab_val
    else:
        # fall back to len(tokens)
        toks = kv.get("tokenizer.ggml.tokens", [])
        vocab = len(toks) if isinstance(toks, list) else 0
    hidden = int(kv.get(f"{arch}.embedding_length") or kv.get(f"{arch}.hidden_size") or 0)
    if vocab == 0 or hidden == 0:
        raise ValueError(f"GGUF header missing vocab/hidden for arch={arch!r}")
    if vocab > CELLS:
        raise ValueError(f"vocab {vocab} > torus capacity {CELLS}")

    # Find the embedding tensor — its name is conventional across GGUF arches:
    EMB_NAMES = ("token_embd.weight", "tok_embeddings.weight", "wte.weight")
    emb_rec = None
    for (n, dims, t, o) in tensors:
        if n in EMB_NAMES:
            emb_rec = (n, dims, t, o); break
    if emb_rec is None:
        names = [t[0] for t in tensors[:8]]
        raise ValueError(f"no embedding tensor in {gguf_path} (saw {names}…)")

    name, dims, dtype_id, off = emb_rec
    abs_off = data_off + off
    print(f"[gguf-holoritify] arch={arch} vocab={vocab} hidden={hidden} "
          f"emb tensor={name!r} dims={dims} dtype_id={dtype_id} off=+{off}")

    if dtype_id in _FP_TYPES:
        emb = _read_fp_tensor(gguf_path, dims, dtype_id, abs_off)
        # emb came out as (vocab, hidden) after the reshape-and-flip — verify
        if emb.shape[0] != vocab or emb.shape[1] != hidden:
            # try the other orientation
            emb = emb.transpose(0, 1).contiguous()
        emb = emb.to(torch.float16).contiguous()
    elif dtype_id == GGML_TYPE_Q8_0:
        n = vocab * hidden
        # one Q8_0 block = 34 bytes per 32 elements
        bs = 34
        nblocks = (n + 31) // 32
        with open(gguf_path, "rb") as f:
            f.seek(abs_off); blob = f.read(nblocks * bs)
        emb = _dequant_q8_0(blob, nblocks * 32)[:n].view(vocab, hidden).to(torch.float16)
    else:
        raise NotImplementedError(
            f"GGUF embedding dtype {dtype_id} not yet decoded — currently the holoritifier "
            f"handles F32/F16/BF16/Q8_0 only. Re-export with --type q8_0 or f16 for now.")

    torus = embedding_to_torus(emb)
    out_dir = out_dir or os.path.join(os.path.dirname(__file__),
                                      f"Holorite-{Path(gguf_path).stem}")
    os.makedirs(out_dir, exist_ok=True)
    torus_path = os.path.join(out_dir, "embeddings_torus.pt")
    torch.save(torus, torus_path)

    manifest = {
        "name": Path(gguf_path).stem,
        "runtime": "gguf",
        "gguf_path": gguf_path,
        "arch": arch,
        "vocab_size": vocab,
        "hidden_dim": hidden,
        "dtype": "float16",
        "torus_shape": list(torus.shape),
        "embeddings_torus": "embeddings_torus.pt",
        # body inference goes through node-llama-cpp on the companion side;
        # the Python server uses this manifest only for the embedding paging
        # stats the visualizer reads. The companion's main.js routes the GGUF
        # body straight to node-llama-cpp for full-speed inference.
        "body_via": "node-llama-cpp",
    }
    mpath = os.path.join(out_dir, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[gguf-holoritify] wrote {mpath}")
    print(f"[gguf-holoritify] wrote {torus_path}  ({torus.numel()*2/1_048_576:.1f} MiB)")
    return out_dir


def _usage():
    print("Usage:  py gguf_holoritify.py <model.gguf> [out_dir]")
    print("Example: py gguf_holoritify.py 'D:/models/qwen2.5-0.5b-instruct.Q4_K_M.gguf'")


if __name__ == "__main__":
    if len(sys.argv) < 2: _usage(); sys.exit(1)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) >= 3 else None
    print(holoritify_gguf(src, out))
