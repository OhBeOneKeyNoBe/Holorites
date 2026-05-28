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
from torus_lattice import (embedding_to_torus, embedding_to_node_chunks,
                           torus_level_for_vocab, CELLS, NESTED_CELLS,
                           SHELLS, RINGS, NODES, SLOTS)

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
    """Decode Q8_0: blocks of 32 int8 + one fp16 scale. Vectorized numpy."""
    import numpy as np
    block = 32
    bs = 2 + block   # 2 fp16 + 32 int8
    nb = n_elem // block
    raw = np.frombuffer(bytes(blob), dtype=np.uint8).reshape(nb, bs)
    scales = raw[:, :2].copy().view(np.float16).reshape(nb).astype(np.float32)
    ints = raw[:, 2:].view(np.int8).astype(np.float32)
    out = ints * scales[:, None]
    return torch.from_numpy(out.reshape(-1)[:n_elem].astype(np.float32))


def _dequant_q4_K(blob: bytes, n_elem: int) -> torch.Tensor:
    """Decode Q4_K (llama.cpp K-quants): blocks of 256 elements at 144 bytes each.

    Layout per block:
      [0:2]     fp16  d     — super-block scale for the quantized 6-bit scales
      [2:4]     fp16  dmin  — super-block scale for the quantized 6-bit mins
      [4:16]    12 bytes    — 8 packed (scale, min) pairs at 6 bits each
      [16:144]  128 bytes   — 256 nibble-packed values (8 sub-blocks × 32 elements)

    Formula:  y[is_lo, l] = d * sc[is_lo] * (q[l] & 0xF) - dmin * m[is_lo]
              y[is_hi, l] = d * sc[is_hi] * (q[l] >> 4) - dmin * m[is_hi]
    where sub-blocks come in pairs sharing the same 32 packed bytes (low/high
    nibbles). All vectorized over blocks for speed — pure-Python loop would
    take hours on an 8B embed.
    """
    import numpy as np
    QK_K = 256
    K_SCALE_SIZE = 12
    BS = 4 + K_SCALE_SIZE + QK_K // 2   # 144 bytes per block
    nb = n_elem // QK_K
    raw = np.frombuffer(bytes(blob), dtype=np.uint8).reshape(nb, BS)
    # super-block scales
    d_dmin = raw[:, :4].copy().view(np.float16).reshape(nb, 2).astype(np.float32)
    d, dmin = d_dmin[:, 0], d_dmin[:, 1]
    scales_bytes = raw[:, 4:4 + K_SCALE_SIZE]   # (nb, 12)
    qs = raw[:, 4 + K_SCALE_SIZE:]              # (nb, 128)
    # Unpack 8 (sc, m) pairs from the 12 scale bytes:
    #   is < 4:  sc = scales[is]   & 0x3F      m = scales[is+4] & 0x3F
    #   is >=4:  sc = (scales[is+4] & 0xF) | ((scales[is-4] >> 6) << 4)
    #            m  = (scales[is+4] >> 4)  | ((scales[is]    >> 6) << 4)
    sc = np.empty((nb, 8), dtype=np.float32)
    m  = np.empty((nb, 8), dtype=np.float32)
    sc[:, 0] = scales_bytes[:, 0] & 0x3F
    sc[:, 1] = scales_bytes[:, 1] & 0x3F
    sc[:, 2] = scales_bytes[:, 2] & 0x3F
    sc[:, 3] = scales_bytes[:, 3] & 0x3F
    m[:, 0]  = scales_bytes[:, 4] & 0x3F
    m[:, 1]  = scales_bytes[:, 5] & 0x3F
    m[:, 2]  = scales_bytes[:, 6] & 0x3F
    m[:, 3]  = scales_bytes[:, 7] & 0x3F
    sc[:, 4] = (scales_bytes[:, 8]  & 0x0F) | ((scales_bytes[:, 0] >> 6) << 4)
    sc[:, 5] = (scales_bytes[:, 9]  & 0x0F) | ((scales_bytes[:, 1] >> 6) << 4)
    sc[:, 6] = (scales_bytes[:, 10] & 0x0F) | ((scales_bytes[:, 2] >> 6) << 4)
    sc[:, 7] = (scales_bytes[:, 11] & 0x0F) | ((scales_bytes[:, 3] >> 6) << 4)
    m[:,  4] = (scales_bytes[:, 8]  >> 4)   | ((scales_bytes[:, 4] >> 6) << 4)
    m[:,  5] = (scales_bytes[:, 9]  >> 4)   | ((scales_bytes[:, 5] >> 6) << 4)
    m[:,  6] = (scales_bytes[:, 10] >> 4)   | ((scales_bytes[:, 6] >> 6) << 4)
    m[:,  7] = (scales_bytes[:, 11] >> 4)   | ((scales_bytes[:, 7] >> 6) << 4)
    out = np.empty((nb, QK_K), dtype=np.float32)
    for pair in range(4):
        is_lo = 2 * pair
        is_hi = 2 * pair + 1
        sc_lo = (sc[:, is_lo] * d)[:, None]
        sc_hi = (sc[:, is_hi] * d)[:, None]
        m_lo  = (m[:,  is_lo] * dmin)[:, None]
        m_hi  = (m[:,  is_hi] * dmin)[:, None]
        q_pair = qs[:, pair*32 : pair*32 + 32].astype(np.float32)
        lo = q_pair % 16.0   # low nibbles  (0..15)
        hi = q_pair // 16.0  # high nibbles (0..15)
        out[:, is_lo*32 : is_lo*32 + 32] = sc_lo * lo - m_lo
        out[:, is_hi*32 : is_hi*32 + 32] = sc_hi * hi - m_hi
    return torch.from_numpy(out.reshape(-1)[:n_elem])


def _dequant_q6_K(blob: bytes, n_elem: int) -> torch.Tensor:
    """Decode Q6_K: blocks of 256 elements at 210 bytes each.

    Layout: 128 bytes lower-4 nibbles + 64 bytes upper-2 bits (packed into 4
    pairs) + 16 bytes int8 scales + 2 bytes fp16 d. Element value is
        y = d * scale[is] * (q6 - 32)
    where q6 is reconstructed from (lower4 | (upper2 << 4)) then sign-shifted.
    """
    import numpy as np
    QK_K = 256
    BS = QK_K//2 + QK_K//4 + 16 + 2   # 128 + 64 + 16 + 2 = 210
    nb = n_elem // QK_K
    raw = np.frombuffer(bytes(blob), dtype=np.uint8).reshape(nb, BS)
    ql = raw[:, :QK_K//2]                   # 128 bytes — low 4 bits × 256
    qh = raw[:, QK_K//2 : QK_K//2 + QK_K//4] # 64  bytes — high 2 bits × 256
    sc = raw[:, QK_K//2 + QK_K//4 : QK_K//2 + QK_K//4 + 16].view(np.int8).astype(np.float32)
    d  = raw[:, -2:].copy().view(np.float16).reshape(nb).astype(np.float32)
    # Reconstruct q6 values: for sub-block of 32 elements, the layout is two
    # passes of 16 elements each, using two ql bytes (16 low-nibbles each)
    # plus shifted qh bits. The standard llama.cpp recipe (vectorized):
    out = np.empty((nb, QK_K), dtype=np.float32)
    for j in range(QK_K // 128):   # 2 outer passes per block (128 elem each)
        ql_lo = ql[:, j*64 : j*64 + 32]
        ql_hi = ql[:, j*64 + 32 : j*64 + 64]
        qh_b  = qh[:, j*32 : j*32 + 32]
        # element pair: low 4 from ql_lo[l] | (((qh_b[l] >> 0) & 3) << 4) ; then ql_hi[l] | (((qh_b[l] >> 2) & 3) << 4)
        # then upper:  ql_lo[l] >> 4 | (((qh_b[l] >> 4) & 3) << 4) ; ql_hi[l] >> 4 | (((qh_b[l] >> 6) & 3) << 4)
        q0 = (ql_lo & 0xF) | ((qh_b & 0x03) << 4)
        q1 = (ql_hi & 0xF) | ((qh_b & 0x0C) << 2)
        q2 = (ql_lo >> 4)  | ((qh_b & 0x30) << 0)
        q3 = (ql_hi >> 4)  | ((qh_b & 0xC0) >> 2)
        # values are stored with bias 32 → real value q - 32
        q0 = q0.astype(np.float32) - 32.0
        q1 = q1.astype(np.float32) - 32.0
        q2 = q2.astype(np.float32) - 32.0
        q3 = q3.astype(np.float32) - 32.0
        # 4 sub-blocks of 32 elements each, two scales per pass (each sub-block uses sc[2j*2 + ...])
        for sub in range(4):
            is_idx = j * 8 + sub * 2
            base = j*128 + sub*32
            scaled = (sc[:, is_idx] * d)[:, None]
            # alternating columns from q0..q3 → 32 values
            # element ordering matches llama.cpp's dequantize_row_q6_K
            interleaved = np.empty((nb, 32), dtype=np.float32)
            interleaved[:, 0:8]   = q0[:, sub*8 : sub*8 + 8]
            interleaved[:, 8:16]  = q1[:, sub*8 : sub*8 + 8]
            interleaved[:, 16:24] = q2[:, sub*8 : sub*8 + 8]
            interleaved[:, 24:32] = q3[:, sub*8 : sub*8 + 8]
            out[:, base : base + 32] = interleaved * scaled
    return torch.from_numpy(out.reshape(-1)[:n_elem])


def _dequant_q4_0(blob: bytes, n_elem: int) -> torch.Tensor:
    """Decode Q4_0: blocks of 32 elements; 18 bytes per block = 2 fp16 d + 16 int8 packed (low/high nibbles)."""
    import numpy as np
    QK = 32
    BS = 2 + QK // 2   # 18 bytes
    nb = n_elem // QK
    raw = np.frombuffer(bytes(blob), dtype=np.uint8).reshape(nb, BS)
    d = raw[:, :2].copy().view(np.float16).reshape(nb).astype(np.float32)
    q = raw[:, 2:]
    lo = (q & 0xF).astype(np.int8) - 8   # signed
    hi = (q >> 4).astype(np.int8) - 8
    # interleave: lo for first 16 elements, hi for next 16
    inter = np.concatenate([lo, hi], axis=1).astype(np.float32)
    out = inter * d[:, None]
    return torch.from_numpy(out.reshape(-1)[:n_elem])


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
    # Nested-torus addressing: a 64³ inner torus (the 13th, capacity 262,144)
    # holds vocabularies up to 262k. When the model has more — like the user's
    # own zioniel-v350 (vocab 262,409) — the enveloping 12th torus extends
    # capacity to 64⁴ = 16,777,216, using a 4th SHELL axis from the
    # bit-slicing. No truncation; the overflow tokens flow into shells 1..63.
    # Storage stays compact via the flat (n_nodes, 64, D) layout (only the
    # actual vocab rows are materialized; padded to a multiple of 64).
    level = torus_level_for_vocab(vocab)
    overflow_vocab = max(0, vocab - CELLS)   # how many tokens spilled into the outer shell

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

    # The on-disk GGUF has all `vocab` rows; we keep every one (no truncation
    # in nested mode). file_vocab is preserved here for back-compat with the
    # manifest field name.
    file_vocab = vocab
    if dtype_id in _FP_TYPES:
        emb = _read_fp_tensor(gguf_path, dims, dtype_id, abs_off)
        # emb came out as (vocab, hidden) after the reshape-and-flip — verify
        if emb.shape[0] != file_vocab or emb.shape[1] != hidden:
            emb = emb.transpose(0, 1).contiguous()
        emb = emb.to(torch.float16).contiguous()
    elif dtype_id == GGML_TYPE_Q8_0:
        n_file = file_vocab * hidden
        bs = 34   # one Q8_0 block = 34 bytes per 32 elements
        nblocks = (n_file + 31) // 32
        with open(gguf_path, "rb") as f:
            f.seek(abs_off); blob = f.read(nblocks * bs)
        emb = _dequant_q8_0(blob, nblocks * 32)[:n_file].view(file_vocab, hidden).to(torch.float16).contiguous()
    elif dtype_id == GGML_TYPE_Q4_K:
        n_file = file_vocab * hidden
        QK_K = 256
        BS = 4 + 12 + QK_K // 2   # 144 bytes per block
        nblocks = (n_file + QK_K - 1) // QK_K
        with open(gguf_path, "rb") as f:
            f.seek(abs_off); blob = f.read(nblocks * BS)
        emb = _dequant_q4_K(blob, nblocks * QK_K)[:n_file].view(file_vocab, hidden).to(torch.float16).contiguous()
    elif dtype_id == GGML_TYPE_Q6_K:
        n_file = file_vocab * hidden
        QK_K = 256
        BS = QK_K//2 + QK_K//4 + 16 + 2   # 210 bytes per block
        nblocks = (n_file + QK_K - 1) // QK_K
        with open(gguf_path, "rb") as f:
            f.seek(abs_off); blob = f.read(nblocks * BS)
        emb = _dequant_q6_K(blob, nblocks * QK_K)[:n_file].view(file_vocab, hidden).to(torch.float16).contiguous()
    elif dtype_id == GGML_TYPE_Q4_0:
        n_file = file_vocab * hidden
        QK = 32
        BS = 2 + QK // 2   # 18 bytes per block
        nblocks = (n_file + QK - 1) // QK
        with open(gguf_path, "rb") as f:
            f.seek(abs_off); blob = f.read(nblocks * BS)
        emb = _dequant_q4_0(blob, nblocks * QK)[:n_file].view(file_vocab, hidden).to(torch.float16).contiguous()
    else:
        raise NotImplementedError(
            f"GGUF embedding dtype {dtype_id} not yet decoded — currently the holoritifier "
            f"handles F32/F16/BF16/Q8_0/Q4_K/Q6_K/Q4_0. Other quants (Q5_K, IQ2/IQ4) need "
            f"adding to _dequant_*. Often the body uses Q4_K_M but the embedding is kept "
            f"at F16 or Q8_0 — check the producer's --output-format flag.")

    # Flat node-chunked layout: works for both inner (≤262k vocab) and nested
    # (≤16.7M vocab) without ever materializing the giant padded torus tensor.
    # Each chunk is 64 consecutive token rows = one paging unit; nesting just
    # changes how the runtime DECOMPOSES the node index into geometric axes.
    chunks = embedding_to_node_chunks(emb)   # (n_nodes, 64, D)
    n_nodes = int(chunks.shape[0])
    out_dir = out_dir or os.path.join(os.path.dirname(__file__),
                                      f"Holorite-{Path(gguf_path).stem}")
    os.makedirs(out_dir, exist_ok=True)
    torus_path = os.path.join(out_dir, "embeddings_torus.pt")
    torch.save(chunks, torus_path)

    if level == 3:
        torus_shape = [RINGS, NODES, SLOTS, hidden]
        addressing  = "ring-node-slot"
    else:
        torus_shape = [SHELLS, RINGS, NODES, SLOTS, hidden]
        addressing  = "shell-ring-node-slot"

    manifest = {
        "name": Path(gguf_path).stem,
        "runtime": "gguf",
        "gguf_path": gguf_path,
        "arch": arch,
        "vocab_size": vocab,           # full vocab — nothing truncated
        "file_vocab_size": file_vocab,
        "overflow_vocab": overflow_vocab,  # how many flowed into the 12th torus (shell>0)
        "hidden_dim": hidden,
        "dtype": "float16",
        # Torus geometry the runtime/visualizer should use to paint this model.
        # level 3 = inner 13th torus only; level 4 = enveloped by 12th torus.
        "torus_level": level,
        "torus_shape": torus_shape,
        "torus_addressing": addressing,
        "n_nodes": n_nodes,            # actual stored chunks (paging units)
        "embeddings_torus": "embeddings_torus.pt",
        "body_via": "node-llama-cpp",
    }
    mpath = os.path.join(out_dir, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[gguf-holoritify] wrote {mpath}")
    print(f"[gguf-holoritify] wrote {torus_path}  ({chunks.numel()*2/1_048_576:.1f} MiB)  "
          f"level={level}  {'inner 64³' if level == 3 else '12th wraps 13th (64⁴)'}")
    return out_dir


def _usage():
    print("Usage:  py gguf_holoritify.py <model.gguf> [out_dir]")
    print("Example: py gguf_holoritify.py 'D:/models/qwen2.5-0.5b-instruct.Q4_K_M.gguf'")


if __name__ == "__main__":
    if len(sys.argv) < 2: _usage(); sys.exit(1)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) >= 3 else None
    print(holoritify_gguf(src, out))
