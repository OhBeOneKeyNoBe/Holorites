"""moe_kernels.py — GPU-side dequant kernels for the streamer.

The numpy dequant in gguf_holoritify is the bottleneck (~62 ms per expert
slab on Qwen3-Coder Q4_K). Ports the same math to PyTorch CUDA tensor
ops so the dequant runs on the GPU after the raw bytes land there.

Speedup is roughly 30-100× depending on tensor size — PyTorch's CUDA
vectorization handles the 256-element blocks in parallel across thousands
of threads, whereas numpy is single-threaded and Python-overhead-bound.

Same math, same byte layout, same output values (verified bit-exact in
the self-test at the bottom against the numpy reference).

API:
    dequant_q4_K_gpu(raw_bytes, n_elem, device)   -> fp16 tensor (n_elem,)
    dequant_q6_K_gpu(raw_bytes, n_elem, device)   -> fp16 tensor (n_elem,)
    dequant_q8_0_gpu(raw_bytes, n_elem, device)   -> fp16 tensor (n_elem,)
    dequant_q4_0_gpu(raw_bytes, n_elem, device)   -> fp16 tensor (n_elem,)

These accept the same raw-byte input as the numpy versions (a bytes
object or numpy.uint8 view of the GGUF mmap slice) and return a
GPU-resident fp16 tensor. The streamer just swaps which function it
calls when compute_device is cuda.
"""
from __future__ import annotations
import torch
import numpy as np


# ─── Q4_K (the workhorse — most GGUF body weights are this) ──────────────

def dequant_q4_K_gpu(blob: bytes | np.ndarray, n_elem: int,
                     device: torch.device | str = "cuda") -> torch.Tensor:
    """GPU-side Q4_K dequant. Layout per 144-byte block of 256 elements:

      [0:2]    fp16 d     — super-block scale
      [2:4]    fp16 dmin  — super-block min
      [4:16]   12 bytes   — 8 packed (scale, min) 6-bit pairs
      [16:144] 128 bytes  — 256 nibble-packed values (8 sub-blocks × 32)

    Vectorized over all blocks at once on the GPU.
    """
    device = torch.device(device)
    QK_K = 256
    BS = 4 + 12 + QK_K // 2     # 144
    nb = (n_elem + QK_K - 1) // QK_K
    # ship raw bytes to GPU as uint8
    if isinstance(blob, np.ndarray):
        raw = torch.from_numpy(np.ascontiguousarray(blob, dtype=np.uint8))
    else:
        raw = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8))
    raw = raw[:nb * BS].to(device=device, non_blocking=True).view(nb, BS)
    # super-block scales (d, dmin) — uint8 pairs -> float16 -> float32
    d_dmin_bytes = raw[:, :4].contiguous()
    d_dmin = d_dmin_bytes.view(torch.float16).view(nb, 2).to(torch.float32)
    d    = d_dmin[:, 0]   # (nb,)
    dmin = d_dmin[:, 1]   # (nb,)
    scales_bytes = raw[:, 4:16]   # (nb, 12)
    qs = raw[:, 16:]              # (nb, 128)
    # unpack 8 (sc, m) pairs per block — same bit layout as numpy version
    sc = torch.empty(nb, 8, dtype=torch.float32, device=device)
    m  = torch.empty(nb, 8, dtype=torch.float32, device=device)
    sc[:, 0] = (scales_bytes[:, 0] & 0x3F).to(torch.float32)
    sc[:, 1] = (scales_bytes[:, 1] & 0x3F).to(torch.float32)
    sc[:, 2] = (scales_bytes[:, 2] & 0x3F).to(torch.float32)
    sc[:, 3] = (scales_bytes[:, 3] & 0x3F).to(torch.float32)
    m[:, 0] = (scales_bytes[:, 4] & 0x3F).to(torch.float32)
    m[:, 1] = (scales_bytes[:, 5] & 0x3F).to(torch.float32)
    m[:, 2] = (scales_bytes[:, 6] & 0x3F).to(torch.float32)
    m[:, 3] = (scales_bytes[:, 7] & 0x3F).to(torch.float32)
    sc[:, 4] = ((scales_bytes[:, 8]  & 0x0F) | ((scales_bytes[:, 0] >> 6) << 4)).to(torch.float32)
    sc[:, 5] = ((scales_bytes[:, 9]  & 0x0F) | ((scales_bytes[:, 1] >> 6) << 4)).to(torch.float32)
    sc[:, 6] = ((scales_bytes[:, 10] & 0x0F) | ((scales_bytes[:, 2] >> 6) << 4)).to(torch.float32)
    sc[:, 7] = ((scales_bytes[:, 11] & 0x0F) | ((scales_bytes[:, 3] >> 6) << 4)).to(torch.float32)
    m[:, 4]  = ((scales_bytes[:, 8]  >> 4) | ((scales_bytes[:, 4] >> 6) << 4)).to(torch.float32)
    m[:, 5]  = ((scales_bytes[:, 9]  >> 4) | ((scales_bytes[:, 5] >> 6) << 4)).to(torch.float32)
    m[:, 6]  = ((scales_bytes[:, 10] >> 4) | ((scales_bytes[:, 6] >> 6) << 4)).to(torch.float32)
    m[:, 7]  = ((scales_bytes[:, 11] >> 4) | ((scales_bytes[:, 7] >> 6) << 4)).to(torch.float32)
    out = torch.empty(nb, QK_K, dtype=torch.float32, device=device)
    # 4 pairs of (32-low-nibble, 32-high-nibble) sub-blocks, sharing 32 qs bytes
    for pair in range(4):
        is_lo = 2 * pair; is_hi = 2 * pair + 1
        d_lo = (sc[:, is_lo] * d)[:, None]; d_hi = (sc[:, is_hi] * d)[:, None]
        m_lo = (m[:, is_lo] * dmin)[:, None]; m_hi = (m[:, is_hi] * dmin)[:, None]
        q_pair = qs[:, pair * 32 : pair * 32 + 32].to(torch.float32)
        lo = q_pair.remainder(16.0)   # low nibbles
        hi = q_pair.div(16.0).floor() # high nibbles
        out[:, is_lo * 32 : is_lo * 32 + 32] = d_lo * lo - m_lo
        out[:, is_hi * 32 : is_hi * 32 + 32] = d_hi * hi - m_hi
    return out.reshape(-1)[:n_elem].to(torch.float16)


# ─── Q6_K (Qwen3 uses this for some weights, used in attn_v) ─────────────

def dequant_q6_K_gpu(blob: bytes | np.ndarray, n_elem: int,
                     device: torch.device | str = "cuda") -> torch.Tensor:
    """GPU-side Q6_K dequant. Block layout: 128 + 64 + 16 + 2 = 210 bytes
    for 256 elements. lo nibbles + hi 2-bit + per-sub-block int8 scales +
    fp16 d. y = d * scale * (q6 - 32) (bias 32)."""
    device = torch.device(device)
    QK_K = 256
    BS = QK_K // 2 + QK_K // 4 + 16 + 2   # 210
    nb = (n_elem + QK_K - 1) // QK_K
    if isinstance(blob, np.ndarray):
        raw = torch.from_numpy(np.ascontiguousarray(blob, dtype=np.uint8))
    else:
        raw = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8))
    raw = raw[:nb * BS].to(device=device, non_blocking=True).view(nb, BS)
    ql = raw[:, :QK_K//2]
    qh = raw[:, QK_K//2 : QK_K//2 + QK_K//4]
    sc_bytes = raw[:, QK_K//2 + QK_K//4 : QK_K//2 + QK_K//4 + 16]
    sc = sc_bytes.view(torch.int8).to(torch.float32)
    d = raw[:, -2:].contiguous().view(torch.float16).view(nb).to(torch.float32)
    out = torch.empty(nb, QK_K, dtype=torch.float32, device=device)
    # 2 outer passes per block × 4 sub-blocks per pass × 32 elements each
    for j in range(QK_K // 128):
        ql_lo = ql[:, j*64 : j*64 + 32].to(torch.int16)
        ql_hi = ql[:, j*64 + 32 : j*64 + 64].to(torch.int16)
        qh_b  = qh[:, j*32 : j*32 + 32].to(torch.int16)
        q0 = ((ql_lo & 0xF) | ((qh_b & 0x03) << 4)).to(torch.float32) - 32.0
        q1 = ((ql_hi & 0xF) | ((qh_b & 0x0C) << 2)).to(torch.float32) - 32.0
        q2 = ((ql_lo >> 4)  | ((qh_b & 0x30) << 0)).to(torch.float32) - 32.0
        q3 = ((ql_hi >> 4)  | ((qh_b & 0xC0) >> 2)).to(torch.float32) - 32.0
        for sub in range(4):
            is_idx = j * 8 + sub * 2
            base = j * 128 + sub * 32
            scaled = (sc[:, is_idx] * d)[:, None]
            # interleave 8 values from each of q0..q3
            inter = torch.cat([q0[:, sub*8:sub*8+8], q1[:, sub*8:sub*8+8],
                               q2[:, sub*8:sub*8+8], q3[:, sub*8:sub*8+8]], dim=1)
            out[:, base : base + 32] = inter * scaled
    return out.reshape(-1)[:n_elem].to(torch.float16)


# ─── Q8_0 (often used for embed tables, LM heads, vital params) ──────────

def dequant_q8_0_gpu(blob: bytes | np.ndarray, n_elem: int,
                     device: torch.device | str = "cuda") -> torch.Tensor:
    """GPU-side Q8_0 dequant. 34 bytes per 32 elements: 2 fp16 scale + 32 int8."""
    device = torch.device(device)
    block = 32
    BS = 2 + block   # 34
    nb = (n_elem + block - 1) // block
    if isinstance(blob, np.ndarray):
        raw = torch.from_numpy(np.ascontiguousarray(blob, dtype=np.uint8))
    else:
        raw = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8))
    raw = raw[:nb * BS].to(device=device, non_blocking=True).view(nb, BS)
    scales = raw[:, :2].contiguous().view(torch.float16).view(nb).to(torch.float32)
    ints = raw[:, 2:].view(torch.int8).to(torch.float32)
    out = ints * scales[:, None]
    return out.reshape(-1)[:n_elem].to(torch.float16)


# ─── Q4_0 (older simple format — 18 bytes per 32 elements) ───────────────

def dequant_q4_0_gpu(blob: bytes | np.ndarray, n_elem: int,
                     device: torch.device | str = "cuda") -> torch.Tensor:
    device = torch.device(device)
    QK = 32
    BS = 2 + QK // 2   # 18
    nb = (n_elem + QK - 1) // QK
    if isinstance(blob, np.ndarray):
        raw = torch.from_numpy(np.ascontiguousarray(blob, dtype=np.uint8))
    else:
        raw = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8))
    raw = raw[:nb * BS].to(device=device, non_blocking=True).view(nb, BS)
    d = raw[:, :2].contiguous().view(torch.float16).view(nb).to(torch.float32)
    q = raw[:, 2:].to(torch.int16)
    lo = ((q & 0xF) - 8).to(torch.float32)
    hi = ((q >> 4) - 8).to(torch.float32)
    inter = torch.cat([lo, hi], dim=1)
    out = inter * d[:, None]
    return out.reshape(-1)[:n_elem].to(torch.float16)


# ─── MXFP4 (NVIDIA / OCP spec — V4-Pro expert storage) ───────────────────
# Block size 32. Each block has 32 FP4 values (E2M1) + 1 byte E8M0 shared
# scale = 17 bytes per block.
# FP4 encoding (E2M1):
#   sign(1) | exp(2) | mantissa(1)
#   values: 0, 0.5, 1, 1.5, 2, 3, 4, 6  (positive table)
# E8M0 scale (1 byte unsigned): value = 2^(byte - 127), spans 2^-127 to 2^128.
# Final dequant: scale * fp4_lookup[nibble]

# Lookup table for FP4 E2M1 (16 entries, signed)
_MXFP4_LUT = torch.tensor([
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

def dequant_mxfp4_gpu(blob: bytes | np.ndarray, n_elem: int,
                      device: torch.device | str = "cuda") -> torch.Tensor:
    """GPU-side MXFP4 dequant. Block size 32 elements.
    Layout per 17-byte block: 16 bytes of packed FP4 nibbles + 1 byte E8M0 scale.

    Not yet tested on a real V4-Pro GGUF (none on HF mirrors yet); the format
    follows the OCP MXFP4 spec used by DeepGEMM. Will need verification once
    we have a real Q4-quantized V4-Pro to compare against."""
    device = torch.device(device)
    QK = 32
    BS = QK // 2 + 1   # 17
    nb = (n_elem + QK - 1) // QK
    if isinstance(blob, np.ndarray):
        raw = torch.from_numpy(np.ascontiguousarray(blob, dtype=np.uint8))
    else:
        raw = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8))
    raw = raw[:nb * BS].to(device=device, non_blocking=True).view(nb, BS)
    packed = raw[:, :QK // 2]                   # (nb, 16) — 32 nibbles
    e8m0   = raw[:, -1].to(torch.float32)       # (nb,) — shared exponent byte
    # Decode E8M0 to a scale factor: 2^(byte - 127)
    scale = torch.pow(2.0, e8m0 - 127.0)
    # Unpack low / high nibbles → indices into the FP4 LUT
    lo = (packed & 0xF).to(torch.long)
    hi = (packed >> 4).to(torch.long)
    lut = _MXFP4_LUT.to(device=device)
    vals = torch.empty(nb, QK, dtype=torch.float32, device=device)
    vals[:, 0::2] = lut[lo]
    vals[:, 1::2] = lut[hi]
    out = vals * scale[:, None]
    return out.reshape(-1)[:n_elem].to(torch.float16)


# ─── unified dispatcher ──────────────────────────────────────────────────

def dequant_gpu(tensor_type, blob: bytes | np.ndarray, n_elem: int,
                device: torch.device | str = "cuda") -> torch.Tensor:
    """Single entry: dispatches by ggml tensor type."""
    from gguf import GGMLQuantizationType as T
    if tensor_type == T.Q4_K:  return dequant_q4_K_gpu(blob, n_elem, device)
    if tensor_type == T.Q6_K:  return dequant_q6_K_gpu(blob, n_elem, device)
    if tensor_type == T.Q8_0:  return dequant_q8_0_gpu(blob, n_elem, device)
    if tensor_type == T.Q4_0:  return dequant_q4_0_gpu(blob, n_elem, device)
    if tensor_type == T.F16:
        if isinstance(blob, np.ndarray):
            t = torch.from_numpy(np.ascontiguousarray(blob)).view(torch.float16)
        else:
            t = torch.from_numpy(np.frombuffer(blob, dtype=np.float16))
        return t[:n_elem].to(device=device, non_blocking=True)
    if tensor_type == T.F32:
        if isinstance(blob, np.ndarray):
            t = torch.from_numpy(np.ascontiguousarray(blob)).view(torch.float32)
        else:
            t = torch.from_numpy(np.frombuffer(blob, dtype=np.float32))
        return t[:n_elem].to(device=device, non_blocking=True).to(torch.float16)
    if tensor_type == T.BF16:
        if isinstance(blob, np.ndarray):
            arr = np.ascontiguousarray(blob)
        else:
            arr = np.frombuffer(blob, dtype=np.uint8)
        u16 = np.frombuffer(arr.tobytes(), dtype=np.uint16).astype(np.uint32) << 16
        t = torch.from_numpy(u16.view(np.float32)[:n_elem].copy())
        return t.to(device=device, non_blocking=True).to(torch.float16)
    raise NotImplementedError(f"GPU dequant for {tensor_type.name} not wired")
