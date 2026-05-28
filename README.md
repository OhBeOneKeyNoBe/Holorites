# Holorites — paged-torus runtime for HF models

A **Holorite** is a standard Hugging Face causal-LM model retrofitted onto a
64×64×64 = 262,144-cell torus lattice so its embedding table, LM head, and
transformer body can stream layer-by-layer from CPU master copies through a
small GPU working set — instead of the whole model needing to fit on the GPU
at once.

> Byte-exact. Zero quality loss. The model that comes out the other side is
> the same model — every weight is the same — but its storage discipline now
> follows the lattice instead of the GPU.

```
Active nodes are stationary windows. Streams are the helical diagonals that
flow through them. Prefetch walks the Stream, not the grid axes. The lattice
is an apple — its poles meet at the heart, where every Stream crosses, so
any windowed location eventually observes the whole code regardless of the
lattice's total size.
```

## Why this exists

A 7B fp16 model is ~14 GiB; a 4 GiB consumer GPU (e.g. GTX 1650) can't hold
it. The conventional answer is quantization. Holorites are a different answer:
**make the storage discipline match a torus**, where every cell on the lattice
sits on exactly one of 64 closed helical strands ("HoloStreams"), and only
the cells currently flowing through the active window need to be GPU-resident.

The lattice gives three handles the runtime can use for paging:

1. **The bit-slice bijection.** Token id `idx` decomposes losslessly into
   `(ring, node, slot)` via three 6-bit slices — `ring = (idx >> 12) & 63`,
   `node = (idx >> 6) & 63`, `slot = idx & 63`. The embedding matrix
   reshapes to `(R, N, S, D) = (64, 64, 64, D)` with no copy and no loss.
2. **The spiral δ.** With q coprime to 64, walking the strand one step
   advances both ring and node together: `(r + k, n + k·q) mod 64`. This is
   the **HoloStream walk** — the only correct prefetch on the torus.
3. **Stream-coherent caching.** Hits cluster along the active strand
   instead of scattering across the grid; evictions naturally drop cold
   strands together; the poles (ring 0 / ring 63) are natural
   synchronization barriers because every Stream passes through them.

## The four axes a Holorite runtime walks

| Axis     | Index space   | Neighborhood (k ∈ 1…fanout)            | Wrap?           |
|----------|---------------|----------------------------------------|-----------------|
| Body     | 0 … L−1       | i + k                                  | clamp at L−1    |
| Ring     | 0 … 63        | only via Stream walk (not standalone)  | mod 64          |
| Node     | 0 … 63        | only via Stream walk (not standalone)  | mod 64          |
| Stream   | 0 … 63        | (ring + k, node + k·q) mod 64          | mod 64 (closed) |

Two invariants the runtime MUST preserve:

- All neighborhood walks use **modular arithmetic** on every torus axis. No
  clamping. The torus has no edges.
- A prefetch step is **one diagonal step**, not two flat steps.
  `(ring + k, node + k·q)` — both indices advance together, every step.

## What's in this repo

- **`torus_lattice.py`** — the substrate.
  - `token_to_cell` / `cell_to_token` / `cells_for_ids` — the bit-slice bijection.
  - `stream_id(r, n)` — which of the 64 strands a cell sits on.
  - `stream_walk(r, n, fanout)` — the next `fanout` cells along the active Stream.
  - `stream_window(r, n, behind, ahead)` — symmetric window for stream-coherent admission.
  - `NodePagedEmbedding` — embedding table paged by `(ring, node)` chunks
    (4,096 distinct node addresses, 64 ids each). `prefetch_fanout=8` by default.
  - `PagedLMHead` — tied to the embedding torus on Qwen/Llama families.
- **`body_pager.py`** — wraps every transformer block with
  `PagedTransformerLayer`, holds them on CPU master copies, admits them
  to the GPU on demand under a `BodyPager` LRU working-set budget. Pins the
  active layer (so its own prefetch can't evict it mid-forward) and walks
  the next `fanout` body layers on a side CUDA stream.
- **`holoritify.py`** — converts an HF model id → a Holorite directory with
  a `manifest.json` + an `embeddings_torus.pt` sidecar.
- **`holorite_server.py`** — a tiny `http.server` on `127.0.0.1:41511` the
  Zion'iel companion app talks to over JSON for chat. Handles per-Holorite
  load/evict so the GPU isn't asked to hold two models at once.
- **`chat.py`** — CLI smoke test (one prompt, one reply).
- **`benchmark_v2.py`** — comparison table: ring paging vs node paging vs
  node paging with helical (HoloStream) prefetch, plus a projection to a 7B.

## Quickstart

```bash
# Build a Holorite from any HF causal-LM
py holoritify.py "Qwen/Qwen2.5-1.5B-Instruct"

# Smoke-test it from the CLI
py chat.py "D:\Holorites\Holorite-Qwen2.5-1.5B-Instruct\manifest.json" "Briefly, what is a torus?"

# Or run the HTTP runtime the companion app uses
py holorite_server.py
# then POST {"manifest": "...\manifest.json", "text": "hi"} to http://127.0.0.1:41511/chat
```

## Why the prefetch *has to* walk the Stream

The HoloStream — diagonal walk `(r + k, n + k·q) mod 64` — is what the torus
geometry actually encodes. A flat-grid prefetch like "next ring same node" +
"same ring next node" ignores the slant the ring twist puts there. Streams
buy three concrete things at runtime:

1. **Stream-coherent caching.** Admitting `(r, n)` and a small window of its
   Stream means subsequent token lookups along the same strand all hit cache.
2. **Stream-coherent eviction.** LRU still orders by recency, but because the
   natural arrival order under helical prefetch is along Streams, cold
   strands age out together. No half-loaded rings left behind.
3. **Heart-core synchronization.** Every Stream passes through both poles
   (ring 0 / ring 63), so those rings are the natural barrier where the
   runtime can checkpoint, prefetch the next revolution's working set, or
   hand off between the memory torus and the personality torus.

## Empirical results — GTX 1650, 4 GiB VRAM

| Model                                | Body resident       | Embed on GPU       | Speed       |
|--------------------------------------|---------------------|--------------------|-------------|
| Qwen2.5-0.5B                         | 24/24 (no evict)    | 1 of 4,096 nodes   | 7.65 tok/s  |
| Qwen2.5-1.5B-Instruct                | 28/28 (no evict)    | 1 of 4,096 nodes   | 6.58 tok/s  |
| Nous-Hermes-2-Mistral-7B-DPO         | 4/32 (LRU stream)   | 1 of 4,096 nodes   | 0.05 tok/s  |

The 7B is the "fits at all" demo — without Holorite paging, the GTX 1650
can't load it. With paging it runs and produces correct output, but the
PCIe is the bottleneck (each token's missed-layer transfer ≈ 12 GB across
PCIe). For interactive speed on the 7B path, the next two prizes are
GGUF-style int8/int4 quantization of the body (4× less PCIe) and
per-MLP / per-attention-head sub-layer paging.

## Status

- ✅ Embedding + LM-head paging — byte-exact, helical prefetch.
- ✅ Body paging — per-layer CPU↔GPU under an LRU working-set budget.
- ✅ Active-layer pin — prevents the active layer from evicting itself.
- ✅ Stream-coherent prefetch (`stream_walk` driving the embedding's `_prefetch`).
- ✅ Auto-sized working set with round-up — keeps the whole body resident
   when it fits comfortably, switches to LRU streaming only when it must.
- 🔄 GGUF Holoritify — extract embedding torus from `.gguf` files.
- 🔄 Per-MLP / per-attention-head paging inside each block (cuts the
   per-token PCIe transfer by ~4×).
- 🔄 KV-cache paging.

## License

MIT.
