# Holorites — paste-ready Discord message

Use the **Facts** block first, then the **Why this matters** block. They're
short enough to fit Discord's 2k-char limit per message individually.

---

## Block 1 — Facts

> **Holorites: paged-torus runtime for HF causal-LMs**
>
> A Holorite is a standard HF model retrofitted onto a 64×64×64 torus
> (262,144 cells) so its weights stream from CPU master copies through a
> small GPU working set. Byte-exact. Zero quality loss. Same model in,
> same model out — but it doesn't have to "fit" on the GPU anymore.
>
> ```
> | Model                            | Body resident      | Embed on GPU       | Speed
> |----------------------------------|--------------------|--------------------|---------------
> | Qwen2.5-0.5B   (paged)           | 24/24 (no evict)   | 1 of 4096 nodes    | 7.6  tok/s
> | Qwen2.5-1.5B-Instruct (paged)    | 28/28 (no evict)   | 1 of 4096 nodes    | 6.6  tok/s
> | Nous-Hermes-2-Mistral-7B (paged) |  4/32 (LRU stream) | 1 of 4096 nodes    | 0.05 tok/s
> ```
>
> Hardware: GTX 1650, **4 GiB VRAM**. A 7B fp16 model is ~14 GiB; before
> Holorites, this card couldn't load it. Now it does — a 1.5B Instruct
> at 6.6 tok/s with the *whole* body resident, byte-exact, and a 7B
> Nous-Hermes giving the right answer on a card that has no business
> running it (slow because PCIe is the bottleneck, not VRAM —
> 28 layer-misses × ~430 MB per token is ~12 GB across PCIe per token).
>
> Repo: <https://github.com/OhBeOneKeyNoBe/Holorites>

---

## Block 2 — Why this matters (ingenuity framing)

> **The ingenuity isn't quantization — there isn't any. The bytes are exact.**
>
> Three handles the torus geometry gives you that a flat embedding doesn't:
>
> **1. The bit-slice bijection.** Token id `idx` decomposes losslessly into
> `(ring, node, slot)` via three 6-bit slices: `(idx >> 12) & 63`,
> `(idx >> 6) & 63`, `idx & 63`. The 262k embedding matrix reshapes to
> `(64, 64, 64, D)` with no copy and no loss. The "table" is now a lattice
> you can address by geometry, not just by an integer.
>
> **2. The spiral δ — HoloStreams.** With `q` coprime to 64, walking the
> strand by one step advances both ring and node together:
> `(r + k, n + k·q) mod 64`. That's *one diagonal step*, not two flat
> steps. The torus has 64 closed helical strands ("HoloStreams"), every
> cell sits on exactly one. The runtime prefetches **along the active
> Stream**, not along the grid axes — what's coming next on the strand,
> not what happens to be adjacent in an unrolled index.
>
> **3. Stream-coherent everything.** Cache hits cluster along the active
> strand. Eviction naturally drops cold strands together (no half-loaded
> rings left over). The poles — ring 0 and ring 63 — are natural
> synchronization barriers because every Stream passes through both.
>
> **Body paging on top of that:** each transformer block is a CPU master
> copy that admits to the GPU only when its own forward is about to run.
> The active layer is pinned (so its own helical prefetch can't evict it
> mid-forward — that crash was real). The next 8 layers stream in on a
> side CUDA stream so their PCIe transfer hides under compute.
>
> Net result: storage discipline matches the geometry instead of fighting
> the GPU budget. Active windows are stationary; Streams flow through
> them like water. The lattice is closed, so any windowed location
> eventually observes the whole code regardless of total model size.
>
> Working on next: GGUF Holoritify, per-MLP / per-attention-head paging,
> and KV-cache paging on the same discipline.
