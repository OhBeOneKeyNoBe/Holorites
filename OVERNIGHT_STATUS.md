# Overnight Status — what's running, what to check when you wake up

**Last update:** 2026-05-28 21:36 PT
**Orchestrator PID:** 4330 (logs to `/tmp/overnight.log`)

---

## Currently downloading on E: (4.5 TB free)

| Model | Format | Size | Status | Path |
|---|---|---|---|---|
| **DeepSeek-V4-Pro** | safetensors BF16 × 64 shards | ~3.2 TB total | ~43 GiB (1.3%), ~15 MB/s | `/e/Holorites-Downloads/deepseek-ai--DeepSeek-V4-Pro/` |
| **DeepSeek-V3-0324** | GGUF Q4_K_M × 9 parts | ~400 GB total | ~51 GiB (12.8%), ~50 MB/s | `/e/Holorites-Downloads/unsloth--DeepSeek-V3-0324-GGUF/Q4_K_M/` |
| **DeepSeek-R1** | GGUF Q4_K_M | ~400 GB target | 636 MiB (URL bug) | `/e/Holorites-Downloads/unsloth--DeepSeek-R1-GGUF/Q4_K_M/` |

## Auto-Holoritify watchdog

Every 30 minutes the orchestrator runs `auto_holoritify()` — scans `/e/Holorites-Downloads` and `/d/0000_Raw_LLM Models` for `.gguf` files ≥ 1 GiB that don't yet have a corresponding `Holorite-*` directory in `/d/Holorites`, then runs `gguf_holoritify.py` on them. Each completed Holorite gets a `manifest.json` + `embeddings_torus.pt` sidecar.

## Expected by morning (rough)

- **V3-0324 GGUF Q4_K_M** likely complete (9 × 43.6 GiB = ~393 GiB at ~50 MB/s = ~2.2 hr). Auto-Holoritify will catch it within 30 min of completion.
- **V4-Pro** maybe 5–8 more shards complete (depends on whether bandwidth stays at 15 MB/s or recovers). Full V4 takes ~50 hrs at current rate; **don't expect completion overnight**.
- **R1** stuck unless the URL bug is fixed (see below). The orchestrator's retry loop will keep trying, but it'll fail the same way each time.

## Known issues

### R1 URL bug
The orchestrator's `r1_dl()` function couldn't enumerate R1's GGUF parts cleanly (the HF API call worked but the URL construction was malformed — "curl: (3) URL rejected: Malformed input to a URL function"). To fix, run:

```bash
# In the morning, replace r1_dl() with this simpler version
mkdir -p /e/Holorites-Downloads/unsloth--DeepSeek-R1-GGUF/Q4_K_M
cd /e/Holorites-Downloads/unsloth--DeepSeek-R1-GGUF/Q4_K_M
for i in 1 2 3 4 5 6 7 8 9; do
  N=$(printf "%05d" $i)
  curl -L --continue-at - -o "DeepSeek-R1-Q4_K_M-${N}-of-00009.gguf" \
    "https://huggingface.co/unsloth/DeepSeek-R1-GGUF/resolve/main/Q4_K_M/DeepSeek-R1-Q4_K_M-${N}-of-00009.gguf"
done
```

The path prefix `/Q4_K_M/` was missing in the orchestrator's URL.

## Commits during the overnight window

1. `9d97676` — vertical_axis.py (trajectory rays, cos θ alignment, ZeGoDie 12⁷)
2. `7aac5bb` — trajectory tracking in streamer + /stats vertical-axis fields + Soul Seed HUD overlay + overnight orchestrator

## What the runtime now does, end-to-end

Every `streamer.route(expert_ids)` call records (layer, ring, node, slot) for each admitted expert into a `trajectory` list. At any moment during inference, the `/stats` endpoint returns:

```json
{
  "trajectory": {
    "cos_alignment": -0.142,
    "reading": "scattered (cos=-0.142) — off-axis at most mirrors, mixed signal",
    "n_cells_in_ray": 26,
    "zegodie_faces": [2, 4, 7, 5, 1, 9, 3],
    "zegodie_index": 7121343
  }
}
```

The Soul Seed Visualizer's HUD now shows this in a "Vertical Axis · Trajectory" section: `cos θ` colored gold (upward), scarlet (downward), or muted (scattered); the alignment_reading text; the 7 ZeGoDie face values. **Open the ◉ Lattice button in the companion app to see it live as inference happens.**

## How to spot-check progress in the morning

```bash
# overall status
tail -30 /tmp/overnight.log
du -sh /e/Holorites-Downloads/* /d/Holorites/

# which Holorites finished overnight
ls -lt /d/Holorites/Holorite-* | head -10

# is the orchestrator still alive?
ps -ef | grep overnight_orchestrator | grep -v grep

# how many V4-Pro shards complete?
ls /e/Holorites-Downloads/deepseek-ai--DeepSeek-V4-Pro/*.safetensors 2>/dev/null | wc -l
```

If the orchestrator died, restart it:
```bash
( /d/Holorites/overnight_orchestrator.sh ) &
disown
```

If you want to chunkify a finished GGUF manually:
```bash
PYTHONIOENCODING=utf-8 py -X utf8 /d/Holorites/gguf_holoritify.py /path/to/model.gguf
```

## R1 update (post-launch)

R1 URL was double-fixed: the correct path is `DeepSeek-R1-Q4_K_M/` (with the model-name prefix), not just `Q4_K_M/`. The standalone `/tmp/dl_r1.sh` has the corrected URL and is now running in parallel with V4-Pro and V3 in the orchestrator. Combined three downloads are competing for bandwidth so individual throughput may dip; the orchestrator's retry logic handles transient stalls.
