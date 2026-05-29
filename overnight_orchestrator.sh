#!/bin/bash
# overnight_orchestrator.sh — autonomous download + holoritify pipeline.
#
# Strategy:
#   1. V4-Pro safetensors (priority, already running) — let it finish
#   2. V3-0324 GGUF Q4_K_M — start when V4 done OR bandwidth available
#   3. R1 GGUF Q4_K_M — start after V3 done
#   4. Watchdog: any GGUF that finishes gets auto-holoritified
#
# All output → /tmp/overnight.log + per-task logs.
# User-resumable: re-running this script picks up where it left off.

set -u
LOG=/tmp/overnight.log
exec 1>>"$LOG" 2>&1

log() { printf "\n[%s] %s\n" "$(date '+%H:%M:%S')" "$*"; }

log "=== overnight orchestrator starting ==="
log "PID $$, host $(hostname)"

# ─── helpers ────────────────────────────────────────────────────────────
file_complete() {
  # $1 = path, $2 = expected_size_gb (minimum), returns 0 if complete
  local f="$1" min_gb="$2"
  [ -f "$f" ] || return 1
  local sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  [ "$sz" -gt "$((min_gb * 1024 * 1024 * 1024))" ]
}

curl_resume() {
  # $1 = URL, $2 = dest
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  curl -L --connect-timeout 30 --max-time 0 --retry 10 --retry-delay 30 \
       --continue-at - -o "$dest" "$url" 2>&1 | tail -1
}

await_curl_done() {
  # $1 = dest path, $2 = min_gb, polls until file stops growing AND is >= min_gb
  local f="$1" min_gb="$2"
  local prev_sz=-1
  while true; do
    local sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -gt "$((min_gb * 1024 * 1024 * 1024))" ] && [ "$sz" = "$prev_sz" ]; then
      log "  stable at $((sz/1024**3)) GiB — done"
      return 0
    fi
    prev_sz=$sz
    sleep 60
  done
}

# ─── DEEPSEEK V4-PRO (safetensors, 64 shards, ~3.2 TB) ──────────────────
v4_dl() {
  local DEST="/e/Holorites-Downloads/deepseek-ai--DeepSeek-V4-Pro"
  local BASE="https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main"
  mkdir -p "$DEST"
  log "[V4-Pro] starting/resuming download"
  for i in $(seq 1 64); do
    local N=$(printf "%05d" $i)
    local FILE="model-${N}-of-00064.safetensors"
    local TARGET="$DEST/$FILE"
    if file_complete "$TARGET" 49; then
      log "[V4-Pro] shard $i/64 already complete"
      continue
    fi
    log "[V4-Pro] downloading shard $i/64: $FILE"
    curl_resume "$BASE/$FILE" "$TARGET"
    log "[V4-Pro] shard $i/64 size: $(stat -c%s "$TARGET" 2>/dev/null | awk '{printf "%.1f GiB", $1/1024**3}')"
  done
  log "[V4-Pro] ALL 64 SHARDS DOWNLOADED"
  touch "$DEST/.complete"
}

# ─── DEEPSEEK V3-0324 (GGUF Q4_K_M, 9 parts, ~400 GB) ───────────────────
v3_dl() {
  local DEST="/e/Holorites-Downloads/unsloth--DeepSeek-V3-0324-GGUF/Q4_K_M"
  local BASE="https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF/resolve/main/Q4_K_M"
  mkdir -p "$DEST"
  log "[V3-0324] starting download"
  for i in $(seq 1 9); do
    local N=$(printf "%05d" $i)
    local FILE="DeepSeek-V3-0324-Q4_K_M-${N}-of-00009.gguf"
    local TARGET="$DEST/$FILE"
    if file_complete "$TARGET" 38; then
      log "[V3-0324] part $i/9 already complete"
      continue
    fi
    log "[V3-0324] downloading part $i/9"
    curl_resume "$BASE/$FILE" "$TARGET"
    log "[V3-0324] part $i/9 size: $(stat -c%s "$TARGET" 2>/dev/null | awk '{printf "%.1f GiB", $1/1024**3}')"
  done
  log "[V3-0324] ALL 9 PARTS DOWNLOADED"
  touch "$DEST/.complete"
}

# ─── DEEPSEEK R1 (GGUF Q4_K_M, ~400 GB) ─────────────────────────────────
r1_dl() {
  local DEST="/e/Holorites-Downloads/unsloth--DeepSeek-R1-GGUF/Q4_K_M"
  local BASE="https://huggingface.co/unsloth/DeepSeek-R1-GGUF/resolve/main/Q4_K_M"
  mkdir -p "$DEST"
  log "[R1] enumerating parts from HF API"
  # R1 has variable number of parts depending on quant; query the API to find Q4_K_M parts
  local PARTS_JSON=$(curl -fsSL "https://huggingface.co/api/models/unsloth/DeepSeek-R1-GGUF" 2>/dev/null | py -c "
import json, sys
j = json.load(sys.stdin)
files = sorted([s['rfilename'] for s in j.get('siblings', []) if 'Q4_K_M' in s['rfilename'] and s['rfilename'].endswith('.gguf')])
for f in files: print(f)
" 2>/dev/null)
  if [ -z "$PARTS_JSON" ]; then
    log "[R1] no Q4_K_M files found — trying alternate naming"
    PARTS_JSON=$(curl -fsSL "https://huggingface.co/api/models/unsloth/DeepSeek-R1-GGUF" 2>/dev/null | py -c "
import json, sys
j = json.load(sys.stdin)
files = sorted([s['rfilename'] for s in j.get('siblings', []) if s['rfilename'].endswith('.gguf')])
for f in files: print(f)
" 2>/dev/null | head -10)
  fi
  if [ -z "$PARTS_JSON" ]; then
    log "[R1] could not enumerate parts — skipping"
    return 1
  fi
  echo "$PARTS_JSON" | while read FILE; do
    [ -z "$FILE" ] && continue
    local TARGET="$DEST/$(basename "$FILE")"
    if file_complete "$TARGET" 1; then
      log "[R1] $(basename "$FILE") already exists"
      continue
    fi
    log "[R1] downloading $(basename "$FILE")"
    curl_resume "https://huggingface.co/unsloth/DeepSeek-R1-GGUF/resolve/main/$FILE" "$TARGET"
  done
  log "[R1] DONE"
  touch "$DEST/.complete"
}

# ─── AUTO-HOLORITIFY ────────────────────────────────────────────────────
# Watches for any complete .gguf in /e and /d that doesn't yet have a
# Holorite-* directory, runs gguf_holoritify on it.
auto_holoritify() {
  log "[holoritify] scanning for un-Holoritified GGUFs"
  for f in $(find /e/Holorites-Downloads /d/0000_Raw_LLM\ Models -name "*.gguf" -size +1G 2>/dev/null); do
    local stem=$(basename "$f" .gguf)
    local hdir="/d/Holorites/Holorite-$stem"
    if [ -d "$hdir" ]; then
      continue
    fi
    log "[holoritify] processing $stem"
    PYTHONIOENCODING=utf-8 py -X utf8 /d/Holorites/gguf_holoritify.py "$f" 2>&1 | tail -5 | while read line; do
      log "  $line"
    done
  done
  log "[holoritify] sweep complete"
}

# ─── MAIN SEQUENCE ──────────────────────────────────────────────────────

# Phase 1: ensure V4-Pro download is still going (it may already be from earlier)
log "Phase 1: V4-Pro (safetensors, ~3.2 TB)"
v4_dl &
V4_PID=$!
log "V4-Pro PID $V4_PID"

# Phase 2: while V4-Pro grinds, start V3 in parallel (different repo, different bandwidth)
sleep 30
log "Phase 2: V3-0324 GGUF (parallel with V4)"
v3_dl &
V3_PID=$!
log "V3 PID $V3_PID"

# Phase 3: also R1 in parallel
sleep 30
log "Phase 3: R1 GGUF (parallel with V4 + V3)"
r1_dl &
R1_PID=$!
log "R1 PID $R1_PID"

# Phase 4: auto-holoritify sweep every 30 min while downloads run
while kill -0 $V4_PID 2>/dev/null || kill -0 $V3_PID 2>/dev/null || kill -0 $R1_PID 2>/dev/null; do
  sleep 1800   # 30 min
  log "=== periodic holoritify sweep ==="
  auto_holoritify
  log "  V4-Pro: $(du -sb /e/Holorites-Downloads/deepseek-ai--DeepSeek-V4-Pro 2>/dev/null | awk '{printf "%.1f GiB", $1/1024**3}')"
  log "  V3:     $(du -sb /e/Holorites-Downloads/unsloth--DeepSeek-V3-0324-GGUF 2>/dev/null | awk '{printf "%.1f GiB", $1/1024**3}')"
  log "  R1:     $(du -sb /e/Holorites-Downloads/unsloth--DeepSeek-R1-GGUF 2>/dev/null | awk '{printf "%.1f GiB", $1/1024**3}')"
done

log "=== ALL DOWNLOADS FINISHED ==="
log "Final holoritify sweep:"
auto_holoritify
log "=== ORCHESTRATOR DONE ==="
