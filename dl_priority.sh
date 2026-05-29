#!/bin/bash
# V3-priority orchestrator: V3 → R1 → V4-Pro, strictly serial.
# Resume-safe (curl --continue-at -).
LOG=/tmp/priority.log
exec 1>>"$LOG" 2>&1
log() { printf "\n[%s] %s\n" "$(date '+%H:%M:%S')" "$*"; }

log "=== V3-priority orchestrator START ==="

# --- V3 PRIORITY ---
log "=== Phase 1: V3-0324 (priority) ==="
V3_DEST="/e/Holorites-Downloads/unsloth--DeepSeek-V3-0324-GGUF/Q4_K_M"
V3_BASE="https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF/resolve/main/Q4_K_M"
mkdir -p "$V3_DEST"
for i in 1 2 3 4 5 6 7 8 9; do
  N=$(printf "%05d" $i)
  FILE="DeepSeek-V3-0324-Q4_K_M-${N}-of-00009.gguf"
  TARGET="$V3_DEST/$FILE"
  sz=$(stat -c%s "$TARGET" 2>/dev/null || echo 0)
  if [ "$sz" -gt 38000000000 ]; then log "[V3] part $i/9 complete ($((sz/1024**3))G)"; continue; fi
  log "[V3] downloading part $i/9 (already $((sz/1024**3))G)"
  curl -L --connect-timeout 30 --max-time 0 --retry 20 --retry-delay 30 \
       --continue-at - -o "$TARGET" "$V3_BASE/$FILE" 2>&1 | tail -1
done
log "[V3] PHASE 1 COMPLETE"
PYTHONIOENCODING=utf-8 py -X utf8 /d/Holorites/gguf_holoritify.py "$V3_DEST/DeepSeek-V3-0324-Q4_K_M-00001-of-00009.gguf" 2>&1 | tail -3
PYTHONIOENCODING=utf-8 py -X utf8 /d/Holorites/consolidate_multipart.py 2>&1 | tail -3

# --- R1 SECOND ---
log "=== Phase 2: R1 ==="
R1_DEST="/e/Holorites-Downloads/unsloth--DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M"
R1_BASE="https://huggingface.co/unsloth/DeepSeek-R1-GGUF/resolve/main/DeepSeek-R1-Q4_K_M"
mkdir -p "$R1_DEST"
for i in 1 2 3 4 5 6 7 8 9; do
  N=$(printf "%05d" $i)
  FILE="DeepSeek-R1-Q4_K_M-${N}-of-00009.gguf"
  TARGET="$R1_DEST/$FILE"
  sz=$(stat -c%s "$TARGET" 2>/dev/null || echo 0)
  if [ "$sz" -gt 38000000000 ]; then log "[R1] part $i/9 complete ($((sz/1024**3))G)"; continue; fi
  log "[R1] downloading part $i/9 (already $((sz/1024**3))G)"
  curl -L --connect-timeout 30 --max-time 0 --retry 20 --retry-delay 30 \
       --continue-at - -o "$TARGET" "$R1_BASE/$FILE" 2>&1 | tail -1
done
log "[R1] PHASE 2 COMPLETE"
PYTHONIOENCODING=utf-8 py -X utf8 /d/Holorites/gguf_holoritify.py "$R1_DEST/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf" 2>&1 | tail -3
PYTHONIOENCODING=utf-8 py -X utf8 /d/Holorites/consolidate_multipart.py 2>&1 | tail -3

# --- V4-Pro LAST ---
log "=== Phase 3: V4-Pro safetensors (64 shards) ==="
V4_DEST="/e/Holorites-Downloads/deepseek-ai--DeepSeek-V4-Pro"
V4_BASE="https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main"
mkdir -p "$V4_DEST"
for i in $(seq 1 64); do
  N=$(printf "%05d" $i)
  FILE="model-${N}-of-00064.safetensors"
  TARGET="$V4_DEST/$FILE"
  sz=$(stat -c%s "$TARGET" 2>/dev/null || echo 0)
  if [ "$sz" -gt 49000000000 ]; then log "[V4] shard $i/64 complete"; continue; fi
  log "[V4] downloading shard $i/64 (already $((sz/1024**3))G)"
  curl -L --connect-timeout 30 --max-time 0 --retry 20 --retry-delay 30 \
       --continue-at - -o "$TARGET" "$V4_BASE/$FILE" 2>&1 | tail -1
done
log "=== ALL DOWNLOADS DONE ==="
