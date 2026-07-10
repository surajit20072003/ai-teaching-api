#!/bin/bash
# ============================================================
# sync_to_cpu.sh — GPU → CPU file sync (runs on GPU host via cron)
# ============================================================
# Rsyncs all pregen media from GPU /sdb-disk/ to CPU /app/storage/
# keeping exactly the same folder structure on both servers.
#
# Cron (every 5 minutes):
#   */5 * * * * /nvme0n1-disk/nvme01/ai-teaching-api/scripts/sync_to_cpu.sh
# ============================================================

set -euo pipefail

# ── Config ──────────────────────────────────────────────────
CPU_USER="root"
CPU_HOST="116.202.230.124"
CPU_PORT="81"
CPU_PATH="/home2/ai-teaching-api/storage"   # host path on CPU (maps to /app/storage in Docker)
GPU_PATH="/sdb-disk/ai-teaching"            # GPU staging area
SSH_KEY="/home/administrator/.ssh/cpu_sync"
LOG_DIR="/sdb-disk/ai-teaching/logs"
LOG_FILE="$LOG_DIR/cpu_sync.log"
MAX_LOG_LINES=5000                          # rotate log when it gets too big

# ── Ensure log dir exists ────────────────────────────────────
mkdir -p "$LOG_DIR"

# ── Log rotation (keep last 5000 lines) ─────────────────────
if [ -f "$LOG_FILE" ] && [ "$(wc -l < "$LOG_FILE")" -gt "$MAX_LOG_LINES" ]; then
    tail -n 1000 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
fi

TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# ── Check SSH key exists ─────────────────────────────────────
if [ ! -f "$SSH_KEY" ]; then
    echo "[$TIMESTAMP] ERROR: SSH key not found at $SSH_KEY — run setup_cpu_sync.sh first" >> "$LOG_FILE"
    exit 1
fi

# ── Check GPU source dir exists ──────────────────────────────
if [ ! -d "$GPU_PATH" ]; then
    echo "[$TIMESTAMP] ERROR: GPU source dir $GPU_PATH not found" >> "$LOG_FILE"
    exit 1
fi

# ── Ensure CPU storage dir exists ───────────────────────────
ssh -i "$SSH_KEY" \
    -p "$CPU_PORT" \
    -o StrictHostKeyChecking=no \
    -o ConnectTimeout=10 \
    "$CPU_USER@$CPU_HOST" \
    "mkdir -p $CPU_PATH" 2>> "$LOG_FILE"

# ── Run rsync ────────────────────────────────────────────────
echo "[$TIMESTAMP] Starting rsync: $GPU_PATH/ → $CPU_USER@$CPU_HOST:$CPU_PATH/" >> "$LOG_FILE"

rsync \
    --archive \
    --compress \
    --ignore-existing \
    --exclude="*.log" \
    --exclude="*.tmp" \
    --exclude="logs/" \
    --stats \
    -e "ssh -i $SSH_KEY -p $CPU_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=30" \
    "$GPU_PATH/" \
    "$CPU_USER@$CPU_HOST:$CPU_PATH/" \
    2>> "$LOG_FILE" | tail -5 >> "$LOG_FILE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$TIMESTAMP] ✓ Sync complete (exit 0)" >> "$LOG_FILE"
else
    echo "[$TIMESTAMP] ✗ Sync FAILED (exit $EXIT_CODE)" >> "$LOG_FILE"
fi

exit $EXIT_CODE
