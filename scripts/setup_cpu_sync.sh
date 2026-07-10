#!/bin/bash
# ============================================================
# setup_cpu_sync.sh — One-time SSH key setup for GPU→CPU sync
# ============================================================
# Run this ONCE on the GPU server to configure passwordless
# SSH access so sync_to_cpu.sh can rsync without prompts.
#
# Usage: bash scripts/setup_cpu_sync.sh
# ============================================================

set -euo pipefail

CPU_USER="administrator"
CPU_HOST="116.202.230.124"
SSH_KEY="/root/.ssh/cpu_sync"
CRON_CMD="*/5 * * * * /nvme0n1-disk/nvme01/ai-teaching-api/scripts/sync_to_cpu.sh >> /sdb-disk/ai-teaching/logs/cron.log 2>&1"

echo "========================================================"
echo " GPU → CPU Sync Setup"
echo "========================================================"

# ── Step 1: Generate SSH key ─────────────────────────────────
if [ -f "$SSH_KEY" ]; then
    echo "[1/4] SSH key already exists at $SSH_KEY — skipping"
else
    echo "[1/4] Generating SSH key at $SSH_KEY..."
    ssh-keygen -t ed25519 -f "$SSH_KEY" -N "" -C "gpu-to-cpu-sync"
    echo "      ✓ Key generated"
fi

# ── Step 2: Copy public key to CPU ──────────────────────────
echo ""
echo "[2/4] Copying public key to CPU server ($CPU_HOST)..."
echo "      You will be prompted for the CPU server password."
ssh-copy-id -i "${SSH_KEY}.pub" "$CPU_USER@$CPU_HOST"
echo "      ✓ Public key installed on CPU"

# ── Step 3: Test SSH connection ──────────────────────────────
echo ""
echo "[3/4] Testing passwordless SSH connection..."
RESULT=$(ssh -i "$SSH_KEY" \
    -o StrictHostKeyChecking=no \
    -o ConnectTimeout=10 \
    "$CPU_USER@$CPU_HOST" \
    echo "ssh_ok")

if [ "$RESULT" = "ssh_ok" ]; then
    echo "      ✓ SSH connection working (no password needed)"
else
    echo "      ✗ SSH test failed — check CPU server access"
    exit 1
fi

# ── Step 4: Install cron job ─────────────────────────────────
echo ""
echo "[4/4] Installing cron job (every 5 minutes)..."

# Make sync script executable
chmod +x /nvme0n1-disk/nvme01/ai-teaching-api/scripts/sync_to_cpu.sh

# Create log dir
mkdir -p /sdb-disk/ai-teaching/logs

# Add to crontab (only if not already there)
CURRENT_CRON=$(crontab -l 2>/dev/null || true)
if echo "$CURRENT_CRON" | grep -q "sync_to_cpu.sh"; then
    echo "      Cron job already installed — skipping"
else
    (echo "$CURRENT_CRON"; echo "$CRON_CMD") | crontab -
    echo "      ✓ Cron job installed: runs every 5 minutes"
fi

# ── Done ────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " ✓ Setup complete!"
echo ""
echo " Files will sync from:"
echo "   GPU: /sdb-disk/ai-teaching/"
echo "   CPU: /home2/ai-teaching-api/storage/"
echo ""
echo " To test immediately:"
echo "   bash /nvme0n1-disk/nvme01/ai-teaching-api/scripts/sync_to_cpu.sh"
echo ""
echo " Check sync log:"
echo "   tail -f /sdb-disk/ai-teaching/logs/cpu_sync.log"
echo "========================================================"
