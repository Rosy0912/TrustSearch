#!/bin/bash
#SBATCH --job-name=ts_static_watch
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=20:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Lightweight watcher: updates ts_static manifest encoder field
# whenever a new checkpoint appears in verl_checkpoints/ts_static/
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1

MANIFEST="probes/live_ts_static/manifest.json"
CKPT_DIR="verl_checkpoints/ts_static"
POLL=60  # seconds

echo "[ts_static_watch] watching $CKPT_DIR -> $MANIFEST"

LAST_STEP=""
# Wait for checkpoint directory to appear
echo "[ts_static_watch] waiting for $CKPT_DIR to appear..."
while [ ! -d "$CKPT_DIR" ]; do
    sleep $POLL
done
echo "[ts_static_watch] $CKPT_DIR found, starting watch loop"

while true; do
    # Find latest actor step dir
    LATEST=$(find "$CKPT_DIR/actor" -maxdepth 1 -name "global_step_*" -type d 2>/dev/null \
        | sort -t_ -k3 -n | tail -1 || true)
    if [ -n "$LATEST" ] && [ "$LATEST" != "$LAST_STEP" ]; then
        ROUND=$(echo "$LATEST" | grep -oP 'global_step_\K[0-9]+')
        echo "[ts_static_watch] new checkpoint: $LATEST (round=$ROUND)"
        # Update manifest (encoder only, probes stay frozen)
        python3 -c "
import json, sys
m = json.load(open('$MANIFEST'))
m['encoder'] = '$LATEST'
m['round'] = int('$ROUND')
json.dump(m, open('$MANIFEST', 'w'))
print(f'[ts_static_watch] manifest updated: encoder={m[\"encoder\"]} round={m[\"round\"]}')
"
        LAST_STEP="$LATEST"
    fi
    sleep $POLL
done
