#!/bin/bash
# Submit ts_static experiment (probe server + watcher + training)
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1

echo "[submit_ts_static] Submitting frozen probe server..."
PSRV_JOB=$(sbatch --parsable ts_static_probe_server.sbatch)
echo "  probe server job: $PSRV_JOB"

echo "[submit_ts_static] Submitting manifest watcher..."
WATCH_JOB=$(sbatch --parsable ts_static_watcher.sbatch)
echo "  watcher job: $WATCH_JOB"

# Wait for probe server to start and get its node
echo "[submit_ts_static] Waiting for probe server to start..."
for i in $(seq 1 60); do
    NODE=$(squeue -j $PSRV_JOB --noheader --format="%N" 2>/dev/null | tr -d ' ')
    STATE=$(squeue -j $PSRV_JOB --noheader --format="%T" 2>/dev/null | tr -d ' ')
    if [ "$STATE" = "RUNNING" ] && [ -n "$NODE" ]; then
        echo "  probe server running on $NODE"
        break
    fi
    echo "  waiting... ($STATE)"
    sleep 5
done

if [ -z "$NODE" ] || [ "$STATE" != "RUNNING" ]; then
    echo "[ERROR] Probe server not running after 5 min. Submitting training anyway with default IP."
    NODE=""
fi

# Resolve node IP for PROBE_URL
if [ -n "$NODE" ]; then
    # Get the IP of the probe server node
    NODE_IP=$(scontrol show node "$NODE" 2>/dev/null | grep -oP 'NodeAddr=\K[^ ]+')
    if [ -z "$NODE_IP" ] || [ "$NODE_IP" = "$NODE" ]; then
        # Fallback: resolve via host command or use the slurm hostname
        NODE_IP=$(getent hosts "$NODE" 2>/dev/null | awk '{print $1}')
    fi
    if [ -z "$NODE_IP" ]; then
        NODE_IP="$NODE"
    fi
    PROBE_URL="http://${NODE_IP}:8004/judge_probe"
    echo "  PROBE_URL=$PROBE_URL"
else
    PROBE_URL="http://192.168.102.19:8004/judge_probe"
fi

# Wait a bit for the probe server to initialize model
echo "[submit_ts_static] Waiting 60s for probe server model init..."
sleep 60

echo "[submit_ts_static] Submitting training job..."
TRAIN_JOB=$(sbatch --parsable --export=ALL,PROBE_URL="$PROBE_URL" ts_static_grpo.sbatch)
echo "  training job: $TRAIN_JOB"

echo ""
echo "=== ts_static submission complete ==="
echo "  Probe server: job $PSRV_JOB on $NODE"
echo "  Watcher:      job $WATCH_JOB"
echo "  Training:     job $TRAIN_JOB"
echo "  PROBE_URL:    $PROBE_URL"
echo ""
echo "Monitor with:"
echo "  tail -f logs/ts_static_probe_*${PSRV_JOB}.out"
echo "  tail -f logs/ts_static_*${TRAIN_JOB}.out"
