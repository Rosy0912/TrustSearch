#!/bin/bash
#SBATCH --job-name=rcp_probe
#SBATCH --partition=RTX4090
#SBATCH --nodelist=node5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SHARED FROZEN probe server for the reward-recipe grid search (6/17).
# Probe heads AND encoder are frozen at round0 (base250b, encoder=base Qwen2.5-3B).
# Every recipe (F1..F5) connects to THIS one server, so the probe signal is
# byte-for-byte identical across recipes -> the only thing that differs between
# runs is the reward coefficient combo. No co-evolution here by design.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="192.168.0.0/16,127.0.0.1,localhost"
export PYTHONUNBUFFERED=1

mkdir -p probes/live_rcp_frozen
cat > probes/live_rcp_frozen/manifest.json << 'EOF'
{"round": 0, "encoder": "/data1/public/hf/Qwen/Qwen2.5-3B-Instruct",
 "commit": "probes/base250b/commit_probe.pkl",
 "boundary": "probes/base250b/boundary_probe.pkl"}
EOF

echo "[rcp_probe] $(hostname) FROZEN round0 probe server on :8005 at $(date)"
# Auto-restart loop so a transient crash doesn't drop the shared service.
while true; do
  python3 probe_server_static.py \
      --manifest probes/live_rcp_frozen/manifest.json \
      --frozen_commit probes/base250b/commit_probe.pkl \
      --frozen_boundary probes/base250b/boundary_probe.pkl \
      --port 8005
  rc=$?
  echo "[rcp_probe] server exited rc=$rc at $(date); restarting in 10s" >&2
  sleep 10
done
