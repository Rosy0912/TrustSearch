#!/bin/bash
#SBATCH --job-name=ts_static_probe
#SBATCH --partition=RTX4090
#SBATCH --nodelist=node7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Static probe server for ts_static ablation.
# Probe heads are FROZEN at r0; encoder tracks the policy checkpoint.
# The manifest is updated manually (or by a simple watcher) to point encoder
# to the latest ts_static checkpoint, but probes never change.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="192.168.0.0/16,127.0.0.1,localhost"

# Create the manifest directory
mkdir -p probes/live_ts_static

# Initialize manifest with the starting encoder
cat > probes/live_ts_static/manifest.json << 'EOF'
{"round": 0, "encoder": "verl_checkpoints/sr1-baseline-em/actor/global_step_250",
 "commit": "probes/base250b/commit_probe.pkl",
 "boundary": "probes/base250b/boundary_probe.pkl"}
EOF

python3 probe_server_static.py \
    --manifest probes/live_ts_static/manifest.json \
    --frozen_commit probes/base250b/commit_probe.pkl \
    --frozen_boundary probes/base250b/boundary_probe.pkl \
    --port 8005
