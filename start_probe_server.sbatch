#!/bin/bash
#SBATCH --job-name=probe_srv
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Hot-reloadable TrustSearch probe server. Set via --export:
#   MANIFEST (probes/live_<exp>/manifest.json)  PORT
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1
echo "[probe_srv] $(hostname) manifest=$MANIFEST port=$PORT at $(date)"
python probe_server.py --manifest "$MANIFEST" --port "${PORT:-8002}"
