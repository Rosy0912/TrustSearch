#!/bin/bash
#SBATCH --job-name=coevolve
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# TrustSearch co-evolution controller: refits each experiment's probe from its
# latest policy checkpoint and hot-swaps it into the probe server.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="192.168.0.0/16,127.0.0.1,localhost"
export PYTHONUNBUFFERED=1
echo "[coevolve] $(hostname) at $(date)"
python ts_coevolve.py --exps "${EXPS:-full,nogate,nocost}" --nq "${NQ:-300}" \
    --retriever "${RETRIEVER_URL:-http://192.168.102.21:8000/retrieve}" --poll 120
