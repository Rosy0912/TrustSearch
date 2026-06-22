#!/bin/bash
#SBATCH --job-name=coevolve
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# TrustSearch co-evolution controller: refits each experiment's probe from its
# latest policy checkpoint and hot-swaps it into the probe server.
set -uo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="192.168.0.0/16,127.0.0.1,localhost"
export PYTHONUNBUFFERED=1
echo "[coevolve] $(hostname) at $(date)"
# Auto-restart: ts_coevolve.py is already fault-tolerant per-refit (a failed refit
# leaves the manifest unchanged), but if the whole controller process dies (e.g. OOM
# during a vLLM dump), relaunch it so co-evolution keeps running for the full window.
while true; do
  python ts_coevolve.py --exps "${EXPS:-full,nogate,nocost}" --nq "${NQ:-300}" \
      --retriever "${RETRIEVER_URL:-http://192.168.102.21:8000/retrieve}" --poll 120
  rc=$?
  echo "[coevolve] controller exited rc=$rc at $(date); restarting in 20s" >&2
  sleep 20
done
