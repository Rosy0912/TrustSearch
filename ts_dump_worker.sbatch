#!/bin/bash
#SBATCH --job-name=tsdump
#SBATCH --output=logs/tsdump_%j.out
#SBATCH --error=logs/tsdump_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
# Single-GPU shard worker for TrustSearch probe data dump.
# Required env (export before sbatch, or pass via --export):
#   TAG DATA OFFSET CAP G LAYERS OUT RETRIEVER_URL MODEL
set -e
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh
conda activate searchr1
cd /home/jmzhang/self-evolution/Search-R1

export RETRIEVER_URL="${RETRIEVER_URL:-http://192.168.102.15:8000/retrieve}"
export NO_PROXY="192.168.0.0/16,127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
MODEL="${MODEL:-verl_checkpoints/sr1-baseline-em/actor/global_step_250}"
G="${G:-5}"
LAYERS="${LAYERS:-21,24,27}"
OUT="${OUT:-dumps/base250}"

echo "[worker] $(hostname) tag=$TAG data=$DATA offset=$OFFSET cap=$CAP G=$G"

python ts_probe_dump.py \
    --model "$MODEL" \
    --data "$DATA" \
    --tag "${TAG}" \
    --offset "${OFFSET:-0}" \
    --cap "${CAP:-100}" \
    --G "$G" \
    --layers "$LAYERS" \
    --gpu_mem "${GPU_MEM:-0.35}" \
    --out "$OUT"
echo "[worker] done"
