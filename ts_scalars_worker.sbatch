#!/bin/bash
#SBATCH --job-name=tsscal
#SBATCH --output=logs/ts_scalars_%j.out
#SBATCH --error=logs/ts_scalars_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
# Commit-token scalar/logit + hidden read-out dump for Exp1.
#   Required env: TAG DATA CAP G OUT RETRIEVER_URL [MODEL]
set -e
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh
conda activate searchr1
cd /home/jmzhang/self-evolution/Search-R1

export RETRIEVER_URL="${RETRIEVER_URL:-http://192.168.102.21:8000/retrieve}"
export NO_PROXY="192.168.0.0/16,127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
MODEL="${MODEL:-/data1/public/hf/Qwen/Qwen2.5-3B-Instruct}"
G="${G:-5}"
LAYERS="${LAYERS:-21,24,27}"

echo "[scalars-worker] $(hostname) tag=$TAG data=$DATA cap=$CAP G=$G model=$MODEL"

python ts_probe_scalars.py \
    --model "$MODEL" \
    --data "$DATA" \
    --tag "$TAG" \
    --cap "${CAP:-300}" \
    --G "$G" \
    --layers "$LAYERS" \
    --out "$OUT"
