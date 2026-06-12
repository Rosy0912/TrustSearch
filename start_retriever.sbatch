#!/bin/bash
#SBATCH --job-name=retriever_svc
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node11
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/retriever_%j.out
#SBATCH --error=logs/retriever_%j.err
# Long-running retriever service. Start BEFORE training jobs.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh
conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1

DATA_DIR=data/retriever_data
echo "[retriever] Starting on $(hostname) at $(date)"
echo "[retriever] Index: $DATA_DIR/e5_Flat.index"
echo "[retriever] Corpus: $DATA_DIR/wiki-18.jsonl"

python search_r1/search/retrieval_server.py \
    --index_path "$DATA_DIR/e5_Flat.index" \
    --corpus_path "$DATA_DIR/wiki-18.jsonl" \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model /home/jmzhang/models/e5-base-v2

echo "[retriever] Exited at $(date)"
