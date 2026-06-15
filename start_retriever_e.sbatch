#!/bin/bash
#SBATCH --job-name=retr_e
#SBATCH --partition=RTX4090
#SBATCH --nodelist=node4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/retr_e_%j.out
#SBATCH --error=logs/retr_e_%j.err
# Dedicated retriever E (node4, 192.168.102.14:8000) for the 'cf_primary' trust run.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1
DATA_DIR=data/retriever_data
echo "[retriever-E] Starting on $(hostname) at $(date)"
python search_r1/search/retrieval_server.py \
    --index_path "$DATA_DIR/e5_Flat.index" \
    --corpus_path "$DATA_DIR/wiki-18.jsonl" \
    --topk 3 --retriever_name e5 \
    --retriever_model /home/jmzhang/models/e5-base-v2
