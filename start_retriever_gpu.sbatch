#!/bin/bash
#SBATCH --job-name=retr_gpu
#SBATCH --partition=RTX3090
#SBATCH --nodelist=node3
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/retr_gpu_%j.out
#SBATCH --error=logs/retr_gpu_%j.err
# GPU-accelerated retriever (node3, 192.168.102.33:8000). FAISS Flat index sharded
# across 4x RTX3090 in fp16 (~30GB). Brings per-query latency from ~19s (CPU) to <0.1s,
# so a SINGLE retriever can serve all training jobs without timeout.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
DATA_DIR=data/retriever_data
echo "[retriever-GPU] Starting on $(hostname) at $(date), faiss_gpu sharded x4"
python search_r1/search/retrieval_server.py \
    --index_path "$DATA_DIR/e5_Flat.index" \
    --corpus_path "$DATA_DIR/wiki-18.jsonl" \
    --topk 3 --retriever_name e5 \
    --retriever_model /home/jmzhang/models/e5-base-v2 \
    --faiss_gpu
