#!/bin/bash
#SBATCH --job-name=eco_v3_nself
#SBATCH --partition=RTX4090
#SBATCH --nodelist=node5
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eco_v3_nself_%j.out
#SBATCH --error=logs/eco_v3_nself_%j.err
# EcoSearch v3 ablation: fixed alpha=0.3 (NO self-evolving). Tests self-evolving value.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
export ECO_FLOOR=0.3 ECO_ALPHA=0.3 ECO_NO_SELF=1
export N_GPU=8 MINI_BATCH=64 GPU_MEM_UTIL=0.35
export EXPERIMENT_NAME=sr1-eco-v3-noself
echo "[train] EcoSearch v3 no-self-evolving (8xRTX4090)"
source eco_train_common.sh
echo "[done] $(date)"
