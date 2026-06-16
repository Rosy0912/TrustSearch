#!/bin/bash
#SBATCH --job-name=eco_v2_b4
#SBATCH --partition=A100
#SBATCH --nodelist=node1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eco_v2_b4_%j.out
#SBATCH --error=logs/eco_v2_b4_%j.err
# EcoSearch v2 ablation: budget=4 (weak cost), floor=0.3
# 4xA100: ppo_mini_batch=32 (128*5/4=160 seqs/gpu, fine)
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
export ECO_BUDGET_INIT=4 ECO_FLOOR=0.3
export N_GPU=4 MINI_BATCH=32 GPU_MEM_UTIL=0.45
export EXPERIMENT_NAME=sr1-eco-v2-b4f03
echo "[train] EcoSearch v2: budget=4 floor=0.3 (4xA100)"
source eco_train_common.sh
echo "[done] $(date)"
