#!/bin/bash
#SBATCH --job-name=eval_ck50
#SBATCH --partition=RTX4090
#SBATCH --nodelist=node7
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=logs/eval_ck50_%j.out
#SBATCH --error=logs/eval_ck50_%j.err
# Offline 3-dim eval of the OLD TrustSearch step-50 checkpoint.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
export ECO_FLOOR=0.3 ECO_ALPHA=0.5 ECO_USE_CF=1
export CF_JUDGE_URL="http://192.168.102.19:8001/judge_batch"
export RETRIEVER_URL="http://192.168.102.15:8000/retrieve"
export N_GPU=4 GPU_MEM_UTIL=0.35
export MODEL_PATH="verl_checkpoints/sr1-eco-full-a100/actor/global_step_50"
export EXPERIMENT_NAME=eval-trustsearch-ckpt50
echo "[eval] 3-dim val_only of TrustSearch ckpt-50"
source eval_3dim_common.sh
echo "[done] $(date)"
