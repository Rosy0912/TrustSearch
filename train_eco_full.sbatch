#!/bin/bash
#SBATCH --job-name=eco_full
#SBATCH --partition=A100
#SBATCH --nodelist=node1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/eco_full_%j.out
#SBATCH --error=logs/eco_full_%j.err
# EcoSearch FULL on fastest hardware (4xA100, ~9min/step). causal D_trust +
# gentle cost (first search free) + self-evolving alpha.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
export ECO_FLOOR=0.3 ECO_ALPHA=0.5
export ECO_USE_CF=1
export CF_JUDGE_URL="http://192.168.102.19:8001/judge_batch"
export N_GPU=4 MINI_BATCH=32 GPU_MEM_UTIL=0.45
export RETRIEVER_URL="http://192.168.102.15:8000/retrieve"
export EXPERIMENT_NAME=sr1-eco-full-a100
echo "[train] EcoSearch FULL + ONLINE counterfactual (4xA100), retriever=node5, cf_judge=node9"
source eco_train_common.sh
echo "[done] $(date)"
