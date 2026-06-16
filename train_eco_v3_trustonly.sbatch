#!/bin/bash
#SBATCH --job-name=eco_v3_trust
#SBATCH --partition=A100
#SBATCH --nodelist=node1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eco_v3_trust_%j.out
#SBATCH --error=logs/eco_v3_trust_%j.err
# EcoSearch v3 ablation: trust-only (no cost factor). Tests if cost dimension matters.
# = behavcf upgraded with causal D_trust. 4xA100.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
export ECO_FLOOR=0.3 ECO_ALPHA=0.3 ECO_TRUST_ONLY=1
export N_GPU=4 MINI_BATCH=32 GPU_MEM_UTIL=0.45
export EXPERIMENT_NAME=sr1-eco-v3-trustonly
echo "[train] EcoSearch v3 trust-only (4xA100)"
source eco_train_common.sh
echo "[done] $(date)"
