#!/bin/bash
#SBATCH --job-name=ts_bnc
#SBATCH --partition=L40S
#SBATCH --nodelist=node10
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/ts_bnc_%j.out
#SBATCH --error=logs/ts_bnc_%j.err
# A2 ABLATION (-cost): balanced reward with ECO_W_COST=0. Isolates the effect of the
# cost term on search/q and oversearch.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
export ECO_TRUST_VARIANT=balanced
export ECO_BAL_NOSEARCH_BONUS=0.5 ECO_BAL_GROUND=0.5 ECO_W_COST=0.0
export ECO_FLOOR=0.3 ECO_USE_CF=1
export CF_JUDGE_URL="http://192.168.102.19:8001/judge_batch"
export N_GPU=4 MINI_BATCH=32 GPU_MEM_UTIL=0.45
export RETRIEVER_URL="http://192.168.102.14:8000/retrieve"
export EXPERIMENT_NAME=sr1-ts-bal-nocost
echo "[train] A2 balanced -cost (w_cost=0) 4xL40S node10, retriever=node4, cf=node9"
source eco_train_common.sh
echo "[done] $(date)"
