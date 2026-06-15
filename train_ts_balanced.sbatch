#!/bin/bash
#SBATCH --job-name=ts_bal
#SBATCH --partition=L40S
#SBATCH --nodelist=node10
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/ts_bal_%j.out
#SBATCH --error=logs/ts_bal_%j.err
# A1 MAIN: balanced reward = perf floor (1.0) + additive grounding bonus + cost.
# Trust is an additive bonus (never zeros a correct answer) -> no perf/trust trade-off.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
export ECO_TRUST_VARIANT=balanced
export ECO_BAL_NOSEARCH_BONUS=0.5 ECO_BAL_GROUND=0.5 ECO_W_COST=0.3
export ECO_FLOOR=0.3 ECO_USE_CF=1
export CF_JUDGE_URL="http://192.168.102.19:8001/judge_batch"
export N_GPU=4 MINI_BATCH=32 GPU_MEM_UTIL=0.45
export RETRIEVER_URL="http://192.168.102.15:8000/retrieve"
export EXPERIMENT_NAME=sr1-ts-balanced
echo "[train] A1 balanced (perf+ground-bonus+cost) 4xL40S node10, retriever=node5, cf=node9"
source eco_train_common.sh
echo "[done] $(date)"
