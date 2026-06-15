#!/bin/bash
#SBATCH --job-name=ts_cfp
#SBATCH --partition=RTX4090
#SBATCH --nodelist=node7
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/ts_cfp_%j.out
#SBATCH --error=logs/ts_cfp_%j.err
# A5 ABLATION (artifact demo): cf_primary penalizes PARAM_HALL (correct-by-memory) with
# a negative reward. Expected to LOWER EM -> demonstrates the induced perf/trust trade-off
# we deliberately avoid in the balanced design.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
export ECO_TRUST_VARIANT=cf_primary
export ECO_HALL_PENALTY=0.5 ECO_W_COST=0.2
export ECO_FLOOR=0.3 ECO_USE_CF=1
export CF_JUDGE_URL="http://192.168.102.19:8001/judge_batch"
export N_GPU=4 MINI_BATCH=32 GPU_MEM_UTIL=0.35
export RETRIEVER_URL="http://192.168.102.17:8000/retrieve"
export EXPERIMENT_NAME=sr1-ts-cfprimary
echo "[train] A5 cf_primary (PARAM_HALL penalized, artifact) 4xRTX4090 node7, retriever=node7, cf=node9"
source eco_train_common.sh
echo "[done] $(date)"
