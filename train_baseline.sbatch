#!/bin/bash
#SBATCH --job-name=sr1_base
#SBATCH --partition=L40S
#SBATCH --nodelist=node11
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/sr1_base_%j.out
#SBATCH --error=logs/sr1_base_%j.err
# Search-R1 BASELINE (plain EM reward). Same hyperparams as TrustSearch full so the
# only variable is the reward function -> direct comparison. Validation now reports
# the three evaluation dimensions (performance/trust/cost) via the cf_judge service.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export VLLM_ATTENTION_BACKEND=XFORMERS TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
export N_GPU=6 MINI_BATCH=60 GPU_MEM_UTIL=0.50
export EVAL_USE_CF=1
export CF_JUDGE_URL="http://192.168.102.19:8001/judge_batch"
export RETRIEVER_URL="http://192.168.102.21:8000/retrieve"
export EXPERIMENT_NAME=sr1-baseline-em
echo "[train] Search-R1 BASELINE plain EM (6xL40S, node11), retriever=node11, 3-dim eval"
source baseline_train_common.sh
echo "[done] $(date)"
