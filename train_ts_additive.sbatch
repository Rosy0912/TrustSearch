#!/bin/bash
#SBATCH --job-name=ts_addi
#SBATCH --partition=L40S
#SBATCH --nodelist=node10
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/ts_addi_%j.out
#SBATCH --error=logs/ts_addi_%j.err
# Method 2: TRUST AS MAIN ADDITIVE TERM.
# reward = w_trust*D_trust + w_perf*correct - w_cost*norm_cost
# Trust is the dominant term (w_trust=1.0), no longer a soft multiplier.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
export ECO_FLOOR=0.3 ECO_ALPHA=0.5
export ECO_TRUST_VARIANT=additive
export ECO_W_TRUST=1.0 ECO_W_PERF=0.3 ECO_W_COST=0.2
export ECO_USE_CF=1
export CF_JUDGE_URL="http://192.168.102.19:8001/judge_batch"
export N_GPU=4 MINI_BATCH=32 GPU_MEM_UTIL=0.45
export RETRIEVER_URL="http://192.168.102.16:8000/retrieve"
export EXPERIMENT_NAME=sr1-ts-additive
echo "[train] TrustSearch ADDITIVE (Method2, trust main term) 4xL40S node10, retriever=node6, cf=node9"
source eco_train_common.sh
echo "[done] $(date)"
