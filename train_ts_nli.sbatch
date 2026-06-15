#!/bin/bash
#SBATCH --job-name=ts_nli
#SBATCH --partition=RTX4090
#SBATCH --nodelist=node6
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/ts_nli_%j.out
#SBATCH --error=logs/ts_nli_%j.err
# T3 trust-signal ablation: NLI / entailment faithfulness (does evidence support the
# answer?). Same balanced framework; trust signal = /judge_faithfulness on the extended
# judge (node6, .16:8001). Validation uses the same judge's /judge_batch (cf) for a
# comparable metric.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
export ECO_TRUST_VARIANT=balanced ECO_TRUST_SIGNAL=nli
export ECO_BAL_NOSEARCH_BONUS=0.5 ECO_BAL_GROUND=0.5 ECO_W_COST=0.3
export ECO_FLOOR=0.3 ECO_USE_CF=1
export CF_JUDGE_URL="http://192.168.102.16:8001/judge_batch"
export N_GPU=4 MINI_BATCH=32 GPU_MEM_UTIL=0.35
export RETRIEVER_URL="http://192.168.102.14:8000/retrieve"
export EXPERIMENT_NAME=sr1-ts-nli
echo "[train] T3 balanced+nli(faithfulness) 4xRTX4090 node6, retriever=node4, judge=node6:.16"
source eco_train_common.sh
echo "[done] $(date)"
