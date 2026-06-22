#!/bin/bash
#SBATCH --job-name=ts_eval_full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# ======================================================================
# Full 3-dim evaluation (Performance × Trust × Cost) on ENTIRE test set.
# Uses all 3610 NQ test samples (vs inline's 192) for statistically
# reliable results.  Evaluates BOTH baseline and TrustSearch at step 50.
#
# Usage:
#   sbatch ts_eval_full.sbatch                 # eval both
#   CKPT=<path> TAG=<name> sbatch ts_eval_full.sbatch   # eval single
# ======================================================================
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="192.168.0.0/16,127.0.0.1,localhost"
export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=XFORMERS TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RETRIEVER_URL="${RETRIEVER_URL:-http://192.168.102.21:8000/retrieve}"
export CF_JUDGE_URL="${CF_JUDGE_URL:-http://192.168.102.19:8001/judge_batch}"
# Eval mode: unified cf-injection for trust measurement
export ECO_TRUST_VARIANT=balanced ECO_TRUST_SIGNAL=cf ECO_USE_CF=1

N_GPU="${N_GPU:-1}"
# A100 80GB can afford higher GPU mem util for faster vLLM inference
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"

# --- Wait for retriever ---
echo "[ts_eval_full] Waiting for retriever at ${RETRIEVER_URL}..."
for i in $(seq 1 60); do
  curl -s -o /dev/null -w "%{http_code}" "$RETRIEVER_URL" -X POST \
    -H "Content-Type: application/json" -d '{"queries":["test"],"topk":1}' \
    | grep -qE "200|422" && break
  sleep 10
done
echo "[ts_eval_full] Retriever ready."

run_eval() {
  local ckpt="$1"
  local tag="$2"
  echo ""
  echo "================================================================"
  echo "[ts_eval_full] START: TAG=$tag CKPT=$ckpt at $(date)"
  echo "================================================================"

  python3 -m verl.trainer.main_ppo_eco \
      data.train_files=data/nq_search/train.parquet \
      data.val_files=data/nq_search/test.parquet \
      data.train_data_num=null \
      data.val_data_num=null \
      data.train_batch_size=128 \
      data.val_batch_size=64 \
      data.max_prompt_length=4096 \
      data.max_response_length=500 \
      data.max_start_length=2048 \
      data.max_obs_length=500 \
      algorithm.adv_estimator=grpo \
      actor_rollout_ref.model.path="${ckpt}" \
      actor_rollout_ref.model.enable_gradient_checkpointing=true \
      actor_rollout_ref.model.use_remove_padding=False \
      actor_rollout_ref.actor.ppo_mini_batch_size=32 \
      actor_rollout_ref.actor.ppo_micro_batch_size=8 \
      actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
      actor_rollout_ref.actor.fsdp_config.param_offload=true \
      actor_rollout_ref.actor.fsdp_config.grad_offload=true \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
      actor_rollout_ref.ref.fsdp_config.param_offload=true \
      actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
      actor_rollout_ref.rollout.n_agent=1 \
      actor_rollout_ref.rollout.temperature=0 \
      actor_rollout_ref.actor.state_masking=true \
      trainer.logger=['console'] \
      +trainer.val_only=true \
      +trainer.val_before_train=true \
      trainer.default_hdfs_dir=null \
      trainer.n_gpus_per_node=${N_GPU} \
      trainer.nnodes=1 \
      trainer.project_name=TrustSearch_eval_full \
      trainer.experiment_name=eval_${tag} \
      trainer.total_epochs=1 \
      trainer.total_training_steps=1 \
      max_turns=3 \
      retriever.url="${RETRIEVER_URL}" \
      retriever.topk=3

  echo "[ts_eval_full] DONE: $tag at $(date)"
}

# --- If user specified a single CKPT, eval that only ---
if [ -n "${CKPT:-}" ]; then
  run_eval "$CKPT" "${TAG:-custom}"
  exit 0
fi

# --- Otherwise: evaluate both baseline and TrustSearch at step 50 ---
BASELINE_CKPT="verl_checkpoints/ts_baseline_v3/actor/global_step_50"
TRUST_CKPT="verl_checkpoints/ts_coevo_F2/actor/global_step_50"

echo "[ts_eval_full] $(hostname) evaluating BOTH checkpoints on FULL test set (3610 samples)"
echo "[ts_eval_full] Baseline: $BASELINE_CKPT"
echo "[ts_eval_full] TrustSearch: $TRUST_CKPT"

run_eval "$BASELINE_CKPT" "baseline_step50_full"
run_eval "$TRUST_CKPT"    "trust_step50_full"

echo ""
echo "================================================================"
echo "[ts_eval_full] ALL EVALUATIONS COMPLETE at $(date)"
echo "================================================================"
