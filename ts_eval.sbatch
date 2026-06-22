#!/bin/bash
#SBATCH --job-name=ts_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Unified EVAL of any policy checkpoint under the SAME measurement protocol
# (eval mode = cf-injection trust + EM + cost), so baseline and trust recipes
# are directly comparable at the same step. val_only -> no training, just one val.
#   CKPT : HF-format checkpoint dir (verl_checkpoints/<exp>/actor/global_step_N)
#   TAG  : label for the run (appears in logs)
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="192.168.0.0/16,127.0.0.1,localhost"
export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=XFORMERS TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa
# ckpt is stored in float32 -> FSDP actor alone is ~17GB; offload params to CPU so
# vLLM has room on a 24GB card, and reduce fragmentation OOMs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="${CKPT:?set CKPT=verl_checkpoints/<exp>/actor/global_step_N}"
TAG="${TAG:-eval}"
N_GPU="${N_GPU:-1}"
RETRIEVER_URL="${RETRIEVER_URL:-http://192.168.102.21:8000/retrieve}"
export CF_JUDGE_URL="${CF_JUDGE_URL:-http://192.168.102.19:8001/judge_batch}"
# eval mode uses cf injection for trust; no probe needed.
export ECO_TRUST_VARIANT=balanced ECO_TRUST_SIGNAL=cf ECO_USE_CF=1
echo "[ts_eval] $(hostname) TAG=$TAG CKPT=$CKPT at $(date)"

for i in $(seq 1 60); do
  curl -s -o /dev/null -w "%{http_code}" "$RETRIEVER_URL" -X POST \
    -H "Content-Type: application/json" -d '{"queries":["test"],"topk":1}' \
    | grep -qE "200|422" && break
  sleep 10
done

python3 -m verl.trainer.main_ppo_eco \
    data.train_files=data/nq_search/train.parquet \
    data.val_files=data/nq_search/test.parquet \
    data.train_data_num=null \
    data.val_data_num=500 \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="${CKPT}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${EVAL_GPU_MEM_UTIL:-0.3} \
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
    trainer.project_name=TrustSearch_eval \
    trainer.experiment_name=eval_${TAG} \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    max_turns=2 \
    retriever.url="${RETRIEVER_URL}" \
    retriever.topk=3
echo "[ts_eval] $TAG done"
