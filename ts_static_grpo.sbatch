#!/bin/bash
#SBATCH --job-name=ts_static
#SBATCH --partition=RTX4090
#SBATCH --nodelist=node4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# TrustSearch STATIC ablation: frozen r0 probe heads + live encoder.
# This is the critical control to test whether co-evolution is necessary.
# If ts_static ≈ ts_full on EM/hallucination, co-evolution is not essential.
# If ts_static << ts_full, co-evolution is validated.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="192.168.0.0/16,127.0.0.1,localhost"
export PYTHONUNBUFFERED=1
# Use XFORMERS attention backend: FlashAttention triggers CUDA illegal memory
# access on ADA6000/RTX4090 cards; XFORMERS is stable (matches baseline on L40S).
export VLLM_ATTENTION_BACKEND=XFORMERS TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa

MODEL_PATH="${MODEL_PATH:-/data1/public/hf/Qwen/Qwen2.5-3B-Instruct}"
N_GPU="${N_GPU:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
RETRIEVER_URL="${RETRIEVER_URL:-http://192.168.102.21:8000/retrieve}"
export PROBE_URL="${PROBE_URL:-http://192.168.102.15:8005/judge_probe}"
export CF_JUDGE_URL="${CF_JUDGE_URL:-http://192.168.102.19:8001/judge_batch}"
export EXPERIMENT_NAME="ts_static"

# Same reward config as ts_full but probe server uses frozen heads
export ECO_TRUST_VARIANT=balanced
export ECO_BAL_NOSEARCH_BONUS=0.5
export ECO_TRUST_SIGNAL=probe
export ECO_BAL_GROUND=0.5
export ECO_W_COST=0.2
export ECO_PROBE_GATE=1

echo "[ts_static] $(hostname) FROZEN PROBE ABLATION"
echo "  model=$MODEL_PATH, probe heads frozen at r0 (base250b)"
echo "  reward: signal=probe gate=1 bal_ground=0.5 w_cost=0.2"

# wait for retriever
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
    data.val_data_num=200 \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['console'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.project_name=TrustSearch \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.total_epochs=15 \
    trainer.total_training_steps=400 \
    trainer.default_local_dir=verl_checkpoints/${EXPERIMENT_NAME} \
    max_turns=2 \
    retriever.url="${RETRIEVER_URL}" \
    retriever.topk=3
echo "[ts_static] done"
