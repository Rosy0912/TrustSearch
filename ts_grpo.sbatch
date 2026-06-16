#!/bin/bash
#SBATCH --job-name=ts_grpo
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# TrustSearch GRPO training, 4 ablation variants selected by $VARIANT:
#   baseline : EM-only reward (no trust, no cost)            [no probe service]
#   full     : EM + gate(1-boundary)*flip - cost  (ours)     [probe service]
#   nogate   : EM + flip - cost  (gate off; tests over-search)
#   nocost   : EM + gate*flip     (cost off; tests cost term)
# All start from the same sr1-baseline-em/step250 init -> isolates the reward effect.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="192.168.0.0/16,127.0.0.1,localhost"
export PYTHONUNBUFFERED=1

VARIANT="${VARIANT:?set VARIANT}"
MODEL_PATH="${MODEL_PATH:-verl_checkpoints/sr1-baseline-em/actor/global_step_250}"
N_GPU="${N_GPU:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
RETRIEVER_URL="${RETRIEVER_URL:-http://192.168.102.21:8000/retrieve}"
export PROBE_URL="${PROBE_URL:-http://192.168.102.19:8002/judge_probe}"
export CF_JUDGE_URL="${CF_JUDGE_URL:-http://192.168.102.19:8001/judge_batch}"
export EXPERIMENT_NAME="ts_${VARIANT}"

# ---- reward config per variant ----
export ECO_TRUST_VARIANT=balanced
export ECO_BAL_NOSEARCH_BONUS=0.5
case "$VARIANT" in
  baseline) export ECO_TRUST_SIGNAL=lexical ECO_BAL_GROUND=0   ECO_W_COST=0   ECO_PROBE_GATE=1 ;;
  full)     export ECO_TRUST_SIGNAL=probe   ECO_BAL_GROUND=0.5 ECO_W_COST=0.2 ECO_PROBE_GATE=1 ;;
  nogate)   export ECO_TRUST_SIGNAL=probe   ECO_BAL_GROUND=0.5 ECO_W_COST=0.2 ECO_PROBE_GATE=0 ;;
  nocost)   export ECO_TRUST_SIGNAL=probe   ECO_BAL_GROUND=0.5 ECO_W_COST=0   ECO_PROBE_GATE=1 ;;
  *) echo "unknown VARIANT=$VARIANT"; exit 1 ;;
esac
echo "[ts_grpo] $(hostname) variant=$VARIANT signal=$ECO_TRUST_SIGNAL gate=$ECO_PROBE_GATE "\
     "bal_ground=$ECO_BAL_GROUND w_cost=$ECO_W_COST model=$MODEL_PATH"

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
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
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
echo "[ts_grpo] $VARIANT done"
