#!/bin/bash
#SBATCH --job-name=ts_recipe
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# TrustSearch REWARD-RECIPE grid search (post-collapse redesign, 6/17).
#
# Goal: with the structure FIXED to "full" (probe signal + gate), sweep the three
# reward coefficients to find the combo that trains stably (no EM collapse) and
# best trades off EM / hallucination / search-cost.
#
#   reward(correct) = 1 + bal_ground*(1-boundary)*flip            - w_cost*norm_cost
#   reward(wrong)   = min(0, wrong_ground_scale*... - w_cost*cost) <= 0   (EM dominates)
#
# Lessons baked in vs the collapsed runs:
#   * grounding bonus is GATED INSIDE correctness (wrong_ground_scale defaults 0)
#   * wrong answers are clamped <= 0 so EM ordering can never be inverted
#   * PROBE IS FROZEN at round0 (no co-evolution) -> isolates the reward recipe
#
# Pick recipe via $RECIPE; each sets (BAL_GROUND, W_COST, WRONG_GROUND_SCALE):
#   F1_emfirst    0.2  0.1  0.0   EM strongly dominant, grounding is garnish
#   F2_balanced   0.3  0.2  0.0   middle ground
#   F3_strongcost 0.3  0.4  0.0   heavy cost to crush over-search
#   F4_lowground  0.15 0.15 0.05  minimal grounding, tiny wrong-variance
#   F5_groundheavy 0.5 0.3  0.0   strong grounding BUT strong cost + wrong=0
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="192.168.0.0/16,127.0.0.1,localhost"
export PYTHONUNBUFFERED=1
# XFORMERS attention: FlashAttention triggers CUDA illegal memory access on
# ADA6000/RTX4090; XFORMERS is stable (matches baseline on L40S).
export VLLM_ATTENTION_BACKEND=XFORMERS TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa

RECIPE="${RECIPE:?set RECIPE (F1_emfirst|F2_balanced|F3_strongcost|F4_lowground|F5_groundheavy)}"
MODEL_PATH="${MODEL_PATH:-/data1/public/hf/Qwen/Qwen2.5-3B-Instruct}"
N_GPU="${N_GPU:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
# Offload toggles (default off for A100/40G+; set =true on 24GB cards like RTX4090).
PARAM_OFFLOAD="${PARAM_OFFLOAD:-false}"
GRAD_OFFLOAD="${GRAD_OFFLOAD:-false}"
OPTIM_OFFLOAD="${OPTIM_OFFLOAD:-false}"
RETRIEVER_URL="${RETRIEVER_URL:-http://192.168.102.21:8000/retrieve}"
# Shared FROZEN probe server (round0). All recipes connect to the same one so the
# probe is identical across recipes -> only the reward coefficients differ.
export PROBE_URL="${PROBE_URL:-http://192.168.102.15:8005/judge_probe}"
export CF_JUDGE_URL="${CF_JUDGE_URL:-http://192.168.102.19:8001/judge_batch}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-rcp_${RECIPE}}"
export SAVE_FREQ="${SAVE_FREQ:-25}"
export TEST_FREQ="${TEST_FREQ:-25}"
# Recipe search: stop at step 100 for fast screening (override via env).
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"

# ---- fixed structure = full (probe + gate); sweep the 3 coefficients ----
export ECO_TRUST_VARIANT=balanced
export ECO_TRUST_SIGNAL=probe
export ECO_PROBE_GATE=1
export ECO_BAL_NOSEARCH_BONUS=0.5
case "$RECIPE" in
  F1_emfirst)     export ECO_BAL_GROUND=0.2  ECO_W_COST=0.1  ECO_WRONG_GROUND_SCALE=0.0  ;;
  F2_balanced)    export ECO_BAL_GROUND=0.3  ECO_W_COST=0.2  ECO_WRONG_GROUND_SCALE=0.0  ;;
  F3_strongcost)  export ECO_BAL_GROUND=0.3  ECO_W_COST=0.4  ECO_WRONG_GROUND_SCALE=0.0  ;;
  F4_lowground)   export ECO_BAL_GROUND=0.15 ECO_W_COST=0.15 ECO_WRONG_GROUND_SCALE=0.05 ;;
  F5_groundheavy) export ECO_BAL_GROUND=0.5  ECO_W_COST=0.3  ECO_WRONG_GROUND_SCALE=0.0  ;;
  # ---- extended grid (queued behind F1-F5; fill in the coefficient space) ----
  F6_nocost)      export ECO_BAL_GROUND=0.3  ECO_W_COST=0.0  ECO_WRONG_GROUND_SCALE=0.0  ;;  # isolate: ground only, no cost
  F7_costheavy)   export ECO_BAL_GROUND=0.2  ECO_W_COST=0.5  ECO_WRONG_GROUND_SCALE=0.0  ;;  # extreme cost, minimal ground
  F8_minground)   export ECO_BAL_GROUND=0.1  ECO_W_COST=0.2  ECO_WRONG_GROUND_SCALE=0.0  ;;  # EM almost solo, tiny ground
  F9_midhigh)     export ECO_BAL_GROUND=0.4  ECO_W_COST=0.25 ECO_WRONG_GROUND_SCALE=0.0  ;;  # between F2 and F5
  *) echo "unknown RECIPE=$RECIPE"; exit 1 ;;
esac
echo "[ts_recipe] $(hostname) recipe=$RECIPE bal_ground=$ECO_BAL_GROUND w_cost=$ECO_W_COST "\
     "wrong_ground_scale=$ECO_WRONG_GROUND_SCALE probe=$PROBE_URL(FROZEN) steps=$TOTAL_TRAINING_STEPS model=$MODEL_PATH"

# wait for retriever
for i in $(seq 1 60); do
  curl -s -o /dev/null -w "%{http_code}" "$RETRIEVER_URL" -X POST \
    -H "Content-Type: application/json" -d '{"queries":["test"],"topk":1}' \
    | grep -qE "200|422" && break
  sleep 10
done
# wait for the frozen probe server
for i in $(seq 1 60); do
  curl -s -o /dev/null -w "%{http_code}" "${PROBE_URL%/judge_probe}/judge_probe" -X POST \
    -H "Content-Type: application/json" -d '{"items":[]}' \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.grad_offload=${GRAD_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIM_OFFLOAD} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${PARAM_OFFLOAD} \
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
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.project_name=TrustSearch \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.total_epochs=15 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.default_local_dir=verl_checkpoints/${EXPERIMENT_NAME} \
    max_turns=4 \
    retriever.url="${RETRIEVER_URL}" \
    retriever.topk=3
echo "[ts_recipe] $RECIPE done"
