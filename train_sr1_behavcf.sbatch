#!/bin/bash
#SBATCH --job-name=sr1_bcf
#SBATCH --partition=L40S
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/sr1_bcf_%j.out
#SBATCH --error=logs/sr1_bcf_%j.err
# Search-R1 GRPO + behavcf grounded-gated reward
# Identical to sr1_k1 (train_grpo_3b.sbatch) EXCEPT: uses main_ppo_behavcf.py
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh
conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS
export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa

BASE_MODEL=/data1/public/hf/Qwen/Qwen2.5-3B-Instruct
EXPERIMENT_NAME=sr1-grpo-behavcf-3b-it-nq
WAND_PROJECT=Search-R1-HALO

# ---- Wait for retriever service ----
echo "[boot] Waiting for retriever service on 192.168.102.21:8000..."
for i in $(seq 1 30); do
    if curl -s -o /dev/null -w "%{http_code}" http://192.168.102.21:8000/retrieve -X POST -H "Content-Type: application/json" -d '{"queries":["test"],"topk":1}' | grep -qE "200|422"; then
        echo "[boot] Retriever ready!"
        break
    fi
    echo "[boot] Waiting... ($i/30)"
    sleep 10
done

# ---- Train: ONLY DIFFERENCE is main_ppo_behavcf (grounded-gated reward) ----
echo "[train] Starting GRPO + behavcf grounded-gated reward training..."
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_behavcf \
    data.train_files=data/nq_search/train.parquet \
    data.val_files=data/nq_search/test.parquet \
    data.train_data_num=null \
    data.val_data_num=100 \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
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
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=500 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=2 \
    retriever.url="http://192.168.102.21:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee logs/${EXPERIMENT_NAME}.log

echo "[done] sr1-behavcf finished $(date)"
