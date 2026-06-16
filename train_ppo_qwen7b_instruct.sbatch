#!/bin/bash
#SBATCH --job-name=sr1_ppo_7b_it
#SBATCH --partition=L40S
#SBATCH --gres=gpu:7
#SBATCH --nodelist=node11
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/sr1_grpo_%j.out
#SBATCH --error=logs/sr1_grpo_%j.err
# Search-R1 GRPO training: Qwen2.5-7B-Instruct on NQ+HotpotQA
# Requires: retriever server running on same node (started inside this script)
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh
conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export VLLM_ATTENTION_BACKEND=XFORMERS

DATA_DIR=data/retriever_data
BASE_MODEL=/data1/public/hf/Qwen/Qwen2.5-7B-Instruct
EXPERIMENT_NAME=sr1-ppo-qwen2.5-7b-it-nq
WAND_PROJECT=Search-R1-HALO

# ---- Start retriever server in background ----
echo "[boot] Starting retriever server..."
python search_r1/search/retrieval_server.py \
    --index_path "$DATA_DIR/e5_Flat.index" \
    --corpus_path "$DATA_DIR/wiki-18.jsonl" \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model /home/jmzhang/models/e5-base-v2 \
    --port 8000 &
RETRIEVER_PID=$!
echo "[boot] Retriever PID=$RETRIEVER_PID, waiting for startup..."
sleep 30  # give retriever time to load index

# Verify retriever is up
for i in $(seq 1 10); do
    if curl -s http://127.0.0.1:8000/retrieve -X POST \
        -H "Content-Type: application/json" \
        -d '{"query": "test", "topk": 1}' | grep -q "contents"; then
        echo "[boot] Retriever ready!"
        break
    fi
    echo "[boot] Waiting for retriever... ($i/10)"
    sleep 10
done

# ---- Train ----
echo "[train] Starting GRPO training..."
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=data/nq_search/train.parquet \
    data.val_files=data/nq_search/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=256 \
    data.val_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
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
    trainer.n_gpus_per_node=7 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=500 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee logs/${EXPERIMENT_NAME}.log

echo "[done] Training finished. Killing retriever..."
kill $RETRIEVER_PID 2>/dev/null || true
echo "[done] $(date)"
