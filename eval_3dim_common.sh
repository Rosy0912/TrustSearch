#!/bin/bash
# Offline 3-dimension (performance/trust/cost) evaluation via val_only.
# Required env: MODEL_PATH, EXPERIMENT_NAME, N_GPU, GPU_MEM_UTIL, RETRIEVER_URL, CF_JUDGE_URL
# Uses the SAME validation set/config as the TrustSearch run (val_batch_size=64,
# val_data_num=100, seed=42) so results are directly comparable.

RETRIEVER_URL="${RETRIEVER_URL:-http://192.168.102.15:8000/retrieve}"
for i in $(seq 1 90); do
    curl -s -o /dev/null -w "%{http_code}" "$RETRIEVER_URL" -X POST \
        -H "Content-Type: application/json" -d '{"queries":["test"],"topk":1}' \
        | grep -qE "200|422" && break
    sleep 10
done

python3 -m verl.trainer.main_ppo_eco \
    data.train_files=data/nq_search/train.parquet \
    data.val_files=data/nq_search/test.parquet \
    data.train_data_num=null \
    data.val_data_num=${VAL_N:-100} \
    data.train_batch_size=128 \
    data.val_batch_size=${VAL_BS:-64} \
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
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=25 \
    trainer.project_name=Search-R1-HALO \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.total_epochs=15 \
    trainer.total_training_steps=500 \
    trainer.default_local_dir=verl_checkpoints/${EXPERIMENT_NAME} \
    max_turns=2 \
    retriever.url="${RETRIEVER_URL}" \
    retriever.topk=3 \
    2>&1 | tee logs/${EXPERIMENT_NAME}.log
