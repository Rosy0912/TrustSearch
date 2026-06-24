#!/bin/bash
# CO-EVOLUTION with the NEW PRIORITY recipe (6/23): EM-first + STRONGER probe + RELAXED cost.
#   recipe F14_emfirst: bal_ground=0.6, w_cost=0.02, wrong_ground_scale=0.0, gate=1, max_turns=4
# The probe CO-EVOLVES: controller refits it from the CURRENT policy checkpoint every
# save_freq=25 steps and hot-swaps it in.
#
# Resource note (6/23): node8/9 (ADA6000) taken by sysun, node10/11 (L40S) memory-starved
# by ylliu. node6/7 (RTX4090 x8) are the only STABLY-FREE GPUs. 24GB OOMs at 4-GPU sharding,
# but 8-GPU FSDP halves per-card memory -> with micro_bs + full offload it fits.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1

RECIPE="${RECIPE:-F14_emfirst}"
EXP="ts_coevo_emfirst"
MANIFEST=probes/live_${EXP}/manifest.json
PORT=8007
PROBE_NODE=node7          # RTX4090: 1 card for probe server
TRAIN_NODE=node6          # RTX4090 x8: 8-GPU FSDP sharding for the 3B training
CTRL_NODE=node3           # RTX3090 for the controller (inference-only dumps)
STEPS=250

echo "[coevo-emf] 1/3 hot-reload probe server on $PROBE_NODE:$PORT (RESET_ROUND0=1)"
PSRV=$(sbatch --parsable -p RTX4090 -w $PROBE_NODE --gres=gpu:1 --mem=80G \
  --export=ALL,MANIFEST=$MANIFEST,PORT=$PORT,RESET_ROUND0=1 \
  start_probe_server.sbatch)
echo "  probe server job: $PSRV"
for i in $(seq 1 60); do
  ST=$(squeue -j $PSRV --noheader --format="%T" 2>/dev/null | tr -d ' ')
  [ "$ST" = "RUNNING" ] && break; echo "  ... $ST"; sleep 5
done
PROBE_IP=$(getent hosts $PROBE_NODE | awk '{print $1}')
PROBE_URL="http://${PROBE_IP}:${PORT}/judge_probe"
echo "  PROBE_URL=$PROBE_URL"; echo "  wait 60s for probe init"; sleep 60

echo "[coevo-emf] 2/3 GRPO training (recipe $RECIPE, CO-EVOLVING probe, 8-GPU) on $TRAIN_NODE"
TRAIN=$(sbatch --parsable -p RTX4090 -w $TRAIN_NODE --gres=gpu:8 --cpus-per-task=64 --mem=400G \
  --export=ALL,RECIPE=$RECIPE,EXPERIMENT_NAME=$EXP,PROBE_URL=$PROBE_URL,CF_JUDGE_URL=http://192.168.102.11:8001/judge_batch,TOTAL_TRAINING_STEPS=$STEPS,SAVE_FREQ=25,TEST_FREQ=25,N_GPU=8,PPO_MICRO_BS=8,LOGPROB_MICRO_BS=8,GPU_MEM_UTIL=0.4,PARAM_OFFLOAD=true,GRAD_OFFLOAD=true,OPTIM_OFFLOAD=true,MAX_TURNS=4 \
  ts_recipe.sbatch)
echo "  training job: $TRAIN"

echo "[coevo-emf] 3/3 co-evolution controller on $CTRL_NODE (refit every 25 steps)"
CTRL=$(sbatch --parsable -p RTX3090 -w $CTRL_NODE \
  --export=ALL,EXPS=coevo_emfirst,NQ=300 \
  ts_coevolve.sbatch)
echo "  controller job: $CTRL"
echo "=== ts_coevo_emfirst submitted: probe=$PSRV train=$TRAIN ctrl=$CTRL (recipe=$RECIPE, refit/25, 8xRTX4090) ==="
echo "Monitor: tail -f logs/ts_recipe_${TRAIN}.out ; tail -f logs/coevolve_${CTRL}.out"
