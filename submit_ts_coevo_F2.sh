#!/bin/bash
# CO-EVOLUTION of the best recipe F2_balanced. Identical to the FROZEN rcp4_F2_balanced
# run (ts_recipe RECIPE=F2_balanced: bal_ground=0.3, w_cost=0.2, gate=1, max_turns=4)
# EXCEPT the probe co-evolves: the controller refits it from the CURRENT policy
# checkpoint every save_freq=25 steps (justified by the drift curve: the frozen probe
# degrades within the first 25 steps). Runs on 48GB ADA6000 to avoid the 24GB OOM.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1

EXP=ts_coevo_F2
MANIFEST=probes/live_${EXP}/manifest.json
PORT=8006
PROBE_NODE=node8        # ADA6000 x8: 1 for probe server + 4 for training
TRAIN_NODE=node8
CTRL_NODE=node3         # RTX3090 for the controller (inference-only dumps)
STEPS=250

echo "[coevo-F2] 1/3 hot-reload probe server on $PROBE_NODE:$PORT (RESET_ROUND0=1)"
PSRV=$(sbatch --parsable -p ADA6000 -w $PROBE_NODE \
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

echo "[coevo-F2] 2/3 GRPO training (recipe F2_balanced, CO-EVOLVING probe) on $TRAIN_NODE"
TRAIN=$(sbatch --parsable -p ADA6000 -w $TRAIN_NODE \
  --export=ALL,RECIPE=F2_balanced,EXPERIMENT_NAME=$EXP,PROBE_URL=$PROBE_URL,TOTAL_TRAINING_STEPS=$STEPS,SAVE_FREQ=25,TEST_FREQ=25 \
  ts_recipe.sbatch)
echo "  training job: $TRAIN"

echo "[coevo-F2] 3/3 co-evolution controller on $CTRL_NODE (refit every 25 steps)"
CTRL=$(sbatch --parsable -p RTX3090 -w $CTRL_NODE \
  --export=ALL,EXPS=coevo_F2,NQ=300 \
  ts_coevolve.sbatch)
echo "  controller job: $CTRL"
echo "=== ts_coevo_F2 submitted: probe=$PSRV train=$TRAIN ctrl=$CTRL (refit/25, 48GB) ==="
