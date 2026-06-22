#!/bin/bash
# Submit the probe<->policy CO-EVOLUTION arm (ts_coevo_full):
#   1) hot-reloadable probe server (port 8002, manifest watched by the controller)
#   2) GRPO training (VARIANT=full == bal_ground0.5/w_cost0.2/gate1, same fixed reward
#      landscape as the frozen rcp3 recipes -> the ONLY difference is the probe co-evolves)
#   3) co-evolution controller: every save_freq steps it refits the probe from the
#      CURRENT policy checkpoint and hot-swaps it in (logs cb_rate = boundary migration).
# round0 == base250b probes (identical start to the frozen recipes), trained to step250
# so it lines up with rcp3_F2/F5/F3 and crosses the baseline step150 death point.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1

EXP=ts_coevo_full
MANIFEST=probes/live_${EXP}/manifest.json
PORT=8002
TRAIN_NODE=node6        # RTX4090 x8 idle: 4 for training + 1 for probe server
PROBE_NODE=node6
CTRL_NODE=node3         # RTX3090 idle: controller does vLLM dumps (inference only)
STEPS=250

echo "[coevo] 1/3 launch hot-reload probe server on $PROBE_NODE:$PORT (RESET_ROUND0=1)"
PSRV=$(sbatch --parsable -p RTX4090 -w $PROBE_NODE \
  --export=ALL,MANIFEST=$MANIFEST,PORT=$PORT,RESET_ROUND0=1 \
  start_probe_server.sbatch)
echo "  probe server job: $PSRV"

echo "[coevo] wait for probe server RUNNING..."
for i in $(seq 1 60); do
  ST=$(squeue -j $PSRV --noheader --format="%T" 2>/dev/null | tr -d ' ')
  [ "$ST" = "RUNNING" ] && break
  echo "  ... $ST"; sleep 5
done
PROBE_IP=$(getent hosts $PROBE_NODE | awk '{print $1}')
PROBE_URL="http://${PROBE_IP}:${PORT}/judge_probe"
echo "  PROBE_URL=$PROBE_URL"
echo "[coevo] wait 60s for probe model init..."; sleep 60

echo "[coevo] 2/3 launch GRPO training (exp=$EXP, steps=$STEPS) on $TRAIN_NODE"
TRAIN=$(sbatch --parsable -p RTX4090 -w $TRAIN_NODE \
  --export=ALL,VARIANT=full,EXPERIMENT_NAME=$EXP,TOTAL_TRAINING_STEPS=$STEPS,PROBE_URL=$PROBE_URL \
  ts_grpo.sbatch)
echo "  training job: $TRAIN"

echo "[coevo] 3/3 launch co-evolution controller on $CTRL_NODE (watches $EXP)"
CTRL=$(sbatch --parsable -p RTX3090 -w $CTRL_NODE \
  --export=ALL,EXPS=coevo_full,NQ=300 \
  ts_coevolve.sbatch)
echo "  controller job: $CTRL"

echo ""
echo "=== ts_coevo_full submitted ==="
echo "  probe server : $PSRV ($PROBE_NODE:$PORT, manifest=$MANIFEST)"
echo "  training     : $TRAIN ($TRAIN_NODE, exp=$EXP, $STEPS steps)"
echo "  controller   : $CTRL ($CTRL_NODE, --exps coevo_full)"
echo "  PROBE_URL    : $PROBE_URL"
echo "Monitor: tail -f logs/ts_grpo_${TRAIN}.out ; tail -f logs/coevolve_${CTRL}.out"
echo "Boundary migration: tail -f logs/boundary_migration.csv  (rows tagged $EXP)"
