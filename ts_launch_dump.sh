#!/bin/bash
# Launch sharded TrustSearch probe dump across many GPUs.
#   bash ts_launch_dump.sh <OUT_DIR> <MODEL> [N_PER_DS] [SHARD] [G]
# Shards NQ + HotpotQA train into single-GPU workers, round-robins retrievers.
set -e
cd /home/jmzhang/self-evolution/Search-R1
mkdir -p logs

OUT="${1:-dumps/base250}"
MODEL="${2:-verl_checkpoints/sr1-baseline-em/actor/global_step_250}"
N_PER_DS="${3:-1500}"   # questions per dataset
SHARD="${4:-100}"       # questions per worker
G="${5:-5}"
LAYERS="21,24,27"

# round-robin retrievers (host:port) that are currently up
RETRIEVERS=(
  "http://192.168.102.15:8000/retrieve"
  "http://192.168.102.14:8000/retrieve"
  "http://192.168.102.17:8000/retrieve"
  "http://192.168.102.21:8000/retrieve"
)
# partitions to spread across (slurm picks a free node within each)
PARTS=(RTX4090 RTX3090 ADA6000 A100)

declare -A DATA
DATA[nq]="data/nq_search/train.parquet"
DATA[hotpotqa]="data/hotpotqa_search/train.parquet"

i=0
for tag in nq hotpotqa; do
  data="${DATA[$tag]}"
  off=0
  while [ "$off" -lt "$N_PER_DS" ]; do
    ret="${RETRIEVERS[$(( i % ${#RETRIEVERS[@]} ))]}"
    part="${PARTS[$(( i % ${#PARTS[@]} ))]}"
    # NOTE: do NOT put LAYERS (which contains commas) in --export; the comma collides
    # with sbatch's variable separator and truncates it. The worker sets LAYERS itself.
    sbatch --partition="$part" \
      --export=ALL,TAG=$tag,DATA=$data,OFFSET=$off,CAP=$SHARD,G=$G,OUT=$OUT,MODEL=$MODEL,RETRIEVER_URL=$ret \
      ts_dump_worker.sbatch >/dev/null
    echo "submitted: tag=$tag offset=$off part=$part ret=${ret##*//}"
    off=$(( off + SHARD ))
    i=$(( i + 1 ))
  done
done
echo "total shards submitted: $i  (out=$OUT)"
