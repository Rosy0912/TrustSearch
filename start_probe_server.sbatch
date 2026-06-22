#!/bin/bash
#SBATCH --job-name=probe_srv
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Hot-reloadable TrustSearch probe server (co-evolution). Set via --export:
#   MANIFEST (probes/live_ts_<exp>/manifest.json)  PORT
#   RESET_ROUND0=1  -> (re)initialize the manifest to the round-0 base probes so the
#                      experiment starts from a clean, identical r0 (encoder=step250,
#                      probes=base250b). The co-evolution controller then bumps rounds.
set -uo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="192.168.0.0/16,127.0.0.1,localhost"
export PYTHONUNBUFFERED=1

: "${MANIFEST:?set MANIFEST}"
: "${PORT:=8002}"
INIT_ENCODER="${INIT_ENCODER:-/data1/public/hf/Qwen/Qwen2.5-3B-Instruct}"
INIT_COMMIT="${INIT_COMMIT:-$(pwd)/probes/base250b/commit_probe.pkl}"
INIT_BOUNDARY="${INIT_BOUNDARY:-$(pwd)/probes/base250b/boundary_probe.pkl}"

mkdir -p "$(dirname "$MANIFEST")"
if [ "${RESET_ROUND0:-0}" = "1" ]; then
  echo "[probe_srv] RESET manifest $MANIFEST -> round0 (encoder=$INIT_ENCODER, probes=base250b)"
  cat > "$MANIFEST" <<EOF
{"round": 0, "encoder": "$INIT_ENCODER", "commit": "$INIT_COMMIT", "boundary": "$INIT_BOUNDARY"}
EOF
fi

echo "[probe_srv] $(hostname) manifest=$MANIFEST port=$PORT at $(date)"

# Auto-restart loop: if the server process dies for any reason, relaunch it.
# The watcher thread re-reads the manifest, so it resumes at the latest co-evolved
# round automatically. SLURM keeps the job (and GPU) for the full 4-day window.
while true; do
  python probe_server.py --manifest "$MANIFEST" --port "$PORT"
  rc=$?
  echo "[probe_srv] server exited rc=$rc at $(date); restarting in 10s" >&2
  sleep 10
done
