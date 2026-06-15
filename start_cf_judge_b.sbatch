#!/bin/bash
#SBATCH --job-name=cf_judge_b
#SBATCH --partition=RTX4090
#SBATCH --nodelist=node6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/cf_judge_b_%j.out
#SBATCH --error=logs/cf_judge_b_%j.err
# Extended judge (node6, 192.168.102.16:8001): /judge_batch (cf), /judge_faithfulness (nli),
# /judge_removal (leave-one-out). Used by the nli (and removal) trust-signal ablations.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1
echo "[cf_judge_b] starting on $(hostname) at $(date)"
python cf_judge_server.py --model /data1/public/hf/Qwen/Qwen2.5-3B-Instruct --port 8001
