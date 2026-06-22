#!/bin/bash
#SBATCH --job-name=cf_judge
#SBATCH --partition=ADA6000
#SBATCH --nodelist=node9
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/cf_judge_%j.out
#SBATCH --error=logs/cf_judge_%j.err
# Counterfactual judge service (node9, 192.168.102.19:8001) for EcoSearch online behavcf.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh && conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1
echo "[cf_judge] starting on $(hostname) at $(date)"
python cf_judge_server.py --model /data1/public/hf/Qwen/Qwen2.5-3B-Instruct --port 8001
