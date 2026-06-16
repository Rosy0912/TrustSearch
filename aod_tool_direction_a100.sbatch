#!/bin/bash
#SBATCH --job-name=aodtool
#SBATCH --output=logs/aodtool_%j.out
#SBATCH --error=logs/aodtool_%j.out
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=3:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate searchr1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy no_proxy NO_PROXY
export no_proxy="*"
export NO_PROXY="*"
export PYTHONUNBUFFERED=1
cd /home/jmzhang/self-evolution/Search-R1

python aod_tool_direction.py \
  --model /data1/public/hf/Qwen/Qwen2.5-3B-Instruct \
  --data data/nq_search/test.parquet \
  --behav behav_cf_300_results.json \
  --layers="-1,-3,-5,-9,-13,-17,-21,-25" \
  --label_mode broad \
  --out aod_tool_direction_broad_results.json
