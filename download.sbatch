#!/bin/bash
#SBATCH --job-name=sr1_download
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
# Download all Search-R1 dependencies (models + data) on CPU node.
set -euo pipefail
cd /home/jmzhang/self-evolution/Search-R1
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh
conda activate agent_evo
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p logs
bash download_all.sh
