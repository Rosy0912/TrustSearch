#!/bin/bash
#SBATCH --job-name=tstrain
#SBATCH --output=logs/tstrain_%j.out
#SBATCH --error=logs/tstrain_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -e
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh
conda activate searchr1
cd /home/jmzhang/self-evolution/Search-R1
python ts_probe_train.py --dumps "${DUMPS:-dumps/base250b}" --layers 21,24,27 \
    --save "${SAVE:-probes/base250b}"
echo "[tstrain] done"
