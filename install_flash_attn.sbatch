#!/bin/bash
#SBATCH --job-name=install_fa
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node11
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/install_fa_%j.out
#SBATCH --error=logs/install_fa_%j.err
set -euo pipefail
source /home/jmzhang/miniconda3/etc/profile.d/conda.sh
conda activate searchr1
export CUDA_HOME=/usr/local/cuda
# If system CUDA not there, use torch bundled
if [ ! -d "$CUDA_HOME" ]; then
    export CUDA_HOME=$(python -c "import torch.utils; print(torch.utils.cmake_prefix_path.replace('/share/cmake',''))" 2>/dev/null || echo "/usr/local/cuda")
fi
echo "CUDA_HOME=$CUDA_HOME"
nvidia-smi | head -5
pip install flash-attn --no-build-isolation 2>&1 | tail -20
echo "DONE"
