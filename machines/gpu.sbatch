#!/bin/bash
# Run a repository command on a quarter Snellius H100 node (1 GPU, 16 cores), from the repository root:
#   sbatch machines/gpu.sbatch train.py lost_sales/zero_shot_large gc_lsn
# Uses the same .venv as machines/genoa.sbatch; its CUDA build of PyTorch also runs on CPU-only nodes.
#SBATCH --job-name=dynaplex
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

module load 2025 Python/3.13.5-GCCcore-14.3.0
export OMP_NUM_THREADS=1
srun --cpu-bind=cores .venv/bin/python -u "$@"
