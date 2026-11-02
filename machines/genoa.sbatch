#!/bin/bash
# Run a repository command on one Snellius Genoa node (192 cores), from the repository root:
#   sbatch machines/genoa.sbatch train.py lost_sales/zero_shot_large gc_lsn
# One-time setup on the login node, also from the repository root:
#   module load 2025 Python/3.13.5-GCCcore-14.3.0
#   python -m venv .venv && .venv/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
#   .venv/bin/pip install -r requirements.txt
#SBATCH --job-name=dynaplex
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

module load 2025 Python/3.13.5-GCCcore-14.3.0
export OMP_NUM_THREADS=1
srun --cpu-bind=cores .venv/bin/python -u "$@"
