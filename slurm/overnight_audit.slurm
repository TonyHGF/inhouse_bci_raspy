#!/usr/bin/env bash
#SBATCH --job-name=night-audit
#SBATCH --partition=bme_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/overnight_environment.sh
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
echo 'Command: python -B scripts/summarize_overnight.py'
python -B scripts/summarize_overnight.py
