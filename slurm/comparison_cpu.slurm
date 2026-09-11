#!/usr/bin/env bash
#SBATCH --partition=bme_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ "$COMPARISON_STAGE" == prepare ]]; then export CONDA_ENV=eeg-preprocessing; fi
source slurm/overnight_environment.sh
export CUDA_VISIBLE_DEVICES=""
echo "Command: python -B scripts/inhouse_comparison.py $COMPARISON_STAGE --config $COMPARISON_CONFIG"
python -B scripts/inhouse_comparison.py "$COMPARISON_STAGE" --config "$COMPARISON_CONFIG"
