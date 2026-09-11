#!/usr/bin/env bash
#SBATCH --partition=bme_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/overnight_environment.sh
export CUDA_VISIBLE_DEVICES=""
echo "Command: python -B scripts/selection_bias.py $SELECTION_STAGE --config $SELECTION_CONFIG"
python -B scripts/selection_bias.py "$SELECTION_STAGE" --config "$SELECTION_CONFIG"
