#!/usr/bin/env bash
#SBATCH --job-name=selection-train
#SBATCH --partition=bme_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/overnight_environment.sh
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
echo "Command: python -B scripts/selection_bias.py run --config $SELECTION_CONFIG"
python -B scripts/selection_bias.py run --config "$SELECTION_CONFIG" &
manager=$!
trap 'kill -TERM "$manager" 2>/dev/null || true' USR1 TERM INT
set +e
wait "$manager"
result=$?
if kill -0 "$manager" 2>/dev/null; then wait "$manager"; result=$?; fi
exit "$result"
