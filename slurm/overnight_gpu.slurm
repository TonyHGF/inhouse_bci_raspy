#!/usr/bin/env bash
#SBATCH --partition=bme_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/overnight_environment.sh
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
command=(python -B scripts/overnight.py run --config "$NIGHT_CONFIG" --dataset "$NIGHT_DATASET")
command+=("$@")
printf 'Command: '; printf '%q ' "${command[@]}"; printf '\n'
# Forward Slurm's pre-timeout signal so workers release locks at interruption.
"${command[@]}" &
manager=$!
trap 'kill -USR1 "$manager" 2>/dev/null || true' USR1
trap 'kill -TERM "$manager" 2>/dev/null || true' TERM INT
set +e
wait "$manager"
result=$?
if kill -0 "$manager" 2>/dev/null; then wait "$manager"; result=$?; fi
exit "$result"
