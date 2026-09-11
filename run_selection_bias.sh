#!/usr/bin/env bash
set -euo pipefail
mode="local"
mode="${1:-$mode}"
export SELECTION_CONFIG="${2:-config/selection_bias.json}"
export CONDA_ENV="${CONDA_ENV:-mirepnet}"
cd "$(dirname "${BASH_SOURCE[0]}")"
set +u
source ~/anaconda3/bin/activate "$CONDA_ENV"
set -u
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -B scripts/selection_bias.py check --config "$SELECTION_CONFIG"
[[ "$mode" != local ]] || exit 0
[[ "$mode" == server ]] || { echo 'Usage: bash run_selection_bias.sh local|server [config.json]' >&2; exit 2; }
mkdir -p logs
export SELECTION_STAGE=prepare
prep=$(sbatch --parsable --export=ALL --job-name=selection-prepare slurm/selection_cpu.slurm)
prep="${prep%%;*}"
cores=$(python -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c["parallel_per_gpu"]*c["threads_per_process"])' "$SELECTION_CONFIG")
export SELECTION_STAGE=run
job=$(sbatch --parsable --export=ALL --cpus-per-task="$cores" --dependency="afterok:$prep" --kill-on-invalid-dep=yes slurm/selection_gpu.slurm)
job="${job%%;*}"
export SELECTION_STAGE=report
report=$(sbatch --parsable --export=ALL --dependency="afterany:$job" --job-name=selection-report slurm/selection_cpu.slurm)
echo "Prepare: $prep bme_cpu slurm/selection_cpu.slurm logs/selection-prepare_${prep}.{out,err}"
echo "Train: $job bme_gpu slurm/selection_gpu.slurm logs/selection-train_${job}.{out,err}"
echo "Report: ${report%%;*} bme_cpu slurm/selection_cpu.slurm logs/selection-report_${report%%;*}.{out,err}"
