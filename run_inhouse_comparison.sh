#!/usr/bin/env bash
set -euo pipefail
mode="local"
mode="${1:-$mode}"
export COMPARISON_CONFIG="${2:-config/inhouse_comparison.json}"
export CONDA_ENV="${CONDA_ENV:-mirepnet}"
cd "$(dirname "${BASH_SOURCE[0]}")"
set +u
source ~/anaconda3/bin/activate "$CONDA_ENV"
set -u
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -B scripts/inhouse_comparison.py check --config "$COMPARISON_CONFIG"
[[ "$mode" != local ]] || exit 0
[[ "$mode" == server ]] || { echo 'Usage: bash run_inhouse_comparison.sh local|server [config.json]' >&2; exit 2; }
mkdir -p logs
export COMPARISON_STAGE=prepare
prep=$(sbatch --parsable --export=ALL --job-name=comparison-prepare slurm/comparison_cpu.slurm)
prep="${prep%%;*}"
cores=$(python -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c["parallel_per_gpu"]*c["threads_per_process"])' "$COMPARISON_CONFIG")
export COMPARISON_STAGE=smoke
smoke=$(sbatch --parsable --export=ALL --cpus-per-task="$cores" --job-name=comparison-smoke --dependency="afterok:$prep" --kill-on-invalid-dep=yes slurm/comparison_gpu.slurm)
smoke="${smoke%%;*}"
export COMPARISON_STAGE=run
job=$(sbatch --parsable --export=ALL --cpus-per-task="$cores" --dependency="afterok:$smoke" --kill-on-invalid-dep=yes slurm/comparison_gpu.slurm)
job="${job%%;*}"
export COMPARISON_STAGE=report
report=$(sbatch --parsable --export=ALL --dependency="afterany:$job" --job-name=comparison-report slurm/comparison_cpu.slurm)
echo "Prepare: $prep bme_cpu slurm/comparison_cpu.slurm logs/comparison-prepare_${prep}.{out,err}"
echo "Train: $job bme_gpu slurm/comparison_gpu.slurm logs/comparison-train_${job}.{out,err}"
echo "Report: ${report%%;*} bme_cpu slurm/comparison_cpu.slurm logs/comparison-report_${report%%;*}.{out,err}"

echo "Smoke: $smoke bme_gpu slurm/comparison_gpu.slurm logs/comparison-smoke_${smoke}.{out,err}"
