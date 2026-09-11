#!/usr/bin/env bash
set -euo pipefail
mode="local" # local: freeze/show plan only; server: two GPU jobs and CPU reports
mode="${1:-$mode}"
config="${2:-config/overnight.json}"
cd "$(dirname "${BASH_SOURCE[0]}")"
export NIGHT_CONFIG="$config" CONDA_ENV="${CONDA_ENV:-mirepnet}"
set +u
source ~/anaconda3/bin/activate "$CONDA_ENV"
set -u
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -B scripts/overnight.py plan --config "$config"
[[ "$mode" != local ]] || exit 0
[[ "$mode" == server ]] || { echo 'Usage: bash run_overnight.sh local|server [config.json]' >&2; exit 2; }
mkdir -p logs
# Experiment semantics/concurrency are stored in config; request matching CPU resources.
cores=$(python -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c["parallel_per_gpu"]*c["threads_per_process"])' "$config")
for dataset in inhouse bci2a; do
    export NIGHT_DATASET="$dataset"
    options=()
    [[ -z "${DEPENDENCY:-}" ]] || options+=("--dependency=afterok:$DEPENDENCY" --kill-on-invalid-dep=yes)
    job=$(sbatch --parsable --export=ALL --cpus-per-task="$cores" \
        --job-name="night-$dataset" --time="${TIME_LIMIT:-12:00:00}" "${options[@]}" slurm/overnight_gpu.slurm)
    job="${job%%;*}"
    echo "GPU job=$job dataset=$dataset partition=bme_gpu script=slurm/overnight_gpu.slurm logs=logs/night-${dataset}_${job}.{out,err}"
    # Report successful tasks even if some failed or the GPU allocation timed out.
    report=$(sbatch --parsable --export=ALL --dependency="afterany:$job" \
        --job-name="night-report-$dataset" slurm/overnight_report.slurm)
    echo "CPU report job=${report%%;*} partition=bme_cpu script=slurm/overnight_report.slurm logs=logs/night-report-${dataset}_${report%%;*}.{out,err}"
done
