#!/usr/bin/env bash
set -euo pipefail
mode="local" # local: configuration checks only; server: submit Slurm
mode="${1:-$mode}"
stage="${2:-check}"
profile="${3:-src/bci_raspy_experiments/profiles/server-bme.json}"
if (( $# >= 3 )); then shift 3; else set --; fi
cd "$(dirname "${BASH_SOURCE[0]}")"
export BCI_PROFILE="$profile"
export BCI_STAGE="$stage"
export CONDA_ENV="${CONDA_ENV:-mirepnet}"
export BCI_SHARDS="${SHARDS:-1}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$stage" == pipeline ]]; then
    [[ "$mode" == server ]] || { echo 'pipeline requires server mode' >&2; exit 2; }
    # One deduplicated suite across all shards. Never run overlapping suites concurrently.
    pipeline_shards="${SHARDS:-64}"
    queue_stage() {
        local receipt
        receipt=$(DEPENDENCY="$2" SHARDS="$3" bash run_experiments.sh server "$1" "$profile" "${@:4}")
        echo "$receipt"
        [[ "$receipt" =~ Job=([0-9]+) ]] || return 1
        queued_id="${BASH_REMATCH[1]}"
    }
    queue_stage verify "${DEPENDENCY:-}" 1
    verify_id="$queued_id"
    queue_stage prepare "$verify_id" 1
    queue_stage smoke "$queued_id" 1 --resume
    queue_stage run "$queued_id" "$pipeline_shards" "$@" --resume
    train_id="$queued_id"
    queue_stage evaluate "$train_id" 1 "$@" --resume
    evaluate_id="$queued_id"
    queue_stage explain "$train_id" "$pipeline_shards" "$@" --resume
    queue_stage report "$evaluate_id:$queued_id" 1 "$@"
    exit 0
fi
case "$stage" in check|verify|prepare|smoke|run|evaluate|explain|report) ;; *) echo "Unknown stage: $stage" >&2; exit 2;; esac
case "$mode" in
  local)
    [[ "$stage" == check ]] || { echo 'Local mode only supports check; use server for computation.' >&2; exit 2; }
    set +u # Existing conda compiler activation hooks read unset variables.
    source ~/anaconda3/bin/activate "$CONDA_ENV"
    set -u
    python -B scripts/server_experiments.py check --profile-file "$profile" "$@"
    ;;
  server)
    [[ "$BCI_SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo 'SHARDS must be positive' >&2; exit 2; }
    mkdir -p logs
    options=(--parsable --export=ALL)
    [[ -z "${DEPENDENCY:-}" ]] || options+=("--dependency=afterok:$DEPENDENCY" --kill-on-invalid-dep=yes)
    [[ -z "${TIME_LIMIT:-}" ]] || options+=("--time=$TIME_LIMIT")
    [[ -z "${MEMORY:-}" ]] || options+=("--mem=$MEMORY")
    case "$stage" in
      smoke|run|explain) job_script=slurm/experiments_gpu.slurm; partition=bme_gpu;;
      *) job_script=slurm/experiments_cpu.slurm; partition=bme_cpu;;
    esac
    if [[ "$stage" == run || "$stage" == explain ]]; then
      options+=("--array=0-$((BCI_SHARDS - 1))%${MAX_PARALLEL:-1}")
    elif [[ "$BCI_SHARDS" != 1 ]]; then
      echo 'SHARDS applies only to run/explain' >&2; exit 2
    fi
    job=$(sbatch "${options[@]}" --job-name="bci-$stage" "$job_script" "$@")
    job="${job%%;*}"
    echo "Job=$job name=bci-$stage partition=$partition script=$job_script"
    echo "Logs: $PWD/logs/bci-${stage}_${job}.out and .err (array tasks have separate job IDs)"
    echo "Profile: $profile"
    ;;
  *) echo 'Usage: bash run_experiments.sh local|server STAGE [PROFILE] [--dataset ... --suite ... --resume ...]' >&2; exit 2;;
esac
