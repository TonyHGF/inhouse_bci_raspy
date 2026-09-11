#!/usr/bin/env bash
# Open a shell in an existing allocation; does not request another GPU/job.
set -euo pipefail
job="${1:?Usage: bash attach_overnight.sh JOB_ID}"
[[ "$job" =~ ^[0-9]+$ ]] || { echo 'JOB_ID must be numeric' >&2; exit 2; }
# Cluster Slurm 19.05 supports --jobid; newer --overlap is unavailable here.
exec srun --jobid="$job" --nodes=1 --ntasks=1 --cpus-per-task=1 --immediate=5 --pty bash
