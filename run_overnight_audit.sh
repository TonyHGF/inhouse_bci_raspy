#!/usr/bin/env bash
set -euo pipefail
mode="local"
mode="${1:-$mode}"
cd "$(dirname "${BASH_SOURCE[0]}")"
case "$mode" in
  local) echo 'Metadata audit of completed overnight-v1 results. Submit with: bash run_overnight_audit.sh server';;
  server) mkdir -p logs; sbatch slurm/overnight_audit.slurm;;
  *) echo 'Usage: bash run_overnight_audit.sh local|server' >&2; exit 2;;
esac
