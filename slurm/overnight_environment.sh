mkdir -p logs
export CONDA_ENV="${CONDA_ENV:-mirepnet}"
set +u
source ~/anaconda3/bin/activate "${CONDA_ENV:-mirepnet}"
set -u
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Manager and report imports stay bounded; workers set their own configured limits.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export MPLCONFIGDIR="/public/home/hugf2022/inhouse_bci_raspy/matplotlib/$SLURM_JOB_ID"
mkdir -p "$MPLCONFIGDIR"
echo "Start: $(date -Is) job=$SLURM_JOB_ID name=$SLURM_JOB_NAME node=$SLURM_JOB_NODELIST env=$CONDA_ENV python=$(command -v python)"
trap 'code=$?; echo "End: $(date -Is) exit=$code"' EXIT
