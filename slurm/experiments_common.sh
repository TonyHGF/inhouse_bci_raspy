# Sourced only inside the CPU/GPU sbatch scripts.
mkdir -p logs
set +u # Existing conda compiler activation hooks read unset variables.
source ~/anaconda3/bin/activate "${CONDA_ENV:?}"
set -u
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="/public/home/hugf2022/inhouse_bci_raspy/matplotlib/$SLURM_JOB_ID"
mkdir -p "$MPLCONFIGDIR"
trap 'code=$?; echo "End: $(date -Is) exit=$code"' EXIT
echo "Start: $(date -Is) job=$SLURM_JOB_ID name=$SLURM_JOB_NAME node=$SLURM_JOB_NODELIST env=$CONDA_ENV python=$(command -v python)"
if [[ "$SLURM_JOB_PARTITION" == bme_gpu ]]; then
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    nvidia-smi
    python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
else
    export CUDA_VISIBLE_DEVICES=""
fi
command=(python -B scripts/server_experiments.py "$BCI_STAGE" --profile-file "$BCI_PROFILE")
if [[ "$BCI_STAGE" == run || "$BCI_STAGE" == explain ]]; then
    command+=(--shards "$BCI_SHARDS" --shard-index "${SLURM_ARRAY_TASK_ID:-0}")
fi
command+=("$@")
printf 'Command: '; printf '%q ' "${command[@]}"; printf '\n'
"${command[@]}"
