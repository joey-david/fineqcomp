#!/usr/bin/env bash
# Source jean_zay_env.sh on the login node, then sbatch --export=ALL this file.
# One Slurm task per GPU. For 12 workers: --nodes=3 --ntasks=12
# --ntasks-per-node=4 --gres=gpu:4. SHARD_OFFSET supports split jobs too.
#SBATCH --job-name=fq-spectral
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:55:00
#SBATCH --output=logs/spectral-%j.out
#SBATCH --error=logs/spectral-%j.err

set -euo pipefail
: "${PYTHON:?Preload the managed environment on the login node}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
phase="$1"
shift
if [[ "$phase" != screen && "$phase" != search && "$phase" != validate && "$phase" != test && "$phase" != likelihood && "$phase" != map ]]; then
  exec "$PYTHON" -u -m fineqcomp.spectral_transfer --phase "$phase" "$@"
fi
workers="${SLURM_NTASKS:?One Slurm task is required for each GPU}"
total_shards="${TOTAL_SHARDS:-$workers}"
offset="${SHARD_OFFSET:-0}"
if (( offset + workers > total_shards )); then
  echo 'Worker count exceeds remaining shards' >&2
  exit 2
fi
srun --mpi=none --kill-on-bad-exit=1 --ntasks "$workers" --gpus-per-task=1 \
  --gpu-bind=single:1 bash -c '
    phase="$1"; total="$2"; offset="$3"; shift 3
    printf "worker=%s node=%s CUDA_VISIBLE_DEVICES=%s\n" "$SLURM_PROCID" "$HOSTNAME" "$CUDA_VISIBLE_DEVICES"
    exec "$PYTHON" -u -m fineqcomp.spectral_transfer --phase "$phase" \
      --shard "$((offset + SLURM_PROCID))" --shards "$total" "$@"
  ' worker "$phase" "$total_shards" "$offset" "$@"

for argument in "$@"; do
  [[ "$argument" != --max-tasks && "$argument" != --max-tasks=* ]] || exit 0
done

# Freeze choices only after every validation worker succeeds. Test reports
# follow the same barrier, so the early report contains every core test row.
# The screen recommendation follows the same shape: workers score their own
# models under --phase screen, and one unsharded call to the DIFFERENT
# recommend_screen phase afterwards reads every cell and writes the table.
# Re-calling screen itself here was tried first and was wrong: its per-model
# loop defaults to shard=0/shards=1, so the "barrier" silently re-walked every
# model and redid a full generation pass on one GPU.
if [[ "$phase" == screen ]]; then
  "$PYTHON" -u -m fineqcomp.spectral_transfer --phase recommend_screen "$@"
elif [[ "$phase" == validate ]]; then
  "$PYTHON" -u -m fineqcomp.spectral_transfer --phase select "$@"
elif [[ "$phase" == test ]]; then
  report_group=all
  for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == --group ]]; then
      next=$((i + 1))
      [[ "${!next}" != core ]] || report_group=core
    fi
  done
  "$PYTHON" -u -m fineqcomp.spectral_transfer --phase report "$@" --group "$report_group"
fi
