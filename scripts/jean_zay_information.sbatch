#!/usr/bin/env bash
# One controlled information-scaling cell per H100.
#
# pilot:  random/p1 at seed 11                    --array=0-1%2
# repeat: random/p1 at seeds 22 and 33            --array=0-3%4
# full:   all four conditions at all three seeds  --array=0-11%12, t3 QoS
# training: random/p1 x three seeds x 8/16/32 epochs --array=0-17%18, t3
#
# Submit, for example:
#   sbatch --array=0-1%2 --export=ALL,MODE=pilot \
#     scripts/jean_zay_information.sbatch
# The dev QoS has a ten-job submit cap. Override it with
# --qos=qos_gpu_h100-t3 for the twelve-cell full array.
#
# Use INFO_DRY_RUN=1 with SLURM_ARRAY_TASK_ID set to check the cell mapping
# without loading modules or requesting a model.
#SBATCH --job-name=fqinfo
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --array=0-1%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"

mode="${MODE:-pilot}"
task="${SLURM_ARRAY_TASK_ID:-0}"
case "$mode" in
pilot)
  cells=(random:11 structured_p1:11)
  ;;
repeat)
  cells=(random:22 random:33 structured_p1:22 structured_p1:33)
  ;;
full)
  conditions=(structured_p1 structured_p4 structured_p16 random)
  seeds=(11 22 33)
  cells=()
  for condition in "${conditions[@]}"; do
    for seed in "${seeds[@]}"; do
      cells+=("$condition:$seed")
    done
  done
  ;;
training)
  conditions=(random structured_p1)
  seeds=(11 22 33)
  epoch_grid=(8 16 32)
  cells=()
  for epochs in "${epoch_grid[@]}"; do
    for condition in "${conditions[@]}"; do
      for seed in "${seeds[@]}"; do
        cells+=("$condition:$seed:$epochs")
      done
    done
  done
  ;;
*)
  echo "unknown MODE=$mode; expected pilot, repeat, full, or training" >&2
  exit 2
  ;;
esac

if ((task < 0 || task >= ${#cells[@]})); then
  echo "array task $task is outside MODE=$mode (0-$((${#cells[@]} - 1)))" >&2
  exit 2
fi
IFS=: read -r condition seed epochs <<<"${cells[$task]}"
run_epochs="${epochs:-${INFO_EPOCHS:-}}"

config="${INFO_CONFIG:-configs/information_scaling.yaml}"
campaign="${INFO_CAMPAIGN:-configs/compressibility.yaml}"
if [[ "$mode" == training ]]; then
  out="${INFO_OUT:-runs_information_scaling/training/e$run_epochs}"
else
  out="${INFO_OUT:-runs_information_scaling/$mode}"
fi
command=(python -m fineqcomp.information_scaling
  --config "$config" --campaign "$campaign" --out "$out"
  --conditions "$condition" --seeds "$seed")
if [[ -n "$run_epochs" ]]; then
  command+=(--epochs "$run_epochs")
fi
if [[ "$mode" == training ]]; then
  command+=(--prefixes 512)
fi

printf 'mode=%s task=%s condition=%s seed=%s epochs=%s out=%s\n' \
  "$mode" "$task" "$condition" "$seed" "${run_epochs:-config}" "$out"
if [[ "${INFO_DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p slurm_logs "$out"
source "$repo_root/scripts/jean_zay_env.sh"
export PYTHONUNBUFFERED=1
command[0]="$PYTHON"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "${command[@]}"
