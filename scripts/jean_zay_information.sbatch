#!/usr/bin/env bash
# Controlled information-scaling grid, one shard of cells per H100.
#
# The grid lives in configs/information_scaling.yaml, not here: this script
# only picks a shard of it. `--shards N` stripes the 51 cells so every shard
# gets a mix of the long four-prefix main cells and the short single-prefix
# rank and cue cells.
#
#   smoke  the two endpoint conditions at the largest prefix, seed 11.
#          Gate: raw recall >= 0.95 on both, and R* defined on both.
#            sbatch --array=0-1%2 --export=ALL,MODE=smoke \
#              scripts/jean_zay_information.sbatch
#
#   full   all 51 cells across twelve shards.
#            sbatch --array=0-11%12 --export=ALL,MODE=full \
#              scripts/jean_zay_information.sbatch
#
# INFO_DRY_RUN=1 prints the cells a shard would run and exits before the
# modules load or a model is requested.
#SBATCH --job-name=fqinfo
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --array=0-11%12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"

mode="${MODE:-smoke}"
task="${SLURM_ARRAY_TASK_ID:-0}"
config="${INFO_CONFIG:-configs/information_scaling.yaml}"
campaign="${INFO_CAMPAIGN:-configs/compressibility.yaml}"
out="${INFO_OUT:-runs_information_scaling/$mode}"

command=(python -m fineqcomp.information_scaling
  --config "$config" --campaign "$campaign" --out "$out")
case "$mode" in
smoke)
  # One cell per array task: the zero-information anchor and the hardest
  # condition, at the largest prefix only.
  conditions=(constant random)
  if ((task < 0 || task >= ${#conditions[@]})); then
    echo "array task $task is outside MODE=smoke (0-1)" >&2
    exit 2
  fi
  command+=(--studies main --conditions "${conditions[$task]}"
    --seeds 11 --prefixes 512)
  ;;
full)
  shards="${INFO_SHARDS:-12}"
  command+=(--shard "$task" --shards "$shards")
  ;;
*)
  echo "unknown MODE=$mode; expected smoke or full" >&2
  exit 2
  ;;
esac

printf 'mode=%s task=%s out=%s\n' "$mode" "$task" "$out"
if [[ "${INFO_DRY_RUN:-0}" == 1 ]]; then
  command+=(--dry-run)
  printf '%q ' "${command[@]}"
  printf '\n'
fi

mkdir -p slurm_logs "$out"
source "$repo_root/scripts/jean_zay_env.sh"
export PYTHONUNBUFFERED=1
command[0]="$PYTHON"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${command[@]}"
