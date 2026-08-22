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
#   probe  learnability diagnostic, one cell per array task. Run this when the
#          smoke fails G1: it separates storage capacity, optimization, and the
#          key-to-readout binding instead of guessing which one broke.
#            sbatch --array=0-12%13 --export=ALL,MODE=probe \
#              scripts/jean_zay_information.sbatch
#
#   validate  certify the measurement inside the range the probe found usable:
#             five conditions spanning 4 to 512 known source bits, three seeds,
#             one cell per task, about half an hour.
#               sbatch --array=0-14%15 --export=ALL,MODE=validate \
#                 scripts/jean_zay_information.sbatch
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
# Retry lever: if G1 fails because the learner never reached the taught map,
# raise the update budget here rather than editing the config.
if [[ -n "${INFO_UPDATES:-}" ]]; then
  command+=(--updates "$INFO_UPDATES")
fi
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
probe)
  # condition:prefix:learning_rate:updates:layout
  #
  # Rows 1-2 ask whether the model can read the key fields at all, at 16 and
  # 64 associations. Rows 3-6 walk the number of arbitrary associations to
  # find where storage stops. Rows 7-9 vary the learning rate and budget at
  # the size that already failed. Rows 10-13 shorten the distance from the key
  # to the answer position, alone and combined with the other levers.
  cells=(
    p1:512:2e-4:2048:fields
    p4:512:2e-4:2048:fields
    random:32:2e-4:2048:fields
    random:64:2e-4:2048:fields
    random:128:2e-4:2048:fields
    random:256:2e-4:2048:fields
    random:512:1e-3:2048:fields
    random:512:3e-3:2048:fields
    random:512:1e-3:8192:fields
    random:512:2e-4:2048:key_last
    random:512:1e-3:2048:key_last
    random:512:3e-3:8192:key_last
    p1:512:1e-3:2048:key_last
  )
  if ((task < 0 || task >= ${#cells[@]})); then
    echo "array task $task is outside MODE=probe (0-$((${#cells[@]} - 1)))" >&2
    exit 2
  fi
  IFS=: read -r condition prefix rate updates layout <<<"${cells[$task]}"
  # Cells differ in more than the run identity covers, so each gets its own
  # output root rather than silently reusing a neighbour's result.json.
  tag="${condition}_n${prefix}_lr${rate//./}_u${updates}_${layout}"
  out="${INFO_OUT:-runs_information_scaling/probe/$tag}"
  command=(python -m fineqcomp.information_scaling
    --config "$config" --campaign "$campaign" --out "$out"
    --studies main --conditions "$condition" --seeds 11
    --prefixes "$prefix" --learning-rate "$rate" --updates "$updates"
    --layout "$layout")
  printf 'probe cell: %s\n' "${cells[$task]}"
  ;;
validate)
  # The shard count has to match the array width, and the array width changes
  # with the grid: the validation grid is fifteen cells, the payload grid is
  # twenty-four. Hardcoding fifteen silently dropped cells off the end.
  config="${INFO_CONFIG:-configs/information_validation.yaml}"
  out="${INFO_OUT:-runs_information_scaling/validate}"
  command=(python -m fineqcomp.information_scaling
    --config "$config" --campaign "$campaign" --out "$out"
    --shard "$task" --shards "${INFO_SHARDS:-15}")
  ;;
full)
  shards="${INFO_SHARDS:-12}"
  command+=(--shard "$task" --shards "$shards")
  ;;
*)
  echo "unknown MODE=$mode; expected smoke, probe, validate or full" >&2
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
