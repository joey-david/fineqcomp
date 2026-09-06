#!/usr/bin/env bash
# Experiment 1 of the algorithm-selection study, one shard of cells per H100.
#
# The grid lives in configs/program_selection.yaml. This script only picks a
# stage and a shard of it. Run the stages in order: the training stage refuses
# to start on a family and model the diagnose stage has not locked.
#
#   audit     families only, no GPU and no model. Proves the training region is
#             ambiguous and the splits are disjoint.
#               sbatch --array=0-0 --export=ALL,MODE=audit \
#                 scripts/jean_zay_program_selection.sbatch
#
#   diagnose  one base model per array task. Measures the version-space
#             posterior of the frozen model and writes the locks.
#               sbatch --array=0-2%3 --export=ALL,MODE=diagnose \
#                 scripts/jean_zay_program_selection.sbatch
#
#   smoke     one family, one model, one seed, four arms. Gate: every arm
#             reaches 0.90 on the training region, and the run finishes inside
#             the walltime.
#               sbatch --array=0-0 --export=ALL,MODE=smoke \
#                 scripts/jean_zay_program_selection.sbatch
#
#   full      all 360 cells across twenty shards.
#               sbatch --array=0-19%20 --export=ALL,MODE=full \
#                 scripts/jean_zay_program_selection.sbatch
#
#   report    read the gates off whatever has finished.
#               sbatch --array=0-0 --export=ALL,MODE=report \
#                 scripts/jean_zay_program_selection.sbatch
#
# PS_DRY_RUN=1 prints the units a shard would run and exits before the modules
# load or a model is requested.
#SBATCH --job-name=fqprog
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --array=0-19%20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"

mode="${MODE:-smoke}"
task="${SLURM_ARRAY_TASK_ID:-0}"
config="${PS_CONFIG:-configs/program_selection.yaml}"
campaign="${PS_CAMPAIGN:-configs/transposed_receiver_panel.yaml}"
out="${PS_OUT:-runs_program_selection}"

mkdir -p slurm_logs "$out"
# Sourced before the model list is read, because a compute node has no usable
# python until the module stack is loaded.
source "$repo_root/scripts/jean_zay_env.sh"
export PYTHONUNBUFFERED=1

command=("$PYTHON" -m fineqcomp.program_selection
  --config "$config" --campaign "$campaign" --out "$out")

# The model list has to match the diagnose array width, and it lives in the
# config, so read it from there rather than repeating it here.
mapfile -t models < <(
  "$PYTHON" - "$config" <<'PY'
import sys, yaml
for name in yaml.safe_load(open(sys.argv[1]))["models"]:
    print(name)
PY
)

case "$mode" in
audit)
  command+=(--audit)
  ;;
diagnose)
  if ((task < 0 || task >= ${#models[@]})); then
    echo "array task $task is outside MODE=diagnose (0-$((${#models[@]} - 1)))" >&2
    exit 2
  fi
  command+=(--stage diagnose --models "${models[$task]}")
  ;;
smoke)
  # The cheapest arms that can still fail a gate: the shared base set, the
  # locked example, and the largest ambiguous arm, on one family and one seed.
  command+=(--stage train --gate --families "${PS_FAMILY:-chain}"
    --models "${models[0]}" --seeds 11
    --arms base targeted matched_difficulty ambiguous_10000)
  ;;
full)
  command+=(--stage train --shard "$task" --shards "${PS_SHARDS:-20}")
  ;;
report)
  command+=(--aggregate)
  ;;
*)
  echo "unknown MODE=$mode; expected audit, diagnose, smoke, full or report" >&2
  exit 2
  ;;
esac

printf 'mode=%s task=%s out=%s\n' "$mode" "$task" "$out"
if [[ "${PS_DRY_RUN:-0}" == 1 ]]; then
  command+=(--dry-run)
fi

printf '%q ' "${command[@]}"
printf '\n'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${command[@]}"
