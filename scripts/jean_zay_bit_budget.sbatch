#!/usr/bin/env bash
# The bit-budget study: does a capped probe predict the budget of a fine-tune?
#
# The panel lives in configs/bit_budget.yaml and trains nothing at full length.
# Every target it predicts is an adapter the campaign already paid for, so the
# whole study is probes and forward-only sweeps and fits in ten GPU-hours.
#
# The array unit is a receiver, not a run: loading a 9B model in NF4 costs more
# than sweeping one of its adapters, so a task that touched every receiver
# would spend most of its time in `from_pretrained`.
#
#   manifest  expand the config. No GPU, no download: every prepared cell this
#             panel needs already exists, which is why it can predict them.
#               sbatch --array=0-0 --export=ALL,MODE=manifest \
#                 scripts/jean_zay_bit_budget.sbatch
#
#   check     prove every cell has a finished adapter to predict and a prepared
#             corpus to read. Non-zero exit stops the chain before any GPU is
#             claimed, which is the whole point of running it first.
#               sbatch --array=0-0 --export=ALL,MODE=check \
#                 scripts/jean_zay_bit_budget.sbatch
#
#   probes    train the 33 capped runs, one receiver per task. One optimizer
#             update over a 256-row batch, or eight for the ladder.
#               sbatch --array=0-6%7 --export=ALL,MODE=probes \
#                 scripts/jean_zay_bit_budget.sbatch
#
#   sweep     the budget surface of every probe and of the finished adapter it
#             predicts, one receiver per task. No training.
#               sbatch --array=0-6%7 --export=ALL,MODE=sweep \
#                 scripts/jean_zay_bit_budget.sbatch
#
#   report    join the two and read the pre-registered gates.
#               sbatch --array=0-0 --export=ALL,MODE=report \
#                 scripts/jean_zay_bit_budget.sbatch
#
# BB_DRY_RUN=1 prints the command and exits before any model is requested.
#SBATCH --job-name=fqbudget
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --array=0-6%7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"

mode="${MODE:-report}"
task="${SLURM_ARRAY_TASK_ID:-0}"
config="${BB_CONFIG:-configs/bit_budget.yaml}"
manifest="${BB_MANIFEST:-prepared/bit-budget-manifest.jsonl}"
runs_root="${BB_RUNS_ROOT:-runs}"
sweeps="${BB_SWEEPS:-reports/bit_budget}"
# Four ranks and seven rungs, not the sweep's wider default. The twelve
# adapters already scored put every 90% crossing at rank one or two and between
# one and two and a bit bits per value, so this grid resolves the budget where
# it actually falls and spends nothing above it.
ranks="${BB_RANKS:-1 2 4 16}"
rates="${BB_RATES:-0:0.5 1:0.0 1:0.25 1:0.5 1:0.75 2:0.0 3:0.0}"

mkdir -p slurm_logs "$sweeps" prepared
source "$repo_root/scripts/jean_zay_env.sh"
export PYTHONUNBUFFERED=1

# The receiver list has to match the array width, and it lives in the config,
# so read it from there rather than repeating it here.
mapfile -t models < <(
  "$PYTHON" - "$config" <<'PY'
import sys, yaml
for name in yaml.safe_load(open(sys.argv[1]))["models"]:
    print(name)
PY
)

case "$mode" in
manifest)
  # --no-data because every corpus this panel touches is already prepared: a
  # compute node has no route to the hub and does not need one.
  command=("$PYTHON" -m fineqcomp prepare
    --config "$config" --manifest "$manifest" --no-data)
  ;;
check)
  # Runs first and holds the chain. This study predicts adapters other
  # campaigns trained and reads corpora they prepared, so both are things it
  # can only find. A missing one is silent at submit time and fatal an hour
  # into the night; here it is a non-zero exit before any GPU is claimed.
  command=("$PYTHON" -m fineqcomp bit-budget --check
    --config "$config" --runs-root "$runs_root")
  ;;
probes | sweep)
  if ((task < 0 || task >= ${#models[@]})); then
    echo "array task $task is outside 0-$((${#models[@]} - 1))" >&2
    exit 2
  fi
  receiver="${models[$task]}"
  if [[ "$mode" == probes ]]; then
    command=("$PYTHON" -m fineqcomp run
      --config "$config" --manifest "$manifest" --runs-root "$runs_root"
      --shards 1 --models "$receiver")
  else
    # From the manifest, not from a glob on study names: the panel grew a
    # `repair_*` family that no `probe*` pattern would have matched, and a
    # sweep that silently skips half the panel is worse than one that fails.
    mapfile -t ids < <("$PYTHON" -c 'import json,sys
for line in open(sys.argv[1]):
    print(json.loads(line)["run_id"])' "$manifest")
    probes=()
    for id in "${ids[@]}"; do
      [[ -f "$runs_root/$id/raw_channel.pt" ]] && probes+=("$runs_root/$id")
    done
    if ((${#probes[@]} == 0)); then
      echo "no swept-able probes under $runs_root; run MODE=probes first" >&2
      exit 2
    fi
    command=("$PYTHON" -m fineqcomp bit-budget
      --config "$config" --out "$sweeps" --with-targets
      --models "$receiver" --ranks $ranks --rates $rates
      --runs "${probes[@]}")
  fi
  ;;
report)
  command=("$PYTHON" -m fineqcomp bit-budget-report
    --config "$config" --sweeps "$sweeps" --runs-root "$runs_root"
    --out results/5_predicting_the_bit_budget)
  ;;
*)
  echo "unknown MODE=$mode; expected check, manifest, probes, sweep or report" >&2
  exit 2
  ;;
esac

printf 'mode=%s task=%s receiver=%s\n' "$mode" "$task" "${receiver:-n/a}"
printf '%q ' "${command[@]}"
printf '\n'
if [[ "${BB_DRY_RUN:-0}" == 1 ]]; then
  exit 0
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${command[@]}"
