#!/usr/bin/env bash
#SBATCH --job-name=fq-grid
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

# One array task per cell of the scale-by-task grid: train the adapter, then
# sweep the seven compression conditions over it. Cells are independent, so the
# array is the whole parallelism -- twenty-one H100s at once if the queue allows,
# and a cell that fails takes nothing else down with it.
#
# The order below is by cost, cheapest first. A grid that is going to fail for a
# structural reason fails in the first twenty minutes on a 1.5B cell rather than
# three hours in on a 32B one, and the small cells' results are readable while
# the large ones are still running.
#
# Walltime is the thing that decides whether any of this runs tonight. gpu_p6
# is at 357 of 364 nodes allocated with two thousand jobs pending, so a job only
# starts by backfilling into a gap, and a gap is short. A blanket 20-hour
# request never fits one; a two-hour request fits constantly. The tiers are in
# scripts/jean_zay_grid_launch.sh, which is how this should be submitted.
#
#   scripts/jean_zay_grid_launch.sh                     # all three tiers
#   SMOKE=1 ARMS_LIST=ptr_h2_1p5b sbatch --array=0 \
#       scripts/jean_zay_grid.sbatch                    # one cell, eight rows


set -uo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
mkdir -p slurm_logs remote_logs
source "$repo_root/scripts/jean_zay_env.sh"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

ARMS=(
  ptr_h2_1p5b ptr_h4_1p5b ptr_h8_1p5b ptr_mm_1p5b
  cot_0p5b cot_1p5b cot_3b
  ptr_h2_7b ptr_h4_7b ptr_h8_7b ptr_mm_7b ptr_h4_clean_7b
  ptr_h4_7b_inst ptr_mm_7b_inst
  cot_7b
  ptr_h2_32b ptr_h4_32b ptr_h8_32b ptr_mm_32b
  cot_14b cot_32b
)
# A tier launcher passes its tier name as the first argument and the cells come
# from slurm_tiers/<tier>.txt. This is deliberately a file and an argument
# rather than `--export=ALL,ARMS_LIST=...`: that flag makes Slurm retrieve the
# user environment at launch, which failed here and left the whole array
# requeued and held at priority zero.
tier="${1:-}"
if [[ -n "$tier" ]]; then
  mapfile -t ARMS < "slurm_tiers/${tier}.txt"
fi
arm="${ARM:-${ARMS[${SLURM_ARRAY_TASK_ID:-0}]}}"

# A second and third argument point the same runner at a different study, so
# the headroom grid reuses this rather than growing a near-identical copy.
SWEEP="${2:-configs/jean_zay/scaling_sweep.yaml}"
OUT="${3:-reports/scaling_grid}"
# Each study stages its own manifest; training a cell against another
# study's manifest fails to find its own run id.
MANIFEST="${4:-prepared/manifest.jsonl}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  SWEEP=configs/jean_zay/scaling_sweep_smoke.yaml
  OUT=reports/scaling_grid_smoke
fi
mkdir -p "$OUT"
LOG="$repo_root/remote_logs/grid_${arm}_$(date +%Y%m%d_%H%M%S).log"

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }
stage() {
  local name="$1"; shift
  say "START $name"
  local t0; t0=$(date +%s)
  if "$@" >>"$LOG" 2>&1; then
    say "OK    $name ($(( $(date +%s) - t0 ))s)"
    return 0
  fi
  say "FAIL  $name ($(( $(date +%s) - t0 ))s); last lines:"
  tail -25 "$LOG" >&2
  return 1
}

field() {
  python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['arms'][sys.argv[2]][sys.argv[3]])" \
    "$SWEEP" "$arm" "$1"
}

say "=== cell $arm (array task ${SLURM_ARRAY_TASK_ID:-none}, job ${SLURM_JOB_ID:-none}) ==="
config=$(field config) || exit 1
study=$(field study) || exit 1
model=$(field model) || exit 1
dataset=$(field dataset) || exit 1

# The run id is a content hash of the run's own identity, so it is the only safe
# handle on "the adapter this cell means". Resolving it here rather than
# guessing a directory name is what stops a cell scoring somebody else's weights.
run_id=$(python - "$config" "$study" "$model" "$dataset" <<'PYX'
import sys
from fineqcomp.campaign import expand_campaign
from fineqcomp.config import load_campaign

config, study, model, dataset = sys.argv[1:5]
runs = [
    run
    for run in expand_campaign(load_campaign(config))
    if run.study == study and run.model.key == model and run.dataset_key == dataset
]
if len(runs) != 1:
    raise SystemExit(f"expected one run, found {len(runs)}")
print(runs[0].run_id)
PYX
) || { say "FAIL resolve run id for $arm"; exit 1; }
say "run id $run_id"

if [[ -f "runs/$run_id/raw_channel.pt" ]]; then
  say "SKIP  train $arm (adapter already on disk)"
else
  stage "train $arm" python -m fineqcomp run \
    --config "$config" --manifest "$MANIFEST" --run-id "$run_id" || exit 1
fi

stage "sweep $arm" python -m fineqcomp.generalisation \
  --config "$SWEEP" --out "$OUT" --phase sweep --arm "$arm" || exit 1

# Every cell rewrites the report. It is cheap, it is idempotent, and it means a
# grid that is still half-running already has a readable table.
python -m fineqcomp.generalisation --config "$SWEEP" --out "$OUT" --phase report \
  >>"$LOG" 2>&1 || say "WARN report failed (other cells may still be missing)"
say "=== cell $arm done ==="
