#!/usr/bin/env bash
#SBATCH --job-name=fq-denoise
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --output=.cache/slurm_logs/%x-%j.out
#SBATCH --error=.cache/slurm_logs/%x-%j.err

# The collaborator's second and third questions on one H100: clean-task
# performance and corruption propensity measured separately, and the compressed
# damaged adapter compared against an adapter trained on clean data.
#
# The order is deliberate. Training the clean counterpart comes first because it
# is work this job has to do regardless and takes about forty minutes, which is
# roughly how long the grid's own 7B cell needs to produce the damaged adapter
# this study reads. By the time the clean arm is trained the damaged one is
# usually there; if it is not, the wait below is bounded and the job fails with
# a clear reason rather than half a study.

set -uo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
mkdir -p .cache/slurm_logs remote_logs
source "$repo_root/scripts/jean_zay_env.sh"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

STUDY=configs/jean_zay/denoise_vs_shrinkage.yaml
ALIGNED=configs/jean_zay/aligned_cot.yaml
GRID=configs/jean_zay/scale_and_tasks.yaml
OUT=.cache/reports/denoise_vs_shrinkage_jz
LOG="$repo_root/remote_logs/denoise_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUT"

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
  tail -30 "$LOG" >&2
  return 1
}
run_id_for() {  # config study model
  python - "$1" "$2" "$3" <<'PYX'
import sys
from fineqcomp.campaign import expand_campaign
from fineqcomp.config import load_campaign

config, study, model = sys.argv[1:4]
runs = [
    run
    for run in expand_campaign(load_campaign(config))
    if run.study == study and run.model.key == model
]
if len(runs) != 1:
    raise SystemExit(f"expected one run, found {len(runs)}")
print(runs[0].run_id)
PYX
}

# STAGE=train trains the clean counterpart and stops; STAGE=study runs the
# measurements. They are separate jobs so neither holds an H100 idle waiting for
# the other: the training is short enough for the two-hour dev queue, and the
# study starts on a Slurm dependency rather than on a sleep loop.
STAGE="${STAGE:-all}"
say "=== denoise/shrinkage, questions two and three, stage $STAGE (job ${SLURM_JOB_ID:-none}) ==="

clean_id=$(run_id_for "$ALIGNED" cot_aligned_full qwen25_7b_base) || exit 1
say "clean arm $clean_id"
if [[ -f ".cache/runs/$clean_id/raw_channel.pt" ]]; then
  say "SKIP  train clean arm (adapter already on disk)"
else
  # Its own manifest, not the grid's. `prepare` rewrites whatever manifest it
  # is pointed at, and the grid's twenty-one-run manifest is being read by
  # array tasks that have not started yet.
  stage "prepare clean corpus" python -m fineqcomp prepare \
    --config "$ALIGNED" --manifest .cache/prepared/aligned-manifest.jsonl || exit 1
  stage "train clean arm" python -m fineqcomp run \
    --config "$ALIGNED" --manifest .cache/prepared/aligned-manifest.jsonl --run-id "$clean_id" || exit 1
fi

damaged_id=$(python - "$GRID" <<'PYX'
import sys
from fineqcomp.campaign import expand_campaign
from fineqcomp.config import load_campaign

runs = [
    run
    for run in expand_campaign(load_campaign(sys.argv[1]))
    if run.study == "scale_ladder"
    and run.model.key == "qwen25_7b_base"
    and run.dataset_key == "cot_math_permuted"
]
if len(runs) != 1:
    raise SystemExit(f"expected one run, found {len(runs)}")
print(runs[0].run_id)
PYX
) || exit 1
say "damaged arm $damaged_id"

# Bounded wait: the grid cell that trains this is an independent job, so this
# one must not assume it has landed and must not hang for ever if it failed.
deadline=$(( $(date +%s) + 1 * 3600 ))
while [[ ! -f ".cache/runs/$damaged_id/raw_channel.pt" ]]; do
  if (( $(date +%s) > deadline )); then
    say "FAIL damaged adapter never appeared: .cache/runs/$damaged_id/raw_channel.pt"
    say "     the grid's cot_7b cell has to finish before this study can run"
    exit 1
  fi
  say "waiting for the grid's cot_7b adapter..."
  sleep 300
done
say "damaged adapter present"

stage "prepare" python -m fineqcomp.denoise_vs_shrinkage \
  --config "$STUDY" --out "$OUT" --phase prepare || exit 1
stage "check" python -m fineqcomp.denoise_vs_shrinkage \
  --config "$STUDY" --out "$OUT" --phase check || exit 1
stage "probe" python -m fineqcomp.denoise_vs_shrinkage \
  --config "$STUDY" --out "$OUT" --phase probe || say "WARN probe failed, continuing"
stage "sweep" python -m fineqcomp.denoise_vs_shrinkage \
  --config "$STUDY" --out "$OUT" --phase sweep || exit 1
stage "report" python -m fineqcomp.denoise_vs_shrinkage \
  --config "$STUDY" --out "$OUT" --phase report || exit 1
say "=== done; report in $OUT ==="
