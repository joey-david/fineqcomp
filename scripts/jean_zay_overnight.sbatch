#!/usr/bin/env bash
# One stage of one overnight cell (model x seed).
#
# t3 rather than dev: the training stage is about three hours at 32k rows, so
# it cannot fit dev's two-hour cap. t3 allows 20 hours and has no tight submit
# limit, at the cost of a real queue wait.
#
#   STAGE=train       train, score base and raw, write the fp16 point
#   STAGE=codecs      the nine remaining fixed-rate codecs
#   STAGE=mdl         adaptive sweep, row dropping allowed
#   STAGE=mdl_nodrop  adaptive sweep, every row keeps at least a sign
#   STAGE=report      collect every cell into one figure and table
#
# The two adaptive arms exist because the shakeout showed the squared-error
# allocator zeroes half the values and retention collapses to zero, while the
# same allocator forbidden from dropping rows is the real test of whether
# adaptive allocation is worth anything.
#
# Submit: sbatch --export=ALL,MODEL=mistral_7b_base,SEED=11,STAGE=train \
#           scripts/jean_zay_overnight.sbatch
#SBATCH --job-name=fqnight
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
mkdir -p slurm_logs reports

seed="${SEED:?set SEED}"
stage="${STAGE:?set STAGE}"
model="${MODEL:-mistral_7b_base}"
config="${CONFIG:-configs/overnight.yaml}"
manifest="${MANIFEST:-prepared/overnight-manifest.jsonl}"

source "$repo_root/scripts/jean_zay_env.sh"
export PYTHONUNBUFFERED=1

echo "stage=$stage model=$model seed=$seed"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_id="$("$PYTHON" - "$manifest" "$seed" "$model" <<'PY'
import json, sys
manifest, seed, model = sys.argv[1], int(sys.argv[2]), sys.argv[3]
for line in open(manifest):
    record = json.loads(line)
    if record["seed"] == seed and record["model"]["key"] == model:
        print(record["run_id"])
        break
PY
)"
if [[ -z "$run_id" ]]; then
  echo "no run for model=$model seed=$seed in $manifest" >&2
  exit 2
fi
echo "run_id=$run_id"

all_codecs="fp16 uniform8 uniform4 uniform3 uniform2 binary midtread3 midtread2 loraquant_2_08 loraquant_3_09"

case "$stage" in
train)
  "$PYTHON" -m fineqcomp run --config "$config" --manifest "$manifest" \
    --runs-root runs --run-id "$run_id" --shards 1 --codecs fp16
  ;;
codecs)
  "$PYTHON" -m fineqcomp run --config "$config" --manifest "$manifest" \
    --runs-root runs --run-id "$run_id" --shards 1 --codecs $all_codecs
  ;;
mdl)
  PYTHONPATH="$repo_root/src:$repo_root/archive/mdl" "$PYTHON" -m mdl "runs/$run_id" \
    --config "$config" --prepared-root prepared \
    --target-rates 0.25 0.5 0.75 1.0 1.25 1.5 2.0 3.0
  ;;
mdl_nodrop)
  # Rates under about one bit per value are unreachable without dropping rows;
  # the sweep reports and skips those rather than failing.
  PYTHONPATH="$repo_root/src:$repo_root/archive/mdl" "$PYTHON" -m mdl "runs/$run_id" --no-drop \
    --config "$config" --prepared-root prepared \
    --target-rates 1.1 1.25 1.5 1.75 2.0 2.5 3.0 4.0
  ;;
report)
  "$PYTHON" -m fineqcomp.pilot_report --runs-root runs \
    --run-prefix overnight__ --out reports/overnight
  ;;
*)
  echo "unknown STAGE=$stage" >&2
  exit 2
  ;;
esac

echo "stage $stage model $model seed $seed complete"
