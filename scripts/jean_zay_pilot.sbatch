#!/usr/bin/env bash
# One stage of one pilot seed.
#
# Split into three stages that each fit the 2-hour dev QoS. Dev carries
# priority 80 against t3's 50 and has started every job here in under ten
# seconds, so three chained instant stages finish sooner than one long job
# sitting in the 260-deep t3 queue.
#
#   STAGE=train   train the adapter, score base and raw, write the fp16 point
#   STAGE=codecs  every remaining fixed-rate codec for one seed
#   STAGE=mdl     a slice of the adaptive sweep, set by RATES
#   STAGE=report  collect all seeds into one figure and table
#
# `fineqcomp run` takes an exclusive lock on a run directory, so only one codec
# job per seed can make progress. The MDL sweep takes no such lock and only
# needs the saved checkpoint, so its slices run beside the codec job and beside
# each other, one model instance per H100.
#
# Submit with: sbatch --export=ALL,SEED=11,STAGE=train scripts/jean_zay_pilot.sbatch
#SBATCH --job-name=fqpilot
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
mkdir -p slurm_logs

seed="${SEED:?set SEED=11|22|33}"
stage="${STAGE:?set STAGE=train|codecs|frontier}"
config=${CONFIG:-configs/pilot_metamath.yaml}
manifest=${MANIFEST:-prepared/pilot-metamath-manifest.jsonl}

source "$repo_root/scripts/jean_zay_env.sh"
export PYTHONUNBUFFERED=1
export FINEQCOMP_PROGRESS=1

echo "stage=$stage seed=$seed python=$PYTHON"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_id="$("$PYTHON" - "$manifest" "$seed" <<'PY'
import json, sys
manifest, seed = sys.argv[1], int(sys.argv[2])
for line in open(manifest):
    record = json.loads(line)
    if record["seed"] == seed:
        print(record["run_id"])
        break
PY
)"
echo "run_id=$run_id"

case "$stage" in
train)
  # fp16 is the cheapest point and forces the base and raw evaluations, so the
  # learning gate is decided in this stage rather than the next one.
  "$PYTHON" -m fineqcomp run --config "$config" --manifest "$manifest" \
    --runs-root runs --run-id "$run_id" --shards 1 --codecs fp16
  ;;
codecs)
  "$PYTHON" -m fineqcomp run --config "$config" --manifest "$manifest" \
    --runs-root runs --run-id "$run_id" --shards 1 \
    --codecs fp16 uniform8 uniform4 uniform3 uniform2 binary \
             midtread3 midtread2 loraquant_2_08 loraquant_3_09
  ;;
mdl)
  PYTHONPATH="$repo_root/src:$repo_root/archive/mdl" "$PYTHON" -m mdl "runs/$run_id" \
    --config "$config" --prepared-root prepared \
    --target-rates ${RATES:?set RATES="0.25 0.5"}
  ;;
report)
  "$PYTHON" -m fineqcomp.pilot_report --runs-root runs --out reports/pilot
  ;;
*)
  echo "unknown STAGE=$stage" >&2
  exit 2
  ;;
esac

echo "stage $stage seed $seed complete"
