#!/usr/bin/env bash
# One pilot seed: train, then every codec point, then the MDL frontier.
#
# Targets qos_gpu_h100-dev deliberately. It caps walltime at 2h but carries
# priority 80 against t3's 50, and allows 10 concurrent jobs, so three seeds
# run in parallel and land in roughly the time of one.
#
# Submit with: sbatch --export=ALL,SEED=11 scripts/jean_zay_pilot.sbatch
#SBATCH --job-name=fineqcomp-pilot
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
config=configs/pilot_gsm8k.yaml
manifest=prepared/pilot-manifest.jsonl

source "$repo_root/scripts/jean_zay_env.sh"
echo "python:  $PYTHON"
echo "seed:    $seed"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ ! -s "$manifest" ]]; then
  echo "missing $manifest; run the staging step on a login node first" >&2
  exit 2
fi

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
echo "run_id:  $run_id"

# Train once, then derive every fixed-rate codec point from that checkpoint.
"$PYTHON" -m fineqcomp run \
  --config "$config" --manifest "$manifest" \
  --runs-root runs --run-id "$run_id" --shards 1

# The adaptive frontier reuses the same checkpoint; no retraining.
"$PYTHON" -m fineqcomp.mdl "runs/$run_id" \
  --config "$config" --prepared-root prepared \
  --target-rates 0.15 0.3 0.5 0.75 1.0 1.5 2.5 4.0

echo "pilot seed $seed complete"
