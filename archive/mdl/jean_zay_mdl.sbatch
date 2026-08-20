#!/usr/bin/env bash
# Adaptive MDL frontier for one already-trained pilot seed.
#
# Split out from the training job so the decisive gate gets its own 2h budget.
# The sweep reads runs/<id>/raw_channel.pt and skips any rate whose metric file
# already exists, so running this after a job that hit the wall resumes rather
# than repeats. Submit with --dependency=afterany:<training job id>.
#SBATCH --job-name=fineqcomp-mdl
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
echo "run_id: $run_id"

if [[ ! -f "runs/$run_id/raw_channel.pt" ]]; then
  echo "no trained adapter for seed $seed; training job did not get far enough" >&2
  exit 2
fi

"$PYTHON" -m fineqcomp.mdl "runs/$run_id" \
  --config "$config" --prepared-root prepared \
  --target-rates 0.15 0.3 0.5 0.75 1.0 1.5 2.5 4.0

echo "mdl seed $seed complete"
