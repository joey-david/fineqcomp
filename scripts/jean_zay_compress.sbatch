#!/usr/bin/env bash
# One cell of the dataset-compressibility experiment (arm x seed).
#
# Every arm trains for the same number of optimizer updates on the same number
# of samples; only the count of distinct samples differs. All arms share one
# baseline per seed, because they share an evaluation set, a calibration split,
# and the same leading training rows.
#
# Submit: sbatch --export=ALL,ARM=arm_a,SEED=11 scripts/jean_zay_compress.sbatch
#SBATCH --job-name=fqcomp
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
mkdir -p slurm_logs reports

arm="${ARM:?set ARM=arm_a|arm_b|arm_c|arm_d|arm_e}"
seed="${SEED:?set SEED}"
config="${CONFIG:-configs/compressibility.yaml}"
manifest="${MANIFEST:-prepared/compressibility-manifest.jsonl}"

source "$repo_root/scripts/jean_zay_env.sh"
export PYTHONUNBUFFERED=1

echo "arm=$arm seed=$seed config=$config"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_id="$("$PYTHON" - "$manifest" "$seed" "$arm" <<'PY'
import json, sys
manifest, seed, arm = sys.argv[1], int(sys.argv[2]), sys.argv[3]
for line in open(manifest):
    record = json.loads(line)
    if record["seed"] == seed and record["study"] == arm:
        print(record["run_id"])
        break
PY
)"
if [[ -z "$run_id" ]]; then
  echo "no run for arm=$arm seed=$seed in $manifest" >&2
  exit 2
fi
echo "run_id=$run_id"

"$PYTHON" -m fineqcomp run --config "$config" --manifest "$manifest" \
  --runs-root runs --run-id "$run_id" --shards 1

echo "arm $arm seed $seed complete"
