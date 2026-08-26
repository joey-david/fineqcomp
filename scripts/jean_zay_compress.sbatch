#!/usr/bin/env bash
# One cell of the dataset-compressibility experiment (arm x seed).
#
# Every arm trains for the same number of optimizer updates on the same number
# of samples; only the count of distinct samples differs. All arms share one
# baseline per seed, because they share an evaluation set, a calibration split,
# and the same leading training rows.
#
# Submit one cell:
#   sbatch --export=ALL,ARM=arm_a,SEED=11 scripts/jean_zay_compress.sbatch
#
# Submit a whole seed sweep as one array, five arms per seed:
#   sbatch --array=0-14%15 --export=ALL,SEEDS="44 55 66" \
#     scripts/jean_zay_compress.sbatch
#
# Submit every run in a manifest by index, for a campaign whose studies do not
# share one arm-by-seed shape:
#   sbatch --array=0-47%16 --export=ALL,BY_INDEX=1,\
#     CONFIG=configs/information_lever.yaml,\
#     MANIFEST=prepared/information-lever-manifest.jsonl \
#     scripts/jean_zay_compress.sbatch
#
# A full cell trains for about 53 minutes and then spends roughly 19 more on
# every rung it scores on the test set, so it does not fit the two-hour dev
# limit these defaults used to carry and every real submission had to override
# them on the command line.
#SBATCH --job-name=fqcomp
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
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
mkdir -p slurm_logs reports

arms=(arm_e arm_a arm_b arm_c arm_d)
if [[ -n "${BY_INDEX:-}" ]]; then
  ARM=""; SEED=""
elif [[ -z "${ARM:-}" || -z "${SEED:-}" ]]; then
  # Array form: walk arms within seeds, so a truncated array still leaves whole
  # seeds finished rather than a ragged grid.
  read -r -a seeds <<<"${SEEDS:-11 22 33}"
  task="${SLURM_ARRAY_TASK_ID:?set ARM and SEED, or submit as an array}"
  total=$((${#arms[@]} * ${#seeds[@]}))
  if ((task < 0 || task >= total)); then
    echo "array task $task is outside 0-$((total - 1))" >&2
    exit 2
  fi
  ARM="${arms[$((task % ${#arms[@]}))]}"
  SEED="${seeds[$((task / ${#arms[@]}))]}"
fi
arm="${ARM:-}"
seed="${SEED:-}"
config="${CONFIG:-configs/compressibility.yaml}"
manifest="${MANIFEST:-prepared/compressibility-manifest.jsonl}"

source "$repo_root/scripts/jean_zay_env.sh"
export PYTHONUNBUFFERED=1

echo "arm=$arm seed=$seed config=$config"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ -n "${BY_INDEX:-}" ]]; then
  index="${SLURM_ARRAY_TASK_ID:?BY_INDEX needs an array}"
  run_id="$("$PYTHON" - "$manifest" "$index" <<'PY'
import json, sys
manifest, index = sys.argv[1], int(sys.argv[2])
records = [json.loads(line) for line in open(manifest) if line.strip()]
if not 0 <= index < len(records):
    raise SystemExit(f"array task {index} is outside 0-{len(records) - 1}")
print(records[index]["run_id"])
PY
)"
else
  run_id="$("$PYTHON" - "$manifest" "$seed" "$arm" <<'PY'
import json, sys
manifest, seed, arm = sys.argv[1], int(sys.argv[2]), sys.argv[3]
for line in open(manifest):
    record = json.loads(line)
    if record["seed"] == seed and record["study"] == arm:
        print(record["run_id"])
        break
PY
PY
)"
fi
if [[ -z "$run_id" ]]; then
  echo "no run for arm=$arm seed=$seed index=${SLURM_ARRAY_TASK_ID:-} in $manifest" >&2
  exit 2
fi
echo "run_id=$run_id"

"$PYTHON" -m fineqcomp run --config "$config" --manifest "$manifest" \
  --runs-root runs --run-id "$run_id" --shards 1

echo "arm $arm seed $seed complete"
