#!/usr/bin/env bash
# Train one fixed manifest run per H100 from an ordered run-ID list.
#SBATCH --job-name=fqtrain
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
source "$repo_root/scripts/jean_zay_env.sh"

config="${CONFIG:?set CONFIG}"
manifest="${MANIFEST:?set MANIFEST}"
run_ids="${RUN_IDS:?set RUN_IDS}"
run_root="${RUNS_ROOT:-runs}"
run_id="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$run_ids")"
if [[ -z "$run_id" ]]; then
  echo "no run ID for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi
mkdir -p "$run_root" slurm_logs

args=(
  -m fineqcomp run
  --config "$config"
  --manifest "$manifest"
  --prepared-root prepared
  --runs-root "$run_root"
  --run-id "$run_id"
)
if [[ -n "${PILOT_ROWS:-}" ]]; then
  args+=(--pilot-rows "$PILOT_ROWS")
fi
if [[ -n "${CODECS:-}" ]]; then
  read -r -a codecs <<< "$CODECS"
  args+=(--codecs "${codecs[@]}")
fi
exec "$PYTHON" "${args[@]}"
