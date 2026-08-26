#!/usr/bin/env bash
# The free arm of the budget-matched rank test: sweep rank against rate on
# adapters that are already trained. No training, no new data -- each cell is a
# truncation, an encode, and a forward pass over 256 held-out rows.
#
# RUNS is a file with one run id per line. The array index picks the line, so
# one job per adapter and the array width is however many you want at once.
#
#   ls runs | grep 'all-linear-r16' | grep -E 'panel-math|kind-code' \
#     > prepared/rank-frontier-runs.txt
#   sbatch --array=0-11%12 --export=ALL,RUNS=prepared/rank-frontier-runs.txt,\
#     CONFIG=configs/campaign.yaml,OUT=reports/rank_frontier \
#     scripts/jean_zay_rank_frontier.sbatch
#SBATCH --job-name=fqrank
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
mkdir -p slurm_logs reports

runs_file="${RUNS:?set RUNS to a file of run ids}"
config="${CONFIG:-configs/campaign.yaml}"
out_root="${OUT:-reports/rank_frontier}"
index="${SLURM_ARRAY_TASK_ID:-0}"
run_id="$(sed -n "$((index + 1))p" "$runs_file")"
if [[ -z "$run_id" ]]; then
  echo "no run id at line $((index + 1)) of $runs_file" >&2
  exit 2
fi

source "$repo_root/scripts/jean_zay_env.sh"
export PYTHONUNBUFFERED=1

printf 'run=%s config=%s out=%s\n' "$run_id" "$config" "$out_root/$run_id"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "$PYTHON" -m fineqcomp rank-frontier \
  --run "runs/$run_id" \
  --config "$config" \
  --prepared-root "${PREPARED_ROOT:-prepared}" \
  --out "$out_root/$run_id" \
  --ranks ${RANKS:-1 2 4 8 16}
