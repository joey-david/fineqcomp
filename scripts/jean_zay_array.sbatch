#!/usr/bin/env bash
#SBATCH --job-name=fineqcomp-run
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --array=0-15%16
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=4-04:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
source "$repo_root/scripts/jean_zay_env.sh"

# Defaults are the original campaign, so an unparameterised submit behaves as
# it always did. A second panel passes its own config and manifest:
#
#   sbatch --array=0-8%9 --export=ALL,\
#     CONFIG=configs/transposed_receiver_panel.yaml,\
#     MANIFEST=prepared/transposed-manifest.jsonl,SHARDS=9 \
#     scripts/jean_zay_array.sbatch
config="${CONFIG:-configs/campaign.yaml}"
manifest="${MANIFEST:-prepared/manifest.jsonl}"
shards="${SHARDS:-16}"
args=(
  -m fineqcomp run
  --config "$config" --manifest "$manifest"
  --shard "$SLURM_ARRAY_TASK_ID" --shards "$shards"
)
if [[ -n "${MODELS:-}" ]]; then
  read -r -a models <<< "$MODELS"
  args+=(--models "${models[@]}")
fi
exec "$PYTHON" "${args[@]}"
