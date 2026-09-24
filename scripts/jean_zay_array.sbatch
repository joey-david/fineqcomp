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
#SBATCH --output=.cache/slurm_logs/%x-%A_%a.out
#SBATCH --error=.cache/slurm_logs/%x-%A_%a.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
source "$repo_root/scripts/jean_zay_env.sh"

# Defaults are the original campaign, so an unparameterised submit behaves as
# it always did. A second panel passes its own config and manifest:
#
#   sbatch --array=0-8%9 --export=ALL,\
#     CONFIG=configs/transposed_receiver_panel.yaml,\
#     MANIFEST=.cache/prepared/transposed-manifest.jsonl,SHARDS=9 \
#     scripts/jean_zay_array.sbatch
config="${CONFIG:-configs/campaign.yaml}"
manifest="${MANIFEST:-.cache/prepared/manifest.jsonl}"
shards="${SHARDS:-16}"
args=(
  -m fineqcomp run
  --config "$config" --manifest "$manifest"
  --shard "$SLURM_ARRAY_TASK_ID" --shards "$shards"
)
# Jean-Zay keeps prepared corpora and finished runs at the repo root, not under
# .cache/, so a campaign launched here has to say where they are.
if [[ -n "${PREPARED_ROOT:-}" ]]; then
  args+=(--prepared-root "$PREPARED_ROOT")
fi
if [[ -n "${RUNS_ROOT:-}" ]]; then
  args+=(--runs-root "$RUNS_ROOT")
fi
# One rank at a time, for a study whose arms differ only by adapter.
if [[ -n "${ADAPTERS:-}" ]]; then
  read -r -a adapters <<< "$ADAPTERS"
  args+=(--adapters "${adapters[@]}")
fi
if [[ -n "${MODELS:-}" ]]; then
  read -r -a models <<< "$MODELS"
  args+=(--models "${models[@]}")
fi
# A config that declares more than one study can be launched a study at a time,
# which is what lets an expensive second tranche wait on the first one's result
# instead of being submitted alongside it. Unset, the shard runs the whole
# manifest exactly as before.
if [[ -n "${STUDIES:-}" ]]; then
  read -r -a studies <<< "$STUDIES"
  args+=(--studies "${studies[@]}")
fi
exec "$PYTHON" "${args[@]}"
