#!/usr/bin/env bash
# Run the real two-row adapter, codec, and model-load smoke for one campaign.
#
# sbatch --export=ALL,CONFIG=configs/external_llama_panel.yaml,\
#   MANIFEST=prepared/external-llama-manifest.jsonl \
#   scripts/jean_zay_preflight.sbatch
#SBATCH --job-name=fqpreflight
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
mkdir -p slurm_logs prepared

config="${CONFIG:?set CONFIG}"
manifest="${MANIFEST:?set MANIFEST}"
report="${REPORT:-prepared/preflight.json}"
source "$repo_root/scripts/jean_zay_env.sh"
export PYTHONUNBUFFERED=1

smoke_args=()
if [[ -n "${MODEL:-}" ]]; then
  smoke_args+=(--model-smoke-model "$MODEL")
fi
if [[ -n "${DATASET:-}" ]]; then
  smoke_args+=(--model-smoke-dataset "$DATASET")
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "$PYTHON" -m fineqcomp preflight \
  --config "$config" \
  --manifest "$manifest" \
  --prepared-root prepared \
  --report "$report" \
  --shards 1 \
  --require-gpus \
  --require-dependencies \
  --gpu-count 1 \
  --min-gpu-memory-gib 75 \
  --model-smoke \
  --min-answer-retention "${MIN_ANSWER_RETENTION:-0.99}" \
  "${smoke_args[@]}"
