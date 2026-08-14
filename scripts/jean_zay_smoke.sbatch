#!/usr/bin/env bash
#SBATCH --job-name=fineqcomp-smoke
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"

source "$repo_root/scripts/jean_zay_env.sh"
python_bin="$PYTHON"

if [[ ! -s prepared/manifest.jsonl ]]; then
  echo 'missing prepared campaign data; stage it before submission' >&2
  exit 2
fi
"$python_bin" -m fineqcomp preflight \
  --require-gpus --tokenizers --model-smoke \
  --gpu-count 1 --min-gpu-memory-gib 75 --shards 16 \
  --report prepared/smoke-preflight.json

"$python_bin" -c 'import json; report=json.load(open("prepared/smoke-preflight.json")); assert report["model_smoke"]["training"]["optimizer_updates"] > 0; assert report["model_smoke"]["storage"]["file_bits"] > 0'
printf 'model, training, and codec smoke complete\n'
