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

run_id="$($python_bin -c 'import json; from pathlib import Path; rows=[json.loads(line) for line in Path("prepared/manifest.jsonl").read_text().splitlines()]; print(next(row["run_id"] for row in rows if row["study"] == "exact_seeded" and row["model"]["key"] == "qwen3_8b_base" and row["family_count"] == 4 and row["adapter"]["key"] == "seeded_last1_r4" and row["seed"] == 11))')"

"$python_bin" -m fineqcomp run --run-id "$run_id" --shard 0 --shards 16
"$python_bin" -m fineqcomp analyze --root runs --out reports
test -s "runs/$run_id/metrics.json"
test -s "runs/$run_id/codecs/adapter_b4.fqcb"
test -s "reports/summary.json"
printf 'smoke run complete: %s\n' "$run_id"
