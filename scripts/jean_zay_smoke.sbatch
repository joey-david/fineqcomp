#!/usr/bin/env bash
#SBATCH --job-name=fineqcomp-smoke
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p slurm_logs

python_bin="${PYTHON:-.venv/bin/python}"
export HF_HOME="${HF_HOME:-${repo_root}/.hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

"$python_bin" -m fineqcomp prepare
"$python_bin" -m fineqcomp preflight \
  --require-gpus --tokenizers --model-smoke \
  --report prepared/smoke-preflight.json

run_id="$($python_bin -c 'import json; from pathlib import Path; rows=[json.loads(line) for line in Path("prepared/manifest.jsonl").read_text().splitlines()]; print(next(row["run_id"] for row in rows if row["study"] == "exact_seeded" and row["model"]["key"] == "qwen3_8b_base" and row["family_count"] == 4 and row["adapter"]["key"] == "seeded_last1_r4" and row["seed"] == 11))')"

"$python_bin" -m fineqcomp run --run-id "$run_id" --shard 0 --shards 2
"$python_bin" -m fineqcomp analyze --root runs --out reports
test -s "runs/$run_id/metrics.json"
test -s "runs/$run_id/codecs/adapter_b4.fqcb"
test -s "reports/summary.json"
printf 'smoke run complete: %s\n' "$run_id"
