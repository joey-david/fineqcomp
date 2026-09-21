#!/usr/bin/env bash
#SBATCH --job-name=fq-func-spectrum
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
mkdir -p slurm_logs
source "$repo_root/scripts/jean_zay_env.sh"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

if [[ -n "${GATE_SUMMARY:-}" ]]; then
  "$PYTHON" - "$GATE_SUMMARY" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
scores = summary["scores"]
if summary.get("status") != "complete" or summary.get("surface_cells") != 25:
    raise SystemExit("functional-spectrum pilot did not complete all 25 cells")
if scores["signed"]["rmse"] >= scores["energy"]["rmse"]:
    raise SystemExit("signed spectrum did not beat the energy baseline")
print("pilot gate passed: signed RMSE", scores["signed"]["rmse"],
      "< energy RMSE", scores["energy"]["rmse"])
PY
fi

args=(
  --config "${CONFIG:-configs/jean_zay/functional_spectrum.yaml}"
  --out "${OUT:-reports/functional_spectrum_v1}"
  --experiment functional_spectrum
)
if [[ -n "${SOURCE_RUN:-}" ]]; then
  args+=(--source-run "$SOURCE_RUN")
fi
if [[ "${SMOKE:-0}" == 1 ]]; then
  args=(
    --config configs/jean_zay/functional_spectrum.yaml
    --out reports/functional_spectrum_smoke
    --experiment functional_spectrum
    --smoke
  )
fi
"$PYTHON" -m fineqcomp.cot_verbosity_mechanism "${args[@]}"
