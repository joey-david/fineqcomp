#!/usr/bin/env bash
set -u -o pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

gpu="${1:?missing GPU index}"
worker="${2:?missing worker index}"
launch_id="${3:?missing launch ID}"
pilot_model="${4:?missing pilot model}"
pilot_dataset="${5:?missing pilot dataset}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
python_bin="${FINEQCOMP_PYTHON:-$repo_root/../reasoning/.venv/bin/python}"
export PYTHONPATH="$repo_root/src:$repo_root/vendor"
export HF_HOME="${HF_HOME:-$repo_root/../fineQComp_hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
coord="$repo_root/remote_logs/$launch_id"
mkdir -p "$coord"

CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m fineqcomp run \
  --models "$pilot_model" --datasets "$pilot_dataset" --limit 1 \
  --pilot-rows 4 --codecs fp16 uniform2 loraquant_2_08 \
  --runs-root "$repo_root/pilot_runs/$launch_id" \
  2>&1 | tee "$coord/pilot_worker${worker}.log"
pilot_status="${PIPESTATUS[0]}"
printf '%s\n' "$pilot_status" >"$coord/pilot_worker${worker}.status"
touch "$coord/pilot_worker${worker}.done"

while [[ "$(find "$coord" -name 'pilot_worker*.done' | wc -l)" -lt 4 ]]; do
  sleep 20
done

CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m fineqcomp run \
  --worker "$worker" --worker-weights 3 3 1 1 \
  --runs-root "$repo_root/runs_literature" \
  2>&1 | tee "$coord/full_worker${worker}.log"
full_status="${PIPESTATUS[0]}"
printf '%s\n' "$full_status" >"$coord/full_worker${worker}.status"
touch "$coord/full_worker${worker}.done"

if [[ "$worker" == 0 ]]; then
  while [[ "$(find "$coord" -name 'full_worker*.done' | wc -l)" -lt 4 ]]; do
    sleep 30
  done
  "$python_bin" -m fineqcomp analyze \
    --root "$repo_root/runs_literature" \
    --out "$repo_root/reports_literature" \
    2>&1 | tee "$coord/analysis.log"
fi

exit "$full_status"
