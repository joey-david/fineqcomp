#!/usr/bin/env bash
# One stage of the reparameterization ablation. Only reparameterization.sh
# submits this file: it passes the account, partition, QoS, time, logs and
# STAGE, FQ_REPO and FQ_STATE. The stages, in dependency order:
#
#   prepare  (prepost) project venv, pinned datasets, dependency preflight
#   smoke    (GPU, dev) 4-row pilot of every receiver, then the gauge sweep
#            and the attenuation probe on one pilot adapter; any failure
#            cancels everything below it
#   train    (GPU array) one manifest run per task, from lists/$LIST
#   sweep    (GPU array) attenuation probe and the full gauge sweep, one
#            seed-11 default adapter per task, from lists/$LIST
#   profile  (GPU) attenuation probe of every init-ablation adapter
#   collect  (prepost) tables, summary and the archive to send back
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
set -euo pipefail

repo_root="${FQ_REPO:?submit with scripts/jean_zay/reparameterization.sh}"
state="${FQ_STATE:?submit with scripts/jean_zay/reparameterization.sh}"
stage="${STAGE:?submit with scripts/jean_zay/reparameterization.sh}"
cd "$repo_root"
# Nothing from the submitting shell's Python or Hugging Face setup may leak
# into the job: a personal HF_HOME can be read-only or hold other revisions.
unset PYTHONPATH PYTHONHOME HF_DATASETS_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HF_HOME="$repo_root/.hf_cache"
config="${FQ_CONFIG:-configs/rate/reparameterization.yaml}"
manifest="$state/manifest.jsonl"
prepared="$state/prepared"
runs="$state/runs"
lists="$state/lists"
profile_script=scripts/analysis/adapter_spectrum_profile.py
task="${SLURM_ARRAY_TASK_ID:-0}"

load_env() {
  # shellcheck disable=SC1090
  source "${FQ_ENV_SCRIPT:-scripts/jean_zay/env.sh}"
  set -euo pipefail
}
need_project_env() {
  "$PYTHON" -c 'import torch, transformers, peft, math_verify, rouge_score' || {
    echo "the project environment is incomplete; the prepare stage builds .venv" >&2
    exit 2
  }
}
helper() { "$PYTHON" scripts/jean_zay/reparameterization.py "$1" --state "$state" "${@:2}"; }
nth() { sed -n "$(($2 + 1))p" "$1"; }

echo "stage=$stage job=${SLURM_JOB_ID:-} task=$task host=$(hostname) start=$(date '+%F %T')"
case "$stage" in
prepare)
  export FINEQCOMP_ONLINE=1
  load_env
  # A .venv that cannot import the module's torch would shadow nothing and
  # break everything; set it aside rather than install into it.
  if [[ -e .venv ]] && ! .venv/bin/python -c 'import torch' >/dev/null 2>&1; then
    mv .venv ".venv.unusable.$(date +%s)"
  fi
  if [[ ! -x .venv/bin/python ]]; then
    "$PYTHON" -m venv --system-site-packages .venv
  fi
  .venv/bin/python -m pip install --no-cache-dir --disable-pip-version-check --quiet \
    'math-verify[antlr4_13_2]==0.9.0' 'rouge-score==0.1.2'
  load_env
  need_project_env
  "$PYTHON" -c 'import torch, transformers, peft, datasets, bitsandbytes
print("torch", torch.__version__, "transformers", transformers.__version__,
      "peft", peft.__version__, "datasets", datasets.__version__,
      "bitsandbytes", bitsandbytes.__version__)'
  "$PYTHON" -m fineqcomp prepare --config "$config" --manifest "$manifest" \
    --prepared-root "$prepared"
  "$PYTHON" -m fineqcomp preflight --config "$config" --manifest "$manifest" \
    --prepared-root "$prepared" --report "$state/preflight.json" \
    --require-dependencies --shards 1 >/dev/null
  helper data-check
  ;;
smoke)
  load_env
  need_project_env
  smoke="$state/smoke"
  rm -rf -- "${smoke:?}/runs" "$smoke"/*.csv "$smoke/PASSED"
  mkdir -p "$smoke/runs"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  while read -r run_id; do
    [[ -n "$run_id" ]] || continue
    echo "pilot $run_id"
    "$PYTHON" -m fineqcomp run --config "$config" --manifest "$manifest" \
      --prepared-root "$prepared" --runs-root "$smoke/runs" --run-id "$run_id" \
      --pilot-rows 4 --force
  done < "$lists/smoke.txt"
  probe="$(tail -n 1 "$lists/smoke.txt")"
  helper cells --run-id "$probe" > "$smoke/cell.csv"
  "$PYTHON" "$profile_script" "$smoke/cell.csv" "$smoke/runs" "$smoke/runs" \
    "$prepared/natural" "$smoke/attenuation" 0 1 attenuation 4 calibration
  "$PYTHON" "$profile_script" "$smoke/cell.csv" "$smoke/runs" "$smoke/runs" \
    "$prepared/natural" "$smoke/gauge" 0 1 gauge 4 calibration "$(cat "$lists/smoke_gauges.txt")"
  helper smoke-check
  ;;
train)
  load_env
  need_project_env
  run_id="$(nth "$lists/${LIST:?}" "$task")"
  [[ -n "$run_id" ]] || { echo "no run at line $((task + 1)) of $lists/$LIST" >&2; exit 2; }
  echo "run $run_id"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  "$PYTHON" -m fineqcomp run --config "$config" --manifest "$manifest" \
    --prepared-root "$prepared" --runs-root "$runs" --run-id "$run_id"
  ;;
sweep)
  load_env
  need_project_env
  run_id="$(nth "$lists/${LIST:?}" "$task")"
  [[ -n "$run_id" ]] || { echo "no run at line $((task + 1)) of $lists/$LIST" >&2; exit 2; }
  if [[ ! -f "$runs/$run_id/raw_channel.pt" ]]; then
    echo "no trained adapter for $run_id: its train task failed or was cancelled" >&2
    exit 3
  fi
  cell="$state/sweep/$run_id"
  mkdir -p "$cell"
  helper cells --run-id "$run_id" > "$cell/cell.csv"
  "$PYTHON" "$profile_script" "$cell/cell.csv" "$runs" "$runs" "$prepared/natural" \
    "$cell/attenuation" 0 1 attenuation 64 calibration
  "$PYTHON" "$profile_script" "$cell/cell.csv" "$runs" "$runs" "$prepared/natural" \
    "$cell/gauge" 0 1 gauge 256 calibration "$(cat "$lists/gauges.txt")"
  ;;
profile)
  load_env
  need_project_env
  mkdir -p "$state/profile"
  cells="$state/profile/cells_${SLURM_JOB_ID:-local}.csv"
  helper present-cells --list profile > "$cells"
  if [[ "$(wc -l < "$cells")" -le 1 ]]; then
    echo "no trained init-ablation adapter left to profile"
    exit 0
  fi
  "$PYTHON" "$profile_script" "$cells" "$runs" "$runs" "$prepared/natural" \
    "$state/profile/attenuation_${SLURM_JOB_ID:-local}" 0 1 attenuation 64 calibration
  ;;
collect)
  load_env
  helper collect
  ;;
*)
  echo "unknown STAGE=$stage" >&2
  exit 2
  ;;
esac
echo "stage=$stage done=$(date '+%F %T')"
