#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

if [[ -n "${PYTHON:-}" ]]; then
  python_bin="$PYTHON"
elif [[ -x .venv/bin/python ]]; then
  python_bin=.venv/bin/python
else
  python_bin=python
fi
if [[ ! -x "$python_bin" ]]; then
  command -v "$python_bin" >/dev/null 2>&1 || {
    echo "missing $python_bin; run scripts/bootstrap.sh first" >&2
    exit 2
  }
fi
if ! import_error="$("$python_bin" -c 'import fineqcomp' 2>&1)"; then
  echo "cannot import fineqcomp with $python_bin" >&2
  printf '%s\n' "$import_error" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-${repo_root}/.hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p remote_logs

if [[ ! -s prepared/manifest.jsonl ]]; then
  echo 'missing prepared campaign data; run scripts/remote.sh push-prepared first' >&2
  exit 2
fi
"$python_bin" -m fineqcomp preflight \
  --require-gpus --tokenizers --model-smoke \
  --shards 2 \
  2>&1 | tee remote_logs/preflight.log

worker_pids=()
stop_workers() {
  trap - INT TERM
  local pid
  for pid in "${worker_pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  wait || true
  exit 130
}
trap stop_workers INT TERM

CUDA_VISIBLE_DEVICES=0 "$python_bin" -m fineqcomp screen \
  --shard 0 --shards 2 --out prepared/baseline_screening_gpu0.json \
  >remote_logs/baseline_screening_gpu0.log 2>&1 &
worker_pids+=("$!")
CUDA_VISIBLE_DEVICES=1 "$python_bin" -m fineqcomp screen \
  --shard 1 --shards 2 --out prepared/baseline_screening_gpu1.json \
  >remote_logs/baseline_screening_gpu1.log 2>&1 &
worker_pids+=("$!")
screen_status=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    screen_status=1
  fi
done
worker_pids=()
if ((screen_status != 0)); then
  echo "base-model screening failed; natural-task training was not started" >&2
  exit 1
fi

(
  set -o pipefail
  CUDA_VISIBLE_DEVICES=0 "$python_bin" -m fineqcomp run --shard 0 --shards 2 \
    2>&1 | tee -a remote_logs/worker0.log
) &
worker_pids+=("$!")
(
  set -o pipefail
  CUDA_VISIBLE_DEVICES=1 "$python_bin" -m fineqcomp run --shard 1 --shards 2 \
    2>&1 | tee -a remote_logs/worker1.log
) &
worker_pids+=("$!")

status=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

"$python_bin" -m fineqcomp analyze 2>&1 | tee -a remote_logs/analysis.log
if ((status != 0)); then
  echo "one or more workers failed; rerun this script to resume" >&2
fi
exit "$status"
