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
export PYTHONPATH="$repo_root/src"
export HF_HOME="${HF_HOME:-$repo_root/../fineQComp_hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

coord="$repo_root/remote_logs/$launch_id"
mkdir -p "$coord"

# These can be overridden from the environment.
gpu_memory_threshold_mb="${FINEQCOMP_GPU_MEMORY_THRESHOLD_MB:-4096}"
gpu_wait_timeout="${FINEQCOMP_GPU_WAIT_TIMEOUT_SECONDS:-86400}"            # 24 h
pilot_barrier_timeout="${FINEQCOMP_PILOT_BARRIER_TIMEOUT_SECONDS:-108000}" # 30 h
full_barrier_timeout="${FINEQCOMP_FULL_BARRIER_TIMEOUT_SECONDS:-604800}"   # 7 d

pilot_status_file="$coord/pilot_worker${worker}.status"
pilot_done_file="$coord/pilot_worker${worker}.done"
full_status_file="$coord/full_worker${worker}.status"
full_done_file="$coord/full_worker${worker}.done"

write_atomic() {
  local path="$1"
  local value="$2"
  local tmp="${path}.tmp.$$"

  printf '%s\n' "$value" >"$tmp"
  mv "$tmp" "$path"
}

mark_done() {
  local path="$1"
  local tmp="${path}.tmp.$$"

  : >"$tmp"
  mv "$tmp" "$path"
}

wait_for_gpu() {
  local started="$SECONDS"
  local used

  while true; do
    used="$(
      nvidia-smi \
        -i "$gpu" \
        --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null
    )"

    # Strip whitespace.
    used="${used//[[:space:]]/}"

    if [[ "$used" =~ ^[0-9]+$ ]] && ((used <= gpu_memory_threshold_mb)); then
      echo "worker $worker: GPU $gpu available (${used} MiB used)"
      return 0
    fi

    if ((SECONDS - started >= gpu_wait_timeout)); then
      echo \
        "worker $worker: timed out waiting for GPU $gpu " \
        "(${used:-unknown} MiB used)" >&2
      return 124
    fi

    echo \
      "worker $worker: GPU $gpu busy (${used:-unknown} MiB used); " \
      "waiting..."
    sleep 60
  done
}

wait_for_markers() {
  local kind="$1"
  local timeout="$2"
  local started="$SECONDS"
  local count
  local missing

  while true; do
    count="$(
      find "$coord" \
        -maxdepth 1 \
        -type f \
        -name "${kind}_worker[0-3].done" |
        wc -l
    )"

    if ((count >= 4)); then
      return 0
    fi

    if ((SECONDS - started >= timeout)); then
      missing=""
      for index in 0 1 2 3; do
        if [[ ! -f "$coord/${kind}_worker${index}.done" ]]; then
          missing="${missing} ${index}"
        fi
      done

      echo \
        "worker $worker: ${kind} barrier timed out; " \
        "missing workers:${missing:- none}" >&2
      return 124
    fi

    sleep 20
  done
}

all_statuses_zero() {
  local kind="$1"
  local index
  local status

  for index in 0 1 2 3; do
    if [[ ! -f "$coord/${kind}_worker${index}.status" ]]; then
      echo \
        "worker $worker: missing ${kind} status for worker $index" >&2
      return 1
    fi

    status="$(<"$coord/${kind}_worker${index}.status")"

    if [[ "$status" != "0" ]]; then
      echo \
        "worker $worker: ${kind} worker $index failed " \
        "with status $status" >&2
      return 1
    fi
  done

  return 0
}

# If the shell exits unexpectedly, publish a failure marker so the other
# workers do not wait forever. SIGKILL / host failure still requires the
# barrier timeout as a backstop.
phase="prepilot"

cleanup() {
  local status=$?

  case "$phase" in
  prepilot | pilot)
    if [[ ! -f "$pilot_done_file" ]]; then
      write_atomic "$pilot_status_file" "$status"
      mark_done "$pilot_done_file"
    fi
    ;;
  full)
    if [[ ! -f "$full_done_file" ]]; then
      write_atomic "$full_status_file" "$status"
      mark_done "$full_done_file"
    fi
    ;;
  esac
}

trap cleanup EXIT

#
# Pilot
#

phase="pilot"

wait_for_gpu
gpu_status=$?

if ((gpu_status != 0)); then
  pilot_status="$gpu_status"
  write_atomic "$pilot_status_file" "$pilot_status"
  mark_done "$pilot_done_file"
else
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m fineqcomp run \
    --models "$pilot_model" \
    --datasets "$pilot_dataset" \
    --limit 1 \
    --pilot-rows 4 \
    --codecs fp16 uniform2 loraquant_2_08 \
    --runs-root "$repo_root/pilot_runs/$launch_id" \
    2>&1 | tee "$coord/pilot_worker${worker}.log"

  pilot_status="${PIPESTATUS[0]}"
  write_atomic "$pilot_status_file" "$pilot_status"
  mark_done "$pilot_done_file"
fi

# Every worker publishes its result first, then waits for all four pilots.
if ! wait_for_markers "pilot" "$pilot_barrier_timeout"; then
  echo "worker $worker: refusing to start full campaign after pilot barrier timeout" >&2
  exit 124
fi

# A failed smoke test is a gate. Do not knowingly launch the expensive run.
if ! all_statuses_zero "pilot"; then
  echo "worker $worker: at least one pilot failed; full campaign will not start" >&2
  exit 1
fi

#
# Full campaign
#

phase="full"

# The device was free for the pilot, but another process could theoretically
# have claimed it while this worker waited at the pilot barrier.
if ! wait_for_gpu; then
  full_status=$?
  write_atomic "$full_status_file" "$full_status"
  mark_done "$full_done_file"
  exit "$full_status"
fi

CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m fineqcomp run \
  --worker "$worker" \
  --worker-weights 3 3 1 1 \
  --runs-root "$repo_root/runs_literature" \
  2>&1 | tee "$coord/full_worker${worker}.log"

full_status="${PIPESTATUS[0]}"
write_atomic "$full_status_file" "$full_status"
mark_done "$full_done_file"

#
# Analysis
#

if [[ "$worker" == "0" ]]; then
  if wait_for_markers "full" "$full_barrier_timeout"; then
    if ! all_statuses_zero "full"; then
      echo \
        "worker 0: one or more full workers failed; " \
        "running analysis on available results anyway" >&2
    fi

    "$python_bin" -m fineqcomp analyze \
      --root "$repo_root/runs_literature" \
      --out "$repo_root/reports_literature" \
      2>&1 | tee "$coord/analysis.log"

    analysis_status="${PIPESTATUS[0]}"

    if ((full_status == 0 && analysis_status != 0)); then
      full_status="$analysis_status"
    fi
  else
    echo \
      "worker 0: full-worker barrier timed out; " \
      "skipping analysis" >&2

    if ((full_status == 0)); then
      full_status=124
    fi
  fi
fi

phase="done"
trap - EXIT

exit "$full_status"
