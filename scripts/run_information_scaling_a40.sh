#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

action=${1:-start}
mode=${2:-pilot}
host=${A40_HOST:-coktailjet}
remote_root=${REMOTE_REPO_ROOT:-/home/lamsade/jdavid/fineQComp}
config=${INFO_CONFIG:-configs/information_scaling.yaml}
remote_out=${INFO_REMOTE_OUT:-$remote_root/runs_information_scaling/$mode}
local_out=${INFO_LOCAL_OUT:-$repo_root/runs_information_scaling/$mode}
session=${INFO_SESSION:-fineqcomp-info-$mode}

if [[ "$action" == worker ]]; then
  gpu=${2:?missing GPU index}
  conditions=${3:?missing conditions}
  seeds=${4:?missing seeds}
  out=${5:?missing output path}
  status=${6:?missing status path}
  log=${7:?missing log path}
  python_bin=${FINEQCOMP_PYTHON:-$repo_root/../reasoning/.venv/bin/python}
  threshold=${FINEQCOMP_GPU_MEMORY_THRESHOLD_MB:-4096}
  mkdir -p "$out" "$(dirname "$status")" "$(dirname "$log")"
  while true; do
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    used=${used//[[:space:]]/}
    if [[ "$used" =~ ^[0-9]+$ ]] && ((used <= threshold)); then
      break
    fi
    printf 'GPU %s busy (%s MiB); waiting\n' "$gpu" "${used:-unknown}" >>"$log"
    sleep 60
  done
  if [[ -f .env ]]; then
    set -a
    source .env
    set +a
  fi
  export PYTHONPATH="${PYTHONPATH:-$repo_root/src}"
  export HF_HOME="${HF_HOME:-$repo_root/../fineQComp_hf_cache}"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  read -ra condition_args <<<"$conditions"
  read -ra seed_args <<<"$seeds"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m fineqcomp.information_scaling \
    --config "$config" --out "$out" \
    --conditions "${condition_args[@]}" --seeds "${seed_args[@]}" \
    >"$log" 2>&1
  rc=$?
  set -e
  tmp="$status.tmp.$$"
  printf '%s\n' "$rc" >"$tmp"
  mv "$tmp" "$status"
  exit "$rc"
fi

case "$mode" in
pilot)
  gpu0_conditions="random"
  gpu1_conditions="structured_p1"
  seeds="11"
  ;;
repeat)
  gpu0_conditions="random"
  gpu1_conditions="structured_p1"
  seeds="22 33"
  ;;
full)
  gpu0_conditions="random structured_p16"
  gpu1_conditions="structured_p1 structured_p4"
  seeds="11 22 33"
  ;;
*)
  echo "usage: $0 start|status|aggregate|fetch [pilot|repeat|full]" >&2
  exit 2
  ;;
esac

case "$action" in
start)
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" bash -s -- \
    "$remote_root" "$session" "$remote_out" "$gpu0_conditions" \
    "$gpu1_conditions" "$seeds" <<'REMOTE'
set -euo pipefail
repo_root=$1
session=$2
out=$3
gpu0_conditions=$4
gpu1_conditions=$5
seeds=$6
cd "$repo_root"
python_bin=${FINEQCOMP_PYTHON:-$repo_root/../reasoning/.venv/bin/python}
export PYTHONPATH="${PYTHONPATH:-$repo_root/src}"
"$python_bin" -c 'import fineqcomp, torch; assert torch.cuda.device_count() >= 2'
if tmux has-session -t "$session" 2>/dev/null; then
  echo "session already exists: $session" >&2
  exit 2
fi
mkdir -p "$out/logs"
rm -f "$out/logs/gpu0.status" "$out/logs/gpu1.status"
tmux new-session -d -s "$session" -n gpu0 -c "$repo_root" \
  bash scripts/run_information_scaling_a40.sh worker 0 \
  "$gpu0_conditions" "$seeds" "$out" "$out/logs/gpu0.status" \
  "$out/logs/gpu0.log"
tmux set-option -t "$session" remain-on-exit on
tmux new-window -d -t "$session:" -n gpu1 -c "$repo_root" \
  bash scripts/run_information_scaling_a40.sh worker 1 \
  "$gpu1_conditions" "$seeds" "$out" "$out/logs/gpu1.status" \
  "$out/logs/gpu1.log"
echo "started $session"
REMOTE
  ;;
status)
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" bash -s -- \
    "$session" "$remote_out" <<'REMOTE'
set -euo pipefail
session=$1
out=$2
tmux list-windows -t "$session" -F '#{window_name} #{pane_dead} #{pane_pid}' 2>/dev/null || true
for gpu in 0 1; do
  printf '\n== gpu%s ==\n' "$gpu"
  [[ -f "$out/logs/gpu${gpu}.status" ]] && printf 'exit: %s\n' "$(<"$out/logs/gpu${gpu}.status")" || printf 'exit: running\n'
  tail -n 12 "$out/logs/gpu${gpu}.log" 2>/dev/null || true
done
REMOTE
  ;;
fetch)
  mkdir -p "$local_out"
  rsync -avz --exclude '*.pt' --exclude '*.fqcb' \
    "$host:$remote_out/" "$local_out/"
  ;;
aggregate)
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" bash -s -- \
    "$remote_root" "$remote_out" <<'REMOTE'
set -euo pipefail
repo_root=$1
out=$2
cd "$repo_root"
python_bin=${FINEQCOMP_PYTHON:-$repo_root/../reasoning/.venv/bin/python}
export PYTHONPATH="${PYTHONPATH:-$repo_root/src}"
"$python_bin" -m fineqcomp.information_scaling --out "$out" --aggregate
REMOTE
  ;;
*)
  echo "usage: $0 start|status|aggregate|fetch [pilot|repeat|full]" >&2
  exit 2
  ;;
esac
