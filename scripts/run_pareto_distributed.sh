#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_dir=${1:-}
if [[ -z "$run_dir" ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi
run_dir=$(realpath "$run_dir")
if [[ ! -f "$run_dir/raw_channel.pt" ]]; then
  echo "missing trained adapter: $run_dir/raw_channel.pt" >&2
  exit 2
fi

python_bin=${FINEQCOMP_PYTHON:-$repo_root/../reasoning/.venv/bin/python}
upnquick=${UPNQUICK_HOST:-upnquick}
ourasi=${OURASI_HOST:-ourasi}
out="$run_dir/pareto"
mkdir -p "$out"

# Stop a previous campaign if requested, but keep its completed artifacts: this
# sweep reuses raw_channel.pt and the already-measured baseline/raw/binary point.
if [[ -n "${OLD_SESSION:-}" ]]; then
  for host in "$upnquick" "$ourasi"; do
    echo "stopping $OLD_SESSION on $host"
    ssh "$host" "tmux kill-session -t '$OLD_SESSION' 2>/dev/null || true"
  done
fi

launch() {
  local host=$1 gpu=$2 bit=$3
  local worker_out="$out/worker_b$bit"
  local log="$out/worker_b$bit.log"
  local status="$out/worker_b$bit.status"
  rm -f "$status"
  mkdir -p "$worker_out"

  echo "launching ${bit}b on $host GPU $gpu"
  ssh "$host" bash -s -- \
    "$repo_root" "$python_bin" "$run_dir" "$worker_out" "$log" "$status" "$gpu" "$bit" <<'REMOTE'
set -euo pipefail
repo_root=$1
python_bin=$2
run_dir=$3
worker_out=$4
log=$5
status=$6
gpu=$7
bit=$8
mkdir -p "$worker_out" "$(dirname "$log")"
nohup bash -c '
  cd "$1"
  set +e
  CUDA_VISIBLE_DEVICES="$7" PYTHONPATH=src "$2" -m fineqcomp.pareto "$3" --bits "$8" --out "$4" >"$5" 2>&1
  rc=$?
  printf "%s\n" "$rc" >"$6"
  exit "$rc"
' _ "$repo_root" "$python_bin" "$run_dir" "$worker_out" "$log" "$status" "$gpu" "$bit" </dev/null >/dev/null 2>&1 &
REMOTE
}

# One full HumanEval pass per GPU. 1-bit is free: it is exactly the existing
# binary sign+mean-scale point and is reused during finalization.
launch "$upnquick" 0 2
launch "$upnquick" 1 3
launch "$ourasi" 0 4
launch "$ourasi" 1 8

bits=(2 3 4 8)
echo "workers launched; waiting for four bit-specific results"
while true; do
  finished=0
  for bit in "${bits[@]}"; do
    [[ -f "$out/worker_b$bit.status" ]] && finished=$((finished + 1))
  done
  printf '\rcompleted workers: %d/4' "$finished"
  [[ "$finished" -eq 4 ]] && break
  sleep 10
done
echo

failed=0
for bit in "${bits[@]}"; do
  rc=$(cat "$out/worker_b$bit.status")
  if [[ "$rc" != 0 ]]; then
    echo "${bit}b worker failed with exit $rc" >&2
    tail -n 40 "$out/worker_b$bit.log" >&2 || true
    failed=1
  fi
done
[[ "$failed" -eq 0 ]] || exit 1

# Merge only after all workers finish, so the four processes never race on the
# aggregate CSV/JSON/PNG.
for bit in "${bits[@]}"; do
  src="$out/worker_b$bit"
  cp "$src/midrise$bit.json" "$out/"
  cp "$src/predictions_midrise$bit.jsonl" "$out/"
  cp "$src/adapter_midrise$bit.fqpm" "$out/"
done

# Every requested point now exists, so this call only aggregates and plots.
PYTHONPATH=src "$python_bin" -m fineqcomp.pareto "$run_dir" \
  --bits 1 2 3 4 8 --out "$out"

echo "done: $out/pareto.png"
