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
out="$run_dir/mdl"
mkdir -p "$out"

launch() {
  local host=$1 gpu=$2 worker=$3
  shift 3
  local rates=("$@")
  local worker_out="$out/worker_$worker"
  local log="$out/worker_$worker.log"
  local status="$out/worker_$worker.status"
  rm -f "$status"
  mkdir -p "$worker_out"

  echo "launching MDL rates ${rates[*]} on $host GPU $gpu"
  ssh "$host" bash -s -- \
    "$repo_root" "$python_bin" "$run_dir" "$worker_out" "$log" "$status" "$gpu" "${rates[@]}" <<'REMOTE'
set -euo pipefail
repo_root=$1
python_bin=$2
run_dir=$3
worker_out=$4
log=$5
status=$6
gpu=$7
shift 7
rates=("$@")
mkdir -p "$worker_out" "$(dirname "$log")"
nohup bash -c '
  repo_root=$1
  python_bin=$2
  run_dir=$3
  worker_out=$4
  log=$5
  status=$6
  gpu=$7
  shift 7
  rates=("$@")
  cd "$repo_root"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src "$python_bin" -m fineqcomp.mdl "$run_dir" \
    --target-rates "${rates[@]}" --out "$worker_out" >"$log" 2>&1
  rc=$?
  printf "%s\n" "$rc" >"$status"
  exit "$rc"
' _ "$repo_root" "$python_bin" "$run_dir" "$worker_out" "$log" "$status" "$gpu" "${rates[@]}" \
  </dev/null >/dev/null 2>&1 &
REMOTE
}

# Eight description-length targets, two full task evaluations per GPU.
launch "$upnquick" 0 u0 0.15 0.30
launch "$upnquick" 1 u1 0.50 0.75
launch "$ourasi" 0 a0 1.00 1.50
launch "$ourasi" 1 a1 2.50 4.00

workers=(u0 u1 a0 a1)
echo "workers launched; waiting for four MDL workers"
while true; do
  finished=0
  for worker in "${workers[@]}"; do
    [[ -f "$out/worker_$worker.status" ]] && finished=$((finished + 1))
  done
  printf '\rcompleted workers: %d/4' "$finished"
  [[ "$finished" -eq 4 ]] && break
  sleep 10
done
echo

failed=0
for worker in "${workers[@]}"; do
  rc=$(cat "$out/worker_$worker.status")
  if [[ "$rc" != 0 ]]; then
    echo "MDL worker $worker failed with exit $rc" >&2
    tail -n 50 "$out/worker_$worker.log" >&2 || true
    failed=1
  fi
done
[[ "$failed" -eq 0 ]] || exit 1

for worker in "${workers[@]}"; do
  src="$out/worker_$worker"
  cp "$src"/mdl_*.json "$out/"
  cp "$src"/predictions_mdl_*.jsonl "$out/"
  cp "$src"/adapter_mdl_*.fqmdl "$out/"
done

# With all eight point files present this is aggregation/plotting only; it does
# not load the model or rerun evaluation.
PYTHONPATH=src "$python_bin" -m fineqcomp.mdl "$run_dir" \
  --target-rates 0.15 0.30 0.50 0.75 1.00 1.50 2.50 4.00 --out "$out"

echo "done: $out/mdl_pareto.png"
