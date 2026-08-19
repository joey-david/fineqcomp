#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

mode=${1:-pilot}
host=${A40_HOST:-coktailjet}
remote_root=${REMOTE_REPO_ROOT:-/home/lamsade/jdavid/fineQComp}
python_bin=${FINEQCOMP_PYTHON:-$remote_root/../reasoning/.venv/bin/python}
config=${INFO_CONFIG:-configs/information_scaling.yaml}
out=${INFO_OUT:-runs_information_scaling/$mode}
mkdir -p "$out"
out=$(realpath "$out")

case "$mode" in
pilot)
  gpu0_conditions=(random)
  gpu1_conditions=(structured_p1)
  seeds=(11)
  ;;
full)
  gpu0_conditions=(random structured_p16)
  gpu1_conditions=(structured_p1 structured_p4)
  seeds=(11 22 33)
  ;;
*)
  echo "usage: $0 [pilot|full]" >&2
  exit 2
  ;;
esac

status0="$out/gpu0.status"
status1="$out/gpu1.status"
log0="$out/gpu0.log"
log1="$out/gpu1.log"
rm -f "$status0" "$status1"

echo "host: $host (2x A40)"
echo "mode: $mode"
echo "output: $out"
echo "gpu0: ${gpu0_conditions[*]} seeds ${seeds[*]}"
echo "gpu1: ${gpu1_conditions[*]} seeds ${seeds[*]}"

ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" bash -s -- \
  "$remote_root" "$python_bin" "$config" "$out" "$log0" "$status0" \
  "$log1" "$status1" \
  "${gpu0_conditions[*]}" "${gpu1_conditions[*]}" "${seeds[*]}" <<'REMOTE'
set -euo pipefail
repo_root=$1
python_bin=$2
config=$3
out=$4
log0=$5
status0=$6
log1=$7
status1=$8
gpu0_text=$9
gpu1_text=${10}
seed_text=${11}

cd "$repo_root"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export PYTHONPATH="${PYTHONPATH:-$repo_root/src:$repo_root/vendor}"
export HF_HOME="${HF_HOME:-$repo_root/../fineQComp_hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
"$python_bin" -c 'import fineqcomp, torch, tqdm; assert torch.cuda.device_count() >= 2'

mkdir -p "$out"
read -ra gpu0_conditions <<<"$gpu0_text"
read -ra gpu1_conditions <<<"$gpu1_text"
read -ra seeds <<<"$seed_text"

launch_worker() {
  local gpu=$1 log=$2 status=$3
  shift 3
  local conditions=("$@")
  rm -f "$status"
  nohup bash -c '
    set +e
    repo_root=$1
    python_bin=$2
    config=$3
    out=$4
    log=$5
    status=$6
    gpu=$7
    seed_text=$8
    shift 8
    conditions=("$@")
    read -ra seeds <<<"$seed_text"
    cd "$repo_root"
    export PYTHONPATH="${PYTHONPATH:-$repo_root/src:$repo_root/vendor}"
    export HF_HOME="${HF_HOME:-$repo_root/../fineQComp_hf_cache}"
    export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m fineqcomp.information_scaling \
      --config "$config" --out "$out" \
      --conditions "${conditions[@]}" --seeds "${seeds[@]}" >"$log" 2>&1
    rc=$?
    printf "%s\n" "$rc" >"$status"
    exit "$rc"
  ' _ "$repo_root" "$python_bin" "$config" "$out" "$log" "$status" "$gpu" \
    "$seed_text" "${conditions[@]}" </dev/null >/dev/null 2>&1 &
}

launch_worker 0 "$log0" "$status0" "${gpu0_conditions[@]}"
launch_worker 1 "$log1" "$status1" "${gpu1_conditions[@]}"
REMOTE

echo "workers launched; watching status"
while true; do
  done0=0; done1=0
  [[ -f "$status0" ]] && done0=1
  [[ -f "$status1" ]] && done1=1
  printf '\rgpu0=%s gpu1=%s' "$([[ $done0 == 1 ]] && echo done || echo running)" \
    "$([[ $done1 == 1 ]] && echo done || echo running)"
  [[ $done0 == 1 && $done1 == 1 ]] && break
  sleep 10
done
echo

rc0=$(cat "$status0")
rc1=$(cat "$status1")
if [[ "$rc0" != 0 || "$rc1" != 0 ]]; then
  echo "worker failure: gpu0=$rc0 gpu1=$rc1" >&2
  echo "===== gpu0 tail =====" >&2
  tail -n 80 "$log0" >&2 || true
  echo "===== gpu1 tail =====" >&2
  tail -n 80 "$log1" >&2 || true
  exit 1
fi

echo "aggregating on $host"
ssh "$host" bash -s -- "$remote_root" "$python_bin" "$out" <<'REMOTE'
set -euo pipefail
repo_root=$1
python_bin=$2
out=$3
cd "$repo_root"
export PYTHONPATH="${PYTHONPATH:-$repo_root/src:$repo_root/vendor}"
"$python_bin" -m fineqcomp.information_scaling --out "$out" --aggregate
REMOTE

echo "done"
echo "  $out/prequential.csv"
echo "  $out/adapter_information.csv"
echo "  $out/source_bits_vs_prequential_bits.png"
echo "  $out/source_bits_vs_adapter_bits.png"
