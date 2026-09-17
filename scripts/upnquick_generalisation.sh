#!/usr/bin/env bash
# The generalisation study on one upnquick GPU, as a queue of train-then-sweep
# stages. Run once per GPU with a different queue:
#
#   GPU=0 QUEUE="cot_arithmetic cot_shuffled" scripts/upnquick_generalisation.sh
#   GPU=1 QUEUE="smoke cls_flipped summary_mismatched cot_permuted" scripts/...
#
# Each arm is trained, then swept, before the next arm starts, so a queue cut
# short leaves finished arms rather than a pile of unscored adapters. Training
# skips an adapter that exists and every sweep cell is cached, so rerunning a
# queue resumes it.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
PY="$root/.venv/bin/python"
export HF_HOME="${HF_HOME:-/home/lamsade/jdavid/fineQComp_hf_cache}"
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU:?set GPU}"
STUDY=configs/upnquick/generalisation.yaml
OUT=reports/generalisation_upnquick
LOG="$root/remote_logs/generalisation_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p remote_logs "$OUT"
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }
stage() {
  local name="$1"; shift; say "START $name"; local t0; t0=$(date +%s)
  if "$@" >>"$LOG" 2>&1; then say "OK    $name ($(( $(date +%s) - t0 ))s)"; return 0; fi
  say "FAIL  $name ($(( $(date +%s) - t0 ))s)"; return 1
}
run_id_for() {
  "$PY" - "$1" "$2" <<'PYX'
import sys
from fineqcomp.campaign import expand_campaign
from fineqcomp.config import load_campaign
runs = [r for r in expand_campaign(load_campaign(sys.argv[1])) if r.study == sys.argv[2]]
assert len(runs) == 1, runs
print(runs[0].run_id)
PYX
}
arm_field() {  # arm field -> value from the study config
  "$PY" -c "import yaml,sys; print(yaml.safe_load(open('$STUDY'))['arms'][sys.argv[1]][sys.argv[2]])" "$1" "$2"
}

say "=== generalisation, GPU$GPU, queue: $QUEUE ==="
for arm in $QUEUE; do
  if [[ "$arm" == smoke ]]; then
    rm -rf reports/generalisation_smoke
    stage "smoke lock" "$PY" -m fineqcomp.generalisation --config configs/upnquick/generalisation_smoke.yaml \
      --out reports/generalisation_smoke --phase prepare || exit 1
    stage "smoke sweep (cot_permuted)" "$PY" -m fineqcomp.generalisation \
      --config configs/upnquick/generalisation_smoke.yaml --out reports/generalisation_smoke \
      --phase sweep --arm cot_permuted || exit 1
    continue
  fi
  config=$(arm_field "$arm" config); study=$(arm_field "$arm" study)
  id=$(run_id_for "$config" "$study") || { say "FAIL resolve $arm"; continue; }
  if [[ -f "runs/$id/raw_channel.pt" ]]; then
    say "SKIP  train $arm (adapter exists)"
  else
    base=$(basename "$config" .yaml)
    stage "train $arm" "$PY" -m fineqcomp run --config "$config" \
      --manifest "prepared/upnquick-$base-manifest.jsonl" --run-id "$id" || continue
  fi
  stage "sweep $arm" "$PY" -m fineqcomp.generalisation --config "$STUDY" --out "$OUT" \
    --phase sweep --arm "$arm"
  "$PY" -m fineqcomp.generalisation --config "$STUDY" --out "$OUT" --phase report >>"$LOG" 2>&1
done
say "=== GPU$GPU queue done ==="
