#!/usr/bin/env bash
# The denoising-or-shrinkage control on one A100, overnight, unattended.
#
# Order is chosen so that whatever the night reaches is a finished experiment.
# The shrinkage control needs no training at all -- it re-codes and rescales an
# adapter that already exists -- and it is the comparison that could overturn
# the recorded mechanistic reading, so it runs first and to completion. Only
# then does the aligned arm train, because that arm adds a reference point
# rather than a refutation: without it the study still answers "is the recovery
# shrinkage?", and with it the study also answers "does it land where clean
# training lands?".
#
# DEADLINE is the guard that makes this safe to leave. No new stage starts after
# it. Every stage is resumable -- training skips a finished run, and each score
# and likelihood probe is cached on its own path -- so a stage cut short resumes
# where it stopped rather than starting over.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

PY="$root/.venv/bin/python"
export HF_HOME="${HF_HOME:-/home/lamsade/jdavid/fineQComp_hf_cache}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# GPU1 is somebody else's job. Never widen this without looking first.

STUDY="${STUDY:-configs/upnquick/denoise_vs_shrinkage.yaml}"
OUT="${OUT:-.cache/reports/denoise_vs_shrinkage_upnquick}"
ALIGNED_CONFIG=configs/upnquick/aligned_cot.yaml
ALIGNED_MANIFEST=.cache/prepared/upnquick-aligned-manifest.jsonl
TRACE_CONFIG=configs/upnquick/conditional_trace_rate.yaml
TRACE_MANIFEST=.cache/prepared/upnquick-trace-manifest.jsonl
DEADLINE="${DEADLINE:-$(date -d 'tomorrow 07:00' +%s 2>/dev/null || echo 0)}"
LOG="$root/remote_logs/denoise_$(date +%Y%m%d_%H%M%S).log"
mkdir -p remote_logs "$OUT"

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

past_deadline() {
  local now; now=$(date +%s)
  if (( DEADLINE > 0 && now >= DEADLINE )); then
    say "DEADLINE reached; starting no further work"
    return 0
  fi
  return 1
}

stage() {
  local name="$1"; shift
  past_deadline && return 1
  say "START $name"
  local began; began=$(date +%s)
  if "$@" >>"$LOG" 2>&1; then
    say "OK    $name ($(( $(date +%s) - began ))s)"
    return 0
  fi
  say "FAIL  $name ($(( $(date +%s) - began ))s) -- see $LOG"
  return 1
}

run_id_for() {  # config study rank seed -> the run id the campaign expands to
  "$PY" - "$1" "$2" "$3" "$4" <<'PYX'
import sys
from fineqcomp.campaign import expand_campaign
from fineqcomp.config import load_campaign
config, study, rank, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
for run in expand_campaign(load_campaign(config)):
    if run.study == study and run.adapter.rank == rank and run.seed == seed:
        print(run.run_id)
        break
else:
    raise SystemExit(f"no run for {study} rank {rank} seed {seed}")
PYX
}

say "=== denoise vs shrinkage, upnquick GPU${CUDA_VISIBLE_DEVICES}, deadline $(date -d "@$DEADLINE" 2>/dev/null || echo none) ==="
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available"' \
  || { say "ABORT: CUDA unavailable"; exit 2; }

# --- corpora and the frozen study ------------------------------------------
# The permuted corpus is what the study scores on; preparing it here is a no-op
# if the matched-budget night already wrote it, and the lock digests it either
# way so a changed corpus is a refusal rather than a quiet re-baseline.
stage "prepare permuted corpus" \
  "$PY" -m fineqcomp prepare --config "$TRACE_CONFIG" --manifest "$TRACE_MANIFEST"
stage "prepare aligned corpus" \
  "$PY" -m fineqcomp prepare --config "$ALIGNED_CONFIG" --manifest "$ALIGNED_MANIFEST"
stage "lock the study" \
  "$PY" -m fineqcomp.denoise_vs_shrinkage --config "$STUDY" --out "$OUT" --phase prepare
stage "check adapters" \
  "$PY" -m fineqcomp.denoise_vs_shrinkage --config "$STUDY" --out "$OUT" --phase check \
  || { say "ABORT: the damaged adapter this study reads is missing"; exit 3; }

# --- one timed pass before anything long -----------------------------------
stage "throughput probe" "$PY" -m fineqcomp.denoise_vs_shrinkage \
  --config "$STUDY" --out "$OUT" --phase probe

# --- the control that could overturn the claim ------------------------------
# The clean conditions are skipped automatically while their adapter is absent,
# so this pass scores every shrinkage and compression condition and stops.
stage "sweep (shrinkage control)" "$PY" -m fineqcomp.denoise_vs_shrinkage \
  --config "$STUDY" --out "$OUT" --phase sweep
stage "report (control only)" "$PY" -m fineqcomp.denoise_vs_shrinkage \
  --config "$STUDY" --out "$OUT" --phase report

# --- the reference point ----------------------------------------------------
aligned_id=$(run_id_for "$ALIGNED_CONFIG" trace_aligned_full 16 11) || aligned_id=""
if [[ -n "$aligned_id" && -f ".cache/runs/$aligned_id/raw_channel.pt" ]]; then
  say "SKIP  train aligned r16 (adapter exists)"
else
  stage "train aligned r16" "$PY" -m fineqcomp run \
    --config "$ALIGNED_CONFIG" --manifest "$ALIGNED_MANIFEST" --run-id "$aligned_id"
fi
stage "sweep (clean arm)" "$PY" -m fineqcomp.denoise_vs_shrinkage \
  --config "$STUDY" --out "$OUT" --phase sweep

# --- always report, deadline or not -----------------------------------------
say "START report"
"$PY" -m fineqcomp.denoise_vs_shrinkage --config "$STUDY" --out "$OUT" --phase report \
  >>"$LOG" 2>&1 && say "OK    report" || say "FAIL  report"
say "=== done; summary at $OUT/summary.json, log at $LOG ==="
