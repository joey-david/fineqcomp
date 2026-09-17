#!/usr/bin/env bash
# The matched-bit-budget study on one A100, overnight, unattended.
#
# Order matters and is not arbitrary. Every sweep needs a trained adapter, and
# the rank-16 arm is the one that answers the headline question on its own --
# accuracy along a fixed `rank x bits` -- so it is trained and swept first. The
# from-scratch arms answer the second question (is a small update good because
# it was cut down, or would one that was never wide do as well?) and each of
# them is useless until its own training has finished, so they are interleaved
# train-then-sweep rather than all trained up front. If the night runs short,
# what survives is a smaller number of *complete* arms rather than a pile of
# adapters nobody scored.
#
# DEADLINE is the guard that makes this safe to leave. No new stage starts
# after it; the report runs on whatever finished. Every stage is resumable --
# training skips a finished run, and each score is cached on its own path --
# so a stage cut short resumes where it stopped on the next invocation rather
# than starting over.
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

STUDY="${STUDY:-configs/upnquick/matched_bit_budget.yaml}"
OUT="${OUT:-reports/matched_bit_budget_upnquick}"
TRACE_CONFIG=configs/upnquick/conditional_trace_rate.yaml
LOWRANK_CONFIG=configs/upnquick/low_rank_permuted.yaml
TRACE_MANIFEST=prepared/upnquick-trace-manifest.jsonl
LOWRANK_MANIFEST=prepared/upnquick-lowrank-manifest.jsonl
# Stop starting new work at this wall-clock time. The report needs minutes, not
# hours, but a sweep cell in flight can take one, so leave real margin.
DEADLINE="${DEADLINE:-$(date -d 'tomorrow 06:30' +%s 2>/dev/null || echo 0)}"
LOG="$root/remote_logs/overnight_$(date +%Y%m%d_%H%M%S).log"
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

# Run a stage, tee its output, and never let one failure kill the night: a
# broken arm should cost that arm, not the arms after it.
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
  "$PY" - "$1" "$2" "$3" "$4" <<'PY'
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
PY
}

cell_for() {  # arm -> its cell index in the prepared lock
  "$PY" - "$OUT/lock.json" "$1" <<'PY'
import json, sys
lock = json.load(open(sys.argv[1]))
for cell in lock["cells"]:
    if cell["arm"] == sys.argv[2]:
        print(cell["id"])
        break
else:
    raise SystemExit(f"no cell for arm {sys.argv[2]}")
PY
}

train_one() {  # config manifest study rank
  local config="$1" manifest="$2" study="$3" rank="$4"
  local id; id=$(run_id_for "$config" "$study" "$rank" 11) || return 1
  if [[ -f "runs/$id/raw_channel.pt" ]]; then
    say "SKIP  train r$rank (adapter exists)"
    return 0
  fi
  stage "train r$rank" "$PY" -m fineqcomp run \
    --config "$config" --manifest "$manifest" --run-id "$id"
}

sweep_one() {  # arm
  local arm="$1" cell
  cell=$(cell_for "$arm") || return 1
  if [[ -f "$OUT/qwen25_7b_base/$arm/seed11/complete.json" ]]; then
    say "SKIP  sweep $arm (complete)"
    return 0
  fi
  stage "sweep $arm (cell $cell)" "$PY" -m fineqcomp.matched_budget \
    --config "$STUDY" --out "$OUT" --phase sweep --cell "$cell"
}

say "=== matched bit budget, upnquick GPU${CUDA_VISIBLE_DEVICES}, deadline $(date -d "@$DEADLINE" 2>/dev/null || echo none) ==="
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available"' \
  || { say "ABORT: CUDA unavailable"; exit 2; }

# --- corpora and the frozen study ------------------------------------------
stage "prepare permuted corpus (rank-16 campaign)" \
  "$PY" -m fineqcomp prepare --config "$TRACE_CONFIG" --manifest "$TRACE_MANIFEST"
stage "prepare low-rank manifest" \
  "$PY" -m fineqcomp prepare --config "$LOWRANK_CONFIG" --manifest "$LOWRANK_MANIFEST"
stage "lock the study" \
  "$PY" -m fineqcomp.matched_budget --config "$STUDY" --out "$OUT" --phase prepare

# --- the headline arm -------------------------------------------------------
train_one "$TRACE_CONFIG" "$TRACE_MANIFEST" trace_permuted_full 16
# One timed pass before the grid, so the log records what this machine actually
# does per row rather than what anyone guessed it would do.
stage "throughput probe" "$PY" -m fineqcomp.matched_budget \
  --config "$STUDY" --out "$OUT" --phase probe --cell "$(cell_for permuted_r16)"
sweep_one permuted_r16

# --- the from-scratch arms, cheapest first ----------------------------------
# Ascending rank is ascending grid size, so if the night runs out it runs out on
# the most expensive arm rather than leaving four half-finished ones.
for rank in 1 2 4 8; do
  train_one "$LOWRANK_CONFIG" "$LOWRANK_MANIFEST" permuted_low_rank "$rank"
  sweep_one "permuted_r$rank"
done

# --- always report, deadline or not -----------------------------------------
say "START report"
"$PY" -m fineqcomp.matched_budget --config "$STUDY" --out "$OUT" --phase report \
  >>"$LOG" 2>&1 && say "OK    report" || say "FAIL  report"
say "=== done; summary at $OUT/summary.json, log at $LOG ==="
