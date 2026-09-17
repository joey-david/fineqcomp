#!/usr/bin/env bash
# The 3B->27B ladder for the repair-headroom study, unattended, on GPU0 only.
#
# One model per process: vLLM does not release a model cleanly inside a live
# interpreter, so each rung is a fresh process that exits and frees the card.
#
# Ascending size. The small rungs are cheap and bank the lower half of the
# curve early; if anything goes wrong it goes wrong on the expensive end, with
# the rest of the ladder already on disk. DEADLINE stops new rungs starting.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

PY="$HOME/.venvs/qwen36-vllm/bin/python"
export CUDA_VISIBLE_DEVICES=0            # GPU1 belongs to someone else.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1                  # everything needed is already cached
export VLLM_LOGGING_LEVEL=ERROR
export TRANSFORMERS_VERBOSITY=error
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

OUT="${OUT:-reports/repair_headroom}"
ITEMS="${ITEMS:-100}"
# Extra flags for a variant arm (e.g. --enable-thinking). Thinking traces need a
# far larger token budget than a direct answer; 512 would cut most of them off
# mid-thought and score the truncation as a wrong answer.
EXTRA_ARGS="${EXTRA_ARGS:-}"
MAXNEW="${MAXNEW:-512}"
DEADLINE="${DEADLINE:-$(date -d 'today 18:30' +%s)}"
LOG="$root/remote_logs/repair_headroom_$(date +%Y%m%d_%H%M%S).log"
mkdir -p remote_logs "$OUT"

# Ascending parameter count. Two families are interleaved by size rather than
# run one after the other, so a short night still leaves both families with a
# low and a middle rung instead of one complete family and one empty.
MODELS="${MODELS:-Qwen/Qwen2.5-3B-Instruct Qwen/Qwen3-4B Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-8B Qwen/Qwen2.5-14B-Instruct Qwen/Qwen3.6-27B}"

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }
say "=== repair headroom ladder, GPU0, deadline $(date -d "@$DEADLINE" +%H:%M) ==="

for model in $MODELS; do
  slug="${model//\//__}"
  if [[ -f "$OUT/$slug/rows.json" ]]; then
    say "SKIP  $model (already done)"
    continue
  fi
  if (( $(date +%s) >= DEADLINE )); then
    say "DEADLINE reached; stopping before $model"
    break
  fi
  # The 27B is 52 GB in bf16 and needs most of the card; the small rungs leave
  # headroom for a longer context instead.
  case "$model" in
    *27B*) util=0.90; len=4096 ;;
    *14B*) util=0.88; len=6144 ;;
    *)     util=0.85; len=6144 ;;
  esac
  say "START $model (util=$util len=$len)"
  began=$(date +%s)
  if "$PY" scripts/repair_headroom.py --model "$model" --out "$OUT/$slug" \
        --items "$ITEMS" --gpu-memory-utilization "$util" \
        --max-new-tokens "$MAXNEW" \
        --max-model-len "$len" $EXTRA_ARGS >>"$LOG" 2>&1; then
    say "OK    $model ($(( $(date +%s) - began ))s)"
  else
    # One rung failing must not take the ladder down: a model that will not fit
    # or will not load is a missing point on the curve, not a dead run.
    say "FAIL  $model ($(( $(date +%s) - began ))s) -- see $LOG"
  fi
done
say "=== ladder done; results under $OUT ==="
