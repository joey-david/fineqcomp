#!/usr/bin/env bash
# Detailed live view of your Jean-Zay jobs. Read-only; prints once and exits.
#
#   scripts/monitor.sh              # newest batch
#   scripts/monitor.sh fqcomp       # one job name
#   scripts/monitor.sh --all        # all batches
#   watch -n 60 scripts/monitor.sh  # refresh in place
#
# The newest batch is the set of jobs with the latest Slurm submission time.
# Cell status reads run in parallel and are printed in stable order.

set -uo pipefail
usage() {
  printf 'usage: %s [--all] [job-name]\n' "$0" >&2
}

show_all=0
name=""
for arg in "$@"; do
  case "$arg" in
    --all) show_all=1 ;;
    --help|-h) usage; exit 0 ;;
    --*) usage; exit 2 ;;
    *)
      [[ -z "$name" ]] || { usage; exit 2; }
      name="$arg"
      ;;
  esac
done
host="${JZ_HOST:-jean-zay}"
root="${JZ_ROOT:-/lustre/fswork/projects/rech/fas/uul94gf/fineQComp}"

root_q="$(printf '%q' "$root")"
name_q="$(printf '%q' "$name")"
all_q="$(printf '%q' "$show_all")"

ssh -o BatchMode=yes -o ConnectTimeout=25 "$host" \
  "bash -s -- $root_q $name_q $all_q" <<'REMOTE'
set -uo pipefail
root="$1"
filter="$2"
show_all="$3"
cd "$root" 2>/dev/null || exit 1

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/fineqcomp-monitor.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT
queue_file="$tmp_dir/queue"
history_file="$tmp_dir/history"

squeue -u "$USER" -h -o '%i|%j|%T|%M|%L|%R|%V' >"$queue_file" 2>/dev/null &
queue_pid=$!
sacct_args=(-u "$USER" -X -n -P -o JobID,JobName,State,Elapsed,ExitCode,Submit)
[[ "$show_all" == 1 ]] || sacct_args+=(-S now-12hours)
sacct "${sacct_args[@]}" >"$history_file" 2>/dev/null &
history_pid=$!
query_status=0
wait "$queue_pid" || query_status=1
wait "$history_pid" || query_status=1

if [[ -n "$filter" ]]; then
  awk -F'|' -v f="$filter" 'index($2, f)' "$queue_file" >"$tmp_dir/q"
  awk -F'|' -v f="$filter" 'index($2, f)' "$history_file" >"$tmp_dir/h"
  mv "$tmp_dir/q" "$queue_file"
  mv "$tmp_dir/h" "$history_file"
fi

latest_submit=""
if [[ "$show_all" != 1 ]]; then
  latest_submit="$(
    { awk -F'|' 'NF >= 7 && $7 != "" {print $7}' "$queue_file"
      awk -F'|' 'NF >= 6 && $6 != "" {print $6}' "$history_file"; } |
    LC_ALL=C sort -r | head -1
  )"
fi
latest_epoch=-1
if [[ -n "$latest_submit" ]]; then
  latest_epoch="$(python3 - "$latest_submit" <<'PY' 2>/dev/null || true
from datetime import datetime
import sys
try:
    print(datetime.fromisoformat(sys.argv[1]).timestamp())
except ValueError:
    pass
PY
  )"
  [[ -n "$latest_epoch" ]] || latest_epoch=-1
fi

printf '\n== queue ==\n'
printf '   %-10s %-8s %-9s %-7s %-7s %s\n' ID NAME STATE USED LIMIT REASON
awk -F'|' -v all="$show_all" -v latest="$latest_submit" '
  all || $7 == latest {
    printf "   %-10s %-8s %-9s %-7s %-7s %s\n", $1, $2, $3, $4, $5, $6
    if ($3 == "RUNNING") running++
    if ($3 == "PENDING") pending++
    shown++
  }
  END {
    if (!shown) print "   (none)"
    printf "   %d shown: %d running, %d pending\n", shown + 0, running + 0, pending + 0
  }
' "$queue_file"

printf '\n== cells ==\n'
printf '   %-26s %-22s %8s %s\n' CELL TRAINING CODECS STATE
cell_tmp="$tmp_dir/cells"
mkdir -p "$cell_tmp"
cell_index=0
cell_pids=()
for d in runs/*/; do
  (
  [ -d "$d" ] || exit 0
  [ -f "$d/config.json" ] || exit 0
  b="$(basename "$d")"
  [[ "$b" == baselines ]] && exit 0
  created="$(stat -c %Y "$d/config.json" 2>/dev/null || echo 0)"
  if [[ "$latest_epoch" != -1 ]] && ! awk -v a="$created" -v b="$latest_epoch" 'BEGIN {exit !(a >= b)}'; then
    exit 0
  fi
  short="$(echo "$b" | sed -E 's/__mistral-7b-base|__qwen25-7b-base|__all-linear-r16//g; s/__[a-f0-9]{10}$//')"
  state="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state","?"))' "$d/status.json" 2>/dev/null || echo -)"
  codecs="$(find "$d/codec_metrics" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')"
  want="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["codecs"]))' "$d/config.json" 2>/dev/null || echo ?)"
  bar="-"
  if [[ -f "$d/logs/training.jsonl" ]]; then
    bar="$(python3 - "$d" <<'PY' 2>/dev/null || echo -
import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
rows = [json.loads(line) for line in (p / "logs" / "training.jsonl").read_text().splitlines() if line.strip()]
if rows:
    last = rows[-1]
    prefix = "[############]" if (p / "training_metrics.json").exists() else "[training...]"
    print(f"{prefix} {last['updates']}u nll={last['validation_nll']:.4f}")
PY
    )"
  fi
  printf '   %-26s %-22s %4s/%-3s %s\n' "$short" "$bar" "$codecs" "$want" "$state"
  ) >"$cell_tmp/$cell_index" 2>/dev/null &
  cell_pids+=("$!")
  cell_index=$((cell_index + 1))
done
for pid in "${cell_pids[@]}"; do
  wait "$pid" || query_status=1
done
if ((cell_index == 0)); then
  printf '   (none)\n'
else
  cat "$cell_tmp"/* | LC_ALL=C sort
fi

printf '\n== recent outcomes ==\n'
if [[ "$show_all" == 1 ]]; then
  awk -F'|' '{printf "   %-14s %-10s %-12s %-9s %s\n", $1, $2, $3, $4, $5}' "$history_file"
else
  awk -F'|' -v latest="$latest_submit" '$6 == latest {
    printf "   %-14s %-10s %-12s %-9s %s\n", $1, $2, $3, $4, $5
  }' "$history_file"
fi
[[ -s "$history_file" ]] || printf '   (none)\n'

printf '\n== crashes ==\n'
found=0
for f in $(find slurm_logs -type f -name "${filter:-*}-*.err" -printf '%T@ %p\n' 2>/dev/null | awk -v all="$show_all" -v cutoff="$latest_epoch" 'all || cutoff < 0 || $1 >= cutoff {sub(/^[^ ]+ /, ""); print}'); do
  if grep -qE 'Traceback|CUDA out of memory|DUE TO TIME LIMIT' "$f" 2>/dev/null; then
    printf '   %s\n' "$f"
    grep -E 'Error|Traceback|DUE TO TIME' "$f" | tail -2 | sed 's/^/      /'
    found=1
  fi
done
[[ $found -eq 0 ]] && printf '   none\n'
printf '\n'
exit "$query_status"
REMOTE
