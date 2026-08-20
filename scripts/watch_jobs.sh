#!/usr/bin/env bash
# Wait for a set of Jean-Zay jobs to finish, then print what happened.
#
# Polls squeue at a low rate, stays silent while jobs run, and on completion
# writes the exit states, artifact counts and any failing logs to stdout and to
# slurm_logs/watch_jobs.log. It does not call Claude and does not notify
# anything: read the log when you want to know, or start a session yourself.
#
#   scripts/watch_jobs.sh fqcomp                   # watch one job name
#   scripts/watch_jobs.sh '' '' 600                # any job, 10-minute poll
#
# Runs in the foreground. To walk away:
#   nohup scripts/watch_jobs.sh fqcomp >/dev/null 2>&1 &
#   tail -f slurm_logs/watch_jobs.log

set -uo pipefail

name="${1:-}"                                   # job-name filter, empty = all
what="${2:-the Jean-Zay jobs}"                  # description passed to Claude
interval="${3:-120}"                            # seconds between polls
host="${JZ_HOST:-jean-zay}"
root="${JZ_ROOT:-/lustre/fswork/projects/rech/fas/uul94gf/fineQComp}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log="$repo/slurm_logs/watch_jobs.log"
mkdir -p "$(dirname "$log")"

filter=(-u '$USER' -h)
[[ -n "$name" ]] && filter+=(--name "$name")

say() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$log"; }

remaining() {
  ssh -o BatchMode=yes -o ConnectTimeout=25 "$host" \
    "squeue -u \$USER -h ${name:+--name $name} 2>/dev/null | wc -l" 2>/dev/null | tr -d ' '
}

say "watching ${name:-all jobs} on $host, polling every ${interval}s"
misses=0
while true; do
  left="$(remaining)"
  if [[ -z "$left" ]]; then
    # A dropped connection must not look like an empty queue.
    misses=$((misses + 1))
    say "squeue unreachable (${misses}/5)"
    [[ "$misses" -ge 5 ]] && { say "giving up on $host"; exit 1; }
    sleep "$interval"; continue
  fi
  misses=0
  [[ "$left" == "0" ]] && break
  sleep "$interval"
done

say "queue empty; collecting outcomes"
summary="$(ssh -o BatchMode=yes "$host" "bash -lc '
  cd $root
  sacct -u \$USER -S now-24hours -X -n -o JobID,JobName%10,State,Elapsed,ExitCode 2>/dev/null ${name:+| grep $name} | tail -30
  echo \"---artifacts---\"
  for d in runs/*/; do
    [ -d \"\$d\" ] || continue
    c=\$(ls \$d/codec_metrics 2>/dev/null | wc -l | tr -d \" \")
    [ \"\$c\" = \"0\" ] && [ ! -f \$d/raw_channel.pt ] && continue
    echo \"\$(basename \$d) codecs=\$c raw=\$([ -f \$d/raw_channel.pt ] && echo y || echo n)\"
  done | tail -20
  echo \"---failures---\"
  # Scoped to this job name and to logs touched in the last day. Grepping the
  # whole directory reported setup failures from three weeks earlier as if they
  # belonged to the run that just finished.
  find slurm_logs -name \"${name:-*}-*.err\" -mtime -1 2>/dev/null \
    | xargs -r grep -l -E \"Traceback|CUDA out of memory|DUE TO TIME LIMIT\" 2>/dev/null | tail -10
'" 2>/dev/null)"

printf '%s\n' "$summary" >> "$log"

failed=0
grep -qE '\b(FAILED|TIMEOUT|CANCELLED|OUT_OF_ME)' <<<"$summary" && failed=1
say "outcome: $([[ $failed -eq 1 ]] && echo 'some jobs did not succeed' || echo 'all jobs succeeded')"

say "summary written to $log"
