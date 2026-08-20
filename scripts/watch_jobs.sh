#!/usr/bin/env bash
# Wait for a set of Jean-Zay jobs to finish, then hand the outcome to Claude.
#
# Polls squeue at a low rate, stays silent while jobs run, and on completion
# calls `claude -p` once with the exit states and a pointer at the artifacts.
# Cheaper than an in-session monitor: no tokens are spent until there is
# something to report.
#
#   scripts/watch_jobs.sh fqcomp "the compressibility smoke test"
#   scripts/watch_jobs.sh '' "everything" 600      # any job, 10-minute poll
#
# Runs in the foreground. To walk away:
#   nohup scripts/watch_jobs.sh fqcomp "the smoke test" >/dev/null 2>&1 &
#
# The Claude it starts inherits no terminal, so it runs with
# --permission-mode auto and is told not to launch jobs on its own. Set
# JZ_CLAUDE_MODE=manual if you would rather it stop and wait for you.

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
  grep -l -E \"Traceback|CUDA out of memory|DUE TO TIME LIMIT\" slurm_logs/*.err 2>/dev/null | tail -10
'" 2>/dev/null)"

printf '%s\n' "$summary" >> "$log"

failed=0
grep -qE '\b(FAILED|TIMEOUT|CANCELLED|OUT_OF_ME)' <<<"$summary" && failed=1
say "outcome: $([[ $failed -eq 1 ]] && echo 'some jobs did not succeed' || echo 'all jobs succeeded')"

cd "$repo"
claude -p --permission-mode "${JZ_CLAUDE_MODE:-auto}" "$(cat <<PROMPT
A Jean-Zay job set has finished: ${what}.

Slurm reported:
${summary}

Pull the results, check them against what the run was supposed to show, and say
plainly whether it worked. If anything failed, diagnose the cause from the slurm
logs rather than guessing. Do not launch new jobs without asking.
PROMPT
)" 2>&1 | tee -a "$log"
