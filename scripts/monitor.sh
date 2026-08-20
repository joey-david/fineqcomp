#!/usr/bin/env bash
# Detailed live view of your Jean-Zay jobs. Read-only; prints once and exits.
#
#   scripts/monitor.sh              # everything
#   scripts/monitor.sh fqcomp       # one job name
#   watch -n 60 scripts/monitor.sh  # refresh in place
#
# Shows the queue, per-cell training progress with a bar, the codec ladder as
# it fills, and the tail of anything that crashed.

set -uo pipefail
name="${1:-}"
host="${JZ_HOST:-jean-zay}"
root="${JZ_ROOT:-/lustre/fswork/projects/rech/fas/uul94gf/fineQComp}"

ssh -o BatchMode=yes -o ConnectTimeout=25 "$host" "bash -lc '
cd $root 2>/dev/null || exit 1
filter=\"${name}\"

printf \"\\n== queue ==\\n\"
squeue -u \$USER -o \"%.10i %.8j %.9T %.7M %.7L %.14R\" \${filter:+--name \$filter} 2>/dev/null | head -25
running=\$(squeue -u \$USER -h \${filter:+--name \$filter} -t RUNNING 2>/dev/null | wc -l | tr -d \" \")
pending=\$(squeue -u \$USER -h \${filter:+--name \$filter} -t PENDING 2>/dev/null | wc -l | tr -d \" \")
printf \"   %s running, %s pending\\n\" \"\$running\" \"\$pending\"

printf \"\\n== cells ==\\n\"
printf \"   %-26s %-22s %8s %s\\n\" CELL TRAINING CODECS STATE
for d in runs/*/; do
  [ -d \"\$d\" ] || continue
  [ -f \"\$d/config.json\" ] || continue
  b=\$(basename \$d); case \"\$b\" in baselines) continue;; esac
  short=\$(echo \"\$b\" | sed -E \"s/__mistral-7b-base|__qwen25-7b-base|__all-linear-r16//g; s/__[a-f0-9]{10}\$//\")
  state=\$(python3 -c \"import json;print(json.load(open(\\\"\$d/status.json\\\")).get(\\\"state\\\",\\\"?\\\"))\" 2>/dev/null || echo -)
  codecs=\$(ls \$d/codec_metrics 2>/dev/null | wc -l | tr -d \" \")
  want=\$(python3 -c \"import json;print(len(json.load(open(\\\"\$d/config.json\\\"))[\\\"codecs\\\"]))\" 2>/dev/null || echo ?)
  # Training progress from the last logged validation record.
  bar=\"-\"
  if [ -f \"\$d/logs/training.jsonl\" ]; then
    bar=\$(python3 - \"\$d\" <<PYEOF 2>/dev/null || echo -
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
rows = [json.loads(l) for l in open(p / \"logs\" / \"training.jsonl\") if l.strip()]
if not rows:
    raise SystemExit
done = rows[-1][\"updates\"]
nll = format(rows[-1][\"validation_nll\"], \".4f\")
# The target is only known for certain once training has written its metrics;
# while it runs, report progress without inventing a denominator.
metrics = p / \"training_metrics.json\"
if metrics.exists():
    print(\"[############] \" + str(done) + \"u nll=\" + nll)
else:
    print(\"[training...] \" + str(done) + \"u nll=\" + nll)
PYEOF
)
  fi
  printf \"   %-26s %-22s %4s/%-3s %s\\n\" \"\$short\" \"\$bar\" \"\$codecs\" \"\$want\" \"\$state\"
done

printf \"\\n== recent outcomes ==\\n\"
sacct -u \$USER -S now-12hours -X -n -o JobID,JobName%9,State,Elapsed,ExitCode 2>/dev/null \${filter:+| grep \$filter} | tail -8

printf \"\\n== crashes (last day) ==\\n\"
found=0
for f in \$(find slurm_logs -name \"\${filter:-*}-*.err\" -mtime -1 2>/dev/null); do
  if grep -qE \"Traceback|CUDA out of memory|DUE TO TIME LIMIT\" \$f 2>/dev/null; then
    printf \"   %s\\n\" \"\$f\"
    grep -E \"Error|Traceback|DUE TO TIME\" \$f | tail -2 | sed \"s/^/      /\"
    found=1
  fi
done
[ \$found -eq 0 ] && printf \"   none\\n\"
printf \"\\n\"
'"
