#!/usr/bin/env bash
set -euo pipefail

usage="usage: scripts/remote.sh push | push-prepared | push-study <name> | pull-study <name> | pull | pull-info | pull-stats | pull-logs"
action="${1:?$usage}"
host="${SSH_SERVER:-lamgate}"
remote_root="${REMOTE_REPO_ROOT:-/home/lamsade/jdavid/fineQComp}"

case "$action" in
push)
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "mkdir -p '$remote_root'"
  rsync -avz --prune-empty-dirs \
    --exclude .git/ \
    --exclude .venv/ \
    --exclude .env \
    --exclude __pycache__/ \
    --exclude .pytest_cache/ \
    --exclude .ruff_cache/ \
    --exclude '*.egg-info/' \
    --exclude '*.fqcb' \
    --exclude '*.fqmdl' \
    --exclude '*.fqpm' \
    --exclude '*.pt' \
    --exclude .cache/ \
    --exclude remote_results/ \
    --exclude tmp/ \
    --exclude remote_logs/ \
    ./ "$host:$remote_root/"
  ;;
push-prepared)
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "mkdir -p '$remote_root/.cache/prepared'"
  rsync -avz --delete \
    --exclude local-preflight.json \
    .cache/prepared/ "$host:$remote_root/.cache/prepared/"
  ;;
push-study)
  # A self-contained study directory under .cache/reports: its frozen probes,
  # prepared splits and lock. Everything in it is produced off the cluster, so
  # the compute nodes never need network access to a dataset hub.
  study="${2:?usage: scripts/remote.sh push-study <name>}"
  local_root=".cache/reports/$study"
  [[ -d "$local_root" ]] || { echo "no such study: $local_root" >&2; exit 2; }
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
    "mkdir -p '$remote_root/.cache/reports/$study'"
  rsync -avz --prune-empty-dirs \
    --exclude '*.pt' \
    --exclude '*.fqcb' \
    "$local_root/" "$host:$remote_root/.cache/reports/$study/"
  ;;
pull-study)
  # Scores, locks and summaries only. Adapters stay on the cluster: every
  # coded file is a deterministic function of them.
  study="${2:?usage: scripts/remote.sh pull-study <name>}"
  mkdir -p ".cache/reports/$study"
  rsync -avz --prune-empty-dirs \
    --exclude '*.pt' \
    --exclude '*.fqcb' \
    --exclude 'prepared/' \
    "$host:$remote_root/.cache/reports/$study/" ".cache/reports/$study/"
  ;;
pull)
  mkdir -p .cache/runs .cache/reports remote_logs
  rsync -avz --prune-empty-dirs \
    --exclude raw_channel.pt \
    --exclude '*.fqcb' \
    --exclude '*.fqmdl' \
    --exclude '*.fqpm' \
    "$host:$remote_root/.cache/runs/" .cache/runs/
  rsync -avz --prune-empty-dirs \
    "$host:$remote_root/.cache/reports/" .cache/reports/ || true
  rsync -avz --prune-empty-dirs \
    "$host:$remote_root/remote_logs/" remote_logs/ || true
  ;;
pull-info)
  # The information-scaling study writes outside runs/. Everything here is
  # small except the raw checkpoints, which are left on the cluster: every
  # coded file is a deterministic function of them.
  mkdir -p .cache/runs_information_scaling
  rsync -avz --prune-empty-dirs \
    --exclude raw_channel.pt \
    --exclude '*.fqcb' \
    "$host:$remote_root/.cache/runs_information_scaling/" .cache/runs_information_scaling/
  ;;
pull-logs)
  mkdir -p .cache/slurm_logs
  rsync -avz --prune-empty-dirs \
    "$host:$remote_root/.cache/slurm_logs/" .cache/slurm_logs/
  ;;
pull-stats)
  mkdir -p .cache/runs remote_logs
  rsync -avz --prune-empty-dirs \
    --include '*/' \
    --include 'config.json' \
    --include 'status.json' \
    --include 'metrics.json' \
    --include 'training_metrics.json' \
    --include 'codec_metrics/*.json' \
    --include 'logs/*.jsonl' \
    --exclude '*' \
    "$host:$remote_root/.cache/runs/" .cache/runs/
  rsync -avz --prune-empty-dirs \
    "$host:$remote_root/remote_logs/" remote_logs/ || true
  ;;
*)
  echo "$usage" >&2
  exit 2
  ;;
esac
