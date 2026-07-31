#!/usr/bin/env bash
set -euo pipefail

usage="usage: scripts/remote.sh push | push-prepared | pull | pull-stats"
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
    --exclude prepared/ \
    --exclude runs/ \
    --exclude reports/ \
    --exclude slurm_logs/ \
    --exclude remote_logs/ \
  ./ "$host:$remote_root/"
  ;;
push-prepared)
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "mkdir -p '$remote_root/prepared'"
  rsync -avz --delete \
    --exclude local-preflight.json \
    prepared/ "$host:$remote_root/prepared/"
  ;;
pull)
  mkdir -p runs reports remote_logs
  rsync -avz --prune-empty-dirs \
    --exclude raw_channel.pt \
    "$host:$remote_root/runs/" runs/
  rsync -avz --prune-empty-dirs \
    "$host:$remote_root/reports/" reports/ || true
  rsync -avz --prune-empty-dirs \
    "$host:$remote_root/remote_logs/" remote_logs/ || true
  ;;
pull-stats)
  mkdir -p runs remote_logs
  rsync -avz --prune-empty-dirs \
    --include '*/' \
    --include 'status.json' \
    --include 'metrics.json' \
    --include 'training_metrics.json' \
    --include 'codec_metrics/*.json' \
    --include 'logs/*.jsonl' \
    --exclude '*' \
    "$host:$remote_root/runs/" runs/
  rsync -avz --prune-empty-dirs \
    "$host:$remote_root/remote_logs/" remote_logs/ || true
  ;;
*)
  echo "$usage" >&2
  exit 2
  ;;
esac
