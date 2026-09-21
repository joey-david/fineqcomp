#!/usr/bin/env bash
set -euo pipefail

usage="usage: scripts/remote.sh push | push-prepared | pull | pull-info | pull-stats | pull-logs"
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
