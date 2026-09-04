#!/usr/bin/env bash
set -euo pipefail

usage="usage: scripts/remote.sh push | push-prepared | pull | pull-info | pull-selection | pull-budget | pull-stats | pull-logs"
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
    --exclude prepared/ \
    --exclude runs/ \
    --exclude reports/ \
    --exclude remote_results/ \
    --exclude tmp/ \
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
    --exclude '*.fqcb' \
    --exclude '*.fqmdl' \
    --exclude '*.fqpm' \
    "$host:$remote_root/runs/" runs/
  rsync -avz --prune-empty-dirs \
    "$host:$remote_root/reports/" reports/ || true
  rsync -avz --prune-empty-dirs \
    "$host:$remote_root/remote_logs/" remote_logs/ || true
  ;;
pull-info)
  # The information-scaling study writes outside runs/. Everything here is
  # small except the raw checkpoints, which are left on the cluster: every
  # coded file is a deterministic function of them.
  mkdir -p runs_information_scaling
  rsync -avz --prune-empty-dirs \
    --exclude raw_channel.pt \
    --exclude '*.fqcb' \
    "$host:$remote_root/runs_information_scaling/" runs_information_scaling/
  ;;
pull-selection)
  # The program-selection study writes outside runs/. Everything it keeps is
  # small: locks, per-cell metrics and probe predictions, no checkpoints.
  mkdir -p runs_program_selection
  rsync -avz --prune-empty-dirs \
    "$host:$remote_root/runs_program_selection/" runs_program_selection/
  ;;
pull-budget)
  # The bit-budget study keeps three small things: the per-cell sweep JSONs,
  # the results folder the report writes, and the slurm logs. The adapters it
  # sweeps stay on the cluster, because every coded file is a function of them.
  mkdir -p reports/bit_budget results/5_predicting_the_bit_budget slurm_logs
  rsync -avz --prune-empty-dirs \
    --exclude '*.fqcb' \
    "$host:$remote_root/reports/bit_budget/" reports/bit_budget/
  rsync -avz --prune-empty-dirs \
    "$host:$remote_root/results/5_predicting_the_bit_budget/" \
    results/5_predicting_the_bit_budget/
  rsync -avz --prune-empty-dirs --include 'fqbudget-*' --exclude '*' \
    "$host:$remote_root/slurm_logs/" slurm_logs/ || true
  # The report classifies a swept adapter by reading its own run record, so the
  # probe run directories have to come down too -- their small files only, not
  # the 160MB adapter each one wrote.
  # Every run directory, small files only. Naming the probe studies here was
  # the same mistake the sweep stage made with its glob: the panel grew a
  # `repair_*` family, the pattern did not match it, and the report silently
  # classified twenty-one swept adapters as strangers.
  mkdir -p runs
  rsync -avz --prune-empty-dirs \
    --include '*/' \
    --include 'config.json' --include 'status.json' \
    --include 'metrics.json' --include 'training_metrics.json' \
    --include 'learning_gate.json' \
    --exclude '*' \
    "$host:$remote_root/runs/" runs/
  ;;
pull-logs)
  mkdir -p slurm_logs
  rsync -avz --prune-empty-dirs \
    "$host:$remote_root/slurm_logs/" slurm_logs/
  ;;
pull-stats)
  mkdir -p runs remote_logs
  rsync -avz --prune-empty-dirs \
    --include '*/' \
    --include 'config.json' \
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
