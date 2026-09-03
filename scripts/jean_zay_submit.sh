#!/usr/bin/env bash
set -euo pipefail

usage="usage: scripts/jean_zay_submit.sh <batch-script|campaign|bit-budget>"
target="${1:?$usage}"

if [[ "$target" == bit-budget ]]; then
  # Five stages, each holding the next. `check` is the one that matters: it
  # proves every panel cell has a finished adapter to predict and a prepared
  # corpus to read, and a non-zero exit there stops the chain before any GPU
  # is spent on a panel that cannot be joined up.
  mkdir -p slurm_logs
  script=scripts/jean_zay_bit_budget.sbatch
  receivers="${BB_ARRAY:-0-6}"
  manifest_id="$(sbatch --parsable --array=0-0 --export=ALL,MODE=manifest "$script")"
  check_id="$(sbatch --parsable --array=0-0 --dependency="afterok:$manifest_id" \
    --export=ALL,MODE=check "$script")"
  probes_id="$(sbatch --parsable --array="$receivers" \
    --dependency="afterok:$check_id" --export=ALL,MODE=probes "$script")"
  sweep_id="$(sbatch --parsable --array="$receivers" \
    --dependency="afterok:$probes_id" --export=ALL,MODE=sweep "$script")"
  report_id="$(sbatch --parsable --array=0-0 \
    --dependency="afterany:$sweep_id" --export=ALL,MODE=report "$script")"
  printf 'manifest=%s check=%s probes=%s sweep=%s report=%s\n' \
    "$manifest_id" "$check_id" "$probes_id" "$sweep_id" "$report_id"
  exit 0
fi

if [[ "$target" != campaign ]]; then
  exec sbatch "$target"
fi

mkdir -p slurm_logs
stage_id="$(sbatch --parsable scripts/jean_zay_stage.sbatch)"
smoke_id="$(sbatch --parsable --dependency="afterok:$stage_id" \
  scripts/jean_zay_smoke.sbatch)"
array_id="$(sbatch --parsable --dependency="afterok:$smoke_id" \
  scripts/jean_zay_array.sbatch)"
analysis_id="$(sbatch --parsable --dependency="afterany:$array_id" \
  scripts/jean_zay_analyze.sbatch)"
printf 'stage=%s smoke=%s array=%s analysis=%s\n' \
  "$stage_id" "$smoke_id" "$array_id" "$analysis_id"
