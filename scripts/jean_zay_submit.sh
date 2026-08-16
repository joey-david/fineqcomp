#!/usr/bin/env bash
set -euo pipefail

usage="usage: scripts/jean_zay_submit.sh <batch-script|campaign>"
target="${1:?$usage}"
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
