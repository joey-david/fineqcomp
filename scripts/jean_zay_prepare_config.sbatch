#!/usr/bin/env bash
# Restage one pinned campaign on a compute node.  This is separate from the
# original campaign bootstrap: it does not install packages or touch models.
#
# sbatch --export=ALL,CONFIG=configs/high_gain_panel.yaml,\
#   MANIFEST=prepared/highgain-manifest.jsonl scripts/jean_zay_prepare_config.sbatch
#SBATCH --job-name=fqprepare
#SBATCH --account=fas@cpu
#SBATCH --partition=prepost
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
mkdir -p slurm_logs

config="${CONFIG:?set CONFIG}"
manifest="${MANIFEST:?set MANIFEST}"
source "$repo_root/scripts/jean_zay_env.sh"

# Only prepost reaches the Hub. Submitting an online prepare to cpu_p1 to skip
# the prepost queue costs a full queue wait and dies on LocalEntryNotFoundError.
partition="${SLURM_JOB_PARTITION:-prepost}"
if [[ "${FINEQCOMP_ONLINE:-0}" == 1 && "$partition" != "prepost" ]]; then
  echo "fineqcomp: online prepare needs the prepost partition, got '$partition'" >&2
  exit 2
fi

echo "config=$config manifest=$manifest offline=${HF_HUB_OFFLINE:-0} partition=$partition"
prepare_args=(
  -m fineqcomp prepare
  --config "$config" \
  --manifest "$manifest" \
  --prepared-root prepared
)
if [[ "${NO_DATA:-0}" == 1 ]]; then
  prepare_args+=(--no-data)
fi
exec "$PYTHON" "${prepare_args[@]}"
