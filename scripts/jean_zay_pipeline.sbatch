#!/usr/bin/env bash
# One tick of the overnight driver: read results, submit what is unblocked,
# re-arm. No GPU -- this reads JSON and calls sbatch.
#
#   sbatch scripts/jean_zay_pipeline.sbatch
#   scripts/pipeline.py --status          # what it has decided so far
#   tail -f pipeline/log.jsonl            # every decision, as it makes them
#
# Stop it by cancelling the queued `fqdriver` job. There is only ever one in
# flight, so cancelling it ends the chain; work already submitted keeps running.
#SBATCH --job-name=fqdriver
#SBATCH --account=fas@cpu
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -uo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
mkdir -p slurm_logs pipeline

source "$repo_root/scripts/jean_zay_env.sh"
export PYTHONUNBUFFERED=1

printf 'driver tick at %s on %s\n' "$(date -Is)" "$(hostname)"
exec "$PYTHON" scripts/pipeline.py
