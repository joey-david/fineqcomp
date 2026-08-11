#!/usr/bin/env bash
#SBATCH --job-name=fineqcomp-analyze
#SBATCH --account=fas@cpu
#SBATCH --partition=prepost
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
source "$repo_root/scripts/jean_zay_env.sh"
exec "$PYTHON" -m fineqcomp analyze --root runs --out reports
