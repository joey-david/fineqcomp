#!/usr/bin/env bash
#SBATCH --job-name=fq-archive
#SBATCH --account=fas@cpu
#SBATCH --partition=archive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# The archive partition is the node class IDRIS provides for moving data
# between WORK and STORE. It burns no GPU hours and keeps 130 GB of tar off the
# login nodes. No compute module is needed: this job only reads, tars and
# verifies files.

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
mkdir -p slurm_logs
exec bash scripts/jean_zay_archive_runs.sh
