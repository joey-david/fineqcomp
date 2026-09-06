#!/usr/bin/env bash
# One immutable study cell per H100; phases share the frozen study lock.
#SBATCH --job-name=fqtrajectory
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:?submit from the isolated study checkout}"
cd "$repo_root"
source scripts/jean_zay_env.sh
export PYTHONUNBUFFERED=1
# Avoid Slurm's implicit login-environment retrieval on the compute node.
export SLURM_EXPORT_ENV=ALL
mode="${MODE:?choose smoke, train, evaluate, field or report}"
task="${SLURM_ARRAY_TASK_ID:-0}"
command=("$PYTHON" -m fineqcomp.behavioral_trajectory
  --config "${STUDY_CONFIG:-configs/behavioral_trajectory.yaml}"
  --out "${STUDY_OUT:-reports/behavioral_trajectory_v1}" --phase "$mode" --cell "$task")
printf 'phase=%s cell=%s\n' "$mode" "$task"
exec "${command[@]}"
