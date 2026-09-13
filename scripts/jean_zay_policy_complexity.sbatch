#!/usr/bin/env bash
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:?submit from the frozen checkout}"
cd "$repo_root"
source scripts/jean_zay_env.sh
export PYTHONUNBUFFERED=1
export SLURM_EXPORT_ENV=ALL
exec "$PYTHON" -m fineqcomp.policy_complexity \
  --config "${CONFIG:-configs/policy_complexity/pilot.json}" \
  --out "${OUTPUT:-outputs/policy_pilot}" --phase "${PHASE:?set phase}" \
  --cell "${CELL:-${SLURM_ARRAY_TASK_ID:-0}}"
