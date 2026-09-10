#!/usr/bin/env bash
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:?submit from the branch checkout}"
cd "$repo_root"
source /lustre/fswork/projects/rech/fas/uul94gf/fineQComp/scripts/jean_zay_env.sh
export PYTHONUNBUFFERED=1
export SLURM_EXPORT_ENV=ALL
export PYTHONPATH="$repo_root:$repo_root/src:${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m fineqcomp.information_budget_search --config "${CONFIG:-configs/information_budget_search/panel.json}" \
  --source-root /lustre/fswork/projects/rech/fas/uul94gf/fineQComp \
  --out "${OUTPUT:-outputs/main_results_v1}" --phase "${PHASE:?set phase}" --cell "${CELL:-${SLURM_ARRAY_TASK_ID:-0}}"
