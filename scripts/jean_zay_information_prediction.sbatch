#!/usr/bin/env bash
#SBATCH --account=fas@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the analysis checkout}"
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
prediction_python=/lustre/fswork/projects/rech/fas/uul94gf/fineQComp/.venv/bin/python
collection="${COLLECTION_ROOT:?set the completed collection output directory}"
config=configs/information_budget_search/panel.json
"$prediction_python" -m fineqcomp.information_budget_prediction --config "$config" \
  --source "$collection" --out "$collection/discovery.json" --phase discover
for stage in development outer_audit; do
  "$prediction_python" -m fineqcomp.information_budget_prediction --config "$config" \
    --source "$collection" --out "$collection/$stage.json" --phase "$stage" \
    --lock "$collection/discovery.json"
done
