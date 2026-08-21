#!/usr/bin/env bash
# One short GPU measurement: a fineqcomp subcommand that needs the base model
# but no training.
#
# Submitting these with `sbatch --wrap` does not work. The wrap shell does not
# reliably pick up the module stack, and the job dies importing numpy two
# seconds in. Everything that runs here goes through a script file.
#
#   sbatch --export=ALL,MEASURE=dataset-information,\
#     CONFIG=configs/behavioural_change.yaml,\
#     MANIFEST=prepared/behavioural-manifest.jsonl,\
#     OUT=reports/behavioural_distance.json,WITH_BASE_MODEL=1 \
#     scripts/jean_zay_measure.sbatch
#SBATCH --job-name=fqmeasure
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=00:40:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
mkdir -p slurm_logs reports

measure="${MEASURE:?set MEASURE, e.g. dataset-information}"
config="${CONFIG:?set CONFIG}"
manifest="${MANIFEST:?set MANIFEST}"
out="${OUT:?set OUT}"

source "$repo_root/scripts/jean_zay_env.sh"
export PYTHONUNBUFFERED=1

command=("$PYTHON" -m fineqcomp "$measure"
  --config "$config" --manifest "$manifest"
  --prepared-root "${PREPARED_ROOT:-prepared}" --out "$out")
if [[ -n "${WITH_BASE_MODEL:-}" ]]; then
  command+=(--with-base-model --sample-rows "${SAMPLE_ROWS:-256}")
fi

printf 'measure=%s config=%s out=%s python=%s\n' "$measure" "$config" "$out" "$PYTHON"
"$PYTHON" -c 'import numpy, torch; print("numpy", numpy.__version__, "torch", torch.__version__)'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "${command[@]}"
