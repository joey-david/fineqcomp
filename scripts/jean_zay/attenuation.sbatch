#!/usr/bin/env bash
# Submit with the Jean-Zay GPU modules already loaded on the login node.
# CELLS, OUT, and PYTHON must be set; ROWS defaults to 64, SHARDS to one.
#SBATCH --job-name=fq-attenuation
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:55:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
srun "$PYTHON" scripts/analysis/adapter_spectrum_profile.py \
  "${CELLS:?}" "${ADAPTERS:-/lustre/fsn1/projects/rech/fas/uul94gf/fq_panel_adapters}" \
  "${RUNS_ROOT:-/lustre/fswork/projects/rech/fas/uul94gf/fineQComp/runs}" \
  /lustre/fswork/projects/rech/fas/uul94gf/fineQComp/prepared/natural \
  "${OUT:?}" "${SLURM_ARRAY_TASK_ID:-0}" "${SHARDS:-1}" "${MODE:-attenuation}" "${ROWS:-64}" "${SPLIT:-train}"
