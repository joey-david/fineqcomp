#!/usr/bin/env bash
# Load the managed H100 stack and the small project overlay.

set -e
source /etc/profile.d/z_modules.sh
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0

python_bin="${repo_root}/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin="$(command -v python)"
fi
export PYTHON="$python_bin"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${repo_root}/.hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
if [[ "${FINEQCOMP_ONLINE:-0}" != 1 ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export DATASETS_OFFLINE=1
else
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE DATASETS_OFFLINE
fi
