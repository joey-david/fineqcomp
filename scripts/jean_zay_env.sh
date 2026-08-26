#!/usr/bin/env bash
# Load the managed GPU stack and the small project overlay.

set -e

# Compute nodes do not carry /etc/profile.d/z_modules.sh, which login nodes do.
# Sourcing it unconditionally is what killed every batch job here inside two
# seconds, with an error no login-node test could reproduce. Try the known init
# scripts in turn and only give up if none of them defines `module`.
if ! command -v module >/dev/null 2>&1 && ! declare -F module >/dev/null 2>&1; then
  for module_init in \
    /etc/profile.d/z_modules.sh \
    /usr/share/lmod/lmod/init/bash \
    /opt/lmod/lmod/init/bash \
    /etc/profile.d/modules.sh
  do
    if [[ -r "$module_init" ]]; then
      # shellcheck disable=SC1090
      source "$module_init" && break
    fi
  done
fi
if ! command -v module >/dev/null 2>&1 && ! declare -F module >/dev/null 2>&1; then
  echo "no module system found on $(hostname); cannot load the H100 stack" >&2
  exit 2
fi

module purge
gpu_arch="${FINEQCOMP_GPU_ARCH:-}"
if [[ -z "$gpu_arch" ]]; then
  case "${SLURM_JOB_CONSTRAINTS:-}" in
    *a100*) gpu_arch="a100" ;;
    *v100*) gpu_arch="v100" ;;
    *) gpu_arch="h100" ;;
  esac
fi
case "$gpu_arch" in
  a100|h100|v100) ;;
  *) echo "unsupported Jean-Zay GPU architecture: $gpu_arch" >&2; exit 2 ;;
esac
# V100 is the site's default GPU stack. A100 and H100 need an architecture
# module to select builds for their newer CUDA capabilities.
if [[ "$gpu_arch" != "v100" ]]; then
  module load "arch/$gpu_arch"
fi
module load pytorch-gpu/py3/2.8.0

# Prefer the repo venv only if it can actually import torch. A venv that exists
# but is missing the stack silently shadows the working module python, which is
# how an earlier batch of jobs died on startup inside a second.
python_bin="$(command -v python)"
# The sbatch scripts set `repo_root` before sourcing this. Sourcing it by hand
# does not, and an empty one silently pointed HF_HOME at /.hf_cache, so an
# interactive `prepare` looked like a network failure in offline mode.
repo_root="${repo_root:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ -x "${repo_root}/.venv/bin/python" ]] &&
   "${repo_root}/.venv/bin/python" -c 'import torch' >/dev/null 2>&1; then
  python_bin="${repo_root}/.venv/bin/python"
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
