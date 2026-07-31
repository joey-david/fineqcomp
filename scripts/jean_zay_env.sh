#!/usr/bin/env bash
# Load the centrally managed Jean-Zay ML stack for batch jobs.
# This file must be sourced from a job after repo_root is known.

set -e
for module_init in \
  /etc/profile.d/z_modules.sh \
  /etc/profile.d/modules.sh \
  /usr/share/Modules/init/bash \
  /usr/share/modules/init/bash; do
  if [[ -f "$module_init" ]]; then
    source "$module_init"
    break
  fi
done
if ! command -v module >/dev/null 2>&1; then
  echo 'Jean-Zay module command is unavailable on this node' >&2
  exit 2
fi
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
if ! command -v python >/dev/null 2>&1; then
  echo 'Jean-Zay H100 Python module did not load' >&2
  exit 2
fi
export PYTHONPATH="${repo_root}/src:${repo_root}/vendor${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${repo_root}/.hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
