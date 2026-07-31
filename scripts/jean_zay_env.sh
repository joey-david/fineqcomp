#!/usr/bin/env bash
# Load the centrally managed Jean-Zay ML stack for batch jobs.
# This file must be sourced from a job after repo_root is known.

set -e
source /etc/profile.d/z_modules.sh
module load pytorch-gpu/py3/2.8.0
export PYTHONPATH="${repo_root}/vendor${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${repo_root}/.hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
