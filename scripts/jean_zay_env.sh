#!/usr/bin/env bash
# Load the centrally managed Jean-Zay ML stack for batch jobs.
# This file must be sourced from a job after repo_root is known.

set -e
if ! command -v python >/dev/null 2>&1; then
  echo 'Jean-Zay H100 Python environment was not inherited from submission' >&2
  exit 2
fi
export PYTHONPATH="${repo_root}/src:${repo_root}/vendor${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${repo_root}/.hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
