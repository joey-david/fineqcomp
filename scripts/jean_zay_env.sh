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
if command -v module >/dev/null 2>&1; then
  # Login nodes expose the modulefiles; compute images may only receive the
  # resulting environment through sbatch --export=ALL.
  module load pytorch-gpu/py3/2.8.0 2>/dev/null || true
fi
if ! command -v python >/dev/null 2>&1; then
  echo 'Jean-Zay Python module was not exported into the job' >&2
  exit 2
fi
# Slurm may strip LD_LIBRARY_PATH while moving from the login node to the
# compute image. Keep the OpenMPI runtime needed by the module's PyTorch.
openmpi_lib=/lustre/fshomisc/sup/hpe/spack_soft/openmpi/4.1.8/gcc-11.5.0-6k5g5tjbenywyt5p6czzbxg7fgo3pvam/lib
if [[ -d "$openmpi_lib" ]]; then
  export LD_LIBRARY_PATH="$openmpi_lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export LD_PRELOAD="$openmpi_lib/libmpi.so.40${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
export PYTHONPATH="${repo_root}/src:${repo_root}/vendor${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${repo_root}/.hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
