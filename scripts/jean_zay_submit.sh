#!/usr/bin/env bash
set -euo pipefail

usage="usage: scripts/jean_zay_submit.sh <batch-script>"
batch_script="${1:?$usage}"
source /etc/profile.d/z_modules.sh
module load pytorch-gpu/py3/2.8.0

# Jean-Zay compute images do not carry the modulefile tree. Pass the full
# dynamic-library environment from the login node into the batch job.
export PYTHONPATH="$(cd "$(dirname "$batch_script")/.." && pwd)/src:$(cd "$(dirname "$batch_script")/.." && pwd)/vendor${PYTHONPATH:+:${PYTHONPATH}}"
exec sbatch --export="ALL,LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-},LIBRARY_PATH=${LIBRARY_PATH:-},CPATH=${CPATH:-},PATH=${PATH}" "$batch_script"
