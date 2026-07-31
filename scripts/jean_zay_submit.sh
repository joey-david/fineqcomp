#!/usr/bin/env bash
set -euo pipefail

usage="usage: scripts/jean_zay_submit.sh <batch-script>"
batch_script="${1:?$usage}"
source /etc/profile.d/z_modules.sh
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
exec sbatch "$batch_script"
