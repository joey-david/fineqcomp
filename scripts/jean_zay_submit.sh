#!/usr/bin/env bash
set -euo pipefail

usage="usage: scripts/jean_zay_submit.sh <batch-script>"
batch_script="${1:?$usage}"
exec sbatch "$batch_script"
