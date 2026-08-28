#!/usr/bin/env bash
#SBATCH --job-name=fineqcomp-stage
#SBATCH --account=fas@cpu
#SBATCH --partition=prepost
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
export FINEQCOMP_ONLINE=1
source "$repo_root/scripts/jean_zay_env.sh"
module_python="$(command -v python)"

if [[ -x .venv/bin/python ]] && ! .venv/bin/python -c 'import torch' 2>/dev/null; then
  if [[ ! -e .venv.unusable ]]; then
    mv .venv .venv.unusable
  else
    echo '.venv is unusable and .venv.unusable already exists' >&2
    exit 2
  fi
fi
if ! command -v uv >/dev/null 2>&1; then
  "$module_python" -m pip install --user uv
  export PATH="$HOME/.local/bin:$PATH"
fi
if [[ ! -x .venv/bin/python ]]; then
  uv venv --system-site-packages --python "$module_python" .venv
fi
uv pip install --python .venv/bin/python \
  'math-verify[antlr4_13_2]==0.9.0' 'reasoning-gym==0.1.25' \
  'rouge-score==0.1.2'

export PYTHON="$repo_root/.venv/bin/python"
if [[ "${INSTALL_ONLY:-0}" == 1 ]]; then
  "$PYTHON" -c 'import importlib.metadata; print(importlib.metadata.version("reasoning-gym"))'
  exit 0
fi
"$PYTHON" -m fineqcomp prepare --config configs/campaign.yaml
"$PYTHON" -m fineqcomp cache-models --config configs/campaign.yaml
"$PYTHON" -m fineqcomp preflight --require-dependencies --tokenizers --shards 16 \
  --report prepared/stage-preflight.json
test -s prepared/model-cache.json
test -s prepared/stage-preflight.json
