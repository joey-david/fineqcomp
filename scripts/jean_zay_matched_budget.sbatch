#!/usr/bin/env bash
# Rank against precision at a matched bit budget, on corrupted-rationale LoRAs.
#
# The study trains nothing. The rank-16 permuted adapters were paid for by
# conditional_trace_rate.yaml; the low- and high-rank arms are trained by the
# ordinary campaign path before this script is useful:
#
#   $PYTHON -m fineqcomp prepare --config configs/low_rank_permuted.yaml \
#     --manifest .cache/prepared/low-rank-permuted-manifest.jsonl
#   sbatch --array=0-5%6 --export=ALL,CONFIG=configs/low_rank_permuted.yaml,\
#     MANIFEST=.cache/prepared/low-rank-permuted-manifest.jsonl,SHARDS=6,\
#     STUDIES=permuted_low_rank scripts/jean_zay_array.sbatch
#
# That is the twelve low-rank runs. The six high-rank ones are the same command
# with STUDIES=permuted_high_rank, and they are worth holding until the low-rank
# arms have said whether there is an effect to find a break point in.
#
# Then, in order. The header carries the sweep's lane and walltime; every other
# mode is minutes of work, so each usage line below overrides both rather than
# sitting in a ten-hour t3 slot waiting behind real jobs.
#
#   prepare   freeze the cell list, the grid and every input digest. No GPU.
#               sbatch --array=0-0 --qos=qos_gpu_h100-dev --time=00:20:00 \
#                 --export=ALL,MODE=prepare scripts/jean_zay_matched_budget.sbatch
#
#   check     prove every cell has a trained adapter to sweep. Non-zero exit
#             holds the chain before a GPU hour is claimed, which is the point
#             of running it before the sweep rather than after.
#               sbatch --array=0-0 --qos=qos_gpu_h100-dev --time=00:20:00 \
#                 --export=ALL,MODE=check scripts/jean_zay_matched_budget.sbatch
#
#   probe     one timed generation pass on the raw corrupted adapter, and the
#             walltime it projects for a cell. Run this before sizing the sweep:
#             a corrupted adapter's answers can run to the token cap, and the
#             difference between that and a short answer is the difference
#             between a right-sized job and a slot that expires mid-grid.
#               sbatch --array=0-0 --qos=qos_gpu_h100-dev --time=00:40:00 \
#                 --export=ALL,MODE=probe scripts/jean_zay_matched_budget.sbatch
#
#   sweep     one study cell per H100. Every score is cached on its own path,
#             so a cell that runs out of walltime resumes at the next unscored
#             candidate; resubmitting the same array finishes it. Set --time
#             from the probe's projection rather than from the ten-hour default
#             in the header.
#               sbatch --array=0-20%5 --time=<from the probe> \
#                 --export=ALL,MODE=sweep scripts/jean_zay_matched_budget.sbatch
#
#   report    join the grid, the equal-budget diagonals and the trained-versus-
#             truncated pairing. No GPU.
#               sbatch --array=0-0 --qos=qos_gpu_h100-dev --time=00:30:00 \
#                 --export=ALL,MODE=report scripts/jean_zay_matched_budget.sbatch
#
# The arms are independent, so the rank-16 arm -- the one needing no new
# training -- can be swept on its own while the others are still training, and
# it already carries the matched-budget diagonals on its own. With the shipped
# config it is cells 3-5; `MODE=cells` prints the mapping, which is the only
# thing to trust after an arm is added or renamed.
#
#   sbatch --array=3-5%3 --time=<from the probe> --export=ALL,MODE=sweep \
#     scripts/jean_zay_matched_budget.sbatch
#
# MB_DRY_RUN=1 prints the command and exits before a model is requested.
#
# The account below is rewritten by scripts/set_slurm_account.sh, which is the
# only supported way to change it: an IDRIS Slurm account is <project>@<family>
# where the project is the three-letter code `idrproj` prints, never an eDARI
# dossier number. Confirm the project that should carry these hours before
# submitting -- a project without h100 hours is accepted and then never runs.
#SBATCH --job-name=fqmatched
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=10:00:00
#SBATCH --output=.cache/slurm_logs/%x-%A_%a.out
#SBATCH --error=.cache/slurm_logs/%x-%A_%a.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"

mode="${MODE:?choose prepare, check, cells, probe, sweep or report}"
task="${SLURM_ARRAY_TASK_ID:-0}"
config="${MB_CONFIG:-configs/matched_bit_budget.yaml}"
out="${MB_OUT:-.cache/reports/matched_bit_budget_v1}"
runs_root="${MB_RUNS_ROOT:-.cache/runs}"

mkdir -p .cache/slurm_logs
source "$repo_root/scripts/jean_zay_env.sh"
export PYTHONUNBUFFERED=1
export SLURM_EXPORT_ENV=ALL

if [[ "$mode" == cells ]]; then
  exec "$PYTHON" - "$out/lock.json" <<'PY'
import json, sys
lock = json.load(open(sys.argv[1]))
for cell in lock['cells']:
    print(f"{cell['id']:3d}  {cell['slug']:44s} trained_rank={cell['trained_rank']:<3d}"
          f" candidates={len(cell['grid'])}")
PY
fi

args=(-m fineqcomp.matched_budget --config "$config" --out "$out"
      --runs "$runs_root" --phase "$mode")
case "$mode" in
prepare | check | report) ;;
probe | sweep) args+=(--cell "$task") ;;
*)
  echo "unknown MODE=$mode; expected prepare, check, cells, probe, sweep or report" >&2
  exit 2
  ;;
esac

printf 'mode=%s cell=%s out=%s\n' "$mode" "$task" "$out"
printf '%q ' "$PYTHON" "${args[@]}"
printf '\n'
if [[ "${MB_DRY_RUN:-0}" == 1 ]]; then
  exit 0
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec "$PYTHON" "${args[@]}"
