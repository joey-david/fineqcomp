#!/usr/bin/env bash
# Submit the scale-by-task grid in three walltime tiers.
#
# gpu_p6 runs at 357 of 364 nodes allocated with two thousand jobs pending, so
# nothing starts by waiting its turn -- it starts by backfilling into whatever
# gap opens next. The length of the request is therefore the single biggest
# lever on when a cell runs, and asking twenty hours for a cell that needs
# eleven minutes is what put the first submission's estimated start more than
# thirty hours out.
#
# So each cell asks for roughly what it needs, measured where possible: the
# 1.5B pointer cell trained in 612 s and swept in 53 s, and cost scales with
# parameters and with how many tokens the sweep has to generate (the
# chain-of-thought cells generate 512 tokens a row against the pointer cells'
# 160).
#
# The dev queue is capped at two hours and ten submitted jobs per user, which is
# exactly the shape of the cheap tier, and it is the lane that turns around
# fastest. The two t3 tiers take everything that cannot fit two hours.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Under two hours: 0.5B and 1.5B everything, plus the 7B pointer cells, whose
# sweep generates 160 tokens a row rather than 512.
DEV_CELLS=(
  ptr_h2_1p5b ptr_h4_1p5b ptr_h8_1p5b ptr_mm_1p5b
  cot_0p5b cot_1p5b
  ptr_h2_7b ptr_h4_7b ptr_h8_7b ptr_mm_7b
)
# Four hours: the 3B and 7B chain-of-thought cells and the rest of the 7B
# pointer family, plus the two cheapest 32B pointer cells.
MID_CELLS=(
  cot_3b ptr_h4_clean_7b ptr_h4_7b_inst ptr_mm_7b_inst
  cot_7b ptr_h2_32b ptr_h4_32b
)
# Six hours: 32B pointer training and the 14B chain-of-thought sweep, which
# are the large cells that do not need a full overnight slot.
BIG_CELLS=(
  ptr_h8_32b ptr_mm_32b cot_14b
)
# One cell on its own: 32B training plus a 512-token sweep over 500 rows is the
# only thing in the grid that genuinely needs ten hours, and asking ten hours
# for the three cells above would have parked them behind it for a day.
HUGE_CELLS=(
  cot_32b
)

submit() {
  local name="$1" qos="$2" time="$3"; shift 3
  local cells=("$@")
  local last=$(( ${#cells[@]} - 1 ))
  mkdir -p slurm_tiers
  printf '%s\n' "${cells[@]}" > "slurm_tiers/${name}.txt"
  local id
  id=$(sbatch --parsable \
    --job-name="fq-$name" \
    --qos="$qos" \
    --time="$time" \
    --array="0-${last}" \
    scripts/jean_zay_grid.sbatch "$name")
  printf '%-6s %-22s %2d cells  %s  job %s\n' "$name" "$qos" "${#cells[@]}" "$time" "$id"
}

submit dev qos_gpu_h100-dev 02:00:00 "${DEV_CELLS[@]}"
submit mid qos_gpu_h100-t3  04:00:00 "${MID_CELLS[@]}"
submit big  qos_gpu_h100-t3 06:00:00 "${BIG_CELLS[@]}"
submit huge qos_gpu_h100-t3 12:00:00 "${HUGE_CELLS[@]}"

total=$(( ${#DEV_CELLS[@]} + ${#MID_CELLS[@]} + ${#BIG_CELLS[@]} + ${#HUGE_CELLS[@]} ))
echo "submitted $total cells in four tiers"
