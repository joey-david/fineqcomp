#!/usr/bin/env bash
set -euo pipefail

usage="usage: scripts/jean_zay_submit.sh <batch-script|campaign|bit-budget>"
target="${1:?$usage}"

# Jean-Zay calls this a QoS, not a partition. The short H100 QoS has a hard
# two-hour wall-time limit and a much shorter queue. Route a batch script whose
# declared wall time fits that limit to the matching architecture-specific QoS.
# A command-line --time passed to sbatch still takes precedence when a caller
# deliberately submits a different wall time.
time_to_seconds() {
  local value="$1" days=0 hours=0 minutes=0 seconds=0
  if [[ "$value" == *-* ]]; then
    days="${value%%-*}"
    value="${value#*-}"
  fi
  IFS=: read -r hours minutes seconds <<< "$value"
  if [[ -z "${seconds:-}" ]]; then
    seconds="$minutes"
    minutes="$hours"
    hours=0
  fi
  printf '%d\n' "$((10#$days * 86400 + 10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))"
}

script_option() {
  local option="$1" script="$2"
  sed -n "s/^[[:space:]]*#SBATCH[[:space:]]*${option}[=[:space:]]*\([^[:space:]]*\).*/\1/p" \
    "$script" | head -n 1
}

submit_script() {
  local script="$1"
  local walltime constraint qos seconds
  walltime="${JZ_EXPECTED_TIME:-$(script_option '--time' "$script")}"
  constraint="$(script_option '--constraint' "$script")"
  if [[ -z "$constraint" ]]; then
    constraint="$(script_option '--C' "$script")"
  fi
  if [[ "${JZ_AUTO_DEV:-1}" == 1 && -n "$walltime" ]]; then
    seconds="$(time_to_seconds "$walltime")"
    if (( seconds > 0 && seconds <= 7200 )); then
      case "$constraint" in
        *h100*) qos="qos_gpu_h100-dev" ;;
        *a100*) qos="qos_gpu_a100-dev" ;;
        *) qos="qos_gpu-dev" ;;
      esac
      printf 'routing %s (%s, %ss) to %s\n' "$script" "$walltime" "$seconds" "$qos" >&2
      sbatch --qos="$qos" "$script"
      return
    fi
  fi
  sbatch "$script"
}

if [[ "$target" == bit-budget ]]; then
  # Five stages, each holding the next. `check` is the one that matters: it
  # proves every panel cell has a finished adapter to predict and a prepared
  # corpus to read, and a non-zero exit there stops the chain before any GPU
  # is spent on a panel that cannot be joined up.
  mkdir -p .cache/slurm_logs
  script=scripts/jean_zay_bit_budget.sbatch
  receivers="${BB_ARRAY:-0-6}"
  manifest_id="$(sbatch --parsable --array=0-0 --export=ALL,MODE=manifest "$script")"
  check_id="$(sbatch --parsable --array=0-0 --dependency="afterok:$manifest_id" \
    --export=ALL,MODE=check "$script")"
  probes_id="$(sbatch --parsable --array="$receivers" \
    --dependency="afterok:$check_id" --export=ALL,MODE=probes "$script")"
  sweep_id="$(sbatch --parsable --array="$receivers" \
    --dependency="afterok:$probes_id" --export=ALL,MODE=sweep "$script")"
  report_id="$(sbatch --parsable --array=0-0 \
    --dependency="afterany:$sweep_id" --export=ALL,MODE=report "$script")"
  printf 'manifest=%s check=%s probes=%s sweep=%s report=%s\n' \
    "$manifest_id" "$check_id" "$probes_id" "$sweep_id" "$report_id"
  exit 0
fi

if [[ "$target" != campaign ]]; then
  submit_script "$target"
  exit $?
fi

mkdir -p .cache/slurm_logs
stage_id="$(sbatch --parsable scripts/jean_zay_stage.sbatch)"
smoke_id="$(sbatch --parsable --dependency="afterok:$stage_id" \
  scripts/jean_zay_smoke.sbatch)"
array_id="$(sbatch --parsable --dependency="afterok:$smoke_id" \
  scripts/jean_zay_array.sbatch)"
analysis_id="$(sbatch --parsable --dependency="afterany:$array_id" \
  scripts/jean_zay_analyze.sbatch)"
printf 'stage=%s smoke=%s array=%s analysis=%s\n' \
  "$stage_id" "$smoke_id" "$array_id" "$analysis_id"
