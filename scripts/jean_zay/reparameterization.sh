#!/usr/bin/env bash
# Queue the whole reparameterization ablation on Jean-Zay with one command.
#
#   bash scripts/jean_zay/reparameterization.sh          # check, plan, submit
#   bash scripts/jean_zay/reparameterization.sh status   # progress
#   bash scripts/jean_zay/reparameterization.sh cancel   # stop every job
#
# Run it from a login node, in a clone under $WORK. It checks the account,
# the GPU family and every model snapshot before submitting anything, then
# queues prepare -> smoke -> train -> sweep/profile -> collect with Slurm
# dependencies. A failed stage cancels the stages that need it, and collect
# always runs and writes the archive. Running it again after a failure or a
# timeout submits only the unfinished work.
#
# Optional overrides: FQ_PROJECT (IDRIS project, default $IDRPROJ), FQ_GPU
# (h100 or a100; default the first the project accepts), FQ_STATE (default
# .cache/reparameterization), FQ_MAX_PARALLEL (array tasks at once, 12).
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
state="${FQ_STATE:-$repo/.cache/reparameterization}"
stage_script="$repo/scripts/jean_zay/reparameterization.sbatch"
helper="scripts/jean_zay/reparameterization.py"
prefix=fq-reparam
action="${1:-launch}"

say() { printf '%s\n' "$*"; }
die() { printf 'ERREUR : %s\n' "$*" >&2; exit 1; }
names="$prefix-prepare,$prefix-smoke,$prefix-train,$prefix-sweep,$prefix-profile,$prefix-collect"
our_jobs() {
  squeue -h -u "$USER" "--name=$names" -o "%.12i %.20j %.9T %.10M %R" 2>/dev/null
}
load_env() {
  unset PYTHONPATH PYTHONHOME
  export PYTHONNOUSERSITE=1
  # shellcheck disable=SC1090
  source "${FQ_ENV_SCRIPT:-$repo/scripts/jean_zay/env.sh}" >/dev/null
  set -euo pipefail
}

case "$action" in
status)
  jobs="$(our_jobs || true)"
  if [[ -n "$jobs" ]]; then
    say "Jobs en file (id, nom, état, durée, raison) :"
    say "$jobs"
  else
    say "Aucun job $prefix en file."
  fi
  load_env
  "$PYTHON" "$helper" status --state "$state"
  exit 0
  ;;
cancel)
  ids="$(our_jobs | awk '{print $1}' | sed 's/_.*//' | sort -u | tr '\n' ' ')"
  if [[ -n "${ids// /}" ]]; then
    # shellcheck disable=SC2086
    scancel $ids
    say "Jobs annulés : $ids"
  else
    say "Aucun job $prefix à annuler."
  fi
  exit 0
  ;;
launch) ;;
*) die "usage : bash scripts/jean_zay/reparameterization.sh [status|cancel]" ;;
esac

command -v sbatch >/dev/null 2>&1 || die "sbatch est introuvable : lance ce script depuis une frontale Jean-Zay."
[[ -f "$stage_script" ]] || die "fichier manquant : $stage_script"
if [[ -n "$(our_jobs || true)" ]]; then
  die "des jobs $prefix sont déjà en file. Attends qu'ils finissent (status) ou arrête-les (cancel)."
fi
project="${FQ_PROJECT:-${IDRPROJ:-}}"
if [[ -z "$project" ]] && command -v idrproj >/dev/null 2>&1; then
  # The default project's three-letter code, if idrproj marks one.
  project="$(idrproj 2>/dev/null | grep -i default | grep -o -E '\b[a-z]{3}\b' | head -n 1 || true)"
fi
[[ "$project" =~ ^[a-z]{3}$ ]] ||
  die "projet IDRIS inconnu ('$project'). Lance 'idrproj', puis : export FQ_PROJECT=<les trois lettres du projet>"
mkdir -p "$state/logs" "$state/lists"

# Check every account/partition/QoS combination with --test-only before any
# real submission: a wrong one is refused here instead of halfway through.
test_job() { sbatch --test-only --export=NONE --time=00:10:00 "$@" "$stage_script" 2>&1; }
gpu=""
for candidate in ${FQ_GPU:-h100 a100}; do
  case "$candidate" in
  h100) partition=gpu_p6 cpus=24 ;;
  a100) partition=gpu_p5 cpus=8 ;;
  *) die "FQ_GPU doit valoir h100 ou a100, pas '$candidate'" ;;
  esac
  gpu_args=(--nodes=1 --ntasks=1 --gres=gpu:1 "--cpus-per-task=$cpus" --hint=nomultithread
            "--partition=$partition" "--constraint=$candidate" "--account=$project@$candidate")
  if answer="$(test_job "${gpu_args[@]}" "--qos=qos_gpu_${candidate}-t3")"; then
    gpu="$candidate"
    break
  fi
  say "GPU $candidate refusé pour le projet $project : $answer"
done
[[ -n "$gpu" ]] || die "aucune partition GPU (h100, a100) n'accepte le projet $project."
smoke_qos="qos_gpu_${gpu}-dev"
test_job "${gpu_args[@]}" "--qos=$smoke_qos" >/dev/null || smoke_qos="qos_gpu_${gpu}-t3"
cpu_args=()
for account in "$project@cpu" "$project@$gpu"; do
  if test_job --nodes=1 --ntasks=1 --cpus-per-task=8 --hint=nomultithread \
       --partition=prepost "--account=$account" >/dev/null; then
    cpu_args=(--nodes=1 --ntasks=1 --cpus-per-task=8 --hint=nomultithread
              --partition=prepost "--account=$account")
    break
  fi
done
[[ ${#cpu_args[@]} -gt 0 ]] || die "la partition prepost refuse le projet $project (@cpu et @$gpu)."
# The planner sets each array's walltime from measured H100 durations (see
# MINUTES in reparameterization.py); A100s get twice as long, within t3's 20 h.
walltime() {  # minutes on an H100
  local minutes="$1"
  if [[ "$gpu" == a100 ]]; then minutes=$((minutes * 2)); fi
  if ((minutes > 1200)); then minutes=1200; fi
  printf '%02d:%02d:00' $((minutes / 60)) $((minutes % 60))
}

load_env
say "Planification (vérifie aussi que chaque modèle de \$DSDIR est lisible)..."
"$PYTHON" "$helper" plan --state "$state" > "$state/lists/plan.out" ||
  die "la planification a échoué ; détails : $state/lists/plan.out"
# shellcheck disable=SC1091
source "$state/lists/counts.env"
[[ "$N_USABLE" -gt 0 ]] || die "aucun modèle n'est lisible par ce compte ; voir $state/lists/plan.json"
if [[ -n "$EXCLUDED_MODELS" ]]; then
  say "ATTENTION : modèles illisibles pour ce compte, leurs runs sont sautés : $EXCLUDED_MODELS"
  say "            (détail dans $state/lists/plan.json ; le reste tourne normalement)"
fi

submitted=()
last_id=""
cancel_submitted() {
  if [[ ${#submitted[@]} -gt 0 ]]; then scancel "${submitted[@]}" 2>/dev/null || true; fi
}
submit() {  # name, extra --export variables, then sbatch options
  local name="$1" exports="$2" log out errors="$state/logs/.sbatch-errors"
  shift 2
  if ! try_submit "$name" "$exports" "$@"; then
    cancel_submitted
    die "sbatch a refusé l'étape $name : $(cat "$errors")"
  fi
}
try_submit() {  # like submit, but returns 1 instead of stopping everything
  local name="$1" exports="$2" log out errors="$state/logs/.sbatch-errors"
  shift 2
  if [[ " $* " == *" --array="* ]]; then log="$state/logs/%x-%A_%a.log"; else log="$state/logs/%x-%j.log"; fi
  out="$(sbatch --parsable "--job-name=$prefix-$name" "--chdir=$repo" "--output=$log" \
           "--export=ALL,FQ_REPO=$repo,FQ_STATE=$state,$exports" "$@" "$stage_script" \
           2>"$errors" </dev/null)" || return 1
  # --parsable prints "id" or "id;cluster" as its last line of stdout. The dev
  # QoS past its submission cap answers with no id at all.
  last_id="$(printf '%s\n' "$out" | tail -n 1)"
  last_id="${last_id%%;*}"
  if [[ ! "$last_id" =~ ^[0-9]+$ ]]; then
    printf 'no job id in: %s\n' "$out" >> "$errors"
    return 1
  fi
  submitted+=("$last_id")
  printf '%s %s %s\n' "$(date '+%F %T')" "$name" "$last_id" >> "$state/jobs.txt"
}

parallel="${FQ_MAX_PARALLEL:-12}"
gpu_work=$((N_TRAIN + N_SWEEP + N_FRONTIER + N_PROFILE))
lines=()
if [[ "$gpu_work" -gt 0 ]]; then
  submit prepare STAGE=prepare "${cpu_args[@]}" --time=02:00:00
  lines+=("$last_id  préparation (venv, données)")
  gate="afterok:$last_id"
  if [[ "$SMOKE_NEEDED" == 1 ]]; then
    smoke_args=("${gpu_args[@]}" "--time=$(walltime 55)" "--dependency=$gate" --kill-on-invalid-dep=yes)
    if ! try_submit smoke STAGE=smoke "${smoke_args[@]}" "--qos=$smoke_qos"; then
      smoke_qos="qos_gpu_${gpu}-t3"
      submit smoke STAGE=smoke "${smoke_args[@]}" "--qos=$smoke_qos"
    fi
    lines+=("$last_id  test rapide sur GPU")
    gate="$gate:$last_id"
  fi
  gpu_t3=("${gpu_args[@]}" "--qos=qos_gpu_${gpu}-t3" --kill-on-invalid-dep=yes)
  # Each line: stage, list, tasks, minutes on an H100, training lists it waits for.
  while read -r stage list tasks minutes needs; do
    [[ -n "$stage" ]] || continue
    waits=""
    if [[ "$needs" != - ]]; then
      for need in ${needs//,/ }; do
        job="job_${need//[^A-Za-z0-9]/_}"
        if [[ -n "${!job:-}" ]]; then waits="$waits:${!job}"; fi
      done
    fi
    submit "$stage" "STAGE=$stage,LIST=$list" "${gpu_t3[@]}" "--time=$(walltime "$minutes")" \
      "--array=0-$((tasks - 1))%$parallel" "--dependency=$gate${waits:+,afterany$waits}"
    printf -v "job_${list//[^A-Za-z0-9]/_}" '%s' "$last_id"
    case "$stage" in
    train) label="entraînement" ;;
    sweep) label="factorisations" ;;
    frontier) label="second codec" ;;
    *) label="profils d'atténuation" ;;
    esac
    lines+=("$last_id  $label, ${list%.txt}, $tasks tâche(s)")
  done < "$state/lists/arrays.tsv"
fi
everything="$(IFS=:; echo "${submitted[*]-}")"
if [[ -n "$everything" ]]; then
  submit collect STAGE=collect "${cpu_args[@]}" --time=01:00:00 "--dependency=afterany:$everything"
else
  submit collect STAGE=collect "${cpu_args[@]}" --time=01:00:00
fi
lines+=("$last_id  collecte et archive")

say ""
say "C'est parti : projet $project, GPU $gpu (test rapide en $smoke_qos)."
for line in "${lines[@]}"; do say "  $line"; done
if [[ "$gpu_work" -eq 0 ]]; then say "  (tout le calcul est déjà fait : seule la collecte est relancée)"; fi
say ""
say "Suivi :     bash scripts/jean_zay/reparameterization.sh status"
say "Journaux :  $state/logs/"
say "Résultat :  $state/fineqcomp_reparameterization_results.tar.gz (écrit par le dernier job)"
