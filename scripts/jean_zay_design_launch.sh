#!/usr/bin/env bash
# Print, or submit, the staged jobs of the 7 September 2026 design pass.
#
# The manifest is the source of truth; this script only reads it, filters it and
# hands the commands to bash in order. Nothing here invents a command.
#
#   scripts/jean_zay_design_launch.sh                 every direction, printed
#   scripts/jean_zay_design_launch.sh F03 F05         two directions, printed
#   scripts/jean_zay_design_launch.sh --stage prepare every prepare stage
#   scripts/jean_zay_design_launch.sh --submit F03    run F03's stages in order
#
# --submit runs only the stages that belong to the repository you are standing
# in. The reasoning-trajectory directions are printed with the checkout they
# need, because their sbatch script and run folders live there.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$repo_root/reports/research_design_2026_09_07/launch/manifest.json"
python_bin="${PYTHON:-$repo_root/.venv/bin/python}"
[[ -x "$python_bin" ]] || python_bin="$(command -v python3)"

submit=0
stage_filter=""
ids=()
while (($#)); do
  case "$1" in
    --submit) submit=1 ;;
    --stage) stage_filter="${2:?--stage needs a stage name}"; shift ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) ids+=("$1") ;;
  esac
  shift
done

commands="$("$python_bin" - "$manifest" "$stage_filter" "${ids[@]+"${ids[@]}"}" <<'PY'
import json, sys
manifest_path, stage_filter, *ids = sys.argv[1:]
manifest = json.loads(open(manifest_path).read())
wanted = {i.upper() for i in ids}
for entry in manifest["directions"]:
    if wanted and entry["id"] not in wanted:
        continue
    if entry["status"] != "ready":
        print(f"# {entry['id']} {entry['title']}: blocked")
        print(f"#   needs: {entry['blocked_by']}")
        continue
    for stage in entry["stages"]:
        if stage_filter and not stage["name"].startswith(stage_filter):
            continue
        print(f"# {entry['id']} / {stage['name']} ({entry['project']})")
        print(f"{entry['project']}\t{stage['command']}")
PY
)"

here="$(basename "$repo_root")"
while IFS= read -r line; do
  if [[ "$line" == \#* || -z "$line" ]]; then
    printf '%s\n' "$line"
    continue
  fi
  project="${line%%$'\t'*}"
  command="${line#*$'\t'}"
  printf '%s\n' "$command"
  if ((submit)); then
    if [[ "$project" != "$here" ]]; then
      echo "# not submitted: run this one from the $project checkout" >&2
      continue
    fi
    bash -c "$command"
  fi
done <<< "$commands"
