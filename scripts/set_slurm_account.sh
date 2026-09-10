#!/usr/bin/env bash
# Point every Jean-Zay batch script at a different IDRIS project.
#
# IDRIS Slurm accounts are written <project>@<partition family>, where the
# project is the short code `idrproj` prints -- never the eDARI dossier number.
# The partition family has to be preserved: an @h100 job moved to @cpu would be
# rejected, and moving it to a project without that family's hours would be
# accepted and then never scheduled.
#
#   scripts/set_slurm_account.sh <project>            # show the diff only
#   scripts/set_slurm_account.sh <project> --apply    # write it
set -euo pipefail

project="${1:-}"
apply="${2:-}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "$project" ]]; then
    echo "usage: $(basename "$0") <project> [--apply]" >&2
    exit 2
fi
if [[ ! "$project" =~ ^[a-z]{3}$ ]]; then
    echo "refusing '$project': an IDRIS project is three lowercase letters," >&2
    echo "as printed by 'idrproj'. A dossier number such as AD011018261 is not" >&2
    echo "a Slurm account and would be rejected at submission." >&2
    exit 2
fi

files=()
while IFS= read -r line; do
    files+=("$line")
done < <(grep -rl -E '^#SBATCH[[:space:]]+--account=' "$root/scripts" 2>/dev/null | sort)
if [[ ${#files[@]} -eq 0 ]]; then
    echo "no batch scripts with an --account line under $root/scripts" >&2
    exit 1
fi

echo "project:  $project"
echo "scripts:  ${#files[@]} under $root/scripts"
echo
for file in "${files[@]}"; do
    current="$(grep -m1 -E '^#SBATCH[[:space:]]+--account=' "$file" | sed 's/.*--account=//')"
    family="${current##*@}"
    printf '  %-44s %s -> %s@%s\n' "$(basename "$file")" "$current" "$project" "$family"
done
echo

if [[ "$apply" != "--apply" ]]; then
    echo "dry run; nothing written. Re-run with --apply to make the change."
    exit 0
fi

for file in "${files[@]}"; do
    perl -pi -e "s/^(#SBATCH\s+--account=)[a-z0-9]+\@/\${1}${project}\@/" "$file"
done
echo "rewritten. Remaining references to any other project:"
grep -rn -E '^#SBATCH[[:space:]]+--account=' "$root/scripts" | grep -v "=$project@" || echo "  none"
