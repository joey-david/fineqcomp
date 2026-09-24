#!/usr/bin/env bash
# Pack finished directories on $WORK into one tar each on $STORE, to stay under
# the project's 500k-file quota on $WORK. Each original is removed only after
# its tar lists exactly as many entries as the directory held. Restore with
#   tar -xf "$STORE/packs/<path>.tar" -C "$WORK"
# Submit on prepost, never on the login node:
#   sbatch -A fas@cpu -p prepost -n1 -c2 --time=02:00:00 -o "$HOME/pack-%j.out" \
#     scripts/jean_zay_pack.sh paper_experiments/2026_09_08 ...
# Paths are relative to $WORK.
#SBATCH --job-name=fq-pack
set -uo pipefail
work=/lustre/fswork/projects/rech/fas/uul94gf
store=/lustre/fsstor/projects/rech/fas/uul94gf/packs
status=0
cd "$work"
for dir in "$@"; do
  [[ -d "$dir" ]] || { echo "skip $dir: not a directory"; continue; }
  archive="$store/$dir.tar"
  mkdir -p "$(dirname "$archive")"
  count=$(lfs find "$dir" | wc -l)
  if ! tar -cf "$archive" "$dir"; then echo "TAR FAILED $dir"; status=1; continue; fi
  listed=$(tar -tf "$archive" | wc -l)
  if [[ "$count" -eq "$listed" ]]; then
    rm -rf "$dir" && echo "packed $dir: $count entries -> $archive"
  else
    echo "COUNT MISMATCH $dir: $count on disk, $listed in tar; original kept"
    status=1
  fi
done
lfs quota -p 20304691 /lustre/fswork | tail -1
exit "$status"
