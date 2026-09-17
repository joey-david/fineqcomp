#!/usr/bin/env bash
# Move the heavy artifacts out of $WORK/fineQComp/runs and into $STORE.
#
# WORK is the small, fast, inode-limited space: the fas project sits at 90% of
# its 500,000 inodes while using only 20% of its 5 TiB, so what has to shrink is
# file count as much as bytes. STORE is empty (50 TiB, 100,000 inodes) and is
# what IDRIS provides for exactly this. Nothing is deleted that is not first
# written into a tar on STORE and verified against the file list that built it.
#
# What moves:
#   *.fqcb, *.fqmdl   every codec output, in every run. These are deterministic
#                     functions of the adapter and the codec settings, and the
#                     measurement they carry -- the file size in bytes -- is
#                     recorded in the manifest before they go.
#   raw_channel.pt    the trained adapter, for every run EXCEPT those whose name
#                     matches KEEP_PATTERN. Retraining one of these costs GPU
#                     hours, so they are archived, never dropped.
#
# What stays on WORK: every .json and .jsonl (run configs, metrics, predictions
# -- the evidence, 1.5 GB) and the adapters of the live study line, so the run
# directories stay browsable and the current work needs no restore.
#
# Idempotent: a family whose tar already exists and verifies is skipped.

set -euo pipefail

runs_root="${RUNS_ROOT:-$WORK/fineQComp/runs}"
archive_root="${ARCHIVE_ROOT:-$STORE/fineQComp-archive/runs}"
keep_pattern="${KEEP_PATTERN:-cot-math}"
dry_run="${DRY_RUN:-0}"

cd "$runs_root"
mkdir -p "$archive_root"
work_list="$(mktemp -d)"
trap 'rm -rf "$work_list"' EXIT

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(stamp)] $*"; }

# The manifest records the exact byte size of every file that is about to move.
# For a codec output that size *is* the measurement the study reports, so it
# must survive on WORK even when the file itself does not.
manifest="$runs_root/ARCHIVE_MANIFEST.tsv"
if [[ ! -s "$manifest" ]]; then
    say "writing manifest to $manifest"
    find . \( -name '*.fqcb' -o -name '*.fqmdl' -o -name 'raw_channel.pt' \) \
        -type f -printf '%s\t%TY-%Tm-%Td\t%p\n' | sort -k3 > "$manifest"
fi
say "manifest holds $(wc -l < "$manifest") files"

families="$(ls -d */ 2>/dev/null | sed 's#/$##' | sed 's/__.*//' | sort -u)"
say "$(echo "$families" | wc -l) run families under $runs_root"

moved_total=0
skipped_total=0

for family in $families; do
    list="$work_list/$family.list"
    : > "$list"

    # Every codec output in this family, whatever the run.
    find . -maxdepth 3 -path "./${family}__*" \( -name '*.fqcb' -o -name '*.fqmdl' \) \
        -type f -printf '%p\n' 2>/dev/null | sed 's#^\./##' >> "$list" || true

    # Adapters, except for the runs the live study line still reads.
    while IFS= read -r adapter; do
        case "$adapter" in
            *"$keep_pattern"*) continue ;;
        esac
        echo "${adapter#./}" >> "$list"
    done < <(find . -maxdepth 2 -path "./${family}__*" -name 'raw_channel.pt' -type f 2>/dev/null)

    if [[ ! -s "$list" ]]; then
        continue
    fi
    sort -u -o "$list" "$list"
    count="$(wc -l < "$list")"
    bytes="$(tr '\n' '\0' < "$list" | xargs -0 stat -c '%s' 2>/dev/null | awk '{s+=$1} END {print s+0}')"
    tarball="$archive_root/$family.tar"

    if [[ "$dry_run" == "1" ]]; then
        printf '%-28s %5d files %8.2f GB -> %s\n' "$family" "$count" \
            "$(awk -v b="$bytes" 'BEGIN {print b/1073741824}')" "$tarball"
        moved_total=$((moved_total + count))
        continue
    fi

    if [[ -f "$tarball" ]]; then
        in_tar="$(tar -tf "$tarball" | grep -c . || true)"
        if [[ "$in_tar" -eq "$count" ]]; then
            say "$family: tar already holds $count files, removing sources"
        else
            say "$family: EXISTING TAR HOLDS $in_tar, EXPECTED $count -- skipping, nothing removed"
            skipped_total=$((skipped_total + count))
            continue
        fi
    else
        say "$family: archiving $count files ($(awk -v b="$bytes" 'BEGIN {printf "%.2f", b/1073741824}') GB)"
        tar -cf "$tarball.partial" -T "$list"
        mv "$tarball.partial" "$tarball"
    fi

    # Verify before deleting: every listed path must be present in the tar with
    # the same byte size. A tar that is short by one file deletes nothing.
    tar -tvf "$tarball" | awk '{print $3, $NF}' | sort -k2 > "$work_list/$family.intar"
    tr '\n' '\0' < "$list" | xargs -0 stat -c '%s %n' | sort -k2 > "$work_list/$family.ondisk"
    if ! diff -q "$work_list/$family.intar" "$work_list/$family.ondisk" >/dev/null; then
        say "$family: VERIFY FAILED (size or membership mismatch) -- nothing removed"
        skipped_total=$((skipped_total + count))
        continue
    fi

    while IFS= read -r path; do
        rm -f "$path"
    done < "$list"
    # Leave no empty codecs/ directories behind; they are pure inode cost.
    find . -maxdepth 3 -path "./${family}__*" -type d -name codecs -empty -delete 2>/dev/null || true
    moved_total=$((moved_total + count))
    say "$family: verified and removed $count files from WORK"
done

say "done: $moved_total files archived, $skipped_total left in place"
if [[ "$dry_run" != "1" ]]; then
    say "runs/ now $(du -sh "$runs_root" 2>/dev/null | cut -f1)"
    say "archive now $(du -sh "$archive_root" 2>/dev/null | cut -f1) in $(ls "$archive_root" | wc -l) tars"
fi
