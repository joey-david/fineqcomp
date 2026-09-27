"""Spectral profile of every archived adapter in the given run tars.

Streams each tar once and reads only `<run_id>/raw_channel.pt`, so nothing is
unpacked onto $WORK. Run on prepost:
    python extract.py OUT.csv TAR [TAR ...]
"""
from __future__ import annotations

import csv
import io
import sys
import tarfile

import torch

from fineqcomp.studies.generalisation import spectral_profile


def main() -> None:
    out, tars = sys.argv[1], sys.argv[2:]
    rows = []
    for path in tars:
        with tarfile.open(path, mode="r|") as archive:
            for member in archive:
                if not member.name.endswith("/raw_channel.pt"):
                    continue
                run_id = member.name.rstrip("/").split("/")[-2]
                tensors = torch.load(io.BytesIO(archive.extractfile(member).read()),
                                     map_location="cpu", weights_only=True)
                rows.append({"run_id": run_id, "tar": path.split("/")[-1], **spectral_profile(tensors)})
                print(run_id, flush=True)
    with open(out, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
