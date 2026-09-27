"""Compute direction contrasts from the recorded window-search CSVs."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2] / "results/recovery"
RANK = 16

def contributions(name: str, skip_first: bool = False) -> list[dict]:
    with (HERE / f"{name}.csv").open() as f:
        windows = [row for row in csv.DictReader(f) if row["low"].isdigit() and row["high"]]
    if skip_first:
        windows = [row for row in windows if int(row["low"]) >= 1]
    rows = []
    for direction in range(1 if skip_first else 0, RANK):
        inside = [float(r["accuracy"]) for r in windows if int(r["low"]) <= direction < int(r["high"])]
        outside = [float(r["accuracy"]) for r in windows if not int(r["low"]) <= direction < int(r["high"])]
        rows.append({"direction": direction + 1, "mean_accuracy_including": float(np.mean(inside)),
                     "mean_accuracy_excluding": float(np.mean(outside)),
                     "difference": float(np.mean(inside) - np.mean(outside)),
                     "bands_including": len(inside), "bands_excluding": len(outside)})
    return rows
