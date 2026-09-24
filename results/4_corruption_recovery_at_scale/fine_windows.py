"""Every contiguous window of the rank-16 corrupted update, scored on the search split.

Usage: python fine_windows.py [STUDY NAME]  (default: spectral_regimes_fine_v1 R6_every_window)
Reads <STUDY>/search. Cell (low, high) keeps singular
directions low..high-1 at full precision and drops the rest. The search split
is 128 reserved MetaMathQA rows, one binomial standard error is about 0.044,
and on it the corrupted adapter barely differs from base, so the map shows what
windows add over base, not how much of the damage they remove.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY, NAME = (sys.argv[1:3] if len(sys.argv) > 2
               else ("spectral_regimes_fine_v1", "R6_every_window"))
SEARCH = ROOT / ".cache/reports" / STUDY / "search"
HERE = Path(__file__).resolve().parent
RANK = 16


def main() -> None:
    accuracy = {path.parent.parent.name: json.loads(path.read_text())["exact_match"]
                for path in SEARCH.glob("*/search/rows*.json")}
    base = accuracy["base"]
    grid = np.full((RANK, RANK), np.nan)
    rows = []
    for low in range(RANK):
        for high in range(low + 1, RANK + 1):
            # The full window is the corrupted adapter itself, which the grid omits.
            value = accuracy.get(f"band{low:02d}_{high:02d}_fp16", accuracy["permuted_raw"])
            grid[high - 1, low] = value - base
            rows.append({"low": low, "high": high, "accuracy": value, "minus_base": value - base})
    for i in range(RANK):
        rows.append({"low": f"all but {i}", "high": "", "accuracy": accuracy[f"drop{i:02d}"],
                     "minus_base": accuracy[f"drop{i:02d}"] - base})
    with (HERE / f"{NAME}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    limit = np.nanmax(np.abs(grid))
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(grid, cmap="RdBu", vmin=-limit, vmax=limit, origin="lower")
    ax.set_xticks(range(0, RANK, 2))
    ax.set_yticks(range(0, RANK, 2))
    ax.set_xlabel("first direction kept")
    ax.set_ylabel("last direction kept")
    fig.colorbar(image, ax=ax, label="accuracy minus frozen base")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / f"{NAME}.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
