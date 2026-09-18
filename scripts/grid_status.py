#!/usr/bin/env python3
"""Print the scale-by-task grid as a table.

A file rather than an inline command: the previous monitor passed a Python
f-string through three layers of shell quoting and produced a SyntaxError on
every poll for five hours, so nothing was watching while the grid sat queued.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLUMNS = [
    ("base", "base"),
    ("raw", "raw"),
    ("rank1_b1", "r1@1b"),
    ("rank1_b16", "r1@16b"),
    ("scale_head_0p5", "a=0.5"),
    ("norm_matched_b1", "norm"),
    ("tail_b16", "tail"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports/scaling_grid"))
    args = parser.parse_args()

    path = args.out / "summary.json"
    if not path.is_file():
        print(f"no summary at {path} yet")
        return
    document = json.loads(path.read_text())
    arms = document.get("arms", {})
    if not arms:
        print("summary exists but holds no arms yet")
        return

    head = f"{'cell':<18}" + "".join(f"{label:>8}" for _, label in COLUMNS)
    print(head + f"{'damage':>9}{'best-rec':>10}")
    print("-" * len(head + " " * 19))
    for name in sorted(arms):
        entry = arms[name]
        scores = entry.get("scores", {})
        row = f"{name:<18}"
        for key, _ in COLUMNS:
            value = scores.get(key)
            row += f"{value:>8.3f}" if isinstance(value, (int, float)) else f"{'-':>8}"
        damage = entry.get("damage")
        row += f"{damage:>9.3f}" if isinstance(damage, (int, float)) else f"{'-':>9}"
        recovery = entry.get("recovery", {})
        scored = {
            key: value
            for key, value in recovery.items()
            if key not in {"base", "raw"} and isinstance(value, (int, float))
        }
        if scored:
            best = max(scored, key=lambda key: scored[key])
            row += f"{scored[best]:>7.2f} {best[:12]}"
        print(row)
    print(f"\nstatus: {document.get('status')}  cells: {len(arms)}")


if __name__ == "__main__":
    main()
