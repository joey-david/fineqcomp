"""Plot the average accuracy contrast for including each SVD direction.

For each singular direction, the mean held-out accuracy of every contiguous
window that contains it, minus the mean of every window that does not. Reads
the window CSVs written by fine_windows.py and draws one figure per study plus
a stacked panel of all of them, each as a vector PDF and a PNG preview.

Windows are contiguous, so any window that contains direction k and starts at
0 also contains direction 1, and the plain contrast for early directions
partly measures direction 1. `R7_all_..._without_direction1` repeats the
contrast over windows that leave direction 1 out, which isolates directions
2-16 from it.

Usage: python plot_band_contribution.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RANK = 16
# (window CSV from fine_windows.py, output tag, panel title)
STUDIES = [
    ("R6_every_window", "R7", "Qwen2.5-7B, MetaMathQA"),
    ("R6c_every_window_qwen14b", "R7c", "Qwen2.5-14B, MetaMathQA"),
    ("R6d_every_window_qwen32b", "R7d", "Qwen2.5-32B, MetaMathQA"),
    ("R6b_every_window_gemma", "R7b", "Gemma-2-9B, NuminaMath-CoT"),
]
# Windows are scored on reserved training-corpus rows, not on the test set.
YLABEL = "mean held-out accuracy:\ninclude minus exclude"


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


def draw(ax, effects: list[float], labels: bool) -> None:
    x = np.arange(RANK - len(effects), RANK)
    colors = ["#d97745" if value < 0 else "#236a9b" for value in effects]
    bars = ax.bar(x, effects, width=0.72, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="#30343b", linewidth=1.1)
    if labels:
        for bar, value in zip(bars, effects):
            offset = 0.006 if value >= 0 else -0.006
            ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.3f}",
                    ha="center", va="bottom" if value >= 0 else "top", fontsize=8, color="#30343b")
    ax.set_xticks(np.arange(RANK), [str(d) for d in range(1, RANK + 1)])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#b8bdc3")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", color="#e7e9eb", linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(HERE / "figures" / f"{name}.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    panels = []
    for source, tag, title in STUDIES:
        if not (HERE / f"{source}.csv").exists():
            continue
        rows = contributions(source)
        with (HERE / f"{tag}_per_direction_contribution.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        effects = [r["difference"] for r in rows]
        panels.append((title, effects))
        fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
        draw(ax, effects, labels=True)
        ax.set_xlabel("singular direction included (one-based label)")
        ax.set_ylabel(YLABEL)
        ax.set_title(title, loc="left", fontsize=13, pad=10)
        ax.set_ylim(min(-0.02, min(effects) - 0.025), max(0.02, max(effects) + 0.03))
        save(fig, f"{tag}_per_direction_contribution")

    stacked(panels, "R7_all_per_direction_contribution")
    stacked([(title, [r["difference"] for r in contributions(source, skip_first=True)])
             for source, _, title in STUDIES if (HERE / f"{source}.csv").exists()],
            "R7_all_per_direction_contribution_without_direction1")


def stacked(panels: list, name: str) -> None:
    low = min(min(e) for _, e in panels) - 0.02
    high = max(max(e) for _, e in panels) + 0.02
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.2 * len(panels)), sharex=True,
                             layout="constrained")
    for ax, (title, effects) in zip(axes, panels):
        draw(ax, effects, labels=False)
        ax.set_ylim(low, high)
        ax.set_title(title, loc="left", fontsize=10)
    axes[-1].set_xlabel("singular direction included (one-based label)")
    fig.supylabel(YLABEL, fontsize=10)
    save(fig, name)


if __name__ == "__main__":
    main()
