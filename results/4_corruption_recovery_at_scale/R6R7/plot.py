"""Window maps (R6) and per-direction contributions (R7) for Qwen2.5-7B, Qwen2.5-14B and Mistral-7B.

Writes, as vector PDF plus PNG preview:
  R6_<model>: accuracy minus frozen base for every contiguous window of the
      corrupted rank-16 update, on the 128-row search split;
  R7_<model>: mean accuracy of windows that include each direction minus the
      mean of windows that exclude it;
  R6R7_collage: window maps on the top row, contributions on the bottom, one
      column per model.
All three maps share one colour scale and all three histograms one y-axis.

Usage: python plot.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from plot_band_contribution import RANK, YLABEL, contributions, draw  # noqa: E402

MODELS = [
    ("R6_every_window", "qwen2.5_7b", "Qwen2.5-7B"),
    ("R6c_every_window_qwen14b", "qwen2.5_14b", "Qwen2.5-14B"),
    # Mistral's adapters were trained and scored on GSM8K, the Qwens' on MetaMathQA.
    ("R6e_every_window_mistral7b", "mistral_7b", "Mistral-7B (GSM8K)"),
]
EDGES = np.arange(RANK + 1) + 0.5
TICKS = [1, 4, 8, 12, 16]


def window_grid(name: str) -> np.ndarray:
    grid = np.full((RANK, RANK), np.nan)
    with (HERE.parent / f"{name}.csv").open() as f:
        for row in csv.DictReader(f):
            if row["low"].isdigit():
                grid[int(row["high"]) - 1, int(row["low"])] = float(row["minus_base"])
    return grid


def draw_map(ax, grid: np.ndarray, limit: float):
    # pcolormesh keeps each cell a vector patch; imshow would embed a raster.
    mesh = ax.pcolormesh(EDGES, EDGES, grid, cmap="RdBu", vmin=-limit, vmax=limit)
    ax.set_aspect("equal")
    ax.set_xticks(TICKS)
    ax.set_yticks(TICKS)
    ax.set_xlabel("first direction kept")
    ax.set_ylabel("last direction kept")
    ax.spines[["top", "right"]].set_visible(False)
    return mesh


def colorbar(fig, mesh, ax, **kwargs) -> None:
    bar = fig.colorbar(mesh, ax=ax, label="accuracy minus frozen base", **kwargs)
    # Matplotlib rasterizes long colorbar gradients by default.
    bar.solids.set_rasterized(False)


def save(fig, name: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    grids = [window_grid(source) for source, _, _ in MODELS]
    effects = [[r["difference"] for r in contributions(source)] for source, _, _ in MODELS]
    limit = max(np.nanmax(np.abs(g)) for g in grids)
    ylim = (min(map(min, effects)) - 0.02, max(map(max, effects)) + 0.02)

    for (_, slug, title), grid, effect in zip(MODELS, grids, effects):
        fig, ax = plt.subplots(figsize=(5.2, 4.4), layout="constrained")
        colorbar(fig, draw_map(ax, grid, limit), ax)
        ax.set_title(title, loc="left")
        save(fig, f"R6_{slug}")

        fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
        draw(ax, effect, labels=True)
        ax.set_ylim(ylim)
        ax.set_xlabel("singular direction included")
        ax.set_ylabel(YLABEL)
        ax.set_title(title, loc="left")
        save(fig, f"R7_{slug}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), layout="constrained",
                             gridspec_kw={"height_ratios": [1.25, 1]})
    for column, ((_, _, title), grid, effect) in enumerate(zip(MODELS, grids, effects)):
        mesh = draw_map(axes[0, column], grid, limit)
        axes[0, column].set_title(title, loc="left", fontsize=12)
        draw(axes[1, column], effect, labels=False)
        axes[1, column].set_ylim(ylim)
        axes[1, column].set_xticks(np.array(TICKS) - 1, [str(t) for t in TICKS])
        axes[1, column].set_xlabel("singular direction included")
        if column:
            axes[0, column].set_ylabel("")
    axes[1, 0].set_ylabel(YLABEL)
    colorbar(fig, mesh, axes[0, :], shrink=0.9)
    save(fig, "R6R7_collage")


if __name__ == "__main__":
    main()
