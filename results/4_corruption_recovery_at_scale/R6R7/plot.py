"""Window maps (R6) and per-direction contributions (R7) across models.

Writes, as vector PDF plus PNG preview:
  R6_<model>: accuracy minus frozen base for every contiguous window [i, j) of
      the corrupted rank-16 update, on the 128-row search split;
  R7_<model>: mean accuracy of windows that include each direction minus the
      mean of windows that exclude it;
  R6R7_collage: window maps on the top row (a), contributions on the bottom
      row (b), one column per model.
All maps share one colour scale and all histograms one y-axis. Directions are
zero-based as in the paper; the maps are A10's layout turned 90 degrees
counter-clockwise, window start on x and window end on y. Style follows
figures4papers (github.com/ChenLiu-1996/figures4papers): Helvetica, thick
left/bottom spines only, black-edged bars, its blue/red palette, no grid.

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
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from plot_band_contribution import RANK, contributions  # noqa: E402

MODELS = [
    ("R6_every_window", "qwen2.5_7b", "Qwen2.5-7B"),
    ("R6c_every_window_qwen14b", "qwen2.5_14b", "Qwen2.5-14B"),
    ("R6d_every_window_qwen32b", "qwen2.5_32b", "Qwen2.5-32B"),
]
BLUE, RED = "#3775BA", "#B64342"
DIVERGING = LinearSegmentedColormap.from_list(
    "red_white_blue", [RED, "#E9A6A1", "#FFFFFF", "#9DBEE0", "#0F4D92"])
MAP_LABEL = "vs base (pp)"
BAR_LABEL = "include minus\nexclude (pp)"
plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 15,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 2,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "pdf.fonttype": 42,
})


def window_grid(name: str) -> np.ndarray:
    """Rows are window starts i, columns window ends j, values in points."""
    grid = np.full((RANK, RANK), np.nan)
    with (HERE.parent / f"{name}.csv").open() as f:
        for row in csv.DictReader(f):
            if row["low"].isdigit():
                grid[int(row["low"]), int(row["high"]) - 1] = 100 * float(row["minus_base"])
    return grid


def draw_map(ax, grid: np.ndarray, limit: float):
    # pcolormesh keeps each cell a vector patch; imshow would embed a raster.
    # Start on x, end on y: A10's layout turned 90 degrees counter-clockwise.
    mesh = ax.pcolormesh(np.arange(RANK + 1) - 0.5, np.arange(RANK + 1) + 0.5, grid.T,
                         cmap=DIVERGING, vmin=-limit, vmax=limit)
    ax.set_aspect("equal")
    ax.set_xticks([0, 4, 8, 12, 15])
    ax.set_yticks([1, 4, 8, 12, 16])
    ax.set_xlabel("window start")
    ax.set_ylabel("window end (exclusive)")
    return mesh


def draw_bars(ax, effects: list[float], ylim: tuple[float, float]) -> None:
    values = 100 * np.asarray(effects)
    ax.bar(np.arange(RANK), values, width=0.75, color=[RED if v < 0 else BLUE for v in values],
           edgecolor="black", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=1.5)
    ax.set_xticks([0, 4, 8, 12, 15])
    ax.set_xlim(-0.8, RANK - 0.2)
    ax.set_ylim(ylim)
    ax.set_xlabel("singular direction")


def colorbar(fig, mesh, ax, **kwargs) -> None:
    bar = fig.colorbar(mesh, ax=ax, label=MAP_LABEL, **kwargs)
    bar.outline.set_linewidth(1.5)
    # Matplotlib rasterizes long colorbar gradients by default.
    bar.solids.set_rasterized(False)


def save(fig, name: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    grids = [window_grid(source) for source, _, _ in MODELS]
    effects = [[r["difference"] for r in contributions(source)] for source, _, _ in MODELS]
    limit = max(np.nanmax(np.abs(g)) for g in grids)
    ylim = (100 * min(map(min, effects)) - 2, 100 * max(map(max, effects)) + 2)

    for (_, slug, title), grid, effect in zip(MODELS, grids, effects):
        fig, ax = plt.subplots(figsize=(5.6, 4.8), layout="constrained")
        colorbar(fig, draw_map(ax, grid, limit), ax)
        ax.set_title(title, loc="left")
        save(fig, f"R6_{slug}")

        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        draw_bars(ax, effect, ylim)
        ax.set_ylabel(BAR_LABEL)
        ax.set_title(title, loc="left")
        save(fig, f"R7_{slug}")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6), layout="constrained",
                             gridspec_kw={"height_ratios": [1.3, 1]})
    for column, ((_, _, title), grid, effect) in enumerate(zip(MODELS, grids, effects)):
        mesh = draw_map(axes[0, column], grid, limit)
        axes[0, column].set_title(title, loc="left", fontsize=17)
        draw_bars(axes[1, column], effect, ylim)
        if column:
            axes[0, column].set_ylabel("")
    axes[1, 0].set_ylabel(BAR_LABEL)
    for row, letter in enumerate("ab"):
        axes[row, 0].annotate(f"({letter})", (0, 1), xycoords="axes fraction",
                              xytext=(-62, 12), textcoords="offset points",
                              fontweight="bold", fontsize=17, va="bottom")
    colorbar(fig, mesh, axes[0, :], shrink=0.85)
    save(fig, "R6R7_collage")


if __name__ == "__main__":
    main()
