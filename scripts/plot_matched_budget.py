#!/usr/bin/env python3
"""Figures for the matched-bit-budget study, in the house style of figures_top10.

Two figures, because the study answers two separate questions and a single panel
that tried to carry both would make neither legible.

  matched-bit-budget-grid   the rank-by-precision surface of one damaged adapter,
                            plus the quantity the experiment is actually for: the
                            accuracy spread across ways of spending one budget.
  compression-not-low-rank  the dissociation. Training narrow and compressing a
                            wide adapter reach the same final rank, bit width and
                            file size, and do not reach the same accuracy.

Numbers are read from summary.json rather than typed in, so a rerun of the study
redraws rather than silently disagreeing with the report.

Ink follows the existing top-ten plots: transparent background, one palette per
mode, and nothing but the ink hexes differing between light and dark.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MODES = {
    "light": dict(ink="#52514e", grid="#e2e1dd", strong="#0b0b0b",
                  orange="#eb6834", blue="#2a78d6", green="#1baf7a",
                  ramp=["#eef3fa", "#b8d0ee", "#6fa2e0", "#2a78d6", "#14406f"],
                  hi_text="#fcfcfb", lo_text="#0b0b0b"),
    # Dark is stepped for the dark surface, not flipped: the ramp keeps one hue
    # but runs dim-to-bright, because on a dark ground brightness reads as "more".
    "dark": dict(ink="#c3c2b7", grid="#3a3a37", strong="#f3f2ec",
                 orange="#d95926", blue="#3987e5", green="#199e70",
                 ramp=["#1d2730", "#22456e", "#2f6bad", "#3987e5", "#8fbcf2"],
                 hi_text="#0b0b0b", lo_text="#c3c2b7"),
}
RANKS = [1, 2, 4, 8, 16]
BITS = [1, 2, 4, 8, 16]


def load(path: Path):
    d = json.load(open(path))
    grid, raw_by_rank, base = {}, {}, None
    for row in d["rows"]:
        base = row["base_accuracy"]["gsm8k"] * 100
        raw_by_rank[row["trained_rank"]] = row["raw_accuracy"]["gsm8k"] * 100
        grid.setdefault(row["arm"], {})[row["key"]] = row["gsm8k_accuracy"] * 100
    spreads = {m["budget_units"]: m["spread"] * 100
               for m in d["matched_budgets"] if m["arm"] == "permuted_r16"}
    return grid, raw_by_rank, base, spreads


def style(ax, c):
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["ink"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=c["ink"], labelsize=8, length=3, width=0.8)
    ax.yaxis.label.set_color(c["ink"])
    ax.xaxis.label.set_color(c["ink"])


def figure_grid(grid, base, spreads, c):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 3.5),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    cmap = LinearSegmentedColormap.from_list("acc", c["ramp"])
    wide = grid["permuted_r16"]
    matrix = [[wide[f"r{r}_b{b}"] for b in BITS] for r in RANKS]
    ax.imshow(matrix, cmap=cmap, vmin=20, vmax=85, aspect="auto")
    for i, r in enumerate(RANKS):
        for j, b in enumerate(BITS):
            v = wide[f"r{r}_b{b}"]
            # Value on every cell: the contrast check warns on the pale steps, and
            # a grid this small is read cell by cell rather than off the ramp.
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color=c["hi_text"] if v > 55 else c["lo_text"])
    ax.set_xticks(range(len(BITS)), [str(b) for b in BITS])
    ax.set_yticks(range(len(RANKS)), [str(r) for r in RANKS])
    ax.set_xlabel("bits per value", fontsize=8.5)
    ax.set_ylabel("truncation rank", fontsize=8.5)
    ax.set_title("GSM8K % after compressing the damaged rank-16 adapter",
                 fontsize=9, color=c["strong"], pad=20)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=c["ink"], labelsize=8, length=0)
    # The anti-diagonal is the comparison: every cell on it stores the same
    # number of coded bits, so differences along it are differences in how the
    # bits were spent, not in how many. Outlining the cells rather than drawing a
    # line through their centres keeps the marker off the values it is pointing at.
    from matplotlib.patches import Rectangle
    for j, b in enumerate(BITS):
        for i, r in enumerate(RANKS):
            if r * b != 16:
                continue
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=c["orange"], lw=1.8, zorder=4))
    # A heatmap has no empty space, so the note explaining the outlines goes
    # above the axes rather than on top of the cells it is describing.
    ax.text(0.5, 1.045, "outlined cells store equal coded bits (16 units)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5,
            color=c["orange"])

    budgets = sorted(spreads)
    bx.bar(range(len(budgets)), [spreads[b] for b in budgets],
           color=c["orange"], width=0.62)
    for i, b in enumerate(budgets):
        bx.text(i, spreads[b] + 1.2, f"{spreads[b]:.1f}", ha="center",
                fontsize=7.5, color=c["ink"])
    bx.set_xticks(range(len(budgets)), [str(b) for b in budgets])
    bx.set_xlabel("budget (rank × bits)", fontsize=8.5)
    bx.set_ylabel("accuracy spread (points)", fontsize=8.5)
    bx.set_title("Spread across equal-budget cells", fontsize=9,
                 color=c["strong"], pad=8)
    bx.set_ylim(0, 44)
    bx.grid(axis="y", color=c["grid"], lw=0.7, zorder=0)
    bx.set_axisbelow(True)
    style(bx, c)
    bx.annotate("how you spend the bits\nstops mattering here",
                xy=(4.0, 3.0), xytext=(3.15, 22), fontsize=7.5, color=c["ink"],
                ha="left", arrowprops=dict(arrowstyle="->", color=c["ink"], lw=0.8))
    fig.tight_layout()
    return fig


def figure_dissociation(grid, raw_by_rank, base, c):
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ranks = [1, 2, 4, 16]
    arms = {1: "permuted_r1", 2: "permuted_r2", 4: "permuted_r4", 16: "permuted_r16"}
    uncompressed = [raw_by_rank[r] for r in ranks]
    compressed = [grid[arms[r]]["r1_b1"] for r in ranks]
    x = range(len(ranks))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], uncompressed, w, color=c["blue"],
                label="as trained (fp16, uncompressed)")
    b2 = ax.bar([i + w / 2 for i in x], compressed, w, color=c["orange"],
                label="compressed to rank 1 @ 1 bit")
    for bars in (b1, b2):
        for rect in bars:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1.5,
                    f"{rect.get_height():.1f}", ha="center", fontsize=7.5,
                    color=c["ink"])
    ax.axhline(base, color=c["green"], lw=1.4, ls=(0, (4, 2)), zorder=1)
    # Anchored left: the right-hand end of this line runs across the tallest
    # bars, and a reference label sitting on a bar reads as if it labels it.
    ax.text(-0.46, base + 1.6, f"base model {base:.1f}",
            fontsize=7.5, color=c["green"], ha="left")
    ax.set_xticks(list(x), [f"rank {r}" for r in ranks])
    ax.set_ylabel("GSM8K %", fontsize=8.5)
    ax.set_ylim(0, 95)
    ax.set_title("Compression is not the same thing as training small",
                 fontsize=9, color=c["strong"], pad=8)
    ax.grid(axis="y", color=c["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    style(ax, c)
    legend = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for text in legend.get_texts():
        text.set_color(c["ink"])
    fig.tight_layout()
    return fig


def main() -> None:
    summary = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    grid, raw_by_rank, base, spreads = load(summary)
    for slug, builder in (
        ("11_matched-bit-budget-grid", lambda c: figure_grid(grid, base, spreads, c)),
        ("12_compression-not-low-rank",
         lambda c: figure_dissociation(grid, raw_by_rank, base, c)),
    ):
        for mode, colours in MODES.items():
            fig = builder(colours)
            fig.savefig(out / f"{slug}.{mode}.svg", transparent=True, format="svg")
            if mode == "light":
                fig.savefig(out / f"{slug}.png", transparent=False, dpi=200,
                            facecolor="#fcfcfb")
            plt.close(fig)
        print(f"wrote {slug} (png + light/dark svg)")


if __name__ == "__main__":
    main()
