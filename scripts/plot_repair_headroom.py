#!/usr/bin/env python3
"""Figures for the repair-headroom ladder, in the house style of figures_top10.

Two panels, one figure. Left: what corrupting a depended-on value costs each
model, against what corrupting an ignored line costs -- the dissociation, across
the ladder. Right: whether the model says anything when the corruption breaks
it, which is the part the prior work (BIG-Bench Mistake) predicts is hard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = {
    "light": dict(ink="#52514e", grid="#e2e1dd", strong="#0b0b0b",
                  orange="#eb6834", blue="#2a78d6", green="#1baf7a"),
    "dark": dict(ink="#c3c2b7", grid="#3a3a37", strong="#f3f2ec",
                 orange="#d95926", blue="#3987e5", green="#199e70"),
}
# Ascending parameter count, which is the x-axis; the label is what a reader
# recognises, the number is what orders them.
ORDER = [("Qwen/Qwen2.5-3B-Instruct", "2.5-3B", 3), ("Qwen/Qwen3-4B", "3-4B", 4),
         ("Qwen/Qwen2.5-7B-Instruct", "2.5-7B", 7), ("Qwen/Qwen3-8B", "3-8B", 8),
         ("Qwen/Qwen2.5-14B-Instruct", "2.5-14B", 14), ("Qwen/Qwen3.6-27B", "3.6-27B", 27)]


def style(ax, c):
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["ink"]); ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=c["ink"], labelsize=8, length=3, width=0.8)
    ax.yaxis.label.set_color(c["ink"]); ax.xaxis.label.set_color(c["ink"])
    ax.grid(axis="y", color=c["grid"], lw=0.7, zorder=0); ax.set_axisbelow(True)


def build(summary, c):
    paired = summary["paired_repair"]
    present = [(key, label) for key, label, _ in ORDER if key in paired]
    x = range(len(present))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 3.4))

    broken = [1 - paired[k]["repair_rate"] for k, _ in present]
    ignored = [1 - paired[k]["distract_survival"] for k, _ in present]
    ax.plot(x, broken, "-o", color=c["orange"], lw=2, ms=7,
            label="corrupted value the answer needs")
    ax.plot(x, ignored, "-o", color=c["blue"], lw=2, ms=7,
            label="corrupted line the answer ignores")
    for i, v in enumerate(broken):
        ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7.5, color=c["ink"])
    ax.set_xticks(list(x), [lab for _, lab in present], fontsize=8)
    ax.set_ylabel("fraction of solvable items broken", fontsize=8.5)
    ax.set_ylim(-0.04, 1.0)
    ax.set_title("Corrupting work the answer depends on", fontsize=9,
                 color=c["strong"], pad=8)
    style(ax, c)
    legend = ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    for text in legend.get_texts():
        text.set_color(c["ink"])

    flags = [paired[k]["flag_when_broken"] for k, _ in present]
    bx.bar(list(x), flags, color=c["green"], width=0.6)
    for i, v in enumerate(flags):
        bx.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5, color=c["ink"])
    bx.set_xticks(list(x), [lab for _, lab in present], fontsize=8)
    bx.set_ylabel("flagged an error when broken", fontsize=8.5)
    bx.set_ylim(0, max(0.3, max(flags) * 1.35) if flags else 1)
    bx.set_title("Did it notice?", fontsize=9, color=c["strong"], pad=8)
    style(bx, c)
    fig.tight_layout()
    return fig


def main() -> None:
    summary = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    slug = "13_repair-headroom-ladder"
    for mode, colours in MODES.items():
        fig = build(summary, colours)
        fig.savefig(out / f"{slug}.{mode}.svg", transparent=True, format="svg")
        if mode == "light":
            fig.savefig(out / f"{slug}.png", dpi=200, facecolor="#fcfcfb")
        plt.close(fig)
    print(f"wrote {slug}")


if __name__ == "__main__":
    main()
