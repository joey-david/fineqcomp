#!/usr/bin/env python3
"""Where in the spectrum of the damaged update does the recovery live?

The compression that recovers keeps the top of the spectrum and drops the rest,
which invites the reading that the top singular direction is signal and the tail
is memorised noise. Slicing the spectrum tests it: a prefix always contains the
top direction, so only the middle and tail bands can falsify the claim.

Each band is evaluated twice — at full precision and through a one-bit file —
because the two axes turn out not to be interchangeable. A band at fp16 says
what that part of the update does; the same band at one bit says what survives
compressing it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODES = {
    "light": dict(ink="#52514e", grid="#e2e1dd", strong="#0b0b0b", orange="#eb6834",
                  blue="#2a78d6", green="#1baf7a", grey="#9b9a95", paper="#fcfcfb"),
    "dark": dict(ink="#c3c2b7", grid="#3a3a37", strong="#f3f2ec", orange="#d95926",
                 blue="#3987e5", green="#199e70", grey="#77766f", paper="#141414"),
}
# Only the disjoint partition is plotted: ranks 1-4 and 1-8 are unions of the
# first two and are reported in the table instead, where nesting is explicit.
BANDS = [((0, 1), "rank 1\n(top direction)"), ((1, 4), "ranks 2–4\n(no top direction)"),
         ((4, 16), "ranks 5–16\n(the tail)")]
BASE, DAMAGED = 74.8, 23.0


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


def figure(root: Path, energy: dict, c):
    rows = []
    for (lo, hi), label in BANDS:
        a, b = root / f"spec_{lo}_{hi}_b16.json", root / f"spec_{lo}_{hi}_b1.json"
        if not (a.exists() and b.exists()):
            continue
        rows.append((label, energy.get(f"{lo}_{hi}", float("nan")) * 100,
                     json.load(open(a))["exact_match"] * 100,
                     json.load(open(b))["exact_match"] * 100))
    if not rows:
        return None
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.4, 4.0),
                                 gridspec_kw={"width_ratios": [1.45, 1]})
    xs = np.arange(len(rows))
    # Both reference labels go at the right-hand end, where no bar reaches.
    for y, name, colour in ((BASE, "base model", "green"),
                            (DAMAGED, "damaged adapter", "strong")):
        ax.axhline(y, color=c[colour], lw=0.9, ls=(0, (4, 3)), zorder=2)
        ax.text(0.988, y + 1.3, name, transform=ax.get_yaxis_transform(),
                fontsize=7.5, color=c[colour], ha="right")
    ax.bar(xs - 0.19, [r[2] for r in rows], width=0.36, color=c["blue"], zorder=3,
           label="kept at full precision")
    ax.bar(xs + 0.19, [r[3] for r in rows], width=0.36, color=c["orange"], zorder=3,
           label="through a 1-bit file")
    for x, r in zip(xs, rows):
        ax.text(x - 0.19, r[2] + 1.5, f"{r[2]:.0f}", ha="center", fontsize=7.5,
                color=c["ink"])
        ax.text(x + 0.19, r[3] + 1.5, f"{r[3]:.0f}", ha="center", fontsize=7.5,
                color=c["ink"])
        # The arrow is the finding: which way compressing this slice moves it.
        lo_y, hi_y = sorted((r[2], r[3]))
        up = r[3] > r[2]
        ax.annotate("", xy=(x, hi_y - 1.5), xytext=(x, lo_y + 1.5),
                    arrowprops=dict(arrowstyle="-|>", lw=1.1,
                                    color=c["orange"] if up else c["ink"],
                                    shrinkA=0, shrinkB=0), zorder=5)
        # Above the group, not between the bars: the gap between them is
        # narrower than the label and the text disappeared behind the bar.
        ax.text(x, hi_y + 6.5, f"{r[3] - r[2]:+.0f}", fontsize=9, ha="center",
                color=c["orange"] if up else c["ink"], fontweight="bold")
    ax.set_xticks(xs, [r[0] for r in rows], fontsize=8)
    ax.set_ylabel("GSM8K %", fontsize=8.5)
    ax.set_ylim(0, 104)
    # Room at the right end for the two reference labels, which otherwise sit
    # on top of the last group's bars.
    ax.set_xlim(-0.62, len(rows) - 1 + 1.05)
    ax.set_title("Compressing a slice helps only where there is damage in it",
                 fontsize=9.5, color=c["strong"], pad=8)
    ax.grid(axis="y", color=c["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    style(ax, c)
    leg = ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    for t in leg.get_texts():
        t.set_color(c["ink"])

    parts = [r for r in rows if r[0].startswith(("rank 1", "ranks 2", "ranks 5"))]
    bx.bar(range(len(parts)), [r[1] for r in parts], width=0.6,
           color=[c["strong"] if r[0].startswith("rank 1") else c["grey"]
                  for r in parts], zorder=3)
    for i, r in enumerate(parts):
        bx.text(i, r[1] + 1.2, f"{r[1]:.1f}%", ha="center", fontsize=8, color=c["ink"])
    bx.set_xticks(range(len(parts)), [r[0].split("\n")[0] for r in parts], fontsize=8)
    bx.set_ylabel("share of the update's energy", fontsize=8.5)
    bx.set_ylim(0, 62)
    bx.set_title("The damage sits where the energy is",
                 fontsize=9.5, color=c["strong"], pad=8)
    bx.grid(axis="y", color=c["grid"], lw=0.7, zorder=0)
    bx.set_axisbelow(True)
    style(bx, c)
    fig.tight_layout()
    return fig


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    energy_path = root / "spectral_energy.json"
    energy = json.load(open(energy_path)) if energy_path.exists() else {}
    out.mkdir(parents=True, exist_ok=True)
    for mode, colours in MODES.items():
        fig = figure(root, energy, colours)
        if fig is None:
            print("no spectral cells yet")
            return
        fig.savefig(out / f"24_spectral-bands.{mode}.svg", transparent=True, format="svg")
        if mode == "light":
            fig.savefig(out / "24_spectral-bands.png", transparent=False, dpi=200,
                        facecolor=colours["paper"])
        plt.close(fig)
    print("wrote 24_spectral-bands")


if __name__ == "__main__":
    main()
