#!/usr/bin/env python3
"""What the rank-one direction reads, measured on activations.

A rank-one update writes b·(a·x), so at every token there is one scalar
coefficient per module saying how strongly the direction fires there. If the
update is a step-boundary detector, that coefficient should differ where a
reasoning step is about to end from where it is not.

Two panels. The left is the evidence that it does: the spread of standardised
differences across modules, against the spread sampling noise alone would
produce. The right is where that detection sits in the stack, plotted against
what each band is actually worth — which do not agree, and the disagreement is
the point.

The signed coefficient matters, not its magnitude: a direction that writes +b at
a boundary and -b elsewhere is perfectly selective and has a magnitude ratio of
exactly one. Measuring the magnitude was the first thing tried here and it
reported nothing.
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
# Accuracy gain of each band, from the layer-ablation experiment.
BAND_GAIN = {(0, 6): 4.8, (7, 13): 4.2, (14, 20): 2.6, (21, 27): 0.0}


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


def figure(data, c):
    gaps = np.array([v["standardised_gap"] for v in data.values()])
    n_at = np.array([v["n_at"] for v in data.values()])
    n_away = np.array([v["n_away"] for v in data.values()])
    noise = np.sqrt(1 / n_at + 1 / n_away).mean()

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    bins = np.linspace(-1.3, 1.3, 46)
    ax.hist(gaps, bins=bins, color=c["orange"], alpha=.85, zorder=3)
    # The null: what the same measurement would spread to if the direction were
    # indifferent to step boundaries. Drawn to the same area as the histogram.
    x = np.linspace(-1.3, 1.3, 400)
    null = np.exp(-0.5 * (x / noise) ** 2)
    null = null / null.max() * np.histogram(gaps, bins=bins)[0].max()
    ax.plot(x, null, color=c["ink"], lw=1.2, ls=(0, (3, 2)), zorder=4)
    ax.text(noise * 2.6, np.histogram(gaps, bins=bins)[0].max() * .88,
            "what noise alone\nwould give", fontsize=7.5, color=c["ink"])
    ax.axvline(0, color=c["ink"], lw=0.8, zorder=2)
    ax.set_xlabel("coefficient at a step boundary minus elsewhere (σ)", fontsize=8.5)
    ax.set_ylabel("modules", fontsize=8.5)
    ax.set_title(f"The direction reads step boundaries\n"
                 f"{abs(gaps).mean() / (np.sqrt(2 / np.pi) * noise):.0f}× above chance, "
                 f"in {(abs(gaps) > 0.2).sum()} of {len(gaps)} modules",
                 fontsize=9, color=c["strong"], pad=8)
    ax.grid(axis="y", color=c["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    style(ax, c)

    labels, reads, gains = [], [], []
    for (lo, hi), gain in BAND_GAIN.items():
        v = np.array([x["standardised_gap"] for x in data.values()
                      if x["layer"] is not None and lo <= x["layer"] <= hi])
        if not len(v):
            continue
        labels.append(f"{lo}–{hi}")
        reads.append(abs(v).mean())
        gains.append(gain)
    xs = np.arange(len(labels))
    bx.bar(xs - 0.19, reads, width=0.36, color=c["blue"], zorder=3,
)
    twin = bx.twinx()
    twin.bar(xs + 0.19, gains, width=0.36, color=c["orange"], zorder=3,
)
    twin.set_ylabel("accuracy gain of the band (points)", fontsize=8.5,
                    color=c["orange"])
    twin.tick_params(colors=c["orange"], labelsize=8, length=3, width=0.8)
    for side in ("top", "left"):
        twin.spines[side].set_visible(False)
    twin.spines["right"].set_color(c["orange"])
    twin.set_facecolor("none")
    bx.set_xticks(xs, labels, fontsize=8.5)
    bx.set_xlabel("transformer layers", fontsize=8.5)
    bx.set_ylabel("mean |standardised gap|", fontsize=8.5, color=c["blue"])
    bx.tick_params(axis="y", colors=c["blue"])
    bx.set_title("Detecting a boundary and being useful\nare not the same thing",
                 fontsize=9, color=c["strong"], pad=8)
    bx.grid(axis="y", color=c["grid"], lw=0.7, zorder=0)
    bx.set_axisbelow(True)
    style(bx, c)
    twin.set_ylim(0, max(gains) * 1.16)
    fig.tight_layout()
    return fig


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    data = json.load(open(src))
    out.mkdir(parents=True, exist_ok=True)
    for mode, colours in MODES.items():
        fig = figure(data, colours)
        fig.savefig(out / f"23_what-the-direction-reads.{mode}.svg", transparent=True,
                    format="svg")
        if mode == "light":
            fig.savefig(out / "23_what-the-direction-reads.png", transparent=False,
                        dpi=200, facecolor=colours["paper"])
        plt.close(fig)
    print("wrote 23_what-the-direction-reads")


if __name__ == "__main__":
    main()
