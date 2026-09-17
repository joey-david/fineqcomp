#!/usr/bin/env python3
"""Corruption retained, with intervals, from per-row likelihoods.

The companion study could only report these gaps as point estimates, because it
stored the token-weighted aggregate rather than per-row values. Recomputing one
row at a time makes the row the unit of analysis, which is the unit the claim is
about, and lets the paired difference be bootstrapped.

Two panels because there are two questions. The left is descriptive: how much of
the corruption does each condition still carry? The right is the comparison the
shrinkage control exists for, and the second row of it is the one that matters —
the coded update beats a rescaling that has a *smaller* norm, so its advantage is
not a magnitude effect.
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
NORMS = {"raw": 20.87, "rank1_b1": 9.18, "norm_matched_b1": 9.18,
         "scale_head_0p5": 7.26, "base": 0.0}
LABEL = {"base": "base model", "raw": "damaged adapter",
         "rank1_b1": "rank 1 @ 1 bit", "norm_matched_b1": "rescaled to that norm",
         "scale_head_0p5": "rescaled smaller (× 0.5)"}


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


def load(root: Path):
    out = {}
    for key in LABEL:
        path = root / f"rownll_{key}.json"
        if path.exists():
            d = json.load(open(path))
            out[key] = np.array(d["aligned"]) - np.array(d["permuted"])
    return out


def ci(rng, x, draws=10000):
    idx = rng.integers(0, len(x), (draws, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


def figure(vals, c):
    rng = np.random.default_rng(9172026)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 3.5),
                                 gridspec_kw={"width_ratios": [1.05, 1]})
    order = ["raw", "norm_matched_b1", "scale_head_0p5", "rank1_b1", "base"]
    order = [k for k in order if k in vals]
    colours = {"raw": "strong", "rank1_b1": "orange", "norm_matched_b1": "blue",
               "scale_head_0p5": "grey", "base": "green"}
    for y, key in enumerate(order):
        lo, hi = ci(rng, vals[key])
        m = vals[key].mean()
        ax.errorbar([m], [y], xerr=[[m - lo], [hi - m]], fmt="o", ms=7,
                    color=c[colours[key]], ecolor=c[colours[key]],
                    elinewidth=1.4, capsize=4, zorder=3)
        ax.text(m, y - 0.30, f"‖ΔW‖ {NORMS[key]:.2f}", ha="center", fontsize=7,
                color=c["ink"])
    ax.set_yticks(range(len(order)), [LABEL[k] for k in order], fontsize=8.5)
    ax.set_xlabel("corruption preference (bits/token)  ← less corrupted",
                  fontsize=8.5)
    ax.set_title("How much corruption each update still carries", fontsize=9,
                 color=c["strong"], pad=8)
    ax.grid(axis="x", color=c["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(y=0.16)
    style(ax, c)

    pairs = [("rank1_b1", "norm_matched_b1", "vs the SAME-norm rescaling\n‖ΔW‖ 9.18 both"),
             ("rank1_b1", "scale_head_0p5", "vs a SMALLER-norm rescaling\n9.18 against 7.26")]
    bx.axvline(0, color=c["ink"], lw=0.9, zorder=2)
    for y, (a, b, label) in enumerate(pairs):
        if a not in vals or b not in vals:
            continue
        d = vals[a] - vals[b]
        lo, hi = ci(rng, d)
        bx.errorbar([d.mean()], [y], xerr=[[d.mean() - lo], [hi - d.mean()]],
                    fmt="o", ms=7, color=c["orange"], ecolor=c["orange"],
                    elinewidth=1.4, capsize=4, zorder=3)
    bx.set_yticks(range(len(pairs)), [p[2] for p in pairs], fontsize=8)
    bx.invert_yaxis()
    bx.set_xlabel("difference in bits/token  ← coding removes more", fontsize=8.5)
    bx.set_title("Coding removes more corruption than scaling,\n"
                 "even at a larger update norm", fontsize=9, color=c["strong"], pad=8)
    bx.set_xlim(-0.085, 0.02)
    bx.grid(axis="x", color=c["grid"], lw=0.7, zorder=0)
    bx.set_axisbelow(True)
    bx.margins(y=0.35)
    style(bx, c)
    fig.tight_layout()
    return fig


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    vals = load(root)
    if not vals:
        print("no per-row likelihoods found")
        return
    out.mkdir(parents=True, exist_ok=True)
    for mode, colours in MODES.items():
        fig = figure(vals, colours)
        fig.savefig(out / f"22_corruption-intervals.{mode}.svg", transparent=True,
                    format="svg")
        if mode == "light":
            fig.savefig(out / "22_corruption-intervals.png", transparent=False,
                        dpi=200, facecolor=colours["paper"])
        plt.close(fig)
    print("wrote 22_corruption-intervals")


if __name__ == "__main__":
    main()
