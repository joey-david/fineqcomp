#!/usr/bin/env python3
"""Figures for the CoT mechanism study, in the house style of figures_top10.

Two panels, one argument. The recovered adapter writes a longer chain than the
base model and scores higher, which invites the reading that it works by
writing more. Both panels refute that reading, from different directions.

  left   Accuracy against chain length, with every way of lengthening the base
         model's chain on the same axes. The forced arms trace out what buying
         length actually costs; the adapter sits far above that curve at the
         same length, so length is not what it is buying.
  right  The same effect localised by depth. Accuracy and chain length are
         drawn together because the point is that they come apart: the middle
         band raises accuracy without lengthening anything.

Numbers are read from the experiment's own metric files rather than typed in.
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
                  orange="#eb6834", blue="#2a78d6", green="#1baf7a",
                  grey="#9b9a95", paper="#fcfcfb"),
    "dark": dict(ink="#c3c2b7", grid="#3a3a37", strong="#f3f2ec",
                 orange="#d95926", blue="#3987e5", green="#199e70",
                 grey="#77766f", paper="#141414"),
}


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


def read(root: Path, stem: str):
    path = root / f"{stem}.json"
    return json.load(open(path)) if path.exists() else None


def figure(root: Path, c):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.6, 3.6),
                                 gridspec_kw={"width_ratios": [1.05, 1]})

    # --- left: what buying length costs -------------------------------------
    biases = []
    for name in sorted(root.glob("forcing_bias*.json")):
        value = float(name.stem.replace("forcing_bias", ""))
        biases.append((value, json.load(open(name))))
    biases.sort()
    if biases:
        ax.plot([m["mean_chain_words"] for _, m in biases],
                [m["exact_match"] * 100 for _, m in biases],
                marker="^", ms=5, lw=1.1, color=c["grey"],
                label="base + marker bias", zorder=3)
        for value, m in biases:
            if value in (0.0, 2.0, 4.0):
                ax.annotate(f"bias {value:g}",
                            (m["mean_chain_words"], m["exact_match"] * 100),
                            textcoords="offset points", xytext=(4, -11),
                            fontsize=7, color=c["ink"])
    holds = []
    for name in sorted(root.glob("forcing_hold*.json")):
        value = int(name.stem.replace("forcing_hold", ""))
        if value:
            holds.append((value, json.load(open(name))))
    if holds:
        ax.scatter([m["mean_chain_words"] for _, m in holds],
                   [m["exact_match"] * 100 for _, m in holds],
                   s=36, marker="v", color=c["grey"], zorder=3,
                   label="base + hard length floor")
    adapter = read(root, "forcing_adapter") or read(root, "band_all")
    base = read(root, "forcing_bias0") or read(root, "band_none")
    if base:
        ax.scatter([base["mean_chain_words"]], [base["exact_match"] * 100],
                   s=60, marker="*", color=c["green"], zorder=6)
        ax.annotate("base model",
                    (base["mean_chain_words"], base["exact_match"] * 100),
                    textcoords="offset points", xytext=(-2, -14), fontsize=7.5,
                    color=c["green"], ha="center")
    if adapter:
        ax.scatter([adapter["mean_chain_words"]], [adapter["exact_match"] * 100],
                   s=70, marker="o", color=c["orange"], zorder=6)
        ax.annotate("rank-1 adapter\n@ 1 bit",
                    (adapter["mean_chain_words"], adapter["exact_match"] * 100),
                    textcoords="offset points", xytext=(9, -2), fontsize=7.5,
                    color=c["orange"])
    # The comparison the panel exists to make: same chain length, different
    # accuracy. Drawn as a bracket between the adapter and the forced arm
    # nearest it in length.
    if adapter and biases:
        near = min(biases, key=lambda kv: abs(
            kv[1]["mean_chain_words"] - adapter["mean_chain_words"]))[1]
        ax.plot([near["mean_chain_words"], adapter["mean_chain_words"]],
                [near["exact_match"] * 100, adapter["exact_match"] * 100],
                lw=1.0, ls=(0, (2, 2)), color=c["strong"], zorder=4)
        ax.annotate(
            f"+{(adapter['exact_match'] - near['exact_match']) * 100:.1f} points\n"
            "at equal length",
            ((near["mean_chain_words"] + adapter["mean_chain_words"]) / 2,
             (near["exact_match"] + adapter["exact_match"]) * 50),
            # To the right of the bracket: the base-model label already owns
            # the space to its left.
            textcoords="offset points", xytext=(10, -4), fontsize=7.5,
            color=c["strong"])
    ax.set_xlabel("reasoning words written before the answer", fontsize=8.5)
    ax.set_ylabel("GSM8K %", fontsize=8.5)
    ax.set_title("Length is not what the adapter is buying", fontsize=9,
                 color=c["strong"], pad=8)
    ax.grid(color=c["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    style(ax, c)
    legend = ax.legend(frameon=False, fontsize=7, loc="lower left")
    for text in legend.get_texts():
        text.set_color(c["ink"])

    # --- right: the same effect by depth ------------------------------------
    bands = []
    for name in sorted(root.glob("band_[0-9]*.json")):
        low, high = name.stem.split("_")[1:]
        bands.append((int(low), int(high), json.load(open(name))))
    bands.sort()
    none = read(root, "band_none")
    if not bands or not none:
        return fig
    labels = [f"{low}–{high}" for low, high, _ in bands]
    gains = [(m["exact_match"] - none["exact_match"]) * 100 for _, _, m in bands]
    words = [m["mean_chain_words"] - none["mean_chain_words"] for _, _, m in bands]
    x = range(len(bands))
    bx.bar([i - 0.19 for i in x], gains, width=0.36, color=c["orange"],
           label="accuracy gain over base (points)", zorder=3)
    twin = bx.twinx()
    twin.bar([i + 0.19 for i in x], words, width=0.36, color=c["blue"],
             label="extra reasoning words", zorder=3)
    twin.set_ylabel("extra words written", fontsize=8.5, color=c["blue"])
    twin.tick_params(colors=c["blue"], labelsize=8, length=3, width=0.8)
    for side in ("top", "left"):
        twin.spines[side].set_visible(False)
    twin.spines["right"].set_color(c["blue"])
    twin.set_facecolor("none")
    bx.axhline(0, color=c["ink"], lw=0.8, zorder=4)
    full = read(root, "band_all")
    if full:
        bx.axhline((full["exact_match"] - none["exact_match"]) * 100,
                   color=c["strong"], lw=0.9, ls=(0, (4, 3)), zorder=2)
        bx.text(0.02, (full["exact_match"] - none["exact_match"]) * 100,
                "all 28 layers", transform=bx.get_yaxis_transform(),
                fontsize=7.5, color=c["strong"], va="bottom")
    bx.set_xticks(list(x), labels, fontsize=8)
    bx.set_xlabel("transformer layers carrying the update", fontsize=8.5)
    bx.set_ylabel("accuracy gain (points)", fontsize=8.5, color=c["orange"])
    bx.set_title("Redundant in depth, and length comes apart from accuracy",
                 fontsize=9, color=c["strong"], pad=8)
    bx.grid(axis="y", color=c["grid"], lw=0.7, zorder=0)
    bx.set_axisbelow(True)
    style(bx, c)
    bx.tick_params(axis="y", colors=c["orange"])
    fig.tight_layout()
    return fig


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for slug, builder in (
        ("16_cot-length-is-not-the-mechanism", figure),
        ("17_step-boundary-selectivity", figure_selectivity),
    ):
        drawn = False
        for mode, colours in MODES.items():
            fig = builder(root, colours)
            if fig is None:
                continue
            fig.savefig(out / f"{slug}.{mode}.svg", transparent=True, format="svg")
            if mode == "light":
                fig.savefig(out / f"{slug}.png", transparent=False, dpi=200,
                            facecolor=colours["paper"])
            plt.close(fig)
            drawn = True
        print(f"wrote {slug}" if drawn else f"skipped {slug} (nothing measured yet)")



# --- the selectivity figure -------------------------------------------------
# Added after the fact, because the result it draws was not anticipated: the
# "." token ends a sentence and separates a decimal, so segmenting reasoning
# steps correctly requires telling those apart, and anything context-blind
# cannot. Plotting both rates against each other separates conditional from
# unconditional versions of the same preference in one picture.

import re as _re

_SENTENCE_BREAK = _re.compile(r"[a-zA-Z\)\$%]\.\s*\n")
_DECIMAL_BREAK = _re.compile(r"\d\.\s*\n\s*\d")


def _rates(path: Path) -> dict[str, float]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    n = max(len(rows), 1)
    return {
        "sentence": sum(bool(_SENTENCE_BREAK.search(r["response"])) for r in rows) / n * 100,
        "decimal": sum(bool(_DECIMAL_BREAK.search(r["response"])) for r in rows) / n * 100,
        "accuracy": sum(r["correct"] for r in rows) / n * 100,
    }


def figure_selectivity(root: Path, c):
    # Label offsets are per point: base and the 1x bias sit almost on top of
    # each other, and so do layers 0-6 and the 3x bias, so they are pushed
    # apart by hand rather than left to collide.
    wanted = [
        ("forcing_bias0", "base model", "green", "*", (-8, -16)),
        ("forcing_adapter", "rank-1 adapter @ 1 bit", "orange", "o", (-6, 12)),
        ("band_00_06", "layers 0–6 only", "orange", "s", (-4, -22)),
        ("segscale_1", "uniform bias 1×", "grey", "^", (8, 4)),
        ("segscale_3", "uniform bias 3×", "grey", "^", (8, 2)),
        ("segscale_6", "uniform bias 6×", "grey", "^", (-9, 2)),
        ("segscale_10", "uniform bias 10×", "grey", "^", (-9, 2)),
    ]
    points = [(label, colour, marker, _rates(root / f"{key}.jsonl"), offset)
              for key, label, colour, marker, offset in wanted
              if (root / f"{key}.jsonl").exists()]
    if not points:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    # The uniform-bias arms form a path; drawing it shows that pushing harder
    # trades selectivity away rather than buying more of it.
    path = [p for p in points if "uniform" in p[0]]
    if len(path) > 1:
        ax.plot([p[3]["sentence"] for p in path], [p[3]["decimal"] for p in path],
                lw=1.0, ls=(0, (3, 2)), color=c["grey"], zorder=2)
    for label, colour, marker, r, offset in points:
        ax.scatter([r["sentence"]], [r["decimal"]], s=90 if marker == "o" else 58,
                   marker=marker, color=c[colour], zorder=4)
        ax.annotate(f"{label}\n{r['accuracy']:.1f}%",
                    (r["sentence"], r["decimal"]),
                    textcoords="offset points", xytext=offset,
                    ha="right" if offset[0] < -5 else "left",
                    fontsize=7.5, color=c[colour])
    ax.set_xlabel("answers that break the line after a sentence (%)", fontsize=8.5)
    ax.set_ylabel("answers that break a line inside a decimal (%)", fontsize=8.5)
    ax.set_title("Segmenting reasoning steps requires knowing which '.' is which",
                 fontsize=9, color=c["strong"], pad=8)
    ax.grid(color=c["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.16, y=0.16)
    style(ax, c)
    ax.text(0.98, 0.04, "lower is more selective at the same segmentation rate",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=c["ink"])
    fig.tight_layout()
    return fig

if __name__ == "__main__":
    main()
