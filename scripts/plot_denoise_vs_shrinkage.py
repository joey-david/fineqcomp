#!/usr/bin/env python3
"""Figures for the denoising-or-shrinkage control, in the house style.

Two figures, because the study asks two questions that a single panel would
blur together.

  shrinkage-control   the two axes the conditions move on. Clean-task accuracy
                      alone cannot separate denoising from shrinkage, because
                      everything that turns the adapter down approaches a base
                      model that is already good at GSM8K. Plotting accuracy
                      against how much of the memorised corruption survives
                      does separate them, and the exact norm-matched pairs are
                      drawn joined so a reader can see whether going through a
                      file moves a point off the shrinkage path.
  distance-to-clean   where each condition lands relative to the adapter that
                      never saw the corruption -- in weight space (cosine) and
                      in behaviour (agreement of extracted answers).

Numbers are read from summary.json rather than typed in, so a rerun of the
study redraws rather than silently disagreeing with the report. Conditions the
run did not reach are skipped rather than drawn as zero.

Ink follows figures_top10: transparent background, one palette per mode, and
nothing but the ink hexes differing between light and dark.
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

# How each family is drawn. The coded arm and its exact shrinkage partners are
# the comparison, so they get the two saturated hues; everything else is
# context and is drawn in grey or in outline.
FAMILY_STYLE = {
    "compression": dict(colour="orange", marker="o", label="rank 1, coded at b bits"),
    "shrinkage_matched": dict(colour="blue", marker="s",
                              label="rank 1, rescaled to the same norm"),
    "shrinkage_head": dict(colour="grey", marker="^",
                           label="rank 1 × α (fp16)"),
    "shrinkage_full": dict(colour="grey", marker="v",
                           label="full rank × α (fp16)"),
}


def load(path: Path):
    summary = json.load(open(path))
    rows = {row["key"]: row for row in summary["rows"]}
    return summary, rows


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


def figure_control(summary, rows, c):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.4, 3.6))

    # --- left: accuracy against surviving corruption ------------------------
    # The x-axis is the likelihood measure rather than the share of answers
    # reciting the modal rationale, because that share is pinned near zero for
    # every arm that recovers at all -- it separates the damaged adapter from
    # the rest and nothing else. The bits/token measure keeps resolving after
    # the behavioural one has saturated, which is where the comparison lives.
    def x_of(row):
        return row["corruption_preference_bits"]

    for family, spec in FAMILY_STYLE.items():
        points = [row for row in rows.values() if row["family"] == family]
        if not points:
            continue
        points.sort(key=lambda row: row["update_norm"])
        ax.plot([x_of(row) for row in points],
                [row["gsm8k_accuracy"] * 100 for row in points],
                marker=spec["marker"], ms=5, lw=1.0,
                color=c[spec["colour"]], label=spec["label"], zorder=3)
    # The two ends of every path, named rather than left to the legend: they are
    # the fixed points a reader checks the rest against.
    for key, label, colour, offset in (
        ("base", "base model", "green", (6, -10)),
        ("raw", "damaged adapter", "strong", (-6, 8)),
        ("clean_raw", "trained on clean data", "green", (6, 4)),
    ):
        row = rows.get(key)
        if row is None:
            continue
        ax.scatter([x_of(row)], [row["gsm8k_accuracy"] * 100], s=46,
                   color=c[colour], zorder=5, marker="X" if key == "raw" else "*")
        ax.annotate(label, (x_of(row), row["gsm8k_accuracy"] * 100),
                    textcoords="offset points", xytext=offset, fontsize=7.5,
                    color=c[colour],
                    ha="right" if key == "raw" else "left")
    # Join each coded cell to the rescaling that matches its update norm
    # exactly. A short join means shrinkage accounts for the cell; a long one
    # means going through the file did something a scalar cannot.
    for pair in summary.get("shrinkage_envelope", []):
        left, right = rows.get(pair["coded_key"]), rows.get(pair["shrunk_key"])
        if not left or not right or left.get("modal_opening_share") is None:
            continue
        if abs((pair.get("norm_ratio") or 0.0) - 1.0) > 0.02:
            continue  # not an exact match; the join would overstate the pairing
        ax.plot([x_of(left), x_of(right)],
                [left["gsm8k_accuracy"] * 100, right["gsm8k_accuracy"] * 100],
                lw=0.9, ls=(0, (2, 2)), color=c["ink"], zorder=2)
    ax.set_xlabel("corruption preference (bits/token)  ← less corrupted",
                  fontsize=8.5)
    ax.set_ylabel("GSM8K %", fontsize=8.5)
    ax.set_title("Task accuracy against corruption still installed", fontsize=9,
                 color=c["strong"], pad=8)
    ax.grid(color=c["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    style(ax, c)
    legend = ax.legend(frameon=False, fontsize=7, loc="lower left",
                       borderaxespad=0.2, handlelength=1.6)
    for text in legend.get_texts():
        text.set_color(c["ink"])
    # The reference markers are named next to the points; leaving room below
    # them keeps those names off the legend in the same corner.
    ax.margins(x=0.12, y=0.14)

    # --- right: the likelihood read of the same thing -----------------------
    # Norm is the quantity a shrinkage account has to work with, so putting it
    # on the x-axis is what makes the coded and rescaled arms commensurable.
    for family, spec in FAMILY_STYLE.items():
        points = [row for row in rows.values() if row["family"] == family]
        if not points:
            continue
        points.sort(key=lambda row: row["update_norm"])
        bx.plot([row["update_norm"] for row in points],
                [row["corruption_preference_bits"] for row in points],
                marker=spec["marker"], ms=5, lw=1.0,
                color=c[spec["colour"]], label=spec["label"], zorder=3)
    # The exact pairs, joined vertically: same directions, same update norm,
    # one through a file and one not. The length of the join is the part of the
    # recovery that shrinkage does not account for.
    for pair in summary.get("shrinkage_envelope", []):
        left, right = rows.get(pair["coded_key"]), rows.get(pair["shrunk_key"])
        if not left or not right:
            continue
        if abs((pair.get("norm_ratio") or 0.0) - 1.0) > 0.02:
            continue
        bx.plot([left["update_norm"], right["update_norm"]],
                [left["corruption_preference_bits"],
                 right["corruption_preference_bits"]],
                lw=0.9, ls=(0, (2, 2)), color=c["ink"], zorder=2)
    # Reference levels, labelled inside the axes: a label hung off the right
    # spine is the first thing a tight bounding box clips.
    for key, colour in (("base", "green"), ("raw", "strong")):
        row = rows.get(key)
        if row is None:
            continue
        bx.axhline(row["corruption_preference_bits"], color=c[colour], lw=0.9,
                   ls=(0, (4, 3)), zorder=1)
        bx.text(0.015, row["corruption_preference_bits"],
                "base model" if key == "base" else "damaged adapter",
                transform=bx.get_yaxis_transform(), fontsize=7.5,
                color=c[colour], va="bottom", ha="left")
    bx.set_xlabel("update norm ‖ΔW‖ (arbitrary common scale)", fontsize=8.5)
    bx.set_ylabel("corruption preference (bits/token)", fontsize=8.5)
    bx.set_title("How much of the corruption is still installed", fontsize=9,
                 color=c["strong"], pad=8)
    bx.grid(color=c["grid"], lw=0.7, zorder=0)
    bx.set_axisbelow(True)
    style(bx, c)
    fig.tight_layout()
    return fig


def figure_distance(summary, rows, c):
    """Does the recovered adapter approach the solution the corruption prevented?

    Two panels, because the question has a behavioural and a weight-space
    answer and they do not agree in kind. Agreement on *answers* is confounded
    with accuracy -- two models that are both right agree by being right -- so
    the left panel plots agreement restricted to items both get wrong against
    accuracy, where a condition that were genuinely converging on the clean
    solution would sit above the trend the others make. The right panel is the
    weight-space reading, which needs no such care.
    """
    clean = rows.get("clean_raw")
    ordered = [row for row in rows.values()
               if row.get("versus_clean") and row["key"] != "clean_raw"
               and (row["versus_clean"].get("agreement_wrong") is not None)]
    if not clean or not ordered:
        return None
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.4, 3.5),
                                 gridspec_kw={"width_ratios": [1.1, 1]})

    for family, spec in FAMILY_STYLE.items():
        pts = [r for r in ordered if r["family"] == family]
        if not pts:
            continue
        ax.scatter([r["gsm8k_accuracy"] * 100 for r in pts],
                   [r["versus_clean"]["agreement_wrong"] * 100 for r in pts],
                   s=34, marker=spec["marker"], color=c[spec["colour"]],
                   label=spec["label"], zorder=3)
    for key, label, colour in (("base", "base model", "green"),
                               ("raw", "damaged", "strong")):
        row = rows.get(key)
        if not row or not row.get("versus_clean"):
            continue
        aw = row["versus_clean"].get("agreement_wrong")
        if aw is None:
            continue
        ax.scatter([row["gsm8k_accuracy"] * 100], [aw * 100], s=52,
                   marker="*" if key == "base" else "X", color=c[colour], zorder=5)
        ax.annotate(label, (row["gsm8k_accuracy"] * 100, aw * 100),
                    textcoords="offset points", xytext=(7, -2), fontsize=7.5,
                    color=c[colour])
    ax.set_xlabel("GSM8K %", fontsize=8.5)
    ax.set_ylabel("agreement with clean adapter\non items both get wrong (%)",
                  fontsize=8.5)
    ax.set_title("Behavioural distance tracks accuracy, nothing more",
                 fontsize=9, color=c["strong"], pad=8)
    ax.grid(color=c["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    style(ax, c)
    legend = ax.legend(frameon=False, fontsize=7, loc="upper left")
    for text in legend.get_texts():
        text.set_color(c["ink"])

    # Right: the weight-space reading. Every update derived from the damaged
    # adapter is near-orthogonal to the clean one, so the bars are shown
    # against the clean adapter's own compression as the only positive control.
    picks = [k for k in ("raw", "rank1_b1", "norm_matched_b1", "scale_head_0p5",
                         "clean_rank1_b1") if k in rows]
    # Short labels: the panel is five bars wide and a "damaged" prefix on four
    # of them is what the note under the axis already says.
    names = {"raw": "raw", "rank1_b1": "r1 @ 1 bit",
             "norm_matched_b1": "rescaled", "scale_head_0p5": "\u00d7 0.5",
             "clean_rank1_b1": "clean,\nr1 @ 1 bit"}
    values = [(rows[k].get("cosine_to_clean") or 0.0) for k in picks]
    colours = [c["orange"] if k != "clean_rank1_b1" else c["green"] for k in picks]
    bx.bar(range(len(picks)), values, color=colours, width=0.6, zorder=3)
    for i, v in enumerate(values):
        bx.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5, color=c["ink"])
    bx.set_xticks(range(len(picks)), [names[k] for k in picks], fontsize=7.5)
    bx.text(0.5, -0.20, "updates derived from the damaged adapter, "
            "except the green control", transform=bx.transAxes,
            ha="center", fontsize=7, color=c["ink"])
    bx.set_ylabel("cosine to the clean update", fontsize=8.5)
    bx.set_ylim(0, 0.42)
    bx.set_title("In weight space it does not move toward clean at all",
                 fontsize=9, color=c["strong"], pad=8)
    bx.grid(axis="y", color=c["grid"], lw=0.7, zorder=0)
    bx.set_axisbelow(True)
    style(bx, c)
    fig.tight_layout()
    return fig


def main() -> None:
    summary_path = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    summary, rows = load(summary_path)
    builders = (
        ("14_shrinkage-control", figure_control),
        ("15_distance-to-clean-adapter", figure_distance),
    )
    for slug, builder in builders:
        drawn = False
        for mode, colours in MODES.items():
            fig = builder(summary, rows, colours)
            if fig is None:
                continue
            fig.savefig(out / f"{slug}.{mode}.svg", transparent=True, format="svg")
            if mode == "light":
                fig.savefig(out / f"{slug}.png", transparent=False, dpi=200,
                            facecolor=colours["paper"])
            plt.close(fig)
            drawn = True
        print(f"wrote {slug}" if drawn else f"skipped {slug} (nothing measured yet)")


if __name__ == "__main__":
    main()
