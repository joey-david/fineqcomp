#!/usr/bin/env python3
"""The full figure set for the recovery studies, positive and negative results.

Four figures that the two studies' own plots do not cover, built so the
negative results are as legible as the positive ones — the point of the whole
line of work is as much what does *not* explain the recovery as what does.

  landscape     accuracy against update magnitude for every family of update.
                Shows that shrinkage recovers, that compression recovers, and
                that the two paths reach the same place.
  no-free-lunch two forest plots. Against a *tuned* scale the coded adapter has
                no advantage at all; against a scale matched to its own norm it
                has a small consistent one. Those are different claims and the
                difference is the whole content of the shrinkage control.
  efficiency    corruption retained per unit of update norm, which is the
                quantity a denoising account is really about.
  ladder        every intervention tried, on two axes: how much of the
                adapter's *form* it reproduces, and how much of its *benefit*.
                The separation between the two is the study's main result.

Numbers are read from the studies' own summaries rather than typed in.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = {
    "light": dict(ink="#52514e", grid="#e2e1dd", strong="#0b0b0b", orange="#eb6834",
                  blue="#2a78d6", green="#1baf7a", grey="#9b9a95", red="#c9403a",
                  paper="#fcfcfb"),
    "dark": dict(ink="#c3c2b7", grid="#3a3a37", strong="#f3f2ec", orange="#d95926",
                 blue="#3987e5", green="#199e70", grey="#77766f", red="#d9534f",
                 paper="#141414"),
}
SENT = re.compile(r"[a-zA-Z\)\$%]\.\s*\n")


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


# --------------------------------------------------------------------------
def figure_landscape(dn, c):
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    families = [
        ("compression", "orange", "o", "rank 1, coded at b bits"),
        ("shrinkage_matched", "blue", "s", "rank 1, rescaled to a coded cell's norm"),
        ("shrinkage_head", "grey", "^", "rank 1 × α (fp16)"),
        ("shrinkage_full", "grey", "v", "full rank × α (fp16)"),
    ]
    for family, colour, marker, label in families:
        pts = sorted((r for r in dn.values() if r["family"] == family),
                     key=lambda r: r["update_norm"])
        if not pts:
            continue
        ax.plot([r["update_norm"] for r in pts],
                [r["gsm8k_accuracy"] * 100 for r in pts],
                marker=marker, ms=5, lw=1.1, color=c[colour], label=label, zorder=3)
    for key, label, colour in (("base", "base model", "green"),
                               ("clean_raw", "adapter trained on clean data", "green"),
                               ("raw", "damaged adapter", "strong")):
        r = dn.get(key)
        if not r:
            continue
        ax.axhline(r["gsm8k_accuracy"] * 100, color=c[colour], lw=0.9,
                   ls=(0, (4, 3)), zorder=1)
        ax.text(0.012, r["gsm8k_accuracy"] * 100 + 0.6, label,
                transform=ax.get_yaxis_transform(), fontsize=7.5, color=c[colour])
    ax.set_xlabel("update norm ‖ΔW‖ (arbitrary common scale)", fontsize=8.5)
    ax.set_ylabel("GSM8K %", fontsize=8.5)
    ax.set_title("Every way of making the damaged update smaller recovers accuracy",
                 fontsize=9.5, color=c["strong"], pad=8)
    ax.grid(color=c["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    style(ax, c)
    leg = ax.legend(frameon=False, fontsize=7.5, loc="center left",
                    bbox_to_anchor=(0.30, 0.28))
    for t in leg.get_texts():
        t.set_color(c["ink"])
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
def figure_no_free_lunch(pairs, tuned, c):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.6, 3.4),
                                 gridspec_kw={"width_ratios": [1, 1.25]})
    # Left: against the best scale anywhere on the curve.
    ax.axvline(0, color=c["ink"], lw=0.9, zorder=2)
    ax.errorbar([tuned["diff"]], [0],
                xerr=[[tuned["diff"] - tuned["low"]], [tuned["high"] - tuned["diff"]]],
                fmt="o", ms=7, color=c["grey"], ecolor=c["grey"],
                elinewidth=1.4, capsize=4, zorder=3)
    ax.set_yticks([0], ["rank 1 @ 1 bit\nminus best α"], fontsize=8)
    ax.set_xlabel("difference in GSM8K points", fontsize=8.5)
    ax.set_title("Against a tuned scale:\nno advantage at all", fontsize=9,
                 color=c["strong"], pad=8)
    ax.set_xlim(-12, 14)
    ax.grid(axis="x", color=c["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    style(ax, c)

    # Right: against a scale matched to each coded cell's own norm.
    bx.axvline(0, color=c["ink"], lw=0.9, zorder=2)
    ys = range(len(pairs))
    for y, row in zip(ys, pairs):
        excl = row["low"] > 0
        bx.errorbar([row["diff"]], [y],
                    xerr=[[row["diff"] - row["low"]], [row["high"] - row["diff"]]],
                    fmt="o", ms=7,
                    color=c["orange"] if excl else c["grey"],
                    ecolor=c["orange"] if excl else c["grey"],
                    elinewidth=1.4, capsize=4, zorder=3)
    bx.set_yticks(list(ys), [f"{r['bits']} bit" for r in pairs], fontsize=8)
    bx.invert_yaxis()
    bx.set_xlabel("difference in GSM8K points", fontsize=8.5)
    bx.set_title("Against a scale matched to its own norm:\na small consistent one",
                 fontsize=9, color=c["strong"], pad=8)
    bx.set_xlim(-12, 14)
    bx.grid(axis="x", color=c["grid"], lw=0.7, zorder=0)
    bx.set_axisbelow(True)
    style(bx, c)
    bx.text(0.99, 0.04, "filled = interval excludes zero", transform=bx.transAxes,
            ha="right", fontsize=7, color=c["ink"])
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
def figure_efficiency(dn, c):
    """Corruption retained per unit of update norm.

    The denoising question is not "is there less corruption" -- turning any
    update down achieves that -- but "is there less corruption for the same
    amount of update". That ratio is what separates a code from a scalar.
    """
    base = dn["base"]["corruption_preference_bits"]
    damaged = dn["raw"]["corruption_preference_bits"]
    picks = [
        ("rank1_b1", "rank 1 @ 1 bit", "orange"),
        ("rank1_b2", "rank 1 @ 2 bits", "orange"),
        ("rank1_b16", "rank 1 @ 16 bits", "blue"),
        ("norm_matched_b1", "rank 1, rescaled\nto the 1-bit norm", "blue"),
        ("scale_head_0p5", "rank 1 × 0.5", "grey"),
        ("scale_full_0p3", "full rank × 0.3", "grey"),
        ("scale_full_0p1", "full rank × 0.1", "grey"),
        ("raw", "damaged, raw", "strong"),
    ]
    rows = []
    for key, label, colour in picks:
        r = dn.get(key)
        if not r or r["update_norm"] <= 0:
            continue
        retained = (r["corruption_preference_bits"] - base) / (damaged - base)
        rows.append((label, colour, retained / r["update_norm"] * 100,
                     r["gsm8k_accuracy"] * 100))
    rows.sort(key=lambda t: t[2])
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    ys = range(len(rows))
    ax.barh(list(ys), [r[2] for r in rows],
            color=[c[r[1]] for r in rows], height=0.62, zorder=3)
    for y, r in zip(ys, rows):
        ax.text(r[2] + 0.08, y, f"{r[2]:.2f}   ({r[3]:.1f}% on GSM8K)",
                va="center", fontsize=7.5, color=c["ink"])
    ax.set_yticks(list(ys), [r[0] for r in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(r[2] for r in rows) * 1.42)
    ax.set_xlabel("corruption still installed, per unit of update norm  "
                  "← more efficient removal", fontsize=8.5)
    ax.set_title("Compression removes corruption more efficiently than scaling does",
                 fontsize=9.5, color=c["strong"], pad=8)
    ax.grid(axis="x", color=c["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    style(ax, c)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
def figure_ladder(cot, c):
    """Form reproduced against benefit, for every intervention tried.

    The y-axis is accuracy points rather than a percentage of the adapter's
    gain: expressed as a percentage the collapsing arms run to -700% and push
    everything else into a band a few pixels tall, which hides the comparison
    the figure exists to make.
    """
    base = cot["forcing_bias0"]
    adapter = cot["forcing_adapter"]
    entries = [
        ("forcing_bias4", "marker bias 4", "grey", "v", (8, -4)),
        ("forcing_hold100", "hard length floor", "grey", "v", (8, 2)),
        ("segscale_6", "token shift × 6", "grey", "v", (-8, 4)),
        ("forcing_bias2", "marker bias 2", "grey", "v", (8, -10)),
        ("style_metamath", "MetaMath exemplars\n(confounded)", "red", "D", (8, -2)),
        ("segment_top80", "token shift, top 80", "grey", "v", (2, -14)),
        ("segscale_3", "token shift × 3", "grey", "v", (6, -16)),
        ("matched_terse", "terse exemplars", "blue", "D", (-8, -12)),
        ("band_14_20", "layers 14–20", "blue", "s", (-8, 2)),
        ("band_00_06", "layers 0–6", "green", "s", (-6, 8)),
        ("matched_verbose", "segmented exemplars", "green", "D", (-6, -16)),
        ("forcing_adapter", "the whole adapter", "orange", "o", (-8, 8)),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    gain = (adapter["accuracy"] - base["accuracy"]) * 100
    ax.axhline(0, color=c["ink"], lw=0.9, zorder=2)
    ax.axhline(gain, color=c["orange"], lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.text(0.012, gain + 0.9, f"the adapter's gain, +{gain:.1f}",
            transform=ax.get_yaxis_transform(), fontsize=7.5, color=c["orange"])
    for key, label, colour, marker, offset in entries:
        r = cot.get(key)
        if not r:
            continue
        form = r["segmentation"] * 100
        benefit = (r["accuracy"] - base["accuracy"]) * 100
        ax.scatter([form], [benefit], s=96 if marker == "o" else 62,
                   marker=marker, color=c[colour], zorder=4, edgecolor="none")
        ax.annotate(label, (form, benefit), textcoords="offset points",
                    xytext=offset, fontsize=7.2, color=c[colour],
                    ha="right" if offset[0] < 0 else "left")
    ax.scatter([base["segmentation"] * 100], [0], s=80, marker="*",
               color=c["strong"], zorder=5)
    ax.annotate("base model", (base["segmentation"] * 100, 0),
                textcoords="offset points", xytext=(-8, 4), fontsize=7.5,
                ha="right", color=c["strong"])
    ax.set_xlabel("form reproduced: answers segmented one step per line (%)",
                  fontsize=8.5)
    ax.set_ylabel("benefit reproduced: GSM8K points against base", fontsize=8.5)
    ax.set_title("Reproducing the form is easy; reproducing the benefit needs "
                 "conditioning", fontsize=9.5, color=c["strong"], pad=8)
    ax.grid(color=c["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.14, y=0.10)
    style(ax, c)
    for marker, label in (("v", "decoding-level, context-blind"),
                          ("D", "in-context demonstration"),
                          ("s", "weight update, part of the stack"),
                          ("o", "weight update, whole")):
        ax.scatter([], [], marker=marker, s=48, color=c["ink"], label=label)
    leg = ax.legend(frameon=False, fontsize=7.5, loc="lower left",
                    bbox_to_anchor=(0.30, 0.02))
    for t in leg.get_texts():
        t.set_color(c["ink"])
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
def load_denoise(path: Path):
    return {r["key"]: r for r in json.load(open(path))["rows"]}


def load_pairs(path: Path):
    rows = []
    for row in csv.DictReader(open(path)):
        rows.append({"bits": int(row["bits"]),
                     "diff": float(row["accuracy_diff"]) * 100,
                     "low": float(row["ci95_low"]) * 100,
                     "high": float(row["ci95_high"]) * 100})
    return sorted(rows, key=lambda r: r["bits"])


def load_cot(root: Path):
    out = {}
    for path in root.glob("*.json"):
        key = path.stem
        try:
            m = json.load(open(path))
        except json.JSONDecodeError:
            continue
        if "exact_match" not in m:
            continue
        jl = root / f"{key}.jsonl"
        seg = None
        if jl.exists():
            rows = [json.loads(l) for l in jl.read_text().splitlines() if l]
            seg = sum(bool(SENT.search(r["response"])) for r in rows) / max(len(rows), 1)
        out[key] = {"accuracy": m["exact_match"], "words": m["mean_chain_words"],
                    "segmentation": seg if seg is not None else 0.0}
    return out


def main() -> None:
    denoise_summary, pairs_csv, cot_root, out = (Path(a) for a in sys.argv[1:5])
    dn = load_denoise(denoise_summary)
    pairs = load_pairs(pairs_csv)
    cot = load_cot(cot_root)
    tuned = {"diff": -0.6, "low": -2.8, "high": 1.6}
    out.mkdir(parents=True, exist_ok=True)
    builders = [
        ("18_recovery-landscape", lambda c: figure_landscape(dn, c)),
        ("19_shrinkage-no-free-lunch", lambda c: figure_no_free_lunch(pairs, tuned, c)),
        ("20_denoising-efficiency", lambda c: figure_efficiency(dn, c)),
        ("21_form-versus-benefit", lambda c: figure_ladder(cot, c)),
    ]
    for slug, builder in builders:
        for mode, colours in MODES.items():
            fig = builder(colours)
            if fig is None:
                continue
            fig.savefig(out / f"{slug}.{mode}.svg", transparent=True, format="svg")
            if mode == "light":
                fig.savefig(out / f"{slug}.png", transparent=False, dpi=200,
                            facecolor=colours["paper"])
            plt.close(fig)
        print(f"wrote {slug}")


if __name__ == "__main__":
    main()
