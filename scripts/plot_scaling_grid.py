#!/usr/bin/env python3
"""Figures for the scale-by-task grid, in the house style.

Eleven panels, read from `reports/scaling_grid/summary.json` rather than typed
in, so a rerun redraws instead of silently disagreeing with the report. Cells a
run did not reach are skipped rather than drawn as zero.

Several of these are negative results and are here for that reason: the
permuted corruption does not damage pointer chasing at all, the overshoot goes
negative on the two largest models, and half the pointer cells cannot support a
"beats the base model" claim because their base model scores near zero. A figure
set that only showed the cells that worked would be the wrong picture.

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
                  red="#d1344b", grey="#9b9a95", paper="#fcfcfb"),
    "dark": dict(ink="#c3c2b7", grid="#3a3a37", strong="#f3f2ec",
                 orange="#d95926", blue="#3987e5", green="#199e70",
                 red="#e0455c", grey="#77766f", paper="#141414"),
}

# The chain-of-thought ladder, in size order. Parameter counts are the x axis
# wherever scale is the variable, because "0.5B to 32B" is a 64-fold range and
# equal spacing would hide where the turn happens.
LADDER = [("cot_0p5b", "0.5B", 0.5), ("cot_1p5b", "1.5B", 1.5),
          ("cot_3b", "3B", 3.0), ("cot_7b", "7B", 7.0),
          ("cot_14b", "14B", 14.0), ("cot_32b", "32B", 32.0)]

CONDITIONS = ["base", "raw", "rank1_b1", "rank1_b16", "scale_head_0p5",
              "norm_matched_b1", "tail_b16"]
NICE = {"base": "frozen base", "raw": "damaged fine-tune",
        "rank1_b1": "rank 1 @ 1 bit", "rank1_b16": "rank 1 @ fp16",
        "scale_head_0p5": "rank 1 × α=0.5", "norm_matched_b1": "norm-matched",
        "tail_b16": "top 4 dropped"}

# Below this, "beats the base model" is not a meaningful claim: a base scoring
# a few per cent is beaten by any adapter that learns the output format.
VACUOUS_BASE = 0.20


def style(ax, mode, *, grid_axis="y"):
    c = MODES[mode]
    ax.set_facecolor("none")
    ax.grid(True, axis=grid_axis, color=c["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["grid"])
    ax.tick_params(colors=c["ink"], labelsize=9)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_color(c["ink"])


def save(fig, out: Path, stem: str, mode: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.{mode}.svg", transparent=True, bbox_inches="tight")
    if mode == "light":
        fig.savefig(out / f"{stem}.png", dpi=170, transparent=False,
                    facecolor=MODES["light"]["paper"], bbox_inches="tight")
    plt.close(fig)


def scores(arms, name):
    return arms.get(name, {}).get("scores", {})


def fig_ladder(arms, out, mode):
    """Base, damaged and recovered at every size on the ladder."""
    c = MODES[mode]
    cells = [(k, lab) for k, lab, _ in LADDER if k in arms]
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    width, xs = 0.26, range(len(cells))
    for offset, key, colour in ((-width, "base", "blue"),
                                (0.0, "raw", "red"),
                                (width, "rank1_b1", "orange")):
        vals = [scores(arms, k).get(key, float("nan")) * 100 for k, _ in cells]
        bars = ax.bar([x + offset for x in xs], vals, width * 0.92,
                      color=c[colour], label=NICE[key], zorder=3)
        for bar, v in zip(bars, vals):
            if v == v:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.0f}",
                        ha="center", fontsize=7.5, color=c["ink"])
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lab for _, lab in cells])
    ax.set_ylabel("GSM8K accuracy (%)", color=c["ink"], fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("Corrupting the rationales destroys every size; compressing the "
                 "damage back out recovers every size",
                 color=c["strong"], fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=c["ink"], ncol=3)
    style(ax, mode)
    save(fig, out, "25_scale-ladder", mode)


def fig_overshoot(arms, out, mode):
    """The reversal: overshoot peaks at 3B and crosses zero by 14B."""
    c = MODES[mode]
    pts = [(n, lab, (scores(arms, k).get("rank1_b1"), scores(arms, k).get("tail_b16"),
                     scores(arms, k).get("base")))
           for k, lab, n in LADDER if k in arms]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.axhline(0, color=c["strong"], linewidth=1.0, zorder=2)
    for key, colour, marker, label in (
            (0, "orange", "o", "rank 1 @ 1 bit"),
            (1, "green", "s", "top 4 directions dropped")):
        xs = [n for n, _, v in pts if v[key] is not None and v[2] is not None]
        ys = [(v[key] - v[2]) * 100 for _, _, v in pts
              if v[key] is not None and v[2] is not None]
        ax.plot(xs, ys, marker=marker, color=c[colour], linewidth=1.8,
                markersize=6, label=label, zorder=4)
    ax.set_xscale("log")
    ax.set_xticks([n for n, _, _ in pts])
    ax.set_xticklabels([lab for _, lab, _ in pts])
    ax.minorticks_off()
    ax.set_xlabel("parameters", color=c["ink"], fontsize=9)
    ax.set_ylabel("points above the frozen base", color=c["ink"], fontsize=9)
    ax.set_title("Beating the base model is a small- and mid-scale phenomenon "
                 "— for the codec, not for the spectrum",
                 color=c["strong"], fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=c["ink"])
    style(ax, mode)
    save(fig, out, "26_overshoot-vs-scale", mode)


def fig_recovery(arms, out, mode):
    """Recovery is near total at every size, which overshoot is not."""
    c = MODES[mode]
    cells = [(k, lab) for k, lab, _ in LADDER if k in arms]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.axhline(100, color=c["grey"], linewidth=1.0, linestyle="--", zorder=2)
    vals = [arms[k].get("recovery", {}).get("rank1_b1", float("nan")) * 100
            for k, _ in cells]
    ax.bar(range(len(cells)), vals, 0.55, color=c["orange"], zorder=3)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([lab for _, lab in cells])
    ax.set_ylabel("% of that cell's own damage undone", color=c["ink"], fontsize=9)
    ax.set_title("Recovery is near total at every size — the scale effect is in "
                 "the overshoot, not in the repair",
                 color=c["strong"], fontsize=10.5, loc="left")
    style(ax, mode)
    save(fig, out, "27_recovery-vs-scale", mode)


def fig_overshoot_vs_base(arms, out, mode):
    """The reframing: overshoot against base competence, every cell."""
    c = MODES[mode]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.axhline(0, color=c["strong"], linewidth=1.0, zorder=2)
    ax.axvspan(0, VACUOUS_BASE * 100, color=c["grey"], alpha=0.16, zorder=1)
    ax.text(VACUOUS_BASE * 50, ax.get_ylim()[1], "", ha="center")
    for name, entry in sorted(arms.items()):
        s = entry.get("scores", {})
        base, best = s.get("base"), None
        if base is None:
            continue
        for key in ("rank1_b1", "tail_b16"):
            if s.get(key) is not None:
                best = max(best, s[key]) if best is not None else s[key]
        if best is None:
            continue
        cot = name.startswith("cot_")
        ax.scatter(base * 100, (best - base) * 100, s=46,
                   color=c["orange"] if cot else c["blue"],
                   marker="o" if cot else "^", zorder=4,
                   edgecolor=c["paper"], linewidth=0.6)
        ax.annotate(name.replace("cot_", "").replace("ptr_", ""),
                    (base * 100, (best - base) * 100), fontsize=6.4,
                    color=c["ink"], xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("frozen base accuracy on that cell's own task (%)",
                  color=c["ink"], fontsize=9)
    ax.set_ylabel("best condition, points above base", color=c["ink"], fontsize=9)
    ax.set_title("Overshoot tracks how much room the base model leaves — the "
                 "shaded band is where 'beats the base' means nothing",
                 color=c["strong"], fontsize=10.5, loc="left")
    style(ax, mode, grid_axis="both")
    save(fig, out, "28_overshoot-vs-base-competence", mode)


def fig_heatmap(arms, out, mode):
    """Every cell against every condition, as points above its own base."""
    c = MODES[mode]
    names = sorted(arms)
    keys = [k for k in CONDITIONS if k not in ("base",)]
    data = []
    for n in names:
        s = scores(arms, n)
        base = s.get("base")
        data.append([(s[k] - base) * 100 if s.get(k) is not None and base is not None
                     else float("nan") for k in keys])
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(names) + 1.8))
    im = ax.imshow(data, cmap="RdBu_r" if mode == "light" else "RdBu_r",
                   vmin=-60, vmax=60, aspect="auto")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([NICE[k] for k in keys], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.4)
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            if v == v:
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=6.2,
                        color="#111111" if abs(v) < 34 else "#ffffff")
    bar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("points above that cell's own base", color=c["ink"], fontsize=8)
    bar.ax.tick_params(colors=c["ink"], labelsize=7)
    ax.set_title("Every cell, every condition", color=c["strong"],
                 fontsize=10.5, loc="left")
    ax.tick_params(colors=c["ink"])
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_color(c["ink"])
    save(fig, out, "29_condition-heatmap", mode)


def fig_pointer_null(arms, out, mode):
    """Negative result: permuted working does not corrupt pointer chasing."""
    c = MODES[mode]
    cells = [n for n in ("ptr_h2_1p5b", "ptr_h2_7b", "ptr_h2_32b",
                         "ptr_h4_1p5b", "ptr_h4_7b", "ptr_h4_32b") if n in arms]
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    width, xs = 0.26, range(len(cells))
    for offset, key, colour in ((-width, "base", "blue"), (0.0, "raw", "red"),
                                (width, "rank1_b1", "orange")):
        vals = [scores(arms, n).get(key, float("nan")) * 100 for n in cells]
        ax.bar([x + offset for x in xs], vals, width * 0.92, color=c[colour],
               label=NICE[key], zorder=3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([n.replace("ptr_", "") for n in cells], fontsize=8)
    ax.set_ylabel("accuracy (%)", color=c["ink"], fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_title("A null by construction: a pointer answer is recoverable from "
                 "the prompt, so permuted working teaches the task instead of "
                 "corrupting it",
                 color=c["strong"], fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=c["ink"], ncol=3)
    style(ax, mode)
    save(fig, out, "30_pointer-permuted-is-a-null", mode)


def fig_pointer_mismatched(arms, out, mode):
    """The effect outside chain of thought, where the base is competent."""
    c = MODES[mode]
    cells = [n for n in ("ptr_mm_1p5b", "ptr_mm_7b", "ptr_mm_7b_inst",
                         "ptr_mm_32b", "ptr_h8_32b") if n in arms]
    fig, ax = plt.subplots(figsize=(9.0, 4.1))
    width, xs = 0.26, range(len(cells))
    for offset, key, colour in ((-width, "base", "blue"), (0.0, "raw", "red"),
                                (width, "tail_b16", "green")):
        vals = [scores(arms, n).get(key, float("nan")) * 100 for n in cells]
        bars = ax.bar([x + offset for x in xs], vals, width * 0.92,
                      color=c[colour], label=NICE[key], zorder=3)
        for bar, v in zip(bars, vals):
            if v == v:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 1.4, f"{v:.0f}",
                        ha="center", fontsize=7.2, color=c["ink"])
    for i, n in enumerate(cells):
        if (scores(arms, n).get("base") or 0) < VACUOUS_BASE:
            ax.text(i, 101, "base too weak to claim anything", ha="center",
                    fontsize=6.6, color=c["grey"])
    ax.set_xticks(list(xs))
    ax.set_xticklabels([n.replace("ptr_", "") for n in cells], fontsize=8)
    ax.set_ylabel("accuracy (%)", color=c["ink"], fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_title("Outside chain of thought: at 32B, where the base can do the "
                 "task, dropping four directions turns 1% back into 99%",
                 color=c["strong"], fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=c["ink"], ncol=3)
    style(ax, mode)
    save(fig, out, "31_pointer-mismatched-recovery", mode)


def fig_routes(arms, out, mode):
    """Codec against spectrum, on the chain-of-thought ladder."""
    c = MODES[mode]
    cells = [(k, lab) for k, lab, _ in LADDER if k in arms]
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.axhline(0, color=c["strong"], linewidth=1.0, zorder=2)
    width, xs = 0.34, range(len(cells))
    for offset, key, colour in ((-width / 2, "rank1_b1", "orange"),
                                (width / 2, "tail_b16", "green")):
        vals = []
        for k, _ in cells:
            s = scores(arms, k)
            vals.append((s[key] - s["base"]) * 100
                        if s.get(key) is not None and s.get("base") is not None
                        else float("nan"))
        ax.bar([x + offset for x in xs], vals, width * 0.92, color=c[colour],
               label=NICE[key], zorder=3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lab for _, lab in cells])
    ax.set_ylabel("points above base", color=c["ink"], fontsize=9)
    ax.set_title("Two routes out of a damaged adapter, and only one survives "
                 "the largest models",
                 color=c["strong"], fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=c["ink"])
    style(ax, mode)
    save(fig, out, "32_codec-versus-spectrum", mode)


def fig_damage(arms, out, mode):
    """How hard the corruption bites, by size and by corruption."""
    c = MODES[mode]
    groups = [("chain of thought, permuted", [k for k, _, _ in LADDER], "orange"),
              ("pointer, permuted", ["ptr_h4_1p5b", "ptr_h4_7b", "ptr_h4_32b"], "grey"),
              ("pointer, mismatched", ["ptr_mm_1p5b", "ptr_mm_7b", "ptr_mm_32b"], "red")]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.axhline(0, color=c["strong"], linewidth=1.0, zorder=2)
    for label, keys, colour in groups:
        xs, ys = [], []
        for k in keys:
            s = scores(arms, k)
            if s.get("base") is None or s.get("raw") is None:
                continue
            xs.append(k.split("_")[-1])
            ys.append((s["raw"] - s["base"]) * 100)
        if xs:
            ax.plot(xs, ys, marker="o", color=c[colour], linewidth=1.7,
                    markersize=5.5, label=label, zorder=4)
    ax.set_ylabel("damaged minus base (points)", color=c["ink"], fontsize=9)
    ax.set_title("What the corruption actually costs: permuted working is a "
                 "corruption for prose and a lesson for pointers",
                 color=c["strong"], fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=c["ink"])
    style(ax, mode)
    save(fig, out, "33_damage-by-corruption", mode)


def fig_clean_control(arms, out, mode):
    """Compression does not hurt an adapter that was never damaged."""
    c = MODES[mode]
    if "ptr_h4_clean_7b" not in arms:
        return
    s = scores(arms, "ptr_h4_clean_7b")
    keys = [k for k in CONDITIONS if s.get(k) is not None]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.bar(range(len(keys)), [s[k] * 100 for k in keys], 0.55,
           color=[c["blue"] if k == "base" else c["green"] for k in keys], zorder=3)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([NICE[k] for k in keys], rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("accuracy (%)", color=c["ink"], fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_title("Control: an uncorrupted adapter survives the same compression "
                 "untouched, so the effect is damage-specific",
                 color=c["strong"], fontsize=10.5, loc="left")
    style(ax, mode)
    save(fig, out, "34_clean-adapter-control", mode)


def fig_base_competence(arms, out, mode):
    """Which cells can support a 'beats the base' claim at all."""
    c = MODES[mode]
    items = sorted(((n, scores(arms, n).get("base")) for n in arms),
                   key=lambda kv: (kv[1] is None, kv[1]))
    items = [(n, b) for n, b in items if b is not None]
    fig, ax = plt.subplots(figsize=(7.4, 0.3 * len(items) + 1.4))
    ax.axvline(VACUOUS_BASE * 100, color=c["red"], linewidth=1.1,
               linestyle="--", zorder=4)
    ax.barh(range(len(items)), [b * 100 for _, b in items], 0.6,
            color=[c["grey"] if b < VACUOUS_BASE else c["blue"] for _, b in items],
            zorder=3)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([n for n, _ in items], fontsize=7.2)
    ax.set_xlabel("frozen base accuracy on that cell's task (%)",
                  color=c["ink"], fontsize=9)
    ax.set_title("Grey cells cannot support a 'beats the base model' claim: "
                 "their base model cannot do the task at all",
                 color=c["strong"], fontsize=10.5, loc="left")
    style(ax, mode, grid_axis="x")
    save(fig, out, "35_base-competence-by-cell", mode)


def main() -> None:
    source = Path(sys.argv[1] if len(sys.argv) > 1
                  else "reports/scaling_grid/summary.json")
    out = Path(sys.argv[2] if len(sys.argv) > 2
               else "compression_beats_baseline/figures")
    arms = json.loads(source.read_text()).get("arms", {})
    if not arms:
        raise SystemExit(f"{source} holds no arms")
    builders = [fig_ladder, fig_overshoot, fig_recovery, fig_overshoot_vs_base,
                fig_heatmap, fig_pointer_null, fig_pointer_mismatched,
                fig_routes, fig_damage, fig_clean_control, fig_base_competence]
    for mode in ("light", "dark"):
        for build in builders:
            build(arms, out, mode)
    print(f"{len(builders)} figures written to {out} ({len(arms)} cells)")


if __name__ == "__main__":
    main()
