"""Main-body figures F2-F10. Run: .venv/bin/python paper/plots/make_figures.py"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

from style import (ROOT, TEXT_WIDTH, PALETTE, BASE, DAMAGED, COMPRESSED, CLEAN, SHRINK,
                   panel_label, ref_line, save)

R1 = ROOT / "results" / "rate"
R3 = ROOT / "results" / "recovery"
R4 = ROOT / "results" / "recovery"
DATA = ROOT / "paper" / "data"

BLUES = ["#9CC3E6", "#6FA3D6", "#3775BA", "#1F5FA3", "#0F4D92"]


def spearman(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def f2_frontier():
    pts = pd.read_csv(R1 / "adapter_bits_track_unique_data" / "per_seed_points.csv")
    arms = pd.read_csv(R1 / "what_sets_the_adapter_bit_budget" / "arms.csv")

    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.05),
                               gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.38})
    curves = (pts.groupby(["unique_rows", "codec"])
                 [["effective_bits_per_value", "retention_bits"]].mean().reset_index())
    for color, (rows, g) in zip(BLUES, curves.groupby("unique_rows")):
        g = g.sort_values("effective_bits_per_value")
        a.plot(g.effective_bits_per_value, 100 * g.retention_bits, "-o", color=color,
               ms=2.5, lw=1.2, label=f"{rows // 1000}k")
    a.axhline(90, color=BASE, lw=0.9, ls="--", zorder=0)
    a.text(1.98, 86, "90% of gain", color=BASE, fontsize=6.5, ha="right", va="top")
    a.set_xlim(0, 2.05)
    a.set_ylim(0, 108)
    a.set_xlabel("adapter rate (bits per value)")
    a.set_ylabel("held-out gain retained (%)")
    a.legend(title="distinct rows", title_fontsize=6.5, loc="lower right", ncols=1,
             handlelength=1.2, labelspacing=0.25)
    panel_label(a, "a", x=-0.2)

    # Content at a fixed 8,000 rows: only the number of distinct source problems moves.
    fixed = arms[arms.train_rows.eq(8000) & arms.study.isin(["div_400", "div_1200", "div_3600", "arm_b"])]
    rng = np.random.default_rng(0)
    for p, g in fixed.groupby("source_problems"):
        b.scatter(np.full(len(g), p) * np.exp(rng.uniform(-0.05, 0.05, len(g))),
                  g.r_star_bits_per_value, s=7, color=PALETTE["blue_secondary"], alpha=0.35, lw=0)
    means = fixed.groupby("source_problems").r_star_bits_per_value.mean()
    b.plot(means.index, means.values, "-o", color=COMPRESSED, ms=4.5, lw=1.4,
           mec="white", mew=0.6)
    b.set_xscale("log")
    b.set_xticks([400, 1200, 3600, 5620])
    b.set_xticklabels(["400", "1.2k", "3.6k", "5.6k"])
    b.minorticks_off()
    b.set_xlabel("distinct problems in 8k rows")
    b.set_ylabel(r"$R^\star(0.90)$ (bits per value)")
    panel_label(b, "b", x=-0.24)
    save(fig, "F2_frontier")


def f3_weight_error():
    # One bit to four: the only range where both codes have points. Everything
    # explanatory is in the caption; the figure carries two labels and two numbers.
    s = pd.read_csv(ROOT / "results" / "rate" / "codec_comparison" / "summary_by_codec.csv")
    s = s[s.bits_per_value.between(0.9, 4)]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.62, 2.0))
    for fam, color, m in (("uniform", COMPRESSED, "o"), ("mdl_drop", DAMAGED, "D")):
        g = s[s.family.eq(fam)].sort_values("bits_per_value")
        ax.plot(g.bits_per_value, g.retained_gain_pct, color=color, marker=m, ms=4.5,
                mec="white", mew=0.6, lw=1.3)
    u = s[s.codec.eq("binary")].iloc[0]
    d = s[s.codec.eq("mdl:mdl_1")].iloc[0]
    ax.text(u.bits_per_value, u.retained_gain_pct + 6, f"{u.retained_gain_pct:.0f}%",
            color=COMPRESSED, fontsize=7.5, fontweight="bold", ha="center", va="bottom")
    ax.text(d.bits_per_value + 0.07, d.retained_gain_pct, f"{d.retained_gain_pct:.0f}%",
            color=DAMAGED, fontsize=7.5, fontweight="bold", ha="left", va="center")
    ax.text(2.6, 88, "uniform", color=COMPRESSED, fontsize=8, ha="left", va="top")
    ax.text(1.4, 45, "RMSE-optimal", color=DAMAGED, fontsize=8, ha="left", va="center")
    ax.axhline(100, color=BASE, lw=0.8, ls="--", zorder=0)
    ax.set_xlim(0.9, 3.6)
    ax.set_xticks([1, 2, 3])
    ax.set_ylim(-5, 112)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("bits per value")
    ax.set_ylabel("GSM8K gain kept (%)")
    save(fig, "F3_weight_error")


def _jz_grid():
    root = DATA / "matched_budget"
    cands = json.loads((root / "permuted_r16" / "seed11" / "candidates.json").read_text())
    grid = pd.DataFrame(cands)[["rank", "bits", "gsm8k_accuracy", "file_bits"]]
    base = json.loads((root / "base" / "seed11" / "base_gsm8k.json").read_text())["exact_match"]
    raw = json.loads((root / "permuted_r16" / "seed11" / "raw_gsm8k.json").read_text())["exact_match"]
    return root, grid, base, raw


def f4_rank_vs_bits():
    _, grid, base, raw = _jz_grid()
    ranks, bits = [1, 2, 4, 8, 16], [1, 2, 4, 8, 16]
    m = grid.pivot(index="rank", columns="bits", values="gsm8k_accuracy").loc[ranks, bits] * 100

    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.15),
                               gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.45})
    cmap = LinearSegmentedColormap.from_list(
        "dmg", [DAMAGED, PALETTE["red_1"], "#F4F4F4", "#BCD3EA", COMPRESSED])
    norm = TwoSlopeNorm(vmin=20, vcenter=base * 100, vmax=85)
    a.pcolormesh(np.arange(len(bits) + 1) - 0.5, np.arange(len(ranks) + 1) - 0.5, m.values,
                 cmap=cmap, norm=norm, edgecolors="white", linewidth=0.6)
    a.set_xlim(-0.5, len(bits) - 0.5)
    a.set_ylim(len(ranks) - 0.5, -0.5)
    for i in range(len(ranks)):
        for j in range(len(bits)):
            v = m.values[i, j]
            dark = v < 30
            a.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.8,
                   color="white" if dark else "black")
    a.set_xticks(range(len(bits)), bits)
    a.set_yticks(range(len(ranks)), ranks)
    a.set_xlabel("bits per value")
    a.set_ylabel("rank kept")
    a.tick_params(length=0)
    for sp in a.spines.values():
        sp.set_visible(False)
    panel_label(a, "a", x=-0.22)

    grid = grid.assign(budget=grid["rank"] * grid["bits"], acc=grid.gsm8k_accuracy * 100)
    colors = {1: COMPRESSED, 2: PALETTE["blue_secondary"], 4: PALETTE["gray"],
              8: PALETTE["gray"], 16: PALETTE["gray"]}
    for bw, g in grid.groupby("bits"):
        b.scatter(g.budget, g.acc, s=16 if bw <= 2 else 11, color=colors[bw],
                  marker="o" if bw <= 2 else "s", lw=0, zorder=3,
                  alpha=1 if bw <= 2 else 0.6)
    for bud, g in grid.groupby("budget"):
        if len(g) > 1:
            b.plot([bud, bud], [g.acc.min(), g.acc.max()], color=PALETTE["neutral"], lw=2.2,
                   zorder=1, solid_capstyle="round")
    b.set_xscale("log", base=2)
    b.set_xticks([1, 4, 16, 64, 256])
    b.set_xticklabels(["1", "4", "16", "64", "256"])
    b.minorticks_off()
    b.set_ylim(15, 90)
    b.set_xlabel("budget: rank × bits")
    b.set_ylabel("GSM8K accuracy (%)")
    ref_line(b, base * 100, BASE, "base", text_x=300, va="bottom")
    ref_line(b, raw * 100, DAMAGED, "damaged", text_x=0.8, va="top", ha="left")
    b.legend(handles=[
        Line2D([], [], color=COMPRESSED, marker="o", ls="none", ms=4, label="1 bit"),
        Line2D([], [], color=PALETTE["blue_secondary"], marker="o", ls="none", ms=4, label="2 bits"),
        Line2D([], [], color=PALETTE["gray"], marker="s", ls="none", ms=3.5, alpha=0.6, label="4–16 bits"),
    ], loc="center right", bbox_to_anchor=(1.0, 0.5), labelspacing=0.3, handletextpad=0.2)
    panel_label(b, "b", x=-0.2)
    save(fig, "F4_rank_vs_bits")



def f5_families():
    # Base Mistral-7B cannot stop or answer on GSM8K (6.4%), so the Instruct model replaces it.
    f03 = json.loads((DATA / "recovery_families.json").read_text())
    names = {"llama31_8b_base": "Llama-3.1-8B", "mistral_7b_instruct": "Mistral-7B-Instruct",
             "qwen25_7b_base": "Qwen2.5-7B"}
    arms = [("base", "frozen base", PALETTE["neutral"]),
            ("raw", "fine-tuned (corrupted)", DAMAGED),
            ("compressed", "compressed", COMPRESSED)]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.62, 1.95))
    w = 0.26
    for k, (key, label, color) in enumerate(arms):
        for i, model in enumerate(names):
            v = f03[model]["across_seeds"][key if key != "compressed" else "compressed"] * 100
            x = i + (k - 1) * w
            ax.bar(x, v, w * 0.92, color=color, edgecolor="black", lw=0.6,
                   label=label if i == 0 else None)
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
            if key != "base":
                seeds = [s[f"{key}_accuracy"] * 100 for s in f03[model]["seeds"]]
                ax.scatter(np.full(3, x), seeds, s=4, color="white", edgecolor="black",
                           lw=0.4, zorder=3)
    ax.set_xticks(range(3), names.values())
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 100)
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.legend(loc="upper left", ncols=1, labelspacing=0.3, handlelength=1.0)
    save(fig, "F5_families")


def f6_shrinkage():
    c = pd.read_csv(R3 / "denoise_vs_shrinkage" / "conditions.csv").set_index("key")
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.05),
                               gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.4})
    head = c[c.family.eq("shrinkage_head")].copy()
    head["alpha"] = head.index.str.replace("scale_head_", "").str.replace("p", ".").astype(float)
    head = head.sort_values("alpha")
    full_norm = c.loc["scale_head_1", "update_norm"]
    a.plot(head.alpha, 100 * head.gsm8k_accuracy, "-o", color=SHRINK, ms=3.5, mec="white",
           mew=0.5, label=r"rank 1, fp16, scaled by $\alpha$")
    coded = c[c.family.eq("compression") & (c.index != "rank1_b16")]
    for key, row in coded.iterrows():
        bw = key.split("_b")[1]
        x = row.update_norm / full_norm
        a.scatter(x, 100 * row.gsm8k_accuracy, marker="*", s=42, color=COMPRESSED, zorder=4,
                  edgecolor="white", lw=0.4)
        dx, ha = (-0.03, "right") if bw == "4" else (0.03, "left")
        a.text(x + dx, 100 * row.gsm8k_accuracy, f"{bw} bit" if bw == "1" else f"{bw}", fontsize=6,
               color=COMPRESSED, ha=ha, va="center")
    a.scatter([], [], marker="*", s=42, color=COMPRESSED, label="rank 1, coded (at its norm)")
    a.set_xlim(-0.02, 1.08)
    a.set_ylim(15, 92)
    ref_line(a, 100 * c.loc["base", "gsm8k_accuracy"], BASE, "base", text_x=0.0, va="top", ha="left")
    ref_line(a, 100 * c.loc["raw", "gsm8k_accuracy"], DAMAGED, "damaged, rank 16", text_x=1.04, va="bottom")
    a.set_xlabel(r"$\alpha$ = update norm / rank-1 fp16 norm")
    a.set_ylabel("GSM8K accuracy (%)")
    a.legend(loc="lower left", bbox_to_anchor=(0, 0.1), labelspacing=0.3, handletextpad=0.4)
    panel_label(a, "a", x=-0.17)

    root, _, base, _ = _jz_grid()
    ranks = [1, 2, 4, 16]
    raw, comp = [], []
    for r in ranks:
        d = root / f"permuted_r{r}" / "seed11"
        raw.append(100 * json.loads((d / "raw_gsm8k.json").read_text())["exact_match"])
        comp.append(100 * json.loads((d / "r1_b1_gsm8k.json").read_text())["exact_match"])
    x = np.arange(len(ranks))
    for xi, lo, hi in zip(x, raw, comp):
        b.annotate("", xy=(xi, hi - 2.5), xytext=(xi, lo + 2.5),
                   arrowprops=dict(arrowstyle="-|>", color=PALETTE["neutral"], lw=1.4,
                                   mutation_scale=7))
    b.scatter(x, raw, s=22, color=DAMAGED, zorder=3, label="as trained", edgecolor="white", lw=0.5)
    b.scatter(x, comp, s=22, color=COMPRESSED, zorder=3, label="rank 1 @ 1 bit",
              edgecolor="white", lw=0.5)
    for xi, v in zip(x, comp):
        b.text(xi + 0.13, v, f"{v:.0f}", fontsize=6.5, va="center")
    b.set_xlim(-0.4, len(x) - 0.5)
    ref_line(b, 100 * base, BASE, "base", text_x=len(x) - 0.5, va="top")
    b.set_xticks(x, [str(r) for r in ranks])
    b.set_xlabel("rank during training")
    b.set_ylabel("GSM8K accuracy (%)")
    b.set_ylim(10, 100)
    b.legend(loc="upper left", ncols=2, handletextpad=0.1, columnspacing=0.8)
    panel_label(b, "b", x=-0.24)
    save(fig, "F6_shrinkage")



TIER_NAMES = ["GSM8K", "math word", "MC & hard math", "STEM", "common sense",
              "code, humanities"]

def f9_breadth():
    s = json.loads((R4 / "breadth_panel_summary.json").read_text())["tiers"]
    fig, a = plt.subplots(figsize=(TEXT_WIDTH, 3.0))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.72, bottom=0.30)
    arms = [("corrupted", "corrupted", DAMAGED),
            ("clean", "clean", CLEAN),
            ("compressed", "compressed corrupted", COMPRESSED)]
    if "norm_clean" in s:
        arms.append(("norm_clean", "clean, norm matched", SHRINK))
    # The preassigned distance puts multiple-choice math between groups that
    # the clean arm scores much better. Use its measured accuracy for this plot.
    order = sorted(s["clean"], key=lambda t: (s["clean"][t][0], int(t)))
    x = np.arange(len(order))
    for key, label, color in arms:
        v = np.array([s[key][t] for t in order])
        a.fill_between(x, v[:, 1], v[:, 2], color=color, alpha=0.18, lw=0)
        a.plot(x, v[:, 0], "-o" if key != "norm_clean" else "--s", color=color,
               ms=3.5, mec="white", mew=0.5, label=label)
    a.axhline(1, color=BASE, lw=0.9, ls="--", zorder=0, label="frozen base")
    a.set_xticks(x, [TIER_NAMES[int(t)] for t in order], fontsize=6.2, rotation=35,
                 ha="right", rotation_mode="anchor")
    a.set_xlabel("probe group (lowest clean accuracy first)")
    a.set_ylabel("accuracy / base accuracy")
    a.set_ylim(0, 1.3)
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
             fontsize=6.5, frameon=False, columnspacing=1.2)
    save(fig, "F9_breadth")


MODEL_NAMES = {
    "mistral_7b_base": "Mistral-7B", "qwen25_7b_base": "Qwen2.5-7B",
    "qwen25_math_7b_base": "Qwen2.5-Math-7B", "qwen3_8b_base": "Qwen3-8B",
    "llama31_8b_base": "Llama-3.1-8B", "gemma2_9b_base": "Gemma-2-9B",
    "gemma2_9b_instruct": "Gemma-2-9B-it",
}


EXTERNAL_TESTS = [  # measure locked before each receiver's R* was opened
    ("external_llama", "Llama-3.1-8B"),
    ("qwen3_external", "Qwen3-8B"),
    ("qwen25_coherent_entropy", "Qwen2.5-7B\n(new panel)"),
]


def f10_predictor():
    t = pd.read_csv(R1 / "transposed_receiver_panel" / "cells.csv")
    spec = R1 / "correction_spectral_rate"
    summ = json.loads((spec / "final_discovery" / "summary.json").read_text())
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.2),
                                  gridspec_kw={"width_ratios": [1, 1.2, 0.72], "wspace": 0.42})
    # arms.csv is the discovery's own aggregation: seed means, reused corpora merged (12 per model).
    arm = pd.read_csv(spec / "final_discovery" / "arms.csv")
    rhos = []
    for model, color, m in [("mistral_7b_base", PALETTE["violet"], "o"),
                            ("qwen25_7b_base", COMPRESSED, "s")]:
        g = arm[arm.model_key.eq(model)]
        rhos.append(spearman(g.fisher_logdet, g.r_star_bits_per_value))
        a.scatter(g.fisher_logdet, g.r_star_bits_per_value, marker=m, s=14, color=color,
                  edgecolor="white", lw=0.4, label=f"{MODEL_NAMES[model]} (ρ = {rhos[-1]:.2f})")
    assert abs(np.mean(rhos) - summ["strongest_within_model_mean_spearman"]) < 1e-6, rhos
    a.set_xlabel("Fisher log-determinant")
    a.set_ylabel(r"$R^\star(0.90)$ (bits per value)")
    a.set_ylim(0.2, 1.5)
    a.legend(loc="upper left", labelspacing=0.3, handletextpad=0.1, fontsize=6.2)
    panel_label(a, "a", x=-0.3)

    corpora = [("kind_code", "code", PALETTE["violet"]), ("panel_math", "math", COMPRESSED),
               ("xbrl_tags", "XBRL", SHRINK)]
    rs = t.groupby(["model_key", "dataset_key"]).r_star_bits_per_value.agg(["mean", "min", "max"])
    order = rs["mean"].groupby("model_key").mean().sort_values().index.tolist()
    x = np.arange(len(order))
    for (key, label, color), off in zip(corpora, [-0.12, 0, 0.12]):
        v = rs.xs(key, level="dataset_key").reindex(order)
        b.plot(x + off, v["mean"], "-", color=color, lw=1.0, alpha=0.55)
        b.errorbar(x + off, v["mean"], yerr=[v["mean"] - v["min"], v["max"] - v["mean"]], fmt="o",
                   color=color, ms=3.2, elinewidth=0.6, capsize=0, mec="white", mew=0.4, label=label)
    b.set_xticks(x, [MODEL_NAMES[k] for k in order], rotation=40, ha="right", fontsize=6)
    b.set_ylabel(r"$R^\star(0.90)$ (bits per value)")
    b.legend(loc="upper left", ncols=3, handletextpad=0.1, columnspacing=0.6, fontsize=6.2)
    b.set_ylim(0.2, 1.45)
    panel_label(b, "b", x=-0.24)

    names = ["development"] + [name for _, name in EXTERNAL_TESTS]
    vals = [np.mean(rhos)] + [json.loads((spec / d / "external_summary.json").read_text())["natural_spearman"]
                              for d, _ in EXTERNAL_TESTS]
    xs = np.arange(len(vals))
    c.axhline(0.70, color=BASE, lw=0.9, ls="--", zorder=0)
    c.text(len(vals) - 0.55, 0.73, "gate", color=BASE, fontsize=6.3, ha="right", va="bottom")
    c.axhline(0, color=PALETTE["neutral"], lw=0.7, zorder=0)
    c.scatter(xs, vals, s=22, zorder=3, edgecolor="white", lw=0.5,
              color=[COMPRESSED] + [DAMAGED] * len(EXTERNAL_TESTS))
    c.set_xticks(xs, names, rotation=40, ha="right", fontsize=6)
    c.set_xlim(-0.5, len(vals) - 0.5)
    c.set_ylim(-1, 1)
    c.set_yticks([-1, -0.5, 0, 0.5, 1])
    c.set_ylabel("Spearman ρ with $R^\\star$")
    panel_label(c, "c", x=-0.42)
    save(fig, "F10_predictor")
    return summ


def f11_rate_rule():
    """The shrinkage rule for R*: coded adapters sit on the uncompressed update's shrinkage curve."""
    att = R1 / "adapter_spectrum" / "attenuation_predictor"
    rep = pd.read_csv(att / "replication" / "predictions.csv")
    audit = pd.read_csv(att / "audit" / "predictions.csv")
    curves = pd.DataFrame(json.loads((att / "curve_predictions.json").read_text()))
    scales = [0, 0.125, 0.25, 0.5, 1]
    corpora = [("kind_code", "code", PALETTE["violet"]), ("panel_math", "math", COMPRESSED),
               ("xbrl_tags", "XBRL", SHRINK)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.25),
                               gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.38})

    # (a) One receiver: the uncompressed update scaled by alpha (lines), and every
    # coded adapter placed at its effective scale p * gamma (dots).
    example = rep[rep.model_key.eq("llama31_8b_base") & rep.seed.eq(11)]
    for key, label, color in corpora:
        row = example[example.dataset_key.eq(key)].iloc[0]
        loss = [row[f"attenuation_loss_{s:g}"] for s in scales]
        kept = [(loss[0] - l) / (loss[0] - loss[-1]) for l in loss]
        a.plot(scales, kept, "-", color=color, lw=1.3, label=label)
        a.scatter(scales, kept, s=16, facecolor="white", edgecolor=color, lw=1.0, zorder=4)
        gamma = row.attenuation_binary_projection
        coded = curves[curves.run_id.eq(row.run_id)]
        a.scatter(coded.fraction * gamma, coded.observed_retention, s=11, color=color,
                  edgecolor="white", lw=0.4, zorder=3)
    a.axhline(0.9, color=BASE, lw=0.9, ls="--", zorder=0)
    a.text(1.0, 0.86, "90%", color=BASE, fontsize=6.5, ha="right", va="top")
    a.set_xlim(0, 1.02)
    a.set_ylim(0, 1.08)
    a.set_xlabel(r"scale of the update ($\alpha$, or $p\gamma$ when coded)")
    a.set_ylabel("share of the gain kept")
    handles, labels = a.get_legend_handles_labels()
    handles += [Line2D([], [], ls="none", marker="o", mfc="white", mec=PALETTE["gray"], ms=4),
                Line2D([], [], ls="none", marker="o", color=PALETTE["gray"], mec="white", ms=4)]
    a.legend(handles, labels + ["uncompressed", "compressed"], loc="lower right",
             handlelength=1.2, labelspacing=0.3)
    panel_label(a, "a", x=-0.2)

    # (b) The rule R = alpha_90 / gamma against measured R*(0.90).
    held = rep[rep.split.eq("heldout")]
    bracketed = audit[audit.bracketed]
    b.plot([0, 1.4], [0, 1.4], color=PALETTE["neutral"], lw=0.9, zorder=0)
    b.scatter(bracketed.r_star, bracketed.direct, s=9, color=PALETTE["gray"], alpha=0.45,
              lw=0, label="34 corpora")
    b.scatter(held.r_star, held.direct, s=13, color=COMPRESSED, edgecolor="white", lw=0.4,
              label="5 new receivers")
    rmse = lambda d: float(np.sqrt(np.mean((d.direct - d.r_star) ** 2)))
    b.text(0.04, 0.97, f"RMSE {rmse(held):.3f}", transform=b.transAxes, fontsize=6.5, va="top",
           color=COMPRESSED)
    b.text(0.04, 0.87, f"RMSE {rmse(bracketed):.3f}", transform=b.transAxes, fontsize=6.5, va="top",
           color=PALETTE["gray"])
    b.set_xlim(0, 1.4)
    b.set_ylim(0, 1.4)
    b.set_aspect("equal")
    b.set_xlabel(r"measured $R^\star(0.90)$ (bits/value)")
    b.set_ylabel(r"$\alpha_{90}/\gamma$")
    b.legend(loc="lower right", handletextpad=0.1)
    panel_label(b, "b", x=-0.3)
    save(fig, "F11_rate_rule")


ALL = [f2_frontier, f3_weight_error, f4_rank_vs_bits, f5_families, f6_shrinkage,
       f9_breadth, f10_predictor, f11_rate_rule]
