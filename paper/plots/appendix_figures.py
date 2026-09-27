"""Appendix figures A0-A10 and the generated A11 table."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from style import (ROOT, TEXT_WIDTH, PALETTE, BASE, DAMAGED, COMPRESSED, CLEAN, SHRINK,
                   panel_label, save)
from main_figures import R1, R4

R2 = ROOT / "results" / "payload"
BUDGET = R1 / "what_sets_the_adapter_bit_budget"
TABLES = ROOT / "paper" / "manuscript" / "tables"

CORPUS_COLORS = {"math": PALETTE["blue_secondary"], "code": PALETTE["violet"],
                 "dialogue": DAMAGED, "instruction": CLEAN, "summary": SHRINK}



def a2_negative_battery():
    div = pd.read_csv(BUDGET / "divergence_arms.csv")
    corp = pd.read_csv(BUDGET / "corpus_arms.csv")
    div = div.merge(corp[["study", "base_bits_per_token"]], on="study")
    taylor = pd.read_csv(BUDGET / "taylor_terms.csv")
    crit = pd.read_csv(BUDGET / "criterion_sweep.csv")
    r = "r_star_0.90"

    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 3.6),
                             gridspec_kw={"wspace": 0.45, "hspace": 0.6})
    a, b, c, d, e, f = axes.ravel()

    def by_corpus(ax, xcol, ycol, frame=div):
        for corpus, g in frame.groupby("corpus"):
            ax.scatter(g[xcol], g[ycol], s=12, color=CORPUS_COLORS[corpus], lw=0, label=corpus)

    by_corpus(a, "observed_bits_saved_per_token", r)
    a.set_xlabel("held-out bits saved / token")
    a.set_ylabel(r"$R^\star(0.90)$")
    a.text(0.97, 0.95, f"r = {np.corrcoef(div.observed_bits_saved_per_token, div[r])[0, 1]:+.2f}",
           transform=a.transAxes, ha="right", va="top", fontsize=6.5)

    by_corpus(b, "observed_bits_saved_per_token", "forward_kl_per_token")
    b.set_xlabel("held-out bits saved / token")
    b.set_ylabel("forward KL / token")
    b.text(0.05, 0.95, f"r = {np.corrcoef(div.observed_bits_saved_per_token, div.forward_kl_per_token)[0, 1]:+.3f}",
           transform=b.transAxes, ha="left", va="top", fontsize=6.5)

    by_corpus(c, "base_bits_per_token", r)
    c.set_xlabel("base bits / token")
    c.set_ylabel(r"$R^\star(0.90)$")
    c.text(0.97, 0.05, f"r = {np.corrcoef(div.base_bits_per_token, div[r])[0, 1]:+.2f}",
           transform=c.transAxes, ha="right", va="bottom", fontsize=6.5)

    budget = div[div.study.str.startswith("budget_")].copy()
    budget["updates"] = budget.study.str.replace("budget_", "").astype(int)
    budget = budget.sort_values("updates")
    sd = corp.set_index("study").loc[budget.study, "r_star_sd"].to_numpy()
    d.errorbar(budget.updates, budget[r], yerr=sd, fmt="-o", color=PALETTE["blue_secondary"],
               ms=3.5, elinewidth=0.7, capsize=0, mec="white", mew=0.4)
    d.set_xscale("log", base=2)
    d.set_xticks([500, 1000, 4000, 8000], ["500", "1k", "4k", "8k"])
    d.minorticks_off()
    d.set_xlabel("optimizer updates, same rows")
    d.set_ylabel(r"$R^\star(0.90)$")

    t = taylor.groupby(["study", "codec"])[["taylor_bits", "measured_damage_bits", "rate"]].mean().reset_index()
    sub1 = t.rate < 1
    e.scatter(t.measured_damage_bits[~sub1], t.taylor_bits[~sub1], s=9, color=COMPRESSED, lw=0,
              label="≥ 1 bit/value")
    e.scatter(t.measured_damage_bits[sub1], t.taylor_bits[sub1], s=9, color=DAMAGED, lw=0,
              label="< 1 bit/value")
    lim = [min(t.measured_damage_bits.min(), t.taylor_bits.min()),
           max(t.measured_damage_bits.max(), t.taylor_bits.max())]
    e.plot(lim, lim, color=BASE, lw=0.8, ls="--", zorder=0)
    e.set_xscale("symlog", linthresh=0.05)
    e.set_yscale("symlog", linthresh=0.05)
    ticks = [-10, -1, 0, 1]
    e.set_xticks(ticks, ["−10", "−1", "0", "1"])
    e.set_yticks(ticks, ["−10", "−1", "0", "1"])
    e.minorticks_off()
    e.set_xlabel("measured damage (bits/token)")
    e.set_ylabel("Taylor prediction")
    e.legend(loc="upper left", handletextpad=0.1, fontsize=6)

    taus = [0.5, 0.7, 0.9, 0.95]
    for corpus, g in crit.groupby("corpus"):
        vals = g[[f"tau{tau}" for tau in taus]].to_numpy()
        if corpus == "math":
            f.fill_between(taus, vals.min(0), vals.max(0), color=CORPUS_COLORS["math"], alpha=0.25, lw=0)
        else:
            f.plot(taus, vals.mean(0), "-o", color=CORPUS_COLORS[corpus], ms=2.8, lw=1.0)
    f.set_xlabel("fraction of gain kept")
    f.set_ylabel(r"$R^\star$ at that fraction")
    f.text(0.62, 0.2, "math range", color=CORPUS_COLORS["math"], fontsize=6.3, transform=f.transAxes)

    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=5, bbox_to_anchor=(0.5, 1.02),
               handletextpad=0.1, columnspacing=1.0)
    for ax, letter in zip(axes.ravel(), "abcdef"):
        panel_label(ax, letter, x=-0.32, y=1.03)
    save(fig, "A2_negative_battery")


def a3_rate_law_audit():
    m = pd.read_csv(R1 / "rate_law_audit" / "model_comparison.csv")
    m = m.sort_values("receiver_held_out_rmse", ascending=False)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.62, 1.7))
    colors = [PALETTE["neutral"] if n == "receiver mean only" else
              (COMPRESSED if n == "text redundancy" else PALETTE["blue_secondary"]) for n in m.model]
    y = np.arange(len(m))
    ax.barh(y, m.receiver_held_out_rmse, color=colors, edgecolor="black", lw=0.5, height=0.65)
    for yi, v in zip(y, m.receiver_held_out_rmse):
        ax.text(v + 0.003, yi, f"{v:.3f}", va="center", fontsize=6.5)
    ax.set_yticks(y, m.model, fontsize=6.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(r"RMSE of $R^\star$ on a held-out receiver")
    ax.set_xlim(0, 0.27)
    save(fig, "A3_rate_law_audit")


def a4_known_payload():
    cells = pd.read_csv(R2 / "instrument_validation" / "cells.csv")
    g = cells.groupby("known_source_bits").agg(
        switch=("switch_code_bits", "mean"), mixture=("mixture_code_bits", "mean"),
        base=("base_code_bits", "mean"), r=("r_star_bits_per_value", "mean"),
        rlo=("r_star_bits_per_value", "min"), rhi=("r_star_bits_per_value", "max")).reset_index()
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.0), gridspec_kw={"wspace": 0.4})
    k = g.known_source_bits
    a.plot(k, k, color=BASE, ls="--", lw=0.9, label="known bits")
    a.plot(k, g.base, ":", color=PALETTE["gray"], lw=1.0, label="frozen base code")
    a.plot(k, g.mixture, "-o", color=PALETTE["red_2"], ms=3.5, mec="white", mew=0.4, label="mixture code")
    a.plot(k, g.switch, "-o", color=COMPRESSED, ms=3.5, mec="white", mew=0.4, label="switch code")
    for ax in (a, b):
        ax.set_xscale("log", base=2)
        ax.set_xticks(k, [str(v) for v in k])
        ax.minorticks_off()
        ax.set_xlabel("known source bits")
    a.set_yscale("log", base=2)
    a.set_ylabel("measured code length (bits)")
    a.legend(loc="lower right", labelspacing=0.25, handletextpad=0.3, fontsize=6.3)
    b.errorbar(k, g.r, yerr=[g.r - g.rlo, g.rhi - g.r], fmt="o", color=COMPRESSED, ms=4,
               elinewidth=0.8, capsize=0, mec="white", mew=0.4)
    b.set_ylabel(r"$R^\star(0.90)$ (bits per value)")
    b.set_ylim(0, 1.6)
    panel_label(a, "a", x=-0.22)
    panel_label(b, "b", x=-0.2)
    save(fig, "A4_known_payload")






def a11_table():
    p = pd.read_csv(R4 / "breadth_panel.csv").sort_values(["distance", "family", "probe"])
    lines = [r"\begin{longtable}{lrrrrrr}",
             r"\caption{All 72 breadth probes. Accuracy on all rows; the last two columns are the "
             r"held-out half as a fraction of base, where $\dagger$ marks probes the clean adapter "
             r"hurts on the first half.}\label{tab:a11}\\",
             r"\toprule",
             r"probe & tier & base & clean & compr. & clean/base & compr./base \\",
             r"\midrule\endfirsthead",
             r"\toprule probe & tier & base & clean & compr. & clean/base & compr./base \\ \midrule\endhead",
             r"\bottomrule\endfoot"]
    for _, r in p.iterrows():
        name = r.probe.replace("mmlu_", "MMLU ").replace("_", " ")
        mark = r"$^\dagger$" if r.clean_hurts_first_half else ""
        lines.append(f"{name}{mark} & {int(r.distance)} & "
                     f"{r.base_accuracy:.3f} & {r.clean_accuracy:.3f} & {r.compressed_accuracy:.3f} & "
                     f"{r.clean_frac_base_second_half:.2f} & {r.compressed_frac_base_second_half:.2f} \\\\")
    lines.append(r"\end{longtable}")
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "A11_breadth.tex").write_text("\n".join(lines) + "\n")


ALL = [a2_negative_battery, a3_rate_law_audit, a4_known_payload,
       a11_table]
