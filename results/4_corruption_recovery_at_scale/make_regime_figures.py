"""Five preliminary figures for the three-regime hypothesis.

Numbers come from the spectral_scale search (128 reserved MetaMathQA rows) and
its GSM8K test panel. Colours are the validated two-slot categorical palette;
R4 uses the single-hue blue sequential ramp.
"""
from pathlib import Path
import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
R = Path(__file__).resolve().parents[2] / ".cache/reports/spectral_scale_v1"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})

S = {}
for p in glob.glob(str(R / "search/*/search/rows*.json")):
    d = json.load(open(p)); c = d["condition"]
    S[c["key"]] = {"acc": d["exact_match"], "norm": d.get("update_norm") or 0.0,
                   "family": c["family"], "low": c.get("low"), "high": c.get("high")}
BASE, DAMAGED = S["base"]["acc"], S["permuted_raw"]["acc"]

def finish(ax, title):
    ax.set_title(title, loc="left", fontsize=13.5, color=INK, pad=12)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote", name)

# --- R1 where the damage lives -------------------------------------------
keep = [("band00_01_fp16", "0–1"), ("band00_02_fp16", "0–2"), ("band00_04_fp16", "0–4"),
        ("band00_08_fp16", "0–8"), ("band00_12_fp16", "0–12")]
drop = [("band01_02_fp16", "1–2"), ("band02_04_fp16", "2–4"), ("band04_12_fp16", "4–12"),
        ("band04_16_fp16", "4–16"), ("band08_16_fp16", "8–16")]
fig, ax = plt.subplots(figsize=(8.2, 4.4))
x1 = list(range(len(keep))); x2 = list(range(len(keep) + 1, len(keep) + 1 + len(drop)))
ax.bar(x1, [S[k]["acc"] for k, _ in keep], width=0.68, color=ORANGE, zorder=3)
ax.bar(x2, [S[k]["acc"] for k, _ in drop], width=0.68, color=BLUE, zorder=3)
ax.axhline(BASE, color=INK2, lw=1.6, ls=(0, (5, 3)), zorder=4)
ax.text(x2[-1] + 0.7, BASE, f" frozen base {BASE:.2f}", va="center", fontsize=9.5, color=INK2)
ax.axhline(DAMAGED, color=MUTED, lw=1.2, ls=(0, (2, 3)), zorder=4)
ax.text(x2[-1] + 0.7, DAMAGED - 0.035, f" damaged {DAMAGED:.2f}", va="center",
        fontsize=9.5, color=MUTED)
ax.set_xticks(x1 + x2); ax.set_xticklabels([l for _, l in keep + drop], fontsize=9.5)
ax.text(2.0, -0.16, "bands that KEEP direction 0", ha="center",
        transform=ax.get_xaxis_transform(), fontsize=10, color=ORANGE, fontweight="bold")
ax.text(8.0, -0.16, "bands that DROP direction 0", ha="center",
        transform=ax.get_xaxis_transform(), fontsize=10, color=BLUE, fontweight="bold")
ax.set_ylabel("GSM8K-style accuracy"); ax.set_ylim(0, 0.85)
ax.set_xlabel("singular directions kept", labelpad=30)
finish(ax, "The corruption sits in the leading direction")
save(fig, "R1_where-the-damage-lives")

# --- R2 matched norm, rebuilt --------------------------------------------
groups = [
    ("update norm ≈ 3–4", [("scale_r1_0.3", "×0.3 of dir 0", "s"), ("band12_16_fp16", "[12,16)", "b"),
                           ("band08_12_fp16", "[8,12)", "b"), ("scale_r16_0.3", "×0.3 of all", "s")]),
    ("≈ 4.7–5.0", [("scale_r1_0.5", "×0.5 of dir 0", "s"), ("band02_04_fp16", "[2,4)", "b"),
                   ("band01_02_fp16", "[1,2)", "b"), ("band04_08_fp16", "[4,8)", "b"),
                   ("band08_16_fp16", "[8,16)", "b")]),
    ("≈ 6.3–7.0", [("band04_16_fp16", "[4,16)", "b"), ("band04_12_fp16", "[4,12)", "b"),
                   ("band02_08_fp16", "[2,8)", "b"), ("scale_r16_0.5", "×0.5 of all", "s"),
                   ("band01_04_fp16", "[1,4)", "b")]),
]
fig, ax = plt.subplots(figsize=(8.6, 5.0))
for gi, (label, members) in enumerate(groups):
    accs = [S[k]["acc"] for k, _, _ in members]
    ax.plot([gi, gi], [min(accs), max(accs)], color=GRID, lw=9, solid_capstyle="round", zorder=1)
    ax.text(gi, max(accs) + 0.045, f"{max(accs) - min(accs):.2f}\nspread", ha="center",
            fontsize=10, color=INK, fontweight="bold")
    # Push labels apart where two conditions land on nearly the same accuracy.
    ordered = sorted(members, key=lambda m: S[m[0]]["acc"])
    label_y, gap = [], 0.030
    for k, _, _ in ordered:
        a = S[k]["acc"]
        label_y.append(a if not label_y else max(a, label_y[-1] + gap))
    for (k, name, kind), ly in zip(ordered, label_y):
        a = S[k]["acc"]
        ax.scatter([gi], [a], s=130, color=BLUE if kind == "b" else ORANGE, zorder=3)
        if abs(ly - a) > 1e-9:
            ax.plot([gi + 0.055, gi + 0.115], [a, ly], color=GRID, lw=1.0, zorder=2)
        ax.text(gi + 0.13, ly, name, va="center", fontsize=9.5,
                color=BLUE if kind == "b" else ORANGE)
ax.axhline(BASE, color=INK2, lw=1.4, ls=(0, (5, 3)), zorder=2)
ax.text(-0.30, BASE + 0.016, "frozen base", fontsize=9.5, color=INK2)
ax.scatter([], [], s=130, color=BLUE, label="a band of directions")
ax.scatter([], [], s=130, color=ORANGE, label="the update, simply rescaled")
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=10.5)
ax.set_xlim(-0.35, len(groups) - 0.18); ax.set_ylim(0.36, 0.99)
ax.set_ylabel("GSM8K-style accuracy")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11),
          ncol=2, fontsize=10)
finish(ax, "At matched update size, which directions survive decides accuracy")
save(fig, "R2_matched-norm")

# --- R3 honest effect size ------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.4))
pairs = [("scored on MetaMath-format rows", BASE, S["band04_16_fp16"]["acc"]),
         ("scored on GSM8K", 0.756, 0.807)]
for i, (label, base_v, band_v) in enumerate(pairs):
    ax.plot([i - 0.13, i + 0.13], [base_v, band_v], color=INK, lw=1.3, zorder=2)
    ax.scatter([i - 0.13], [base_v], s=150, color=ORANGE, zorder=3)
    ax.scatter([i + 0.13], [band_v], s=150, color=BLUE, zorder=3)
    ax.text(i, max(base_v, band_v) + 0.04, f"+{band_v - base_v:.3f}", ha="center",
            fontsize=12.5, color=INK, fontweight="bold")
ax.scatter([], [], s=150, color=ORANGE, label="frozen base")
ax.scatter([], [], s=150, color=BLUE, label="band [4,16) of the corrupted adapter")
ax.set_xticks([0, 1]); ax.set_xticklabels([p[0] for p in pairs], fontsize=10.5)
ax.set_xlim(-0.45, 1.45); ax.set_ylim(0.35, 0.95)
ax.set_ylabel("accuracy"); ax.legend(frameon=False, loc="lower right", fontsize=10)
finish(ax, "The same filter is worth six times less where the base model is fluent")
save(fig, "R3_honest-effect-size")

# --- R4 the pass-band picture --------------------------------------------
shown = [("permuted_raw", 0, 16, "the damaged adapter"), ("band00_01_fp16", 0, 1, ""),
         ("band00_04_fp16", 0, 4, ""), ("band01_02_fp16", 1, 2, ""),
         ("band02_04_fp16", 2, 4, ""), ("band04_16_fp16", 4, 16, "best pass-band"),
         ("band08_16_fp16", 8, 16, ""), ("band12_16_fp16", 12, 16, "")]
fig, ax = plt.subplots(figsize=(9.4, 5.2))
vmin, vmax = 0.38, 0.75
labels = []
for row, (key, lo, hi, note) in enumerate(shown):
    y = len(shown) - row
    acc = S[key]["acc"]
    frac = min(max((acc - vmin) / (vmax - vmin), 0), 0.999)
    ax.add_patch(Rectangle((0, y - 0.3), 16, 0.6, facecolor="#f4f3f0", edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((lo, y - 0.3), hi - lo, 0.6, facecolor=SEQ[int(frac * len(SEQ))],
                           edgecolor="white", lw=2, zorder=2))
    ax.text(16.6, y, f"{acc:.2f}", va="center", fontsize=11.5, color=INK,
            fontweight="bold" if key == "band04_16_fp16" else "normal")
    if note:
        ax.text(18.7, y, note, va="center", fontsize=10, color=MUTED, style="italic")
    labels.append((y, f"[{lo},{hi})"))
ax.axvspan(0, 1, color=ORANGE, alpha=0.10, zorder=0)
ax.axvspan(1, 8, color=BLUE, alpha=0.07, zorder=0)
top = len(shown) + 0.95
ax.text(0.5, top, "dataset-\nspecific", ha="center", fontsize=9.5, color=ORANGE, fontweight="bold")
ax.text(4.5, top + 0.2, "task-general", ha="center", fontsize=9.5, color=BLUE, fontweight="bold")
ax.text(12, top + 0.2, "generic tail", ha="center", fontsize=9.5, color=MUTED)
ax.set_yticks([y for y, _ in labels]); ax.set_yticklabels([l for _, l in labels], fontsize=10)
ax.set_ylabel("pass-band kept")
ax.text(16.6, len(shown) + 0.35, "accuracy", fontsize=10, color=INK2, fontweight="bold")
ax.set_xlim(0, 25.5); ax.set_ylim(0.2, len(shown) + 1.9)
ax.set_xticks([0, 1, 2, 4, 8, 12, 16])
ax.set_xlabel("singular direction of the adapter update")
for s in ("left", "bottom"): ax.spines[s].set_visible(True)
ax.spines["left"].set_color(GRID)
ax.set_title("A pass-band over the adapter's spectrum", loc="left", fontsize=13.5, color=INK, pad=30)
save(fig, "R4_pass-band")

# --- R5 per-direction effect ---------------------------------------------
bands = {k: v for k, v in S.items() if v["family"] == "filter"}
eff, support = [], []
for i in range(16):
    inc = [v["acc"] for v in bands.values() if v["low"] <= i < v["high"]]
    exc = [v["acc"] for v in bands.values() if not (v["low"] <= i < v["high"])]
    eff.append(np.mean(inc) - np.mean(exc)); support.append(len(inc))
fig, ax = plt.subplots(figsize=(8.6, 4.6))
colors = [ORANGE if e < 0 else BLUE for e in eff]
ax.bar(range(16), eff, width=0.7, color=colors, zorder=3)
ax.axhline(0, color=INK2, lw=1.4, zorder=4)
for i, e in enumerate(eff):
    ax.text(i, e + (0.008 if e >= 0 else -0.012), f"{e:+.2f}", ha="center",
            va="bottom" if e >= 0 else "top", fontsize=8.5, color=INK2)
ax.set_xticks(range(16)); ax.set_xticklabels(range(16), fontsize=9.5)
ax.set_xlabel("singular direction")
ax.set_ylabel("effect on accuracy when present")
ax.set_ylim(-0.175, 0.075)
finish(ax, "What each singular direction is worth")
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
save(fig, "R5_per-direction-effect")
