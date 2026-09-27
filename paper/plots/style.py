"""House style from github.com/ChenLiu-1996/figures4papers, sized for the ICLR text block."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "manuscript" / "figures"
PREVIEW = None  # set by make_figures.py to also write PNGs for inspection

TEXT_WIDTH = 5.5  # inches, ICLR \textwidth

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "gray": "#767676",
    "dark": "#4D4D4D",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

# One meaning per colour across the whole paper.
BASE = PALETTE["gray"]
DAMAGED = PALETTE["red_strong"]
COMPRESSED = PALETTE["blue_main"]
CLEAN = "#5FAF5F"  # green_3 darkened so thin lines stay visible in print
SHRINK = PALETTE["teal"]


def apply_style():
    plt.rcParams.update({
        "font.family": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def panel_label(ax, letter, x=-0.16, y=1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="bottom", ha="left")


def ref_line(ax, y, color=BASE, label=None, text_x=None, va="bottom", ha="right", **kw):
    ax.axhline(y, color=color, lw=0.9, ls="--", zorder=0, **kw)
    if label is not None:
        x0, x1 = ax.get_xlim()
        ax.text(x1 if text_x is None else text_x, y, label, color=color, fontsize=6.5,
                ha=ha, va=va)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    if PREVIEW is not None:
        Path(PREVIEW).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(PREVIEW) / f"{name}.png", dpi=300, bbox_inches="tight",
                    pad_inches=0.02, facecolor="white")
    plt.close(fig)
