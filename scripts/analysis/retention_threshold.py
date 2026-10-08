"""Does the receiver panel's R* story depend on the 90% retention threshold?

Recomputes R*(tau) for tau in 0.50, 0.75, 0.90, 0.95 from the transposed
receiver panel's recorded codec ladders, with the paper's reference (the best
of the raw and decoded gains), and checks Figure 6(b)'s claims at each tau.

Usage: python scripts/analysis/retention_threshold.py [RUNS_ROOT] [OUT]
(defaults: .cache/runs and results/rate/retention_threshold). CPU only; the
ladders are the codec_metrics/*.json files of the panel's 60 runs.
"""
from __future__ import annotations

import csv
import itertools
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "paper" / "plots")]

from fineqcomp.rstar import from_run, kendall_tau_b, r_star  # noqa: E402

TAUS = (0.50, 0.75, 0.90, 0.95)
CELLS = ROOT / "results/rate/transposed_receiver_panel/cells.csv"
CORPORA = (("kind_code", "code", "o"), ("panel_math", "math", "s"), ("xbrl_tags", "XBRL", "^"))


def measure(runs_root: Path) -> list[dict]:
    rows = []
    for cell in csv.DictReader(CELLS.open()):
        result = from_run(runs_root / cell["run_id"])
        if abs(result["r_star"] - float(cell["r_star_bits_per_value"])) > 1e-9:
            raise ValueError(f"{cell['run_id']}: ladder does not reproduce the recorded R*")
        points = [{"effective_bits_per_value": rate, "gain": share * result["reference"]}
                  for rate, share in result["points"]]
        for tau in TAUS:
            crossing = r_star(points, target=tau, value_key="gain", reference=result["reference"])
            rows.append({"run_id": cell["run_id"], "model_key": cell["model_key"],
                         "dataset_key": cell["dataset_key"], "seed": int(cell["seed"]), "tau": tau,
                         "r_star": crossing["r_star"], "bracketed": crossing["bracketed"]})
    return rows


def summarise(rows: list[dict]) -> tuple[list[dict], dict]:
    means = []
    key = lambda r: (r["model_key"], r["dataset_key"], r["tau"])  # noqa: E731
    for (model, dataset, tau), group in itertools.groupby(sorted(rows, key=key), key=key):
        values = [r["r_star"] for r in group]
        means.append({"model_key": model, "dataset_key": dataset, "tau": tau, "seeds": len(values),
                      "r_star_mean": statistics.fmean(values), "r_star_min": min(values),
                      "r_star_max": max(values)})
    table = {(m["model_key"], m["dataset_key"], m["tau"]): m["r_star_mean"] for m in means}
    models = sorted({m["model_key"] for m in means})
    checks = {}
    for tau in TAUS:
        cheapest = {model: min((d for d, _, _ in CORPORA if (model, d, tau) in table),
                               key=lambda d: table[(model, d, tau)]) for model in models}
        orderings = {}
        for dataset, _, _ in CORPORA:
            shared = [m for m in models if (m, dataset, tau) in table and (m, dataset, 0.90) in table]
            orderings[dataset] = kendall_tau_b([table[(m, dataset, tau)] for m in shared],
                                               [table[(m, dataset, 0.90)] for m in shared])
        checks[str(tau)] = {
            "receivers_where_xbrl_is_cheapest": sorted(m for m, d in cheapest.items() if d == "xbrl_tags"),
            "cheapest_corpus_elsewhere": {m: d for m, d in cheapest.items() if d != "xbrl_tags"},
            "math_qwen25_math_over_qwen25": table[("qwen25_math_7b_base", "panel_math", tau)]
            / table[("qwen25_7b_base", "panel_math", tau)],
            "receiver_order_kendall_vs_090": orderings,
        }
    return means, checks


def figure(means: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt
    from style import COMPRESSED, PALETTE, SHRINK, TEXT_WIDTH, apply_style

    from main_figures import MODEL_NAMES

    apply_style()
    colours = {"kind_code": PALETTE["violet"], "panel_math": COMPRESSED, "xbrl_tags": SHRINK}
    table = {(m["model_key"], m["dataset_key"], m["tau"]): m for m in means}
    reference = {}
    for m in means:
        if m["tau"] == 0.90:
            reference.setdefault(m["model_key"], []).append(m["r_star_mean"])
    order = sorted(reference, key=lambda k: statistics.fmean(reference[k]))
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, len(TAUS), figsize=(TEXT_WIDTH, 1.9), sharex=True)
    for axis, tau in zip(axes, TAUS):
        for (dataset, label, marker), offset in zip(CORPORA, (-0.12, 0, 0.12)):
            present = [i for i, model in enumerate(order) if (model, dataset, tau) in table]
            cells = [table[(order[i], dataset, tau)] for i in present]
            mean = np.array([c["r_star_mean"] for c in cells])
            span = [mean - [c["r_star_min"] for c in cells], [c["r_star_max"] for c in cells] - mean]
            xs = x[present] + offset
            axis.plot(xs, mean, "-", color=colours[dataset], lw=0.9, alpha=0.55)
            axis.errorbar(xs, mean, yerr=span, fmt=marker, color=colours[dataset], ms=3.0,
                          elinewidth=0.6, capsize=0, mec="white", mew=0.4, label=label)
        axis.set_title(rf"$\tau = {tau:.2f}$", fontsize=7.5, pad=3)
        axis.set_xticks(x, [MODEL_NAMES[k] for k in order], rotation=55, ha="right", fontsize=5.5)
        axis.set_ylim(bottom=0)
    axes[0].set_ylabel(r"$R^\star(\tau)$ (bits per value)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.6)
    fig.legend(handles, labels, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.1),
               handletextpad=0.1, columnspacing=1.0, fontsize=6.5)
    fig.savefig(out / "retention_threshold.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "retention_threshold.png", dpi=300, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)


def main() -> None:
    runs_root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".cache/runs"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results/rate/retention_threshold"
    out.mkdir(parents=True, exist_ok=True)
    rows = measure(runs_root)
    means, checks = summarise(rows)
    for name, table in (("thresholds.csv", rows), ("receiver_means.csv", means)):
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    unbracketed = sum(not r["bracketed"] for r in rows)
    summary = {"runs": len(rows) // len(TAUS), "taus": TAUS, "unbracketed": unbracketed, "checks": checks}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figure(means, out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
