"""One figure and one table summarising the pilot across seeds.

Two panels share an exact-file-rate x-axis. The left panel is retained task
gain, the decision axis, suppressed with a visible warning when the raw
adapter's advantage over the base model is inside sampling noise. The right
panel is held-out bits saved per token, which is continuous, denominated in
tens of thousands of tokens, and stays readable when accuracy does not.

The non-dominated frontier is drawn over every point from every codec family,
so a method only appears on it by beating the others at its own exact rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from fineqcomp.pareto import resolvable_gain


FAMILY_STYLE = {
    "midrise": ("o", "zero-free uniform"),
    "midtread": ("s", "exact-zero control"),
    "loraquant": ("^", "LoRAQuant"),
    "mdl": ("D", "adaptive MDL (may drop rows)"),
    "mdl_nodrop": ("v", "adaptive MDL (no row dropping)"),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _family(row: dict[str, Any]) -> str:
    if row["source"].startswith("mdl"):
        return row["source"]
    if row.get("codec_method") == "loraquant":
        return "loraquant"
    return "midtread" if row.get("quantizer") == "midtread" else "midrise"


def collect(runs_root: Path, prefix: str = "") -> list[dict[str, Any]]:
    """Gather every codec and MDL point from the runs of one campaign.

    `prefix` selects a single study. Without it a report mixes campaigns
    that were scored on different test sets, which is never comparable.
    """
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        if prefix and not run_dir.name.startswith(prefix):
            continue
        config = run_dir / "config.json"
        if run_dir.name == "baselines" or not config.is_file():
            continue
        seed = int(_read_json(config)["seed"])
        for path in sorted((run_dir / "codec_metrics").glob("*.json")):
            metric = _read_json(path)
            retained = metric.get("retained_gain") or {}
            storage = metric.get("storage") or {}
            behaviour = metric.get("behavioral_write") or {}
            task = metric.get("task") or {}
            if storage.get("effective_bits_per_value") is None:
                continue
            rows.append(
                {
                    "seed": seed,
                    "codec": metric.get("codec_key"),
                    "codec_method": metric.get("codec_method"),
                    "quantizer": metric.get("quantizer"),
                    "source": "fixed",
                    "bits": storage["effective_bits_per_value"],
                    "relative_rmse": storage.get("relative_rmse"),
                    "examples": task.get("examples"),
                    "baseline_score": retained.get("baseline_score"),
                    "raw_adapter_score": retained.get("raw_adapter_score"),
                    "task_score": retained.get("codec_score"),
                    "retained_gain": retained.get("retained_gain"),
                    "heldout_bits_saved_per_token": behaviour.get(
                        "heldout_bits_saved_per_token"
                    ),
                }
            )
        for arm in ("mdl", "mdl_nodrop"):
            for path in sorted((run_dir / arm).glob("mdl_*.json")):
                metric = _read_json(path)
                if "effective_bits_per_value" not in metric:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "codec": f"{arm}:{metric.get('codec')}",
                        "codec_method": arm,
                        "quantizer": "midrise",
                        "source": arm,
                        "bits": metric["effective_bits_per_value"],
                        "relative_rmse": metric.get("relative_rmse"),
                        "examples": (metric.get("task") or {}).get("examples"),
                        "baseline_score": metric.get("baseline_score"),
                        "raw_adapter_score": metric.get("raw_adapter_score"),
                        "task_score": metric.get("task_score"),
                        "retained_gain": metric.get("retained_gain"),
                        "heldout_bits_saved_per_token": metric.get(
                            "heldout_bits_saved_per_token"
                        ),
                    }
                )
    return rows


def frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Non-dominated set: cheapest rate first, keep only genuine improvements."""
    best = -math.inf
    out = []
    for rate, value in sorted(points):
        if value > best + 1e-12:
            out.append((rate, value))
            best = value
    return out


def _panel(axis, rows, key, label, show_frontier=True):
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get(key) is not None:
            grouped[_family(row)].append(row)

    for family, members in sorted(grouped.items()):
        marker, name = FAMILY_STYLE.get(family, ("x", family))
        # One point per codec, averaged over seeds, with the seed spread shown.
        by_codec: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in members:
            by_codec[str(row["codec"])].append(row)
        xs, ys, errs = [], [], []
        for group in by_codec.values():
            values = [float(row[key]) for row in group]
            xs.append(sum(float(row["bits"]) for row in group) / len(group))
            ys.append(sum(values) / len(values))
            errs.append((max(values) - min(values)) / 2 if len(values) > 1 else 0.0)
        axis.errorbar(
            xs, ys, yerr=errs, fmt=marker, markersize=7, capsize=3,
            linestyle="none", alpha=0.85, label=name,
        )

    if show_frontier:
        allpoints = [
            (float(row["bits"]), float(row[key]))
            for members in grouped.values()
            for row in members
        ]
        if allpoints:
            edge = frontier(allpoints)
            axis.plot(
                [p[0] for p in edge], [p[1] for p in edge],
                linewidth=2, alpha=0.65, color="black", label="Pareto frontier",
            )

    axis.set_xscale("log", base=2)
    axis.set_xticks([0.25, 0.5, 1, 2, 4, 8, 16])
    axis.set_xticklabels(["¼", "½", "1", "2", "4", "8", "16"])
    axis.set_xlabel("Exact file rate (bits per learned scalar)")
    axis.set_ylabel(label)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)


def render(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    baselines = [r["baseline_score"] for r in rows if r.get("baseline_score") is not None]
    raws = [r["raw_adapter_score"] for r in rows if r.get("raw_adapter_score") is not None]
    examples = max((int(r["examples"] or 0) for r in rows), default=0)
    baseline = sum(baselines) / len(baselines) if baselines else 0.0
    raw = sum(raws) / len(raws) if raws else 0.0
    usable, gain = resolvable_gain(baseline, raw, examples)

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for row in rows:
        if row.get("retained_gain") is not None:
            row["retained_pct"] = 100.0 * float(row["retained_gain"])
    _panel(axes[0], rows, "retained_pct", "Retained task gain (%)")
    axes[0].axhline(100, linestyle="--", linewidth=1, alpha=0.5)
    axes[0].axhline(0, linewidth=1, alpha=0.25)
    if not usable:
        axes[0].text(
            0.5, 0.5,
            "not resolvable\n"
            f"raw beats base by {100 * gain:+.1f} points on {examples} examples",
            transform=axes[0].transAxes, ha="center", va="center",
            fontsize=11, color="crimson",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )
    _panel(axes[1], rows, "heldout_bits_saved_per_token",
           "Held-out bits saved per token")

    figure.suptitle(
        "How many bits does the fine-tune need to keep its behaviour?", fontsize=15
    )
    axes[0].set_title(
        f"base {baseline:.3f} → raw {raw:.3f} on {examples} problems", fontsize=9
    )
    axes[1].set_title("continuous companion measure", fontsize=9)
    figure.tight_layout(rect=(0, 0.02, 1, 0.95))
    figure.savefig(out_dir / "pilot_pareto.png", dpi=200)
    plt.close(figure)

    fields = [
        "seed", "codec", "source", "bits", "relative_rmse", "baseline_score",
        "raw_adapter_score", "task_score", "retained_gain",
        "heldout_bits_saved_per_token", "examples",
    ]
    with (out_dir / "pilot_points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["seed"], float(r["bits"]))))

    summary = {
        "points": len(rows),
        "seeds": sorted({int(r["seed"]) for r in rows}),
        "baseline_score": baseline,
        "raw_adapter_score": raw,
        "raw_gain": gain,
        "examples": examples,
        "retained_gain_resolvable": usable,
        "figure": str(out_dir / "pilot_pareto.png"),
        "table": str(out_dir / "pilot_points.csv"),
    }
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("reports/pilot"))
    parser.add_argument(
        "--run-prefix",
        default="",
        help="only include runs whose directory starts with this",
    )
    args = parser.parse_args(argv)
    rows = collect(args.runs_root, args.run_prefix)
    if not rows:
        raise SystemExit(
            f"no points under {args.runs_root} matching {args.run_prefix!r}"
        )
    print(json.dumps(render(rows, args.out), indent=2))


if __name__ == "__main__":
    main()
