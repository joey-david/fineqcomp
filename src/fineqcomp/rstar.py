"""Locate R*: the smallest adapter file that still holds a target of the gain.

The rate axis is measured on exact file size. The retention axis defaults to
held-out bits saved rather than task accuracy, because a bits-saved measurement
is a forward pass over a few hundred examples and costs seconds, while a GSM8K
accuracy pass costs about nineteen minutes at the full test set. That gap is
what makes a dense ladder affordable, and a dense ladder is what R* needs: the
whole-bit rungs leave a 0.95-bit hole exactly where the crossing falls.

Accuracy is still measured, at a few anchor rates, to check that the two axes
order the codecs the same way. They are not interchangeable: on the Mistral runs
R*(0.90) came to 0.98/1.00/1.03 bits on bits-saved against 1.30/1.11/1.40 on
accuracy, so bits-saved reaches its ceiling sooner. Comparing arms is therefore
fine — the offset is shared — but an absolute R* must say which axis produced
it. Both of those figures interpolate across the 0.95-bit hole in the whole-bit
ladder, which is what the blended rungs exist to close.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _crossing(points: list[tuple[float, float]], target: float) -> float | None:
    """Rate at which retention first reaches `target`, linearly interpolated.

    Points are (rate, retention). Retention is not guaranteed monotone in rate,
    so this takes the *lowest* rate whose running maximum reaches the target;
    a dip at a higher rate cannot lower the answer.
    """
    ordered = sorted(points)
    best = float("-inf")
    previous: tuple[float, float] | None = None
    for rate, value in ordered:
        running = max(best, value)
        if running >= target:
            if previous is None:
                return rate
            rate0, value0 = previous
            if value <= value0:
                return rate
            span = value - value0
            return rate0 + (target - value0) / span * (rate - rate0)
        best = running
        previous = (rate, running)
    return None


def r_star(
    points: Iterable[dict[str, Any]],
    target: float = 0.90,
    rate_key: str = "effective_bits_per_value",
    value_key: str = "heldout_bits_saved_per_token",
    reference: float | None = None,
) -> dict[str, Any]:
    """R* against a fraction of the raw adapter's own gain on `value_key`.

    `reference` is the raw adapter's value; without it the largest measured
    value stands in, which is what the highest-rate point converges to.
    """
    usable = [
        (float(p[rate_key]), float(p[value_key]))
        for p in points
        if p.get(rate_key) is not None and p.get(value_key) is not None
    ]
    if not usable:
        return {"target": target, "r_star": None, "reason": "no measured points"}
    ceiling = reference if reference is not None else max(v for _, v in usable)
    if ceiling <= 0:
        return {"target": target, "r_star": None, "reason": "no gain to retain"}
    fractions = [(rate, value / ceiling) for rate, value in usable]
    crossing = _crossing(fractions, target)
    return {
        "target": target,
        "value_key": value_key,
        "reference": ceiling,
        "r_star": crossing,
        "bracketed": crossing is not None
        and min(r for r, _ in fractions) < crossing < max(r for r, _ in fractions),
        "points": sorted(fractions),
    }


def from_run(run_dir: Path, target: float = 0.90) -> dict[str, Any]:
    """R* for one finished run, read from its codec metric files."""
    points = []
    for path in sorted((run_dir / "codec_metrics").glob("*.json")):
        metric = json.loads(path.read_text())
        storage = metric.get("storage") or {}
        behaviour = metric.get("behavioral_write") or {}
        if storage.get("effective_bits_per_value") is None:
            continue
        points.append(
            {
                "codec": metric.get("codec_key"),
                "effective_bits_per_value": storage["effective_bits_per_value"],
                "heldout_bits_saved_per_token": behaviour.get(
                    "heldout_bits_saved_per_token"
                ),
            }
        )
    raw = run_dir / "metrics.json"
    raw_reference = None
    if raw.is_file():
        record = json.loads(raw.read_text())
        raw_reference = (record.get("raw_behavioral_write") or {}).get(
            "heldout_bits_saved_per_token"
        )
    measured = [
        float(point["heldout_bits_saved_per_token"])
        for point in points
        if point.get("heldout_bits_saved_per_token") is not None
    ]
    candidates = [*measured]
    if raw_reference is not None:
        candidates.append(float(raw_reference))
    reference = max(candidates) if candidates else None
    result = r_star(points, target=target, reference=reference)
    result["reference_mode"] = "best_decoded_utility"
    result["raw_reference"] = raw_reference
    # Retention above one came from measuring against the raw adapter while a
    # coded adapter beat it: the low-data arms overfit, and quantization strips
    # the overfit part. Referencing the frontier bounds retention at one, and
    # the gap that used to show as ">100%" is reported here instead, as the
    # size of the overfit rather than as a broken ratio.
    if reference and raw_reference is not None:
        result["raw_retained_gain"] = float(raw_reference) / reference
        result["overfit_bits_per_token"] = reference - float(raw_reference)
        result["overfit_codec"] = next(
            (
                point["codec"]
                for point in points
                if point.get("heldout_bits_saved_per_token") is not None
                and float(point["heldout_bits_saved_per_token"]) == reference
            ),
            None,
        )
    return result


def _fit(xs: list[float], ys: list[float]) -> dict[str, Any]:
    """Least-squares slope of R* on log2 of an information measure."""
    import math

    if len(xs) < 3:
        return {"n": len(xs)}
    logs = [math.log2(value) for value in xs if value > 0]
    if len(logs) != len(xs):
        return {"n": len(xs), "reason": "a measure was not positive"}
    mean_x = sum(logs) / len(logs)
    mean_y = sum(ys) / len(ys)
    sxx = sum((x - mean_x) ** 2 for x in logs)
    if sxx == 0:
        return {"n": len(xs), "reason": "the measure does not vary"}
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(logs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(logs, ys))
    total = sum((y - mean_y) ** 2 for y in ys)
    return {
        "n": len(xs),
        "slope_bits_per_doubling": slope,
        "intercept": intercept,
        "r_squared": 1 - residual / total if total > 0 else None,
    }


def collect(
    runs_root: Path,
    information: list[dict[str, Any]] | None = None,
    target: float = 0.90,
    keep: set[str] | None = None,
) -> list[dict[str, Any]]:
    """One row per finished run: R*, its overfit gap, and both information measures.

    `keep` restricts to the run IDs the current campaign expands to. The runs
    directory accumulates: it holds runs from earlier configs, and a run ID is
    a hash of the settings, so an unfiltered sweep silently averages arms that
    were trained under different row caps. On the Mistral campaign that meant
    34 directories for 15 live cells.
    """
    by_arm = {
        (str(row["dataset_key"]), int(row["seed"])): row
        for row in (information or [])
    }
    rows = []
    for config_path in sorted(Path(runs_root).glob("*/config.json")):
        config = json.loads(config_path.read_text())
        if keep is not None and str(config["run_id"]) not in keep:
            continue
        result = from_run(config_path.parent, target=target)
        measures = by_arm.get((str(config["dataset_key"]), int(config["seed"])), {})
        rows.append(
            {
                "run_id": config["run_id"],
                "study": config["study"],
                "dataset_key": config["dataset_key"],
                "seed": config["seed"],
                "epochs": config["training"]["epochs"],
                "r_star_bits_per_value": result.get("r_star"),
                "bracketed": result.get("bracketed"),
                "reference_bits_saved_per_token": result.get("reference"),
                "raw_bits_saved_per_token": result.get("raw_reference"),
                "raw_retained_gain": result.get("raw_retained_gain"),
                "overfit_bits_per_token": result.get("overfit_bits_per_token"),
                "overfit_codec": result.get("overfit_codec"),
                "distinct_rows": measures.get("distinct_rows"),
                "samples_seen": measures.get("samples_seen"),
                "lzma_bits": measures.get("lzma_bits"),
                "zlib_bits": measures.get("zlib_bits"),
                "base_stream_bits": measures.get("base_stream_bits"),
            }
        )
    return rows


# Ordinal ramp reused from results/adapter_bits_track_unique_data, where it
# passed the sequential palette checks.
MEASURE_COLOURS = ("#2a78d6", "#9ca3af")


def report(
    runs_root: Path,
    information_path: Path | None,
    out_dir: Path,
    target: float = 0.90,
    keep: set[str] | None = None,
) -> dict[str, Any]:
    """Write the R*-against-information table, fits, and figure."""
    import csv
    import statistics

    information = (
        json.loads(Path(information_path).read_text()) if information_path else None
    )
    rows = collect(Path(runs_root), information, target, keep)
    usable = [row for row in rows if row["r_star_bits_per_value"] is not None]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out_dir / "r_star_information.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    fits = {}
    for measure in ("lzma_bits", "base_stream_bits", "distinct_rows"):
        selected = [row for row in usable if row.get(measure)]
        fits[measure] = _fit(
            [float(row[measure]) for row in selected],
            [float(row["r_star_bits_per_value"]) for row in selected],
        )
    overfit = [
        row["raw_retained_gain"]
        for row in usable
        if row["raw_retained_gain"] is not None
    ]
    summary = {
        "runs": len(rows),
        "with_r_star": len(usable),
        "target": target,
        "fits": fits,
        # Below one means a coded adapter beat the raw one on held-out bits,
        # which is the overfit the old ">100% retention" was reporting.
        "raw_retained_gain": {
            "min": min(overfit) if overfit else None,
            "median": statistics.median(overfit) if overfit else None,
            "arms_where_coding_beat_raw": sum(1 for value in overfit if value < 1.0),
        },
    }
    write = out_dir / "r_star_information.json"
    write.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    try:
        _plot(usable, out_dir / "r_star_information.png", fits)
    except Exception as error:  # a missing backend must not lose the table
        summary["figure_error"] = str(error)
    return summary


def _plot(rows: list[dict[str, Any]], path: Path, fits: dict[str, Any]) -> None:
    import math
    import statistics

    import matplotlib.pyplot as plt

    measures = (
        ("lzma_bits", "compressed bits of the distinct rows", "duplication-aware"),
        ("base_stream_bits", "base-model bits over every sample", "duplication-blind"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for axis, (key, label, kind), colour in zip(axes, measures, MEASURE_COLOURS):
        selected = [row for row in rows if row.get(key)]
        grouped: dict[str, list[tuple[float, float]]] = {}
        for row in selected:
            grouped.setdefault(str(row["dataset_key"]), []).append(
                (float(row[key]), float(row["r_star_bits_per_value"]))
            )
        for arm in sorted(grouped, key=lambda name: statistics.fmean(
            x for x, _ in grouped[name]
        )):
            xs = [x for x, _ in grouped[arm]]
            ys = [y for _, y in grouped[arm]]
            axis.scatter(xs, ys, s=22, color=colour, alpha=0.55, linewidth=0)
            axis.scatter(
                [statistics.fmean(xs)], [statistics.fmean(ys)],
                s=70, color=colour, edgecolor="white", linewidth=1.2, zorder=3,
            )
        fit = fits.get(key, {})
        # A degenerate fit still needs its panel labelled: a control that does
        # not vary is the expected outcome, not a missing measurement.
        axis.set_title(kind)
        if fit.get("slope_bits_per_doubling") is not None and selected:
            span = [
                min(float(row[key]) for row in selected),
                max(float(row[key]) for row in selected),
            ]
            axis.plot(
                span,
                [
                    fit["intercept"] + fit["slope_bits_per_doubling"] * math.log2(x)
                    for x in span
                ],
                color=colour, linewidth=1.2, linestyle="--",
            )
            axis.set_title(
                f"{kind}\n{fit['slope_bits_per_doubling']:+.3f} bits per doubling,"
                f" R² = {fit['r_squared']:.2f}"
            )
        axis.set_xscale("log", base=2)
        axis.set_xlabel(label)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("R*(0.90), adapter bits per value")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)
