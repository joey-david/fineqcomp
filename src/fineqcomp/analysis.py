"""Fixed campaign tables, confidence intervals, and paper figures."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def rate_bound(source_symbols: float, distortion: float, classes: int = 16) -> float:
    """M-ary Hamming rate-distortion lower bound in total bits."""
    distortion = min(max(distortion, 0.0), (classes - 1) / classes)
    if distortion in {0.0, 1.0}:
        binary_entropy = 0.0
    else:
        binary_entropy = -distortion * math.log2(distortion) - (
            1 - distortion
        ) * math.log2(1 - distortion)
    per_symbol = (
        math.log2(classes) - binary_entropy - distortion * math.log2(classes - 1)
    )
    return max(per_symbol, 0.0) * source_symbols


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_correct(path: Path) -> list[int]:
    if not path.is_file():
        return []
    values = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            values.append(int(bool(row.get("correct", row.get("passed", False)))))
    return values


def _binomial_ci(values: list[int], seed: int = 20260731) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = rng.binomial(len(values), np.mean(values), size=2_000) / len(values)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def collect_rows(
    root: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    runs_root = Path(root)
    baselines = {
        path.parent.name: _read_json(path)
        for path in (runs_root / "baselines").glob("*/metrics.json")
    }
    rows = []
    for metrics_path in sorted(runs_root.glob("*/metrics.json")):
        if metrics_path.parent.name == "baselines":
            continue
        run = _read_json(metrics_path)
        for codec in run.get("codecs", []):
            task = codec["task"]
            bits = int(codec["bits"])
            predictions = metrics_path.parent / "predictions" / f"task_b{bits}.jsonl"
            correct = _read_correct(predictions)
            ci_low, ci_high = _binomial_ci(correct)
            row = {
                "run_id": run["run_id"],
                "study": run["study"],
                "kind": run["kind"],
                "model": run["model"],
                "model_key": run["model_key"],
                "backbone": run["backbone"],
                "adapter": run["adapter"],
                "adapter_method": run["adapter_method"],
                "seed": run["seed"],
                "family_count": run.get("family_count"),
                "dataset_key": run.get("dataset_key"),
                "quant_bits": bits,
                "clip_percentile": codec["selected_clip_percentile"],
                "file_bits": codec["storage"]["file_bits"],
                "raw_payload_bits": codec["storage"]["raw_payload_bits"],
                "accuracy": task.get("accuracy"),
                "distortion": task.get("distortion"),
                "label_nll": task.get("label_nll"),
                "exact_match": task.get("exact_match"),
                "pass_at_1": task.get("pass_at_1"),
                "heldout_nll": task.get("heldout_nll"),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ifeval_prompt_strict": codec.get("ifeval", {}).get(
                    "prompt_level_strict_accuracy"
                ),
                "ifeval_prompt_loose": codec.get("ifeval", {}).get(
                    "prompt_level_loose_accuracy"
                ),
                "task_entropy_bits": run.get("data", {}).get("task_entropy_bits"),
                "train_zlib_bits": run.get("data", {}).get("train_zlib_bits"),
                "training_seconds": run.get("training", {}).get("elapsed_seconds"),
                "peak_memory_bytes": run.get("training", {}).get("peak_memory_bytes"),
            }
            if row["kind"] == "synthetic" and row["distortion"] is not None:
                source_symbols = int(row["family_count"]) * 16
                bound = rate_bound(source_symbols, float(row["distortion"]))
                row["theory_min_bits"] = bound
                row["rate_over_bound"] = (
                    float(row["file_bits"]) / bound if bound > 0 else math.inf
                )
                row["bits_per_source_symbol"] = float(row["file_bits"]) / source_symbols
            rows.append(row)
    return rows, baselines


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _save_rate_distortion(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["kind"] == "synthetic"]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for adapter in sorted({row["adapter"] for row in selected}):
        group = [row for row in selected if row["adapter"] == adapter]
        ax.scatter(
            [row["bits_per_source_symbol"] for row in group],
            [row["distortion"] for row in group],
            s=18,
            alpha=0.65,
            label=adapter,
        )
    distortions = np.linspace(0, 15 / 16, 300)
    theory = [rate_bound(1, float(value)) for value in distortions]
    ax.plot(theory, distortions, color="black", linewidth=2, label="theory lower bound")
    ax.set_xscale("log")
    ax.set_xlabel("coded update bits per hidden source symbol")
    ax.set_ylabel("label error (distortion)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_bits_targets(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    main = [
        row
        for row in rows
        if row["kind"] == "synthetic"
        and row["model"] == "Qwen/Qwen3-8B-Base"
        and row["backbone"] == "nf4"
    ]
    targets = [0.70, 0.90, 0.99]
    summary = []
    for (seed, family_count), group in _group(main, "seed", "family_count").items():
        entropy = 64 * int(family_count)
        for target in targets:
            reached = [row for row in group if float(row["accuracy"]) >= target]
            summary.append(
                {
                    "seed": seed,
                    "family_count": family_count,
                    "task_entropy_bits": entropy,
                    "target_accuracy": target,
                    "minimum_file_bits": min(
                        (int(row["file_bits"]) for row in reached), default=None
                    ),
                    "censored": not bool(reached),
                    "censor_at_bits": max(int(row["file_bits"]) for row in group),
                }
            )
    if not summary:
        return summary
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    markers = {0.70: "o", 0.90: "s", 0.99: "^"}
    for target in targets:
        group = [row for row in summary if row["target_accuracy"] == target]
        observed = [row for row in group if not row["censored"]]
        ax.scatter(
            [row["task_entropy_bits"] for row in observed],
            [row["minimum_file_bits"] for row in observed],
            marker=markers[target],
            alpha=0.3,
        )
        medians = []
        for (entropy,), cell in _group(group, "task_entropy_bits").items():
            cell_observed = [row for row in cell if not row["censored"]]
            if len(cell_observed) == len(cell):
                values = np.asarray(
                    [row["minimum_file_bits"] for row in cell_observed], dtype=float
                )
                medians.append(
                    (
                        entropy,
                        float(np.median(values)),
                        float(np.min(values)),
                        float(np.max(values)),
                    )
                )
            else:
                for row in cell:
                    if row["censored"]:
                        ax.scatter(
                            row["task_entropy_bits"],
                            row["censor_at_bits"],
                            marker="x",
                            color="black",
                            s=22,
                        )
        if medians:
            medians.sort()
            x = np.asarray([row[0] for row in medians], dtype=float)
            y = np.asarray([row[1] for row in medians], dtype=float)
            low = y - np.asarray([row[2] for row in medians], dtype=float)
            high = np.asarray([row[3] for row in medians], dtype=float) - y
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([low, high]),
                marker=markers[target],
                linewidth=1.5,
                capsize=3,
                label=f"accuracy {target:.2f}",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("known task information (bits)")
    ax.set_ylabel("minimum measured adapter bits")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return summary


def _group(
    rows: list[dict[str, Any]], *keys: str
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    return groups


def _save_allocation(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if row["kind"] == "synthetic" and row["model"] == "Qwen/Qwen3-8B-Base"
    ]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for adapter, group in _group(selected, "adapter").items():
        label = str(adapter[0])
        ax.scatter(
            [row["file_bits"] for row in group],
            [row["accuracy"] for row in group],
            s=[16 + int(row["quant_bits"]) * 2 for row in group],
            alpha=0.7,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("actual coded adapter bits")
    ax.set_ylabel("synthetic accuracy")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_model_checks(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["kind"] == "synthetic"]
    if not selected:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for axis, family_count in zip(axes, (32, 1024), strict=True):
        subset = [row for row in selected if row["family_count"] == family_count]
        for key, group in _group(subset, "model", "backbone").items():
            axis.scatter(
                [row["file_bits"] for row in group],
                [row["accuracy"] for row in group],
                marker=".",
                alpha=0.65,
                label=f"{str(key[0]).split('/')[-1]} / {key[1]}",
            )
        axis.set_xscale("log")
        axis.set_title(f"K={family_count}")
        axis.set_xlabel("coded adapter bits")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("accuracy")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_natural(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["kind"] == "natural"]
    if not selected:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, dataset, metric in zip(
        axes, ("gsm8k", "mbpp"), ("exact_match", "pass_at_1"), strict=True
    ):
        subset = [row for row in selected if row["dataset_key"] == dataset]
        for key, group in _group(subset, "model", "adapter").items():
            axis.scatter(
                [row["file_bits"] for row in group],
                [row[metric] for row in group],
                s=24,
                alpha=0.7,
                label=f"{str(key[0]).split('/')[-1]} / {key[1]}",
            )
        axis.set_xscale("log")
        axis.set_title(dataset.upper())
        axis.set_xlabel("coded adapter bits")
        axis.set_ylabel(metric.replace("_", " "))
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_retention(
    rows: list[dict[str, Any]], baselines: dict[str, dict[str, Any]], path: Path
) -> None:
    import matplotlib.pyplot as plt

    points = []
    for row in rows:
        if row["kind"] != "natural" or row["ifeval_prompt_strict"] is None:
            continue
        key = f"{row['model_key']}__{row['backbone']}__{row['dataset_key']}"
        baseline = baselines.get(key)
        if not baseline:
            continue
        metric = "exact_match" if row["dataset_key"] == "gsm8k" else "pass_at_1"
        baseline_ifeval = baseline.get("ifeval", {}).get("prompt_level_strict_accuracy")
        if (
            row[metric] is None
            or baseline.get(metric) is None
            or baseline_ifeval is None
        ):
            continue
        points.append(
            {
                **row,
                "task_gain": float(row[metric]) - float(baseline[metric]),
                "ifeval_drop": float(baseline_ifeval)
                - float(row["ifeval_prompt_strict"]),
            }
        )
    if not points:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for dataset, group in _group(points, "dataset_key").items():
        ax.scatter(
            [row["task_gain"] for row in group],
            [row["ifeval_drop"] for row in group],
            label=str(dataset[0]),
            alpha=0.7,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("task score gain over no-adapter baseline")
    ax.set_ylabel("IFEval strict prompt accuracy drop")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def analyze(root: str | Path = "runs", out: str | Path = "reports") -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    rows, baselines = collect_rows(root)
    _write_csv(output / "summary.csv", rows)
    efficiency = [
        {
            key: row.get(key)
            for key in (
                "run_id",
                "model",
                "backbone",
                "adapter",
                "seed",
                "family_count",
                "quant_bits",
                "accuracy",
                "file_bits",
                "theory_min_bits",
                "rate_over_bound",
            )
        }
        for row in rows
        if row["kind"] == "synthetic"
    ]
    _write_csv(output / "efficiency.csv", efficiency)
    _save_rate_distortion(rows, output / "rate_distortion.png")
    target_rows = _save_bits_targets(rows, output / "bits_vs_information.png")
    _write_csv(output / "accuracy_targets.csv", target_rows)
    _save_allocation(rows, output / "rate_allocation.png")
    _save_model_checks(rows, output / "model_checks.png")
    _save_natural(rows, output / "natural_pareto.png")
    _save_retention(rows, baselines, output / "ifeval_retention.png")
    complete_runs = len({row["run_id"] for row in rows})
    summary = {
        "complete_runs": complete_runs,
        "codec_rows": len(rows),
        "baselines": len(baselines),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
