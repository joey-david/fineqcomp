"""Fixed campaign tables, confidence intervals, and paper figures."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_correct(path: Path, evaluator: str | None = None) -> list[int]:
    if not path.is_file():
        return []
    values = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if evaluator is not None and row.get("evaluator") != evaluator:
                continue
            if "correct" in row or "passed" in row:
                values.append(int(bool(row.get("correct", row.get("passed")))))
    return values


def _binomial_ci(values: list[int], seed: int = 20260731) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = rng.binomial(len(values), np.mean(values), size=2_000) / len(values)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _paired_stats(
    baseline: list[int], adapted: list[int], seed: int = 20260731
) -> dict[str, float | int | None]:
    if not baseline or len(baseline) != len(adapted):
        return {
            "paired_gain": None,
            "paired_ci_low": None,
            "paired_ci_high": None,
            "mcnemar_p": None,
        }
    delta = np.asarray(adapted, dtype=float) - np.asarray(baseline, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(2_000, len(delta)))
    boot = delta[indices].mean(axis=1)
    wrong_to_right = sum(
        before == 0 and after == 1
        for before, after in zip(baseline, adapted, strict=True)
    )
    right_to_wrong = sum(
        before == 1 and after == 0
        for before, after in zip(baseline, adapted, strict=True)
    )
    discordant = wrong_to_right + right_to_wrong
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(wrong_to_right, right_to_wrong) + 1)
        ) / 2**discordant
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0
    return {
        "paired_gain": float(delta.mean()),
        "paired_ci_low": float(np.quantile(boot, 0.025)),
        "paired_ci_high": float(np.quantile(boot, 0.975)),
        "mcnemar_p": p_value,
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
    }


def _primary_metric(task: dict[str, Any]) -> str | None:
    for metric in ("exact_match", "accuracy", "pass_at_1", "rouge_l"):
        if task.get(metric) is not None:
            return metric
    return None


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
            # Rungs scored on held-out bits alone have no task row to report.
            if not codec.get("task"):
                continue
            task = codec["task"]
            primary_metric = _primary_metric(task)
            primary_evaluator = task.get("primary_evaluator")
            bits = codec.get("bits")
            codec_key = str(
                codec.get("codec_key", f"b{bits}" if bits is not None else "unknown")
            )
            predictions = (
                metrics_path.parent / "predictions" / f"task_{codec_key}.jsonl"
            )
            if not predictions.is_file() and bits is not None:
                predictions = (
                    metrics_path.parent / "predictions" / f"task_b{bits}.jsonl"
                )
            correct = _read_correct(predictions, primary_evaluator)
            ci_low, ci_high = _binomial_ci(correct)
            baseline_key = (
                f"{run['model_key']}__{run['backbone']}__"
                f"{run.get('dataset_key')}-seed{run['seed']}"
            )
            if baseline_key not in baselines:
                legacy_key = (
                    f"{run['model_key']}__{run['backbone']}__"
                    f"{run.get('dataset_key')}"
                )
                if legacy_key in baselines:
                    baseline_key = legacy_key
            baseline = baselines.get(baseline_key, {})
            baseline_predictions = (
                runs_root / "baselines" / baseline_key / "predictions.jsonl"
            )
            paired = _paired_stats(
                _read_correct(baseline_predictions, primary_evaluator), correct
            )
            task_score = task.get(primary_metric) if primary_metric else None
            baseline_test_score = (
                baseline.get(primary_metric) if primary_metric else None
            )
            task_evaluations = task.get("evaluations", {})
            baseline_evaluations = baseline.get("evaluations", {})
            raw_task = run.get("raw_task", {})
            raw_evaluations = raw_task.get("evaluations", {})
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
                "dataset_key": run.get("dataset_key"),
                "codec_key": codec_key,
                "codec_method": codec.get("codec_method", "uniform"),
                "quant_bits": bits,
                "high_bits": codec.get("high_bits"),
                "low_bits": codec.get("low_bits"),
                "variance_ratio": codec.get("variance_ratio"),
                "quantizer": codec.get("quantizer"),
                "relative_rmse": codec["storage"].get("relative_rmse"),
                "file_bits": codec["storage"]["file_bits"],
                "raw_payload_bits": codec["storage"]["raw_payload_bits"],
                "header_bits": codec["storage"].get("header_bits"),
                "value_bits": codec["storage"].get("value_bits"),
                "scale_bits": codec["storage"].get("scale_bits"),
                "padding_bits": codec["storage"].get("padding_bits"),
                "effective_bits_per_value": codec["storage"].get(
                    "effective_bits_per_value", bits
                ),
                "accuracy": task.get("accuracy"),
                "distortion": task.get("distortion"),
                "label_nll": task.get("label_nll"),
                "exact_match": task.get("exact_match"),
                "pass_at_1": task.get("pass_at_1"),
                "rouge_l": task.get("rouge_l"),
                "math_exact_match": task_evaluations.get("math", {}).get(
                    "exact_match"
                ),
                "baseline_math_exact_match": baseline_evaluations.get(
                    "math", {}
                ).get("exact_match"),
                "raw_math_exact_match": raw_evaluations.get("math", {}).get(
                    "exact_match"
                ),
                "heldout_nll": task.get("heldout_nll"),
                "primary_metric": primary_metric,
                "primary_evaluator": primary_evaluator,
                "task_score": task_score,
                "baseline_test_score": baseline_test_score,
                "test_gain": (
                    float(task_score) - float(baseline_test_score)
                    if task_score is not None and baseline_test_score is not None
                    else None
                ),
                "math_gain": (
                    float(task_evaluations["math"]["exact_match"])
                    - float(baseline_evaluations["math"]["exact_match"])
                    if task_evaluations.get("math", {}).get("exact_match")
                    is not None
                    and baseline_evaluations.get("math", {}).get("exact_match")
                    is not None
                    else None
                ),
                "ci_low": ci_low,
                "ci_high": ci_high,
                **paired,
                "task_entropy_bits": run.get("data", {}).get("task_entropy_bits"),
                "train_zlib_bits": run.get("data", {}).get("train_zlib_bits"),
                "training_seconds": run.get("training", {}).get("elapsed_seconds"),
                "peak_memory_bytes": run.get("training", {}).get("peak_memory_bytes"),
                "baseline_score": run.get("baseline_screening", {}).get(
                    "baseline_score"
                ),
                "baseline_headroom": run.get("baseline_screening", {}).get("headroom"),
                "baseline_status": run.get("baseline_screening", {}).get("status"),
                "raw_calibration_gain": run.get("learning_gate", {}).get("gain"),
                "learning_status": run.get("learning_gate", {}).get("status"),
                "retained_gain": codec.get("retained_gain", {}).get(
                    "retained_gain"
                ),
                "train_bits_saved": codec.get("behavioral_write", {}).get(
                    "train_bits_saved"
                ),
                "heldout_bits_saved": codec.get("behavioral_write", {}).get(
                    "heldout_bits_saved"
                ),
                "excess_train_bits_per_token": codec.get(
                    "behavioral_write", {}
                ).get("excess_train_bits_per_token"),
            }
            rows.append(row)
    return rows, baselines


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _group(
    rows: list[dict[str, Any]], *keys: str
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    return groups


def _save_natural(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["kind"] == "natural"]
    if not selected:
        return
    datasets = sorted({str(row["dataset_key"]) for row in selected})
    columns = min(2, len(datasets))
    rows_count = math.ceil(len(datasets) / columns)
    fig, axes = plt.subplots(
        rows_count, columns, figsize=(6 * columns, 4.6 * rows_count), squeeze=False
    )
    for axis, dataset in zip(axes.flat, datasets, strict=False):
        subset = [row for row in selected if row["dataset_key"] == dataset]
        for key, group in _group(subset, "model", "adapter").items():
            axis.scatter(
                [row["file_bits"] / 8 / 2**20 for row in group],
                [row["test_gain"] for row in group],
                s=24,
                alpha=0.65,
                label=f"{str(key[0]).split('/')[-1]} / {key[1]}",
            )
        axis.set_xscale("log")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(dataset.replace("_", " ").upper())
        axis.set_xlabel("actual adapter file size (MiB)")
        axis.set_ylabel("test score gain over base model")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6)
    for axis in list(axes.flat)[len(datasets) :]:
        axis.remove()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_quantization_retention(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["kind"] == "natural"]
    if not selected:
        return
    datasets = sorted({str(row["dataset_key"]) for row in selected})
    columns = min(2, len(datasets))
    rows_count = math.ceil(len(datasets) / columns)
    fig, axes = plt.subplots(
        rows_count, columns, figsize=(6 * columns, 4.6 * rows_count), squeeze=False
    )
    for axis, dataset in zip(axes.flat, datasets, strict=False):
        subset = [row for row in selected if row["dataset_key"] == dataset]
        for key, group in _group(subset, "model", "adapter").items():
            retained: dict[str, list[float]] = defaultdict(list)
            rates: dict[str, list[float]] = defaultdict(list)
            for row in group:
                if row["retained_gain"] is None:
                    continue
                retained[str(row["codec_key"])].append(float(row["retained_gain"]))
                rates[str(row["codec_key"])].append(
                    float(row["effective_bits_per_value"])
                )
            if not rates:
                continue
            codecs = sorted(rates, key=lambda name: np.median(rates[name]))
            x = [float(np.median(rates[name])) for name in codecs]
            y = [float(np.median(retained[name])) for name in codecs]
            low = [min(retained[name]) for name in codecs]
            high = [max(retained[name]) for name in codecs]
            axis.errorbar(
                x,
                y,
                yerr=[
                    np.asarray(y) - np.asarray(low),
                    np.asarray(high) - np.asarray(y),
                ],
                marker="o",
                capsize=3,
                label=f"{str(key[0]).split('/')[-1]} / {key[1]}",
            )
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.set_title(dataset.replace("_", " ").upper())
        axis.set_xlabel("effective serialized bits per adapter value")
        axis.set_ylabel("fraction of raw adapter gain retained")
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, fontsize=6)
    for axis in list(axes.flat)[len(datasets) :]:
        axis.remove()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_behavioral_write(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if row["kind"] == "natural" and row["heldout_bits_saved"] is not None
    ]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (dataset,), group in _group(selected, "dataset_key").items():
        ax.scatter(
            [row["file_bits"] for row in group],
            [row["heldout_bits_saved"] for row in group],
            label=str(dataset),
            alpha=0.65,
        )
    ax.set_xscale("log")
    ax.set_xlabel("serialized adapter bits")
    ax.set_ylabel("held-out code bits saved over the base model")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_placement(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["study"] == "placement_control"]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (adapter,), group in _group(selected, "adapter").items():
        ax.scatter(
            [row["file_bits"] for row in group],
            [row["test_gain"] for row in group],
            label=str(adapter),
            alpha=0.7,
        )
    ax.set_xscale("log")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("serialized adapter bits")
    ax.set_ylabel("test score gain over the base model")
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
    _save_natural(rows, output / "natural_pareto.png")
    _save_quantization_retention(rows, output / "quantization_retention.png")
    _save_behavioral_write(rows, output / "behavioral_write.png")
    _save_placement(rows, output / "placement_control.png")
    screening = [
        {
            "model": baseline.get("model"),
            "model_key": baseline.get("model_key"),
            "backbone": baseline.get("backbone"),
            "dataset_key": baseline.get("dataset_key"),
            "seed": baseline.get("seed"),
            **baseline.get("screening", {}),
        }
        for baseline in baselines.values()
        if baseline.get("kind") == "natural"
    ]
    _write_csv(output / "baseline_screening.csv", screening)
    learning_gates = []
    for path in sorted(Path(root).glob("*/learning_gate.json")):
        config = _read_json(path.parent / "config.json")
        learning_gates.append(
            {
                "run_id": config["run_id"],
                "model": config["model"]["name"],
                "dataset_key": config.get("dataset_key"),
                "adapter": config["adapter"]["key"],
                "seed": config["seed"],
                **_read_json(path),
            }
        )
    _write_csv(output / "learning_gates.csv", learning_gates)
    status_counts: dict[str, int] = defaultdict(int)
    for path in Path(root).glob("*/status.json"):
        status_counts[str(_read_json(path).get("state", "unknown"))] += 1
    complete_runs = len({row["run_id"] for row in rows})
    summary = {
        "complete_runs": complete_runs,
        "codec_rows": len(rows),
        "baselines": len(baselines),
        "status_counts": dict(sorted(status_counts.items())),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
