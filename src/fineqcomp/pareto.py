"""Fast codec-only Pareto sweep over an already-trained natural-task run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fineqcomp.adapters import apply_adapter_tensors
from fineqcomp.codec import pack_unsigned
from fineqcomp.config import RunSpec, load_campaign
from fineqcomp.evaluation import write_predictions
from fineqcomp.modeling import ModelSession
from fineqcomp.runner import RunEngine


MAGIC = b"FQPM1\n"
DEFAULT_BITS = (1, 2, 3, 4)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _primary_metric(task: dict[str, Any]) -> str:
    for key in ("pass_at_1", "exact_match", "accuracy", "rouge_l"):
        if task.get(key) is not None:
            return key
    raise ValueError("task metrics do not contain a supported primary metric")


def _existing_controls(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run_dir / "codec_metrics").glob("*.json")):
        metric = _read_json(path)
        retained = metric.get("retained_gain") or {}
        storage = metric.get("storage") or {}
        if retained.get("codec_score") is None or storage.get("effective_bits_per_value") is None:
            continue
        rows.append(
            {
                "codec": str(metric.get("codec_key", path.stem)),
                "family": str(metric.get("codec_method", "existing")),
                "nominal_bits": metric.get("bits"),
                "effective_bits_per_value": float(storage["effective_bits_per_value"]),
                "file_bits": int(storage["file_bits"]),
                "metric": str(retained["metric"]),
                "baseline_score": float(retained["baseline_score"]),
                "raw_adapter_score": float(retained["raw_adapter_score"]),
                "task_score": float(retained["codec_score"]),
                "retained_gain": float(retained["retained_gain"]),
                "source": "existing",
            }
        )
    return rows


def _reference_scores(controls: list[dict[str, Any]]) -> tuple[str, float, float]:
    if not controls:
        raise RuntimeError(
            "fast Pareto sweep needs at least one existing codec metric so it can "
            "reuse the already-measured baseline and raw-adapter scores"
        )
    first = controls[0]
    metric = str(first["metric"])
    baseline = float(first["baseline_score"])
    raw = float(first["raw_adapter_score"])
    for row in controls[1:]:
        if str(row["metric"]) != metric:
            raise ValueError("existing codec metrics disagree on the primary metric")
        if not math.isclose(float(row["baseline_score"]), baseline, abs_tol=1e-12):
            raise ValueError("existing codec metrics disagree on the baseline score")
        if not math.isclose(float(row["raw_adapter_score"]), raw, abs_tol=1e-12):
            raise ValueError("existing codec metrics disagree on the raw-adapter score")
    if raw <= baseline:
        raise ValueError("raw adapter does not improve over baseline")
    return metric, baseline, raw


def _midrise_tensor(
    tensor: torch.Tensor, bits: int, iterations: int = 8
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Zero-free symmetric quantization with an MSE-refit row scale.

    Positive magnitudes use odd reconstruction levels 1, 3, ..., 2**bits-1.
    At one bit this reduces exactly to sign(w) * mean(abs(w)) per row.
    """
    if bits not in {1, 2, 3, 4, 8}:
        raise ValueError("midrise bits must be one of 1, 2, 3, 4, 8")
    matrix = tensor.detach().cpu().float().reshape(tensor.shape[0], -1)
    absolute = matrix.abs()
    positive_levels = 1 << (bits - 1)
    max_level = 2 * positive_levels - 1
    row_max = absolute.amax(dim=1)
    scale = row_max / max_level

    for _ in range(iterations):
        safe_scale = scale.clamp_min(1e-12)
        index = torch.round((absolute / safe_scale[:, None] - 1.0) / 2.0)
        index = index.clamp(0, positive_levels - 1)
        level = 2.0 * index + 1.0
        scale = (absolute * level).sum(dim=1) / level.square().sum(dim=1)
        scale = torch.where(row_max > 0, scale, torch.zeros_like(scale))

    safe_scale = scale.clamp_min(1e-12)
    index = torch.round((absolute / safe_scale[:, None] - 1.0) / 2.0)
    index = index.clamp(0, positive_levels - 1)
    level = 2.0 * index + 1.0
    scale = (absolute * level).sum(dim=1) / level.square().sum(dim=1)
    scale = torch.where(row_max > 0, scale, torch.zeros_like(scale))

    scale16 = scale.to(torch.float16).cpu().numpy()
    decoded_scale = torch.from_numpy(scale16.astype(np.float32))
    signed_level = torch.where(matrix >= 0, level, -level)
    reconstructed = (signed_level * decoded_scale[:, None]).reshape(tensor.shape)

    code = torch.where(
        matrix >= 0,
        index.to(torch.int64) + positive_levels,
        index.to(torch.int64),
    )
    return (
        scale16,
        code.numpy().astype(np.uint16).reshape(-1),
        reconstructed,
    )


def _encode_midrise(
    tensors: dict[str, torch.Tensor], path: Path, bits: int
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = bytearray()
    entries = []
    decoded: dict[str, torch.Tensor] = {}
    squared_error = 0.0
    squared_norm = 0.0
    value_bits = 0
    scale_bits = 0
    padding_bits = 0
    total_values = 0

    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().float().contiguous()
        scales, codes, reconstructed = _midrise_tensor(tensor, bits)
        scales_bytes = scales.tobytes()
        data = pack_unsigned(codes, bits)
        scale_offset = len(payload)
        payload.extend(scales_bytes)
        data_offset = len(payload)
        payload.extend(data)
        count = tensor.numel()
        packed_bits = len(data) * 8
        entries.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "count": count,
                "scale_offset": scale_offset,
                "scale_nbytes": len(scales_bytes),
                "data_offset": data_offset,
                "data_nbytes": len(data),
            }
        )
        decoded[name] = reconstructed
        squared_error += float((tensor - reconstructed).square().sum().item())
        squared_norm += float(tensor.square().sum().item())
        total_values += count
        value_bits += count * bits
        scale_bits += len(scales_bytes) * 8
        padding_bits += packed_bits - count * bits

    header = {
        "version": 1,
        "quantizer": "row_midrise_mse_v1",
        "bits": bits,
        "compression": "zlib-9",
        "tensors": entries,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    compressed = zlib.compress(bytes(payload), level=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<I", len(header_bytes)))
        stream.write(header_bytes)
        stream.write(compressed)
    temporary.replace(path)

    file_bits = path.stat().st_size * 8
    return (
        {
            "file_bits": file_bits,
            "header_bits": (len(MAGIC) + 4 + len(header_bytes)) * 8,
            "compressed_payload_bits": len(compressed) * 8,
            "raw_payload_bits": len(payload) * 8,
            "value_bits": value_bits,
            "scale_bits": scale_bits,
            "padding_bits": padding_bits,
            "tensor_values": total_values,
            "effective_bits_per_value": file_bits / max(total_values, 1),
            "relative_rmse": math.sqrt(squared_error / max(squared_norm, 1e-30)),
        },
        decoded,
    )


def _capped(value: float) -> float:
    return min(1.0, max(0.0, value))


def _frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["effective_bits_per_value"]),
            -_capped(float(row["retained_gain"])),
        ),
    )
    frontier = []
    best = -math.inf
    for row in ordered:
        score = _capped(float(row["retained_gain"]))
        if score > best + 1e-12:
            frontier.append(row)
            best = score
    return frontier


def _save_plot(
    rows: list[dict[str, Any]], run: RunSpec, metric: str, baseline: float, raw: float, path: Path
) -> None:
    import matplotlib.pyplot as plt

    midrise = [row for row in rows if row["source"] == "midrise"]
    reused_binary = any(row.get("reused_from") == "binary" for row in midrise)
    controls = [
        row
        for row in rows
        if row["source"] == "existing"
        and not (reused_binary and row["codec"] == "binary")
    ]
    frontier = _frontier(rows)

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    if controls:
        ax.scatter(
            [row["effective_bits_per_value"] for row in controls],
            [100 * _capped(float(row["retained_gain"])) for row in controls],
            marker="x",
            s=42,
            alpha=0.55,
            label="existing codecs",
        )
    if midrise:
        ax.scatter(
            [row["effective_bits_per_value"] for row in midrise],
            [100 * _capped(float(row["retained_gain"])) for row in midrise],
            s=72,
            label="zero-free mid-rise",
        )
        for row in midrise:
            ax.annotate(
                f"{row['nominal_bits']}b",
                (row["effective_bits_per_value"], 100 * _capped(float(row["retained_gain"]))),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    if frontier:
        ax.plot(
            [row["effective_bits_per_value"] for row in frontier],
            [100 * _capped(float(row["retained_gain"])) for row in frontier],
            linewidth=2,
            alpha=0.8,
            label="Pareto frontier",
        )

    ax.axhline(100, linestyle="--", linewidth=1, alpha=0.55)
    ax.axhline(0, linewidth=1, alpha=0.25)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.set_xlim(1, 17)
    ax.set_ylim(-4, 104)
    ax.set_xlabel("Effective transmitted bits per LoRA value")
    ax.set_ylabel("Retained task gain (%) — capped at raw adapter")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("From Reasoning Trajectories to Information Budgets", fontsize=15)
    ax.set_title("Pareto frontier of learned behavior vs. transmitted adapter bits", fontsize=10)
    fig.text(
        0.5,
        0.015,
        f"{run.model.key} · {run.dataset_key} · seed {run.seed} · {metric}: "
        f"baseline {baseline:.3f} → raw {raw:.3f}",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "codec",
        "family",
        "nominal_bits",
        "effective_bits_per_value",
        "file_bits",
        "metric",
        "baseline_score",
        "raw_adapter_score",
        "task_score",
        "retained_gain",
        "retained_gain_capped",
        "relative_rmse",
        "source",
        "reused_from",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(
    run_dir: Path,
    config_path: Path,
    prepared_root: Path,
    bits: tuple[int, ...],
    out_dir: Path,
    force: bool,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if not (run_dir / "raw_channel.pt").is_file():
        raise FileNotFoundError(f"missing trained adapter: {run_dir / 'raw_channel.pt'}")
    if not (run_dir / "config.json").is_file():
        raise FileNotFoundError(f"missing run config: {run_dir / 'config.json'}")

    run = RunSpec.from_dict(_read_json(run_dir / "config.json"))
    if run.kind != "natural":
        raise ValueError("fast Pareto sweep currently targets natural-task runs")
    campaign = load_campaign(config_path)
    controls = _existing_controls(run_dir)
    metric, baseline, raw_score = _reference_scores(controls)
    out_dir.mkdir(parents=True, exist_ok=True)

    binary = next((row for row in controls if row["codec"] == "binary"), None)
    midrise_rows: list[dict[str, Any]] = []
    pending = []
    for bit_width in bits:
        metric_path = out_dir / f"midrise{bit_width}.json"
        if metric_path.is_file() and not force:
            midrise_rows.append(_read_json(metric_path))
        elif bit_width == 1 and binary is not None and not force:
            row = {
                **binary,
                "codec": "midrise1",
                "family": "midrise",
                "nominal_bits": 1,
                "source": "midrise",
                "reused_from": "binary",
                "relative_rmse": None,
            }
            row["retained_gain_capped"] = _capped(float(row["retained_gain"]))
            metric_path.write_text(json.dumps(row, indent=2, sort_keys=True))
            midrise_rows.append(row)
        else:
            pending.append(bit_width)

    if pending:
        engine = RunEngine(campaign, prepared_root, run_dir.parent)
        data, _ = engine._load_data(run)
        session = ModelSession.load(run.model)
        try:
            session.attach(run.adapter, run.seed)
            raw_tensors = torch.load(
                run_dir / "raw_channel.pt", map_location="cpu", weights_only=True
            )
            apply_adapter_tensors(session.model, raw_tensors)
            for index, bit_width in enumerate(pending, start=1):
                print(
                    f"[{index}/{len(pending)}] midrise {bit_width}-bit: encode + task eval",
                    flush=True,
                )
                storage, decoded = _encode_midrise(
                    raw_tensors,
                    out_dir / f"adapter_midrise{bit_width}.fqpm",
                    bit_width,
                )
                apply_adapter_tensors(session.model, decoded)
                task, predictions = engine._evaluate_natural(
                    session, run, data["test"]
                )
                if task.get(metric) is None:
                    raise ValueError(f"task evaluation did not produce metric {metric!r}")
                task_score = float(task[metric])
                retained_gain = (task_score - baseline) / (raw_score - baseline)
                row = {
                    "codec": f"midrise{bit_width}",
                    "family": "midrise",
                    "nominal_bits": bit_width,
                    "effective_bits_per_value": float(
                        storage["effective_bits_per_value"]
                    ),
                    "file_bits": int(storage["file_bits"]),
                    "metric": metric,
                    "baseline_score": baseline,
                    "raw_adapter_score": raw_score,
                    "task_score": task_score,
                    "retained_gain": retained_gain,
                    "retained_gain_capped": _capped(retained_gain),
                    "relative_rmse": float(storage["relative_rmse"]),
                    "source": "midrise",
                    "reused_from": None,
                    "storage": storage,
                    "task": task,
                }
                (out_dir / f"midrise{bit_width}.json").write_text(
                    json.dumps(row, indent=2, sort_keys=True)
                )
                write_predictions(
                    out_dir / f"predictions_midrise{bit_width}.jsonl", predictions
                )
                midrise_rows.append(row)
                apply_adapter_tensors(session.model, raw_tensors)
        finally:
            try:
                session.unload()
            except Exception:
                pass

    rows = controls + midrise_rows
    for row in rows:
        row["retained_gain_capped"] = _capped(float(row["retained_gain"]))
        row.setdefault("relative_rmse", None)
        row.setdefault("reused_from", None)
    rows.sort(key=lambda row: (float(row["effective_bits_per_value"]), str(row["codec"])))

    _write_csv(out_dir / "pareto.csv", rows)
    summary = {
        "title": "From Reasoning Trajectories to Information Budgets",
        "subtitle": "Pareto frontier of learned behavior vs. transmitted adapter bits",
        "run_id": run.run_id,
        "model": run.model.key,
        "dataset": run.dataset_key,
        "seed": run.seed,
        "metric": metric,
        "baseline_score": baseline,
        "raw_adapter_score": raw_score,
        "midrise_bits": list(bits),
        "frontier": [row["codec"] for row in _frontier(rows)],
        "rows": rows,
    }
    (out_dir / "pareto.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    _save_plot(rows, run, metric, baseline, raw_score, out_dir / "pareto.png")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reuse a trained adapter and measure a fast zero-free rate/performance curve."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/campaign.yaml"))
    parser.add_argument("--prepared-root", type=Path, default=Path("prepared"))
    parser.add_argument("--bits", type=int, nargs="+", default=list(DEFAULT_BITS))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    bits = tuple(dict.fromkeys(args.bits))
    if not bits or any(bit not in {1, 2, 3, 4, 8} for bit in bits):
        raise SystemExit("--bits must be chosen from 1 2 3 4 8")
    out = args.out or args.run_dir / "pareto"
    summary = run_sweep(
        args.run_dir,
        args.config,
        args.prepared_root,
        bits,
        out,
        args.force,
    )
    print(
        json.dumps(
            {
                "figure": str(out / "pareto.png"),
                "table": str(out / "pareto.csv"),
                "frontier": summary["frontier"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
