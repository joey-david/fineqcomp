"""Adaptive minimum-description-length sweep for an already-trained LoRA run.

The base model, adapter architecture, and decoder are treated as shared side
information. We measure the exact serialized length of the learned adapter.
Each LoRA row may be dropped (0 bit) or encoded with the zero-free mid-rise
family at 1/2/3/4/8 bits. A Lagrange multiplier chooses the rowwise allocation
that minimizes weight reconstruction error + lambda * proxy code length.
"""

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
from fineqcomp.codec import pack_unsigned, unpack_unsigned
from fineqcomp.config import RunSpec, load_campaign
from fineqcomp.evaluation import write_predictions
from fineqcomp.modeling import ModelSession
from fineqcomp.pareto import (
    _capped,
    _existing_controls,
    _frontier,
    _midrise_tensor,
    _reference_scores,
)
from fineqcomp.runner import RunEngine


MAGIC = b"FQMDL1\n"
BIT_OPTIONS = (0, 1, 2, 3, 4, 8)
SELECTOR_BITS = 3
DEFAULT_TARGET_RATES = (0.15, 0.30, 0.50, 0.75, 1.0, 1.5, 2.5, 4.0)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _slug(rate: float) -> str:
    return f"{rate:g}".replace(".", "p")


def _row_candidates(
    tensors: dict[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], int]:
    """Precompute only each row's distortion/rate table, not reconstructions."""
    rows: list[dict[str, Any]] = []
    total_values = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().float().contiguous()
        matrix = tensor.reshape(tensor.shape[0], -1)
        total_values += tensor.numel()
        for row_index, row in enumerate(matrix):
            count = row.numel()
            candidates: dict[int, dict[str, Any]] = {
                0: {
                    "distortion": float(row.square().sum().item()),
                    "proxy_bits": SELECTOR_BITS,
                }
            }
            for bits in BIT_OPTIONS[1:]:
                _, _, reconstructed = _midrise_tensor(row.reshape(1, -1), bits)
                packed_bits = ((count * bits + 7) // 8) * 8
                candidates[bits] = {
                    "distortion": float(
                        (row - reconstructed.reshape(-1)).square().sum().item()
                    ),
                    "proxy_bits": SELECTOR_BITS + 16 + packed_bits,
                }
            rows.append(
                {
                    "tensor": name,
                    "row": row_index,
                    "count": count,
                    "candidates": candidates,
                }
            )
    return rows, total_values


def _assignment(
    rows: list[dict[str, Any]], penalty: float
) -> tuple[list[int], int, float]:
    chosen: list[int] = []
    rate = 0
    distortion = 0.0
    for row in rows:
        candidates = row["candidates"]
        bits = min(
            BIT_OPTIONS,
            key=lambda value: (
                float(candidates[value]["distortion"])
                + penalty * int(candidates[value]["proxy_bits"]),
                int(candidates[value]["proxy_bits"]),
            ),
        )
        chosen.append(bits)
        rate += int(candidates[bits]["proxy_bits"])
        distortion += float(candidates[bits]["distortion"])
    return chosen, rate, distortion


def _allocate(
    rows: list[dict[str, Any]], total_values: int, target_rate: float
) -> tuple[list[int], dict[str, Any]]:
    """Find a Lagrangian allocation at or below the requested proxy rate."""
    if target_rate <= 0:
        raise ValueError("target rate must be positive")
    target_bits = target_rate * total_values

    zero_assignment = [0] * len(rows)
    minimum_bits = sum(
        int(row["candidates"][0]["proxy_bits"]) for row in rows
    )
    zero_distortion = sum(
        float(row["candidates"][0]["distortion"]) for row in rows
    )
    if target_bits <= minimum_bits:
        return zero_assignment, {
            "penalty": None,
            "proxy_bits": minimum_bits,
            "proxy_bits_per_value": minimum_bits / max(total_values, 1),
            "distortion": zero_distortion,
        }

    full_assignment, full_bits, full_distortion = _assignment(rows, 0.0)
    if full_bits <= target_bits:
        return full_assignment, {
            "penalty": 0.0,
            "proxy_bits": full_bits,
            "proxy_bits_per_value": full_bits / max(total_values, 1),
            "distortion": full_distortion,
        }

    high = 1e-18
    _, high_bits, _ = _assignment(rows, high)
    while high_bits > target_bits and high < 1e18:
        high *= 10.0
        _, high_bits, _ = _assignment(rows, high)
    if high_bits > target_bits:
        raise RuntimeError("failed to find an MDL penalty below the target rate")

    low = 0.0
    best = _assignment(rows, high)
    best_penalty = high
    for _ in range(80):
        mid = (low + high) / 2.0
        candidate = _assignment(rows, mid)
        if candidate[1] > target_bits:
            low = mid
        else:
            high = mid
            best = candidate
            best_penalty = mid
    chosen, proxy_bits, distortion = best
    return chosen, {
        "penalty": best_penalty,
        "proxy_bits": proxy_bits,
        "proxy_bits_per_value": proxy_bits / max(total_values, 1),
        "distortion": distortion,
    }


def _encode_mdl(
    tensors: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    assignment: list[int],
    path: Path,
    allocation: dict[str, Any],
) -> dict[str, Any]:
    """Serialize a reloadable variable-rate adapter and return exact file rate."""
    by_tensor: dict[str, list[tuple[dict[str, Any], int]]] = {}
    for row, bits in zip(rows, assignment, strict=True):
        by_tensor.setdefault(str(row["tensor"]), []).append((row, bits))

    payload = bytearray()
    entries: list[dict[str, Any]] = []
    bit_histogram = {str(bits): 0 for bits in BIT_OPTIONS}
    value_histogram = {str(bits): 0 for bits in BIT_OPTIONS}
    total_values = 0
    squared_error = 0.0
    squared_norm = 0.0

    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().float().contiguous()
        matrix = tensor.reshape(tensor.shape[0], -1)
        tensor_rows = by_tensor[name]
        selectors = np.asarray(
            [BIT_OPTIONS.index(bits) for _, bits in tensor_rows], dtype=np.uint16
        )
        selector_bytes = pack_unsigned(selectors, SELECTOR_BITS)
        selector_offset = len(payload)
        payload.extend(selector_bytes)

        scale_offset = len(payload)
        scale_buffer = bytearray()
        data_buffer = bytearray()
        for source_row, bits in tensor_rows:
            row = matrix[int(source_row["row"])]
            count = int(source_row["count"])
            bit_histogram[str(bits)] += 1
            value_histogram[str(bits)] += count
            total_values += count
            squared_norm += float(row.square().sum().item())
            if bits == 0:
                squared_error += float(row.square().sum().item())
                continue
            scales, codes, reconstructed = _midrise_tensor(row.reshape(1, -1), bits)
            squared_error += float(
                (row - reconstructed.reshape(-1)).square().sum().item()
            )
            scale_buffer.extend(scales.tobytes())
            data_buffer.extend(pack_unsigned(codes, bits))
        payload.extend(scale_buffer)
        data_offset = len(payload)
        payload.extend(data_buffer)

        entries.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "selector_offset": selector_offset,
                "selector_nbytes": len(selector_bytes),
                "scale_offset": scale_offset,
                "scale_nbytes": len(scale_buffer),
                "data_offset": data_offset,
                "data_nbytes": len(data_buffer),
            }
        )

    header = {
        "version": 1,
        "codec": "conditional_mdl_row_midrise_v1",
        "shared_side_information": "base model + LoRA architecture + decoder",
        "bit_options": list(BIT_OPTIONS),
        "selector_bits": SELECTOR_BITS,
        "compression": "zlib-9",
        "allocation": allocation,
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
    return {
        "file_bits": file_bits,
        "description_bits": file_bits,
        "effective_bits_per_value": file_bits / max(total_values, 1),
        "tensor_values": total_values,
        "header_bits": (len(MAGIC) + 4 + len(header_bytes)) * 8,
        "compressed_payload_bits": len(compressed) * 8,
        "raw_payload_bits": len(payload) * 8,
        "relative_rmse": math.sqrt(squared_error / max(squared_norm, 1e-30)),
        "row_bit_histogram": bit_histogram,
        "value_bit_histogram": value_histogram,
        "proxy_bits": int(allocation["proxy_bits"]),
        "proxy_bits_per_value": float(allocation["proxy_bits_per_value"]),
        "penalty": allocation["penalty"],
    }


def decode_mdl(path: str | Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Decode an FQMDL file without access to the original learned tensors."""
    source = Path(path)
    with source.open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{source}: invalid MDL adapter magic")
        header_size = struct.unpack("<I", stream.read(4))[0]
        header = json.loads(stream.read(header_size))
        payload = zlib.decompress(stream.read())

    bit_options = tuple(map(int, header["bit_options"]))
    selector_bits = int(header["selector_bits"])
    tensors: dict[str, torch.Tensor] = {}
    for entry in header["tensors"]:
        shape = tuple(map(int, entry["shape"]))
        row_count = shape[0]
        columns = math.prod(shape[1:])
        selector_start = int(entry["selector_offset"])
        selector_stop = selector_start + int(entry["selector_nbytes"])
        selector_ids = unpack_unsigned(
            payload[selector_start:selector_stop], row_count, selector_bits
        )
        selectors = [bit_options[int(index)] for index in selector_ids]

        scale_cursor = int(entry["scale_offset"])
        data_cursor = int(entry["data_offset"])
        reconstructed: list[torch.Tensor] = []
        for bits in selectors:
            if bits == 0:
                reconstructed.append(torch.zeros(columns, dtype=torch.float32))
                continue
            scale = float(
                np.frombuffer(
                    payload[scale_cursor : scale_cursor + 2], dtype=np.float16, count=1
                )[0]
            )
            scale_cursor += 2
            data_nbytes = (columns * bits + 7) // 8
            codes = unpack_unsigned(
                payload[data_cursor : data_cursor + data_nbytes], columns, bits
            ).astype(np.int32)
            data_cursor += data_nbytes
            positive_levels = 1 << (bits - 1)
            positive = codes >= positive_levels
            magnitude_index = np.where(positive, codes - positive_levels, codes)
            levels = 2 * magnitude_index + 1
            signed = np.where(positive, levels, -levels).astype(np.float32)
            reconstructed.append(torch.from_numpy(signed * scale))
        tensors[str(entry["name"])] = torch.stack(reconstructed).reshape(shape)
    return header, tensors


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "codec",
        "target_rate",
        "description_bits",
        "effective_bits_per_value",
        "proxy_bits_per_value",
        "metric",
        "baseline_score",
        "raw_adapter_score",
        "task_score",
        "retained_gain",
        "retained_gain_capped",
        "relative_rmse",
        "source",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_plot(
    rows: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    run: RunSpec,
    metric: str,
    baseline: float,
    raw: float,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    if controls:
        ax.scatter(
            [row["effective_bits_per_value"] for row in controls],
            [100 * _capped(float(row["retained_gain"])) for row in controls],
            marker="x",
            s=38,
            alpha=0.4,
            label="fixed-rate controls",
        )
    ax.scatter(
        [row["effective_bits_per_value"] for row in rows],
        [100 * _capped(float(row["retained_gain"])) for row in rows],
        s=72,
        label="adaptive MDL",
    )
    frontier = _frontier(rows)
    if frontier:
        ax.plot(
            [row["effective_bits_per_value"] for row in frontier],
            [100 * _capped(float(row["retained_gain"])) for row in frontier],
            linewidth=2,
            label="MDL Pareto frontier",
        )
    ax.axhline(100, linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(0, linewidth=1, alpha=0.25)
    ax.set_xscale("log", base=2)
    ax.set_ylim(-4, 104)
    ax.set_xlabel("Conditional adapter description length (effective bits / learned scalar)")
    ax.set_ylabel("Retained task gain (%) — capped at raw adapter")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Minimum Description Length of Learned Behavior", fontsize=15)
    ax.set_title("Adaptive LoRA code length vs. retained capability", fontsize=10)
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


def run_sweep(
    run_dir: Path,
    config_path: Path,
    prepared_root: Path,
    target_rates: tuple[float, ...],
    out_dir: Path,
    force: bool,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if not (run_dir / "raw_channel.pt").is_file():
        raise FileNotFoundError(f"missing trained adapter: {run_dir / 'raw_channel.pt'}")
    run = RunSpec.from_dict(_read_json(run_dir / "config.json"))
    if run.kind != "natural":
        raise ValueError("MDL sweep currently targets natural-task runs")

    campaign = load_campaign(config_path)
    controls = _existing_controls(run_dir)
    metric, baseline, raw_score = _reference_scores(controls)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    pending: list[float] = []
    for target in target_rates:
        metric_path = out_dir / f"mdl_{_slug(target)}.json"
        if metric_path.is_file() and not force:
            rows.append(_read_json(metric_path))
        else:
            pending.append(target)

    if pending:
        raw_tensors = torch.load(
            run_dir / "raw_channel.pt", map_location="cpu", weights_only=True
        )
        candidates, total_values = _row_candidates(raw_tensors)
        engine = RunEngine(campaign, prepared_root, run_dir.parent)
        data, _ = engine._load_data(run)
        session = ModelSession.load(run.model)
        try:
            session.attach(run.adapter, run.seed)
            for index, target in enumerate(pending, start=1):
                print(
                    f"[{index}/{len(pending)}] MDL target {target:g} bits/value: allocate + encode + task eval",
                    flush=True,
                )
                assignment, allocation = _allocate(candidates, total_values, target)
                adapter_path = out_dir / f"adapter_mdl_{_slug(target)}.fqmdl"
                storage = _encode_mdl(
                    raw_tensors, candidates, assignment, adapter_path, allocation
                )
                _, decoded = decode_mdl(adapter_path)
                apply_adapter_tensors(session.model, decoded)
                task, predictions = engine._evaluate_natural(session, run, data["test"])
                task_score = float(task[metric])
                retained_gain = (task_score - baseline) / (raw_score - baseline)
                row = {
                    "codec": f"mdl_{target:g}",
                    "target_rate": target,
                    "description_bits": int(storage["description_bits"]),
                    "effective_bits_per_value": float(storage["effective_bits_per_value"]),
                    "proxy_bits_per_value": float(storage["proxy_bits_per_value"]),
                    "metric": metric,
                    "baseline_score": baseline,
                    "raw_adapter_score": raw_score,
                    "task_score": task_score,
                    "retained_gain": retained_gain,
                    "retained_gain_capped": _capped(retained_gain),
                    "relative_rmse": float(storage["relative_rmse"]),
                    "row_bit_histogram": storage["row_bit_histogram"],
                    "value_bit_histogram": storage["value_bit_histogram"],
                    "source": "mdl",
                    "storage": storage,
                    "task": task,
                }
                (out_dir / f"mdl_{_slug(target)}.json").write_text(
                    json.dumps(row, indent=2, sort_keys=True)
                )
                write_predictions(
                    out_dir / f"predictions_mdl_{_slug(target)}.jsonl", predictions
                )
                rows.append(row)
                apply_adapter_tensors(session.model, raw_tensors)
        finally:
            try:
                session.unload()
            except Exception:
                pass

    rows.sort(key=lambda row: float(row["effective_bits_per_value"]))
    _write_csv(out_dir / "mdl_pareto.csv", rows)
    summary = {
        "title": "Minimum Description Length of Learned Behavior",
        "definition": "conditional description length of the learned LoRA update given the base model, adapter architecture, and decoder",
        "run_id": run.run_id,
        "model": run.model.key,
        "dataset": run.dataset_key,
        "seed": run.seed,
        "metric": metric,
        "baseline_score": baseline,
        "raw_adapter_score": raw_score,
        "target_rates": list(target_rates),
        "frontier": [row["codec"] for row in _frontier(rows)],
        "rows": rows,
    }
    (out_dir / "mdl_pareto.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    _save_plot(
        rows,
        controls,
        run,
        metric,
        baseline,
        raw_score,
        out_dir / "mdl_pareto.png",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adaptive conditional-MDL sweep over a trained LoRA adapter."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/campaign.yaml"))
    parser.add_argument("--prepared-root", type=Path, default=Path("prepared"))
    parser.add_argument(
        "--target-rates",
        type=float,
        nargs="+",
        default=list(DEFAULT_TARGET_RATES),
        help="proxy adapter description budgets in bits per learned scalar",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rates = tuple(dict.fromkeys(float(value) for value in args.target_rates))
    if not rates or any(value <= 0 for value in rates):
        raise SystemExit("--target-rates must be positive")
    out = args.out or args.run_dir / "mdl"
    summary = run_sweep(
        args.run_dir,
        args.config,
        args.prepared_root,
        rates,
        out,
        args.force,
    )
    print(
        json.dumps(
            {
                "figure": str(out / "mdl_pareto.png"),
                "table": str(out / "mdl_pareto.csv"),
                "frontier": summary["frontier"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
