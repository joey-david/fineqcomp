"""Fast adaptive description-length sweep for an already-trained LoRA run.

The base model, LoRA architecture, and decoder are shared side information.
Only the learned adapter is transmitted. Each LoRA row can be omitted or
encoded at 1/2/3/4/8 bits. A Lagrange multiplier selects the row precision that
minimizes reconstruction error plus a penalty on code length.

Candidate distortions are computed tensor-wide (not row-by-row), cached once,
and reused by every GPU worker.
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
from tqdm.auto import tqdm

from fineqcomp.adapters import apply_adapter_tensors
from fineqcomp.codec import (
    container_header_bits,
    midrise_dequantize,
    midrise_quantize,
    pack_unsigned,
    read_container,
    unpack_unsigned,
    write_container,
)
from fineqcomp.config import RunSpec, load_campaign
from fineqcomp.evaluation import write_predictions
from fineqcomp.modeling import ModelSession
from fineqcomp.pareto import (
    _capped,
    _existing_controls,
    _frontier,
    _reference_scores,
)
from fineqcomp.runner import RunEngine


MAGIC = b"FQMDL2\n"
BIT_OPTIONS = (0, 1, 2, 3, 4, 8)
SELECTOR_BITS = 3
CACHE_VERSION = 2
DEFAULT_TARGET_RATES = (0.15, 0.30, 0.50, 0.75, 1.0, 1.5, 2.5, 4.0)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _slug(rate: float) -> str:
    return f"{rate:g}".replace(".", "p")


def _row_candidates(
    tensors: dict[str, torch.Tensor], *, show_progress: bool = False
) -> tuple[dict[str, Any], int]:
    """Vectorized row rate/distortion table for all adapter tensors."""
    tensor_meta: list[dict[str, Any]] = []
    distortion_parts: list[torch.Tensor] = []
    proxy_parts: list[torch.Tensor] = []
    total_values = 0
    row_start = 0
    names = sorted(tensors)
    progress = tqdm(
        total=len(names) * (len(BIT_OPTIONS) - 1),
        desc="MDL preprocessing",
        unit="tensor-bit",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    try:
        for name in names:
            tensor = tensors[name].detach().cpu().float().contiguous()
            if tensor.ndim < 1:
                raise ValueError(f"cannot encode scalar tensor {name}")
            matrix = tensor.reshape(tensor.shape[0], -1)
            rows, columns = map(int, matrix.shape)
            total_values += tensor.numel()
            distortions = torch.empty((rows, len(BIT_OPTIONS)), dtype=torch.float64)
            proxies = torch.empty((rows, len(BIT_OPTIONS)), dtype=torch.int64)
            distortions[:, 0] = matrix.square().sum(dim=1).double()
            proxies[:, 0] = SELECTOR_BITS
            for option_index, bits in enumerate(BIT_OPTIONS[1:], start=1):
                _, _, reconstructed = midrise_quantize(matrix, bits)
                distortions[:, option_index] = (
                    (matrix - reconstructed).square().sum(dim=1).double()
                )
                packed_bits = ((columns * bits + 7) // 8) * 8
                proxies[:, option_index] = SELECTOR_BITS + 16 + packed_bits
                progress.update(1)
            tensor_meta.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "row_start": row_start,
                    "row_count": rows,
                    "columns": columns,
                }
            )
            row_start += rows
            distortion_parts.append(distortions)
            proxy_parts.append(proxies)
    finally:
        progress.close()

    all_distortions = torch.cat(distortion_parts, dim=0)
    cache = {
        "version": CACHE_VERSION,
        "bit_options": list(BIT_OPTIONS),
        "selector_bits": SELECTOR_BITS,
        "total_values": total_values,
        "total_squared_norm": float(all_distortions[:, 0].sum().item()),
        "tensors": tensor_meta,
        "distortions": all_distortions,
        "proxy_bits": torch.cat(proxy_parts, dim=0),
    }
    return cache, total_values


def _save_candidate_cache(cache: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(cache, temporary)
    temporary.replace(path)


def _load_candidate_cache(path: Path) -> dict[str, Any]:
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if int(cache.get("version", -1)) != CACHE_VERSION:
        raise ValueError(f"{path}: stale MDL cache version")
    if tuple(map(int, cache["bit_options"])) != BIT_OPTIONS:
        raise ValueError(f"{path}: MDL bit options do not match this decoder")
    return cache


def prepare_candidate_cache(
    run_dir: Path, cache_path: Path, *, force: bool = False, show_progress: bool = True
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    raw_path = run_dir / "raw_channel.pt"
    if not raw_path.is_file():
        raise FileNotFoundError(f"missing trained adapter: {raw_path}")
    if cache_path.is_file() and not force:
        cache = _load_candidate_cache(cache_path)
        print(
            f"reusing MDL candidate cache: {cache_path} "
            f"({cache['distortions'].shape[0]} rows)",
            flush=True,
        )
        return cache
    raw_tensors = torch.load(raw_path, map_location="cpu", weights_only=True)
    cache, _ = _row_candidates(raw_tensors, show_progress=show_progress)
    cache["raw_channel_size"] = raw_path.stat().st_size
    cache["raw_channel_mtime_ns"] = raw_path.stat().st_mtime_ns
    _save_candidate_cache(cache, cache_path)
    print(
        f"wrote MDL candidate cache: {cache_path} "
        f"({cache['distortions'].shape[0]} rows)",
        flush=True,
    )
    return cache


def _assignment(
    cache: dict[str, Any], penalty: float
) -> tuple[list[int], int, float]:
    distortions = cache["distortions"]
    proxy_bits = cache["proxy_bits"]
    costs = distortions + float(penalty) * proxy_bits.double()
    option_ids = costs.argmin(dim=1)
    row_ids = torch.arange(option_ids.numel())
    rate = int(proxy_bits[row_ids, option_ids].sum().item())
    distortion = float(distortions[row_ids, option_ids].sum().item())
    options = torch.tensor(BIT_OPTIONS, dtype=torch.int64)
    chosen = options[option_ids].tolist()
    return [int(value) for value in chosen], rate, distortion


def _allocate(
    cache: dict[str, Any], total_values: int, target_rate: float
) -> tuple[list[int], dict[str, Any]]:
    """Find a Lagrangian row allocation at or below the requested proxy rate."""
    if target_rate <= 0:
        raise ValueError("target rate must be positive")
    target_bits = target_rate * total_values
    proxy_bits = cache["proxy_bits"]
    distortions = cache["distortions"]

    minimum_bits = int(proxy_bits[:, 0].sum().item())
    zero_distortion = float(distortions[:, 0].sum().item())
    if target_bits <= minimum_bits:
        return [0] * int(proxy_bits.shape[0]), {
            "penalty": None,
            "proxy_bits": minimum_bits,
            "proxy_bits_per_value": minimum_bits / max(total_values, 1),
            "distortion": zero_distortion,
        }

    full_assignment, full_bits, full_distortion = _assignment(cache, 0.0)
    if full_bits <= target_bits:
        return full_assignment, {
            "penalty": 0.0,
            "proxy_bits": full_bits,
            "proxy_bits_per_value": full_bits / max(total_values, 1),
            "distortion": full_distortion,
        }

    high = 1e-18
    _, high_bits, _ = _assignment(cache, high)
    while high_bits > target_bits and high < 1e18:
        high *= 10.0
        _, high_bits, _ = _assignment(cache, high)
    if high_bits > target_bits:
        raise RuntimeError("failed to find an MDL penalty below the target rate")

    low = 0.0
    best = _assignment(cache, high)
    best_penalty = high
    for _ in range(50):
        mid = (low + high) / 2.0
        candidate = _assignment(cache, mid)
        if candidate[1] > target_bits:
            low = mid
        else:
            high = mid
            best = candidate
            best_penalty = mid
    chosen, used_bits, distortion = best
    return chosen, {
        "penalty": best_penalty,
        "proxy_bits": used_bits,
        "proxy_bits_per_value": used_bits / max(total_values, 1),
        "distortion": distortion,
    }


def _encode_mdl(
    tensors: dict[str, torch.Tensor],
    cache: dict[str, Any],
    assignment: list[int],
    path: Path,
    allocation: dict[str, Any],
    *,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Write a reloadable variable-rate adapter using tensor-wide operations."""
    if len(assignment) != int(cache["distortions"].shape[0]):
        raise ValueError("MDL assignment length does not match candidate cache")
    assignment_array = np.asarray(assignment, dtype=np.int16)
    payload = bytearray()
    entries: list[dict[str, Any]] = []
    row_histogram = {str(bits): 0 for bits in BIT_OPTIONS}
    value_histogram = {str(bits): 0 for bits in BIT_OPTIONS}
    total_values = int(cache["total_values"])

    iterator = tqdm(
        cache["tensors"],
        desc="encoding adapter",
        unit="tensor",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for meta in iterator:
        name = str(meta["name"])
        tensor = tensors[name].detach().cpu().float().contiguous()
        matrix = tensor.reshape(int(meta["row_count"]), int(meta["columns"]))
        start = int(meta["row_start"])
        stop = start + int(meta["row_count"])
        selected = assignment_array[start:stop]

        selector_ids = np.asarray(
            [BIT_OPTIONS.index(int(bits)) for bits in selected], dtype=np.uint16
        )
        selector_bytes = pack_unsigned(selector_ids, SELECTOR_BITS)
        selector_offset = len(payload)
        payload.extend(selector_bytes)

        groups: dict[str, Any] = {}
        for bits in BIT_OPTIONS:
            rows_for_bits = np.flatnonzero(selected == bits)
            row_histogram[str(bits)] += int(rows_for_bits.size)
            value_histogram[str(bits)] += int(rows_for_bits.size) * int(meta["columns"])
            if bits == 0 or rows_for_bits.size == 0:
                continue
            row_index = torch.from_numpy(rows_for_bits.astype(np.int64, copy=False))
            submatrix = matrix.index_select(0, row_index)
            scales, codes, _ = midrise_quantize(submatrix, bits)
            scale_bytes = scales.tobytes()
            data_bytes = pack_unsigned(codes, bits)
            scale_offset = len(payload)
            payload.extend(scale_bytes)
            data_offset = len(payload)
            payload.extend(data_bytes)
            groups[str(bits)] = {
                "rows": int(rows_for_bits.size),
                "scale_offset": scale_offset,
                "scale_nbytes": len(scale_bytes),
                "data_offset": data_offset,
                "data_nbytes": len(data_bytes),
            }

        entries.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "selector_offset": selector_offset,
                "selector_nbytes": len(selector_bytes),
                "groups": groups,
            }
        )

    header = {
        "version": 2,
        "codec": "conditional_mdl_row_midrise_v2",
        "shared_side_information": "base model + LoRA architecture + decoder",
        "bit_options": list(BIT_OPTIONS),
        "selector_bits": SELECTOR_BITS,
        "compression": "zlib-9",
        "allocation": allocation,
        "tensors": entries,
    }
    compressed = zlib.compress(bytes(payload), level=9)
    file_bits = write_container(path, MAGIC, header, compressed)
    total_norm = float(cache["total_squared_norm"])
    distortion = float(allocation["distortion"])
    return {
        "file_bits": file_bits,
        "description_bits": file_bits,
        "effective_bits_per_value": file_bits / max(total_values, 1),
        "tensor_values": total_values,
        "header_bits": container_header_bits(MAGIC, header),
        "compressed_payload_bits": len(compressed) * 8,
        "raw_payload_bits": len(payload) * 8,
        "relative_rmse": math.sqrt(distortion / max(total_norm, 1e-30)),
        "row_bit_histogram": row_histogram,
        "value_bit_histogram": value_histogram,
        "proxy_bits": int(allocation["proxy_bits"]),
        "proxy_bits_per_value": float(allocation["proxy_bits_per_value"]),
        "penalty": allocation["penalty"],
    }


def decode_mdl(path: str | Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Decode an FQMDL2 file without access to the original learned tensors."""
    header, payload = read_container(path, MAGIC)
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
        selectors = np.asarray([bit_options[int(index)] for index in selector_ids])
        matrix = torch.zeros((row_count, columns), dtype=torch.float32)

        for bits_text, group in entry.get("groups", {}).items():
            bits = int(bits_text)
            row_ids = np.flatnonzero(selectors == bits)
            expected_rows = int(group["rows"])
            if row_ids.size != expected_rows:
                raise ValueError(f"{path}: corrupt MDL selector/group counts")
            scale_start = int(group["scale_offset"])
            scale_stop = scale_start + int(group["scale_nbytes"])
            scales = np.frombuffer(
                payload[scale_start:scale_stop], dtype=np.float16, count=expected_rows
            )
            data_start = int(group["data_offset"])
            data_stop = data_start + int(group["data_nbytes"])
            codes = unpack_unsigned(
                payload[data_start:data_stop], expected_rows * columns, bits
            )
            reconstructed = midrise_dequantize(
                codes, scales, bits, expected_rows, columns
            )
            row_index = torch.from_numpy(row_ids.astype(np.int64, copy=False))
            matrix.index_copy_(0, row_index, torch.from_numpy(reconstructed.copy()))
        tensors[str(entry["name"])] = matrix.reshape(shape)
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
    if rows:
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
    cache_path: Path,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if not (run_dir / "raw_channel.pt").is_file():
        raise FileNotFoundError(f"missing trained adapter: {run_dir / 'raw_channel.pt'}")
    if not (run_dir / "config.json").is_file():
        raise FileNotFoundError(f"missing run config: {run_dir / 'config.json'}")
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
        cache = prepare_candidate_cache(
            run_dir, cache_path, force=False, show_progress=True
        )
        raw_tensors = torch.load(
            run_dir / "raw_channel.pt", map_location="cpu", weights_only=True
        )
        total_values = int(cache["total_values"])
        engine = RunEngine(campaign, prepared_root, run_dir.parent)
        data, _ = engine._load_data(run)
        session = ModelSession.load(run.model)
        try:
            session.attach(run.adapter, run.seed)
            targets = tqdm(
                pending,
                desc="MDL task evaluations",
                unit="point",
                dynamic_ncols=True,
            )
            for target in targets:
                targets.set_postfix_str(f"target={target:g} b/value")
                assignment, allocation = _allocate(cache, total_values, target)
                adapter_path = out_dir / f"adapter_mdl_{_slug(target)}.fqmdl"
                storage = _encode_mdl(
                    raw_tensors,
                    cache,
                    assignment,
                    adapter_path,
                    allocation,
                    show_progress=True,
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
        "frontier": [row["codec"] for row in _frontier(rows)] if rows else [],
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
        description="Fast adaptive conditional-MDL sweep over a trained LoRA adapter."
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
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    out = (args.out or run_dir / "mdl").resolve()
    cache = (args.cache or out / "candidates_v2.pt").resolve()
    if args.prepare_only:
        prepared = prepare_candidate_cache(
            run_dir, cache, force=args.force, show_progress=True
        )
        print(
            json.dumps(
                {
                    "cache": str(cache),
                    "rows": int(prepared["distortions"].shape[0]),
                    "tensor_values": int(prepared["total_values"]),
                },
                indent=2,
            )
        )
        return

    rates = tuple(dict.fromkeys(float(value) for value in args.target_rates))
    if not rates or any(value <= 0 for value in rates):
        raise SystemExit("--target-rates must be positive")
    summary = run_sweep(
        run_dir,
        args.config,
        args.prepared_root,
        rates,
        out,
        args.force,
        cache,
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
