"""The rank-rate frontier of a finished adapter, at no training cost.

Every rate in this campaign has been measured at one LoRA rank. That fixes the
number of values and sweeps the precision, which is one slice of the surface
the project's question is really about: given a serialized budget, how should
it be split between how many directions the adapter keeps and how well each one
is resolved?

The two axes are commensurable because the file shrinks in proportion to both.
A rank-4 adapter at four bits and a rank-16 adapter at one bit are the same
number of bytes, and which of them keeps more behaviour is a measurement rather
than an opinion. The one comparison already on disk says the answer is not
obvious: on `lever_div_400` a rank-16 adapter returns 1.2 to 3.1 times the
held-out bits of a rank-64 adapter at every matched file size, and the gap
widens as the budget falls.

Nothing here trains. It reads a finished run's stored adapter, truncates it,
re-encodes, and scores held-out code length -- a forward pass over a few
hundred rows, seconds rather than the twenty minutes a task pass costs. The
frozen-model baseline is recovered from the run's own record so every number
is comparable with the rates already published.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch

from fineqcomp.adapters import apply_adapter_tensors
from fineqcomp.codec import (
    decode_adapter_tensor_map,
    encode_tensor_map,
    pad_lora_rank,
    truncate_lora_rank,
)
from fineqcomp.config import RunSpec, load_campaign
from fineqcomp.modeling import ModelSession

# The whole-bit rungs, plus the sub-bit blends the dense ladder already uses.
# The quarter-bit rungs between one and two bits were added once the first
# twelve swept adapters showed where the 90% crossing actually falls: with
# informed truncation it lands at rank one or two and between 1.3 and 2.2 bits
# per value, so a ladder that jumps straight from 1.0 to 2.0 brackets the
# budget inside a factor of 1.7 in file size and cannot resolve it.
DEFAULT_RATES: tuple[tuple[int, float], ...] = (
    (0, 0.25),
    (0, 0.5),
    (0, 0.75),
    (1, 0.0),
    (1, 0.25),
    (1, 0.5),
    (1, 0.75),
    (2, 0.0),
    (2, 0.5),
    (3, 0.0),
    (4, 0.0),
    (8, 0.0),
)
DEFAULT_RANKS: tuple[int, ...] = (1, 2, 4, 8, 16)


def baseline_heldout_bits(record: dict[str, Any]) -> float:
    """The frozen model's code length on the rows the run scored.

    Recovered rather than remeasured: the raw adapter's own held-out total and
    the bits it saved add back to the base model's total, so a sweep does not
    have to re-run a baseline that is already recorded and cannot drift from it.
    """
    raw = (record.get("raw_information") or {}).get("heldout") or {}
    saved = (record.get("raw_behavioral_write") or {}).get("heldout_bits_saved")
    if raw.get("total_bits") is None or saved is None:
        raise ValueError("run record has no held-out baseline to recover")
    return float(raw["total_bits"]) + float(saved)


def _rate_key(bits: int, blend: float) -> str:
    if blend:
        return f"b{bits}_{int(round(blend * 100)):03d}"
    return f"b{bits}"


def sweep_tensors(
    engine: Any,
    session: ModelSession,
    run: RunSpec,
    data: dict[str, list[Any]],
    raw: dict[str, torch.Tensor],
    out_dir: Path,
    *,
    baseline_bits: float,
    ceiling: float,
    ranks: Iterable[int] = DEFAULT_RANKS,
    rates: Iterable[tuple[int, float]] = DEFAULT_RATES,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Score every (rank, rate) cell of one adapter tensor map.

    The caller owns the session and has already attached an adapter of the
    container rank, so the same sweep serves a finished run and an adapter the
    probe built without training.  `ceiling` is the uncoded gain the cells are
    scored against, and it is the caller's because the two uses differ: a
    finished run is scored against its own trained gain, a probe against its
    own.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    full_rank = max(
        int(tensor.shape[0]) for name, tensor in raw.items() if ".lora_A." in name
    )
    # The container's own rank is normally swept, because it is the only cell
    # that can reach the uncoded gain and leaving it out would make the 90%
    # crossing unreachable. It is left out when the caller asked for a grid
    # that stops below it: a probe built in a wider container than the adapter
    # it predicts has to be scored on the same containers as that adapter, or
    # the two budgets are files of different kinds.
    grid = {min(int(value), full_rank) for value in ranks}
    if full_rank <= max(int(value) for value in ranks):
        grid.add(full_rank)
    for rank in sorted(grid):
        truncated = truncate_lora_rank(raw, rank)
        for bits, blend in rates:
            key = f"r{rank}_{_rate_key(bits, blend)}"
            cell = out_dir / f"{key}.json"
            if cell.is_file() and not force:
                rows.append(json.loads(cell.read_text()))
                continue
            path = out_dir / f"adapter_{key}.fqcb"
            storage = encode_tensor_map(
                truncated, path, bits, blend=blend,
                metadata={"rank": rank, "run_id": run.run_id},
            )
            _, decoded = decode_adapter_tensor_map(path)
            # The model carries the trained rank, so the decoded factors go
            # back into that container. The file on disk stays the small
            # one, which is the number this sweep is about.
            apply_adapter_tensors(
                session.model, pad_lora_rank(decoded, full_rank)
            )
            measured = engine._information_measure(
                session, run, data, parts=("heldout",)
            )["heldout"]
            path.unlink(missing_ok=True)
            row = {
                "run_id": run.run_id,
                "model_key": run.model.key,
                "dataset_key": run.dataset_key,
                "seed": run.seed,
                "trained_rank": full_rank,
                "rank": rank,
                "nominal_bits": bits,
                "blend": blend,
                "effective_bits_per_value": float(
                    storage["effective_bits_per_value"]
                ),
                "file_bits": int(storage["file_bits"]),
                "tensor_values": int(storage["tensor_values"]),
                "relative_rmse": float(storage["relative_rmse"]),
                "heldout_total_bits": float(measured["total_bits"]),
                "heldout_bits_saved": baseline_bits
                - float(measured["total_bits"]),
                "baseline_heldout_bits": baseline_bits,
                "ceiling_heldout_bits_saved": ceiling,
                # Against the gain the full trained adapter reached, never
                # against a truncated adapter's own weaker gain: the point
                # of the comparison is which file keeps more of the same
                # behaviour, so the denominator has to be shared.
                "retained_of_trained_gain": (
                    (baseline_bits - float(measured["total_bits"])) / ceiling
                    if ceiling
                    else float("nan")
                ),
            }
            cell.write_text(json.dumps(row, indent=2, sort_keys=True))
            rows.append(row)
            print(
                f"rank {rank:2d} at {row['effective_bits_per_value']:.3f} b/v"
                f" -> {row['file_bits'] / 8e6:6.2f} MB,"
                f" {row['retained_of_trained_gain']:6.3f} of the trained gain",
                flush=True,
            )
    return rows



def sweep_run(
    run_dir: Path,
    config_path: Path,
    prepared_root: Path,
    out_dir: Path,
    *,
    ranks: Iterable[int] = DEFAULT_RANKS,
    rates: Iterable[tuple[int, float]] = DEFAULT_RATES,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Score every (rank, rate) cell of one finished run."""
    from fineqcomp.runner import RunEngine

    run_dir = Path(run_dir).resolve()
    adapter_path = run_dir / "raw_channel.pt"
    if not adapter_path.is_file():
        raise FileNotFoundError(f"missing trained adapter: {adapter_path}")
    record = json.loads((run_dir / "metrics.json").read_text())
    run = RunSpec.from_dict(json.loads((run_dir / "config.json").read_text()))
    if run.kind != "natural":
        raise ValueError("the rank frontier targets natural-task runs")
    campaign = load_campaign(config_path)
    engine = RunEngine(campaign, prepared_root, run_dir.parent)
    data, _ = engine._load_data(run)
    session = ModelSession.load(run.model)
    try:
        session.attach(run.adapter, run.seed)
        return sweep_tensors(
            engine,
            session,
            run,
            data,
            torch.load(adapter_path, map_location="cpu", weights_only=True),
            out_dir,
            baseline_bits=baseline_heldout_bits(record),
            ceiling=float(
                (record.get("raw_behavioral_write") or {}).get("heldout_bits_saved")
                or 0.0
            ),
            ranks=ranks,
            rates=rates,
            force=force,
        )
    finally:
        del session
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The best cell at or under each file size, and the rank that won it.

    This is the object the campaign has never had: the lower envelope of the
    rank-rate surface. A point on it is an instruction -- at this budget, keep
    this many directions at this precision.
    """
    ordered = sorted(rows, key=lambda row: row["file_bits"])
    best: list[dict[str, Any]] = []
    top = float("-inf")
    for row in ordered:
        if row["heldout_bits_saved"] > top:
            top = row["heldout_bits_saved"]
            best.append(row)
    return best


def random_mask_ladder(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The sub-bit rungs the campaign already measured, for comparison.

    Below one bit the shipped ladder is not a precision code: `blend_widths`
    keeps a random fraction of rank directions and drops the rest, keyed on the
    pair name so it knows nothing about the weights. Every sub-one-bit point in
    this campaign is therefore random rank pruning, which makes the informed
    truncation above its natural control at the same file size.
    """
    rungs = []
    for entry in record.get("codecs") or []:
        trial = (entry.get("calibration_trials") or [{}])[0]
        saved = (entry.get("behavioral_write") or {}).get("heldout_bits_saved")
        if trial.get("file_bits") is None or saved is None:
            continue
        rungs.append(
            {
                "codec": entry.get("codec_key"),
                "file_bits": int(trial["file_bits"]),
                "effective_bits_per_value": float(
                    trial["effective_bits_per_value"]
                ),
                "heldout_bits_saved": float(saved),
                "random_rank_mask": int(entry.get("bits") or 0) == 0,
            }
        )
    return sorted(rungs, key=lambda rung: rung["file_bits"])


def _interpolate(points: list[tuple[float, float]], at: float) -> float | None:
    ordered = sorted(points)
    if not ordered or at < ordered[0][0] or at > ordered[-1][0]:
        return None
    for index in range(1, len(ordered)):
        left, right = ordered[index - 1], ordered[index]
        if left[0] <= at <= right[0]:
            span = right[0] - left[0]
            if span <= 0:
                return right[1]
            return left[1] + (at - left[0]) * (right[1] - left[1]) / span
    return None


def compare_against_random_mask(
    cells: list[dict[str, Any]], record: dict[str, Any]
) -> list[dict[str, Any]]:
    """Informed truncation against the mask the ladder already uses, by bytes.

    The two are the same operation -- keep some rank directions, drop the rest
    -- differing only in whether the choice looks at the weights. So the
    comparison is paired at matched file size and needs no model of anything:
    at this many bytes, does picking the strongest directions beat picking at
    random, on the same adapter?
    """
    ladder = [
        (float(rung["file_bits"]), float(rung["heldout_bits_saved"]))
        for rung in random_mask_ladder(record)
    ]
    rows = []
    for cell in sorted(cells, key=lambda cell: cell["file_bits"]):
        masked = _interpolate(ladder, float(cell["file_bits"]))
        if masked is None:
            continue
        ceiling = float(cell["ceiling_heldout_bits_saved"]) or float("nan")
        rows.append(
            {
                "run_id": cell["run_id"],
                "model_key": cell["model_key"],
                "dataset_key": cell["dataset_key"],
                "seed": cell["seed"],
                "rank": cell["rank"],
                "effective_bits_per_value": cell["effective_bits_per_value"],
                "file_megabytes": cell["file_bits"] / 8e6,
                "truncated_bits_saved": cell["heldout_bits_saved"],
                "random_mask_bits_saved": masked,
                "truncated_retained": cell["heldout_bits_saved"] / ceiling,
                "random_mask_retained": masked / ceiling,
                "retained_gain_points": (
                    100.0 * (cell["heldout_bits_saved"] - masked) / ceiling
                ),
            }
        )
    return rows


def write_rank_frontier_report(
    results_root: Path, runs_root: Path, out_dir: Path
) -> dict[str, Any]:
    """Join every swept adapter and score truncation against the random mask."""
    from fineqcomp.artifacts import write_json
    from fineqcomp.relative_validation import _write_csv

    results_root = Path(results_root)
    cells: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        run_cells = [
            json.loads(path.read_text())
            for path in sorted(run_dir.glob("r*_b*.json"))
        ]
        if not run_cells:
            continue
        metrics = Path(runs_root) / run_dir.name / "metrics.json"
        if not metrics.is_file():
            continue
        cells.extend(run_cells)
        paired.extend(
            compare_against_random_mask(run_cells, json.loads(metrics.read_text()))
        )
    out_dir = Path(out_dir)
    _write_csv(out_dir / "cells.csv", cells)
    _write_csv(out_dir / "truncation_vs_random_mask.csv", paired)
    _write_csv(out_dir / "frontier.csv", frontier(cells))
    wins = [row["retained_gain_points"] for row in paired]
    summary = {
        "adapters": len({row["run_id"] for row in cells}),
        "cells": len(cells),
        "paired_comparisons": len(paired),
        "ranks": sorted({int(row["rank"]) for row in cells}),
        "median_gain_points_over_random_mask": (
            float(sorted(wins)[len(wins) // 2]) if wins else None
        ),
        "share_where_truncation_wins": (
            sum(1 for value in wins if value > 0) / len(wins) if wins else None
        ),
        "ranks_on_the_frontier": sorted(
            {int(row["rank"]) for row in frontier(cells)}
        ),
    }
    write_json(out_dir / "summary.json", summary)
    return summary
