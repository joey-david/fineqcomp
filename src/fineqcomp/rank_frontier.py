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
DEFAULT_RATES: tuple[tuple[int, float], ...] = (
    (0, 0.25),
    (0, 0.5),
    (0, 0.75),
    (1, 0.0),
    (1, 0.5),
    (2, 0.0),
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
    baseline_bits = baseline_heldout_bits(record)
    ceiling = float(
        (record.get("raw_behavioral_write") or {}).get("heldout_bits_saved") or 0.0
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    campaign = load_campaign(config_path)
    engine = RunEngine(campaign, prepared_root, run_dir.parent)
    data, _ = engine._load_data(run)
    session = ModelSession.load(run.model)
    rows: list[dict[str, Any]] = []
    try:
        session.attach(run.adapter, run.seed)
        raw = torch.load(adapter_path, map_location="cpu", weights_only=True)
        full_rank = max(
            int(tensor.shape[0])
            for name, tensor in raw.items()
            if ".lora_A." in name
        )
        for rank in sorted({min(int(value), full_rank) for value in ranks}):
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
    finally:
        del session
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


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
