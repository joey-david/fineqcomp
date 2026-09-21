"""Rank against precision at a fixed bit budget, on corrupted-rationale LoRAs.

A LoRA fine-tuned on MetaMathQA rows whose rationale was taken from a *different*
problem loses most of its GSM8K accuracy. Truncating that finished rank-16
adapter to rank one and re-encoding it at one bit puts the accuracy back, above
the base model. The recorded result sweeps precision at one rank, so it cannot
separate two explanations that both predict it:

* the *code* matters -- coarse quantization discards the part of the update that
  memorised the mismatched rationales; or
* the *size* matters -- any sufficiently small update would do, and rank one at
  one bit is simply the smallest thing tested.

Rank and precision are commensurable because the serialized file shrinks in
proportion to both, so `rank x bits` names a budget that several different
adapters can meet. A rank-16 adapter at one bit, a rank-4 adapter at four bits
and a rank-1 adapter at sixteen bits move the same number of bits and differ
only in how those bits are spent. If the recovery is about size, accuracy is
flat along such a diagonal. If it is about the code, the coarse end wins.

This module sweeps that surface and scores it on the task rather than on
held-out code length, which is what separates it from `rank_frontier`. It adds
a third arm the compression sweep cannot supply on its own: adapters that were
*trained* at low rank at full precision, from the first optimizer step, so that
"low rank because it was truncated" and "low rank because it was never wider"
can be read off the same budget axis.

Nothing here trains. Training the low-rank arm is an ordinary campaign,
`configs/low_rank_permuted.yaml`; this module reads the adapters it writes.

Evaluation, corpora and scoring are imported from `behavioral_trajectory` rather
than restated, so the numbers land on the same axis as the recorded result: the
same permuted-rationale corpus, the same disjoint calibration split, the same
1,319 GSM8K test rows, the same GSM-Symbolic release, and the same paired
bootstrap clustered by Symbolic template.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from fineqcomp.adapters import adapter_tensors, apply_adapter_tensors
from fineqcomp.artifacts import claim_run, read_json, write_json
from fineqcomp.behavioral_trajectory import paired_interval, score, sha256, study_data
from fineqcomp.campaign import expand_campaign
from fineqcomp.codec import (
    ALLOWED_BITS,
    decode_adapter_tensor_map,
    encode_tensor_map,
    pad_lora_rank,
    truncate_lora_rank,
)
from fineqcomp.config import RunSpec, load_campaign
from fineqcomp.modeling import ModelSession

SPLITS = ("calibration", "gsm8k", "symbolic")


def candidate_key(rank: int, bits: int) -> str:
    return f"r{rank}_b{bits}"


def budget_units(rank: int, bits: int) -> int:
    """Bits per stored LoRA direction-slot: the axis the diagonals run along.

    This is the nominal budget, exact and integer, so two cells either match or
    do not. The measured file is reported alongside it and is only proportional
    to this number -- the per-row scales and the header do not scale with rank.
    """
    return rank * bits


def candidate_grid(
    native_rank: int, ranks: Iterable[int], bits: Iterable[int]
) -> list[tuple[int, int]]:
    """The (rank, bits) cells worth scoring for an adapter of `native_rank`.

    `truncate_lora_rank` clamps a requested rank to the rank the adapter
    actually has, so a rank-4 adapter asked for rank 8 and rank 16 returns the
    same tensors three times. Scoring those repeats would cost three full GSM8K
    passes to re-measure one number, so the grid clamps first and then drops the
    duplicates. The surviving cell keeps the adapter's real rank as its label,
    because that is the rank whose bits were paid for.
    """
    if native_rank < 1:
        raise ValueError("native rank must be at least one")
    cells = set()
    for rank in ranks:
        if int(rank) < 1:
            raise ValueError("every swept rank must be at least one")
        for width in bits:
            if int(width) not in ALLOWED_BITS or int(width) < 1:
                raise ValueError(f"bits must be one of {sorted(ALLOWED_BITS - {0})}")
            cells.add((min(int(rank), native_rank), int(width)))
    return sorted(cells)


def arm_runs(arm: dict[str, Any], seeds: Iterable[int]) -> list[RunSpec]:
    """Resolve one arm's run ids by expanding the campaign that defines them.

    Run ids are content hashes of the run's own identity, so re-expanding the
    campaign recovers them exactly without the runs existing yet. That lets the
    study be prepared and checked before its training campaign has finished --
    and makes a silently edited training config a missing run rather than a
    mismatched comparison.
    """
    wanted = set(int(seed) for seed in seeds)
    runs = [
        run
        for run in expand_campaign(load_campaign(arm["config"]))
        if run.study == arm["study"]
        and run.model.key == arm["model"]
        and run.adapter.rank == int(arm["rank"])
        and run.seed in wanted
    ]
    if len(runs) != len(wanted):
        raise ValueError(
            f"arm {arm['study']} rank {arm['rank']} expands to {len(runs)} runs"
            f" for {len(wanted)} seeds in {arm['config']}"
        )
    return sorted(runs, key=lambda run: run.seed)


def prepare(config: dict, out: Path, runs: Path, prepared: Path) -> dict:
    """Freeze the cell list, the grid and every input digest before any GPU."""
    cells = []
    for name, arm in sorted(config["arms"].items()):
        for run in arm_runs(arm, config["seeds"]):
            cells.append(
                {
                    "id": len(cells),
                    "arm": name,
                    "trained_rank": run.adapter.rank,
                    "run": run.to_dict(),
                    "slug": f"{run.model.key}/{name}/seed{run.seed}",
                    "grid": [
                        list(cell)
                        for cell in candidate_grid(
                            run.adapter.rank, config["ranks"], config["bits"]
                        )
                    ],
                }
            )
    if not cells:
        raise ValueError("no arms declared")
    inputs = [Path(config["symbolic_source"])]
    inputs.extend(Path(arm["config"]) for arm in config["arms"].values())
    for seed in config["seeds"]:
        study_data(config, seed, prepared)
        inputs.extend(
            (prepared / "natural" / "cot_math_permuted" / f"seed{seed}").glob("*.jsonl")
        )
    record = {
        "config": config,
        "cells": cells,
        "inputs": {str(path): sha256(path) for path in sorted(set(inputs))},
        "implementation": {
            str(path): sha256(path)
            for path in sorted(Path("src/fineqcomp").glob("*.py"))
        },
    }
    previous = read_json(out / "lock.json")
    if previous is not None and previous != record:
        raise ValueError("study changed: use a new output directory")
    write_json(out / "lock.json", record)
    return record


def check(lock: dict, runs: Path) -> dict:
    """Prove every cell has a trained adapter before a GPU hour is claimed.

    This study evaluates adapters other campaigns trained. A missing one is
    silent at submit time and fatal an hour into the job, so it is worth one
    stat call per cell here.
    """
    missing = [
        cell["slug"]
        for cell in lock["cells"]
        if not (runs / cell["run"]["run_id"] / "raw_channel.pt").is_file()
    ]
    passes = sum(len(cell["grid"]) for cell in lock["cells"])
    return {
        "cells": len(lock["cells"]),
        "candidates": passes,
        "missing_adapters": missing,
        "passed": not missing,
    }


def encode_candidate(
    tensors: dict[str, torch.Tensor], rank: int, bits: int, path: Path
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Write one cell's real file and read back what the decoder reconstructs.

    Scoring the decoded tensors rather than the pre-encode ones is the point:
    the claim is about what survives a file of this size, so the measurement has
    to go through the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    storage = encode_tensor_map(truncate_lora_rank(tensors, rank), path, bits)
    _, decoded = decode_adapter_tensor_map(path)
    return storage, decoded


def base_root(out: Path, model: str, seed: int) -> Path:
    """Where the base-model scores for one model and seed live, shared by arms."""
    return out / model / "base" / f"seed{seed}"


def sweep_cell(cell: dict, config: dict, out: Path, runs: Path, prepared: Path) -> None:
    """Score the base model, the raw adapter and every grid cell, resumably.

    Every score is cached on its own path, so a job that runs out of walltime
    mid-grid resumes at the next unscored cell rather than repeating the pass.
    """
    root = out / cell["slug"]
    run = RunSpec.from_dict(cell["run"])
    splits = tuple(config["candidate_splits"])
    with claim_run(root) as acquired:
        if not acquired:
            raise RuntimeError(f"cell already running: {root}")
        if (root / "complete.json").exists():
            return
        data = study_data(config, run.seed, prepared)
        if config.get("smoke"):
            data = {key: value[:4] for key, value in data.items()}
        session = ModelSession.load(run.model)
        try:
            session.attach(run.adapter, run.seed)
            trained = torch.load(
                runs / run.run_id / "raw_channel.pt", map_location="cpu",
                weights_only=True,
            )
            # The attached adapter is zero-initialised, so its tensors are the
            # base model. Reading them here rather than detaching keeps the
            # base, raw and candidate passes on one generation path.
            base = adapter_tensors(session.model, "full_lora")
            # The base model is the same object for every arm of one model and
            # seed, and scoring it is three generation passes. Write it once to
            # a shared path so fifteen cells do not pay for it fifteen times;
            # `score` caches on the path, so whichever cell reaches it first
            # does the work and the rest read the file.
            shared = base_root(out, run.model.key, run.seed)
            apply_adapter_tensors(session.model, base)
            for split in SPLITS:
                score(session, run, data[split], config, shared / f"base_{split}.json")
            apply_adapter_tensors(session.model, trained)
            for split in SPLITS:
                score(session, run, data[split], config, root / f"raw_{split}.json")
            rows = []
            for rank, bits in [tuple(pair) for pair in cell["grid"]]:
                key = candidate_key(rank, bits)
                storage, decoded = encode_candidate(
                    trained, rank, bits, root / f"{key}.fqcb"
                )
                apply_adapter_tensors(
                    session.model, pad_lora_rank(decoded, run.adapter.rank)
                )
                measured = {
                    split: score(session, run, data[split], config, root / f"{key}_{split}.json")
                    for split in splits
                }
                rows.append(
                    {
                        "key": key,
                        "rank": rank,
                        "bits": bits,
                        "budget_units": budget_units(rank, bits),
                        "file_bits": int(storage["file_bits"]),
                        "effective_bits_per_value": float(
                            storage["effective_bits_per_value"]
                        ),
                        **{
                            f"{split}_accuracy": measured[split]["exact_match"]
                            for split in splits
                        },
                    }
                )
                write_json(root / "candidates.json", rows)
                # The coded file is a deterministic function of the raw adapter
                # and the two integers in its name, and the whole grid of them
                # runs to several gigabytes per cell. The measured byte count is
                # the part worth keeping, and it is already in the row above.
                (root / f"{key}.fqcb").unlink(missing_ok=True)
            write_json(root / "complete.json", {"complete": True, "candidates": len(rows)})
        finally:
            session.unload()


def probe(cell: dict, config: dict, out: Path, runs: Path, prepared: Path) -> dict:
    """Time one short generation pass and project the study's walltime.

    Walltime here is set by generation, not by training or by the codec, and
    the rate depends on the model, the batch size and how long the corrupted
    adapter's answers run. Measuring it on one cell costs minutes and is the
    difference between a right-sized job and a queue slot that expires with the
    grid half scored.
    """
    run = RunSpec.from_dict(cell["run"])
    rows = int(config["probe_rows"])
    data = study_data(config, run.seed, prepared)
    session = ModelSession.load(run.model)
    try:
        session.attach(run.adapter, run.seed)
        trained = torch.load(
            runs / run.run_id / "raw_channel.pt", map_location="cpu", weights_only=True
        )
        # Probe the raw corrupted adapter, not the base model: its answers are
        # the degenerate ones, and whether they run short or run to the token
        # cap is exactly what the projection turns on.
        apply_adapter_tensors(session.model, trained)
        examples = data["gsm8k"][:rows]
        started = time.monotonic()
        score(session, run, examples, config, out / "probe" / f"{run.run_id}.json")
        elapsed = time.monotonic() - started
    finally:
        session.unload()
    per_row = elapsed / max(rows, 1)
    split_rows = {
        "calibration": config["calibration_rows"],
        "gsm8k": config["gsm8k_test_rows"],
        "symbolic": 100 * len(config["symbolic_instances"]),
    }
    # One raw pass over every split per cell; the base passes are shared across
    # the arms of a seed, so they are not charged to each cell.
    reference = sum(split_rows.values())
    candidates = len(cell["grid"]) * sum(
        split_rows[split] for split in config["candidate_splits"]
    )
    return {
        "probe_rows": rows,
        "seconds": elapsed,
        "seconds_per_row": per_row,
        "rows_per_cell": reference + candidates,
        "projected_cell_hours": per_row * (reference + candidates) / 3600,
        "note": "generation only; model load and encoding are not included",
    }


def matched_sets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group a cell's candidates into equal-budget diagonals.

    A budget met by only one cell says nothing about how bits should be split,
    so singletons are dropped. `spread` is the quantity the experiment is for:
    the accuracy range across ways of spending one budget. Near zero means only
    the size mattered; large means the split does.
    """
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(int(row["budget_units"]), []).append(row)
    sets = []
    for budget, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda row: row["rank"])
        scored = [row for row in members if row.get("gsm8k_accuracy") is not None]
        best = max(scored, key=lambda row: row["gsm8k_accuracy"], default=None)
        worst = min(scored, key=lambda row: row["gsm8k_accuracy"], default=None)
        files = [row["file_bits"] for row in members]
        sets.append(
            {
                "budget_units": budget,
                "cells": [row["key"] for row in members],
                "accuracy": {row["key"]: row.get("gsm8k_accuracy") for row in members},
                "best_key": best["key"] if best else None,
                "best_rank": best["rank"] if best else None,
                "spread": (best["gsm8k_accuracy"] - worst["gsm8k_accuracy"])
                if best and worst
                else None,
                # Nominal budgets are matched exactly; real files are not, so
                # carry the disagreement rather than implying a tie that the
                # bytes do not support.
                "file_bits_ratio": max(files) / min(files),
            }
        )
    return sets


def scratch_versus_compressed(
    cells: list[dict[str, Any]], rows_by_slug: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Pair each trained-at-rank-r adapter against a truncated one of equal budget.

    Both sides are read off the grid, which is why the comparison is available
    at every budget rather than only at the one the recovery result used: the
    trained-low-rank arm contributes its own fp16 cell, and the trained-wide arm
    contributes whichever (rank, bits) cell meets the same nominal budget.
    """
    # Only arms that were actually swept can take part. The lock declares every
    # arm the study will ever have, including ones a short night never reached,
    # and taking the widest *declared* rank would point every pairing at an arm
    # with no rows -- producing an empty comparison that reads like "no effect"
    # rather than "not measured yet".
    available = [cell for cell in cells if rows_by_slug.get(cell["slug"])]
    if not available:
        return []
    widest = max(cell["trained_rank"] for cell in available)
    comparisons = []
    for cell in available:
        if cell["trained_rank"] == widest:
            continue
        seed = cell["run"]["seed"]
        model = cell["run"]["model"]["key"]
        partner = next(
            (
                other
                for other in available
                if other["trained_rank"] == widest
                and other["run"]["seed"] == seed
                and other["run"]["model"]["key"] == model
            ),
            None,
        )
        if partner is None:
            continue
        mine = {row["key"]: row for row in rows_by_slug.get(cell["slug"], [])}
        theirs = {row["key"]: row for row in rows_by_slug.get(partner["slug"], [])}
        native = mine.get(candidate_key(cell["trained_rank"], 16))
        if native is None:
            continue
        for row in theirs.values():
            if row["budget_units"] != native["budget_units"] or row["rank"] == native["rank"]:
                continue
            if row.get("gsm8k_accuracy") is None or native.get("gsm8k_accuracy") is None:
                continue
            comparisons.append(
                {
                    "model": model,
                    "seed": seed,
                    "budget_units": native["budget_units"],
                    "trained_rank": cell["trained_rank"],
                    "trained_key": native["key"],
                    "trained_accuracy": native["gsm8k_accuracy"],
                    "compressed_from_rank": partner["trained_rank"],
                    "compressed_key": row["key"],
                    "compressed_accuracy": row["gsm8k_accuracy"],
                    "trained_minus_compressed": native["gsm8k_accuracy"]
                    - row["gsm8k_accuracy"],
                }
            )
    return sorted(
        comparisons, key=lambda row: (row["seed"], row["budget_units"], row["trained_rank"])
    )


def report(lock: dict, out: Path) -> dict:
    config = lock["config"]
    rows_by_slug: dict[str, list[dict[str, Any]]] = {}
    rows, missing, diagonals = [], [], []
    for cell in lock["cells"]:
        root = out / cell["slug"]
        if not (root / "complete.json").exists():
            missing.append(cell["slug"])
            continue
        candidates = read_json(root / "candidates.json", [])
        rows_by_slug[cell["slug"]] = candidates
        shared = base_root(out, cell["run"]["model"]["key"], cell["run"]["seed"])
        base = {split: read_json(shared / f"base_{split}.json")["exact_match"] for split in SPLITS}
        raw = {split: read_json(root / f"raw_{split}.json")["exact_match"] for split in SPLITS}
        for row in candidates:
            entry = {
                **row,
                "cell": cell["slug"],
                "arm": cell["arm"],
                "trained_rank": cell["trained_rank"],
                "seed": cell["run"]["seed"],
            }
            for split in config["candidate_splits"]:
                accuracy = row.get(f"{split}_accuracy")
                if accuracy is None:
                    continue
                entry[f"{split}_change_from_base"] = accuracy - base[split]
                entry[f"{split}_change_from_raw"] = accuracy - raw[split]
                entry[f"{split}_change_from_base_ci95"] = paired_interval(
                    root / f"{row['key']}_{split}.jsonl",
                    shared / f"base_{split}.jsonl",
                    config["bootstrap_draws"],
                    config["analysis_seed"],
                )
            entry["base_accuracy"] = base
            entry["raw_accuracy"] = raw
            rows.append(entry)
        for entry in matched_sets(candidates):
            diagonals.append({**entry, "cell": cell["slug"], "arm": cell["arm"]})
    result = {
        "status": "complete" if not missing else "partial",
        "missing": missing,
        "rows": rows,
        "matched_budgets": diagonals,
        "scratch_versus_compressed": scratch_versus_compressed(lock["cells"], rows_by_slug),
        "claim_boundary": (
            "Qwen2.5-7B, MetaMathQA rationales permuted across problems, GSM8K and "
            "GSM-Symbolic, three training seeds. Budgets are matched on nominal "
            "rank x bits; measured files differ by the reported ratio. Every cell in "
            "the grid is tested, so no arm is selected on the test split, but the "
            "intervals are unadjusted for the number of cells."
        ),
    }
    write_json(out / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/matched_bit_budget.yaml"))
    parser.add_argument("--out", type=Path, default=Path(".cache/reports/matched_bit_budget_v1"))
    parser.add_argument("--runs", type=Path, default=Path(".cache/runs"))
    parser.add_argument("--prepared", type=Path, default=Path(".cache/prepared"))
    parser.add_argument(
        "--phase", required=True,
        choices=["prepare", "check", "probe", "sweep", "report"],
    )
    parser.add_argument("--cell", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.phase == "prepare":
        lock = prepare(config, args.out, args.runs, args.prepared)
        print(json.dumps({"cells": len(lock["cells"]),
                          "candidates": sum(len(c["grid"]) for c in lock["cells"]),
                          "lock": str(args.out / "lock.json")}, indent=2))
        return
    lock = read_json(args.out / "lock.json")
    if lock is None or lock["config"] != config:
        raise ValueError("prepare this exact config before running")
    for path, digest in {**lock["implementation"], **lock["inputs"]}.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"{path} changed after the study was locked")
    if args.phase == "check":
        result = check(lock, args.runs)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["passed"] else 1)
    if args.phase == "report":
        print(json.dumps({k: v for k, v in report(lock, args.out).items()
                          if k != "rows"}, indent=2)[:4000])
        return
    if args.cell is None or not 0 <= args.cell < len(lock["cells"]):
        raise ValueError(f"--cell must be in 0-{len(lock['cells']) - 1}")
    cell = lock["cells"][args.cell]
    if args.phase == "probe":
        print(json.dumps(probe(cell, config, args.out, args.runs, args.prepared), indent=2))
        return
    sweep_cell(cell, config, args.out, args.runs, args.prepared)


if __name__ == "__main__":
    main()
