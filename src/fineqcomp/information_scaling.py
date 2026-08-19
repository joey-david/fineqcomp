"""Controlled dataset-information vs adapter-description-length experiment.

The experiment keeps prompts, example counts, labels, model, adapter, and
optimizer fixed while changing only how many independent label bits the task
contains.

``random`` assigns an independent one-of-16 label to every mapping (4 source
bits/mapping). ``structured_pK`` samples K hidden item->label prototype tables
and reuses them periodically across families.  With 16 items/family this gives
at most 64*K independent task bits even as the number of examples grows.

Dataset compressibility is measured with a conditional prequential code: the
pretrained model is shared side information, the first block is coded by the
base model, and each later block is coded by a fresh LoRA trained only on the
preceding prefix.  Adapter complexity is measured independently by the exact
serialized size of reloadable adaptive-MDL adapter files at a rate sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm.auto import tqdm

import fineqcomp.mdl as mdl
import fineqcomp.mdl_fast as mdl_fast
from fineqcomp.adapters import (
    adapter_tensors,
    apply_adapter_tensors,
    restore_trainable_state,
    trainable_state,
)
from fineqcomp.campaign import _adapter, _model
from fineqcomp.config import TrainingSpec, load_campaign
from fineqcomp.data import Example, _read_codebook, _render_prompt, _ticket
from fineqcomp.evaluation import evaluate_synthetic
from fineqcomp.modeling import ModelSession, validate_single_token_labels
from fineqcomp.training import train_adapter


@dataclass(frozen=True)
class Condition:
    name: str
    prototype_count: int | None


@dataclass
class DatasetBundle:
    train_by_mapping: list[list[Example]]
    calibration: list[Example]
    prequential: list[Example]
    test: list[Example]
    label_indices: list[int]
    source_keys: list[tuple[int, int] | tuple[str, int]]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_info_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or int(raw.get("version", 0)) != 1:
        raise ValueError(f"{path}: expected information-scaling config version 1")
    return raw


def _condition(raw: dict[str, Any], name: str) -> Condition:
    for entry in raw["conditions"]:
        if str(entry["name"]) == name:
            prototype = entry.get("prototype_count")
            return Condition(
                name=name,
                prototype_count=int(prototype) if prototype is not None else None,
            )
    raise KeyError(f"unknown condition {name!r}")


def _mapping_label_indices(
    condition: Condition,
    symbols: list[int],
    max_mappings: int,
    items_per_family: int,
) -> tuple[list[int], list[tuple[int, int] | tuple[str, int]]]:
    if condition.prototype_count is None:
        return (
            symbols[:max_mappings],
            [("random", index) for index in range(max_mappings)],
        )

    prototype_count = condition.prototype_count
    if prototype_count < 1:
        raise ValueError("prototype_count must be positive")
    required = prototype_count * items_per_family
    prototypes = symbols[:required]
    labels: list[int] = []
    keys: list[tuple[int, int]] = []
    for mapping in range(max_mappings):
        family, item = divmod(mapping, items_per_family)
        prototype = family % prototype_count
        key = (prototype, item)
        keys.append(key)
        labels.append(prototypes[prototype * items_per_family + item])
    return labels, keys


def _example(
    *,
    mapping: int,
    label_index: int,
    labels: list[str],
    items_per_family: int,
    seed: int,
    split: str,
    instance: int,
) -> Example:
    family, item = divmod(mapping, items_per_family)
    ticket = _ticket(seed, family, item, instance, f"info-{split}")
    render_split = split if split in {"train", "calibration"} else "test"
    return Example(
        example_id=f"{split}-m{mapping}-n{instance}",
        prompt=_render_prompt(family, item, ticket, render_split, instance),
        response=labels[label_index],
        metadata={
            "split": split,
            "mapping": mapping,
            "family": family,
            "item": item,
            "label_index": label_index,
        },
    )


def build_dataset(
    raw: dict[str, Any], condition: Condition, seed: int
) -> DatasetBundle:
    labels = list(map(str, raw["labels"]))
    if len(labels) != 16 or len(set(labels)) != 16:
        raise ValueError("information scaling requires exactly 16 unique labels")
    max_mappings = max(map(int, raw["prefix_mappings"]))
    items = int(raw.get("items_per_family", 16))
    repeats = int(raw.get("repeats_per_mapping", 4))
    if items < 1 or repeats < 1:
        raise ValueError("items_per_family and repeats_per_mapping must be positive")

    needed_symbols = max_mappings
    if condition.prototype_count is not None:
        needed_symbols = max(
            needed_symbols, condition.prototype_count * items
        )
    symbols, _ = _read_codebook(
        {"codebook_dir": str(raw.get("codebook_dir", "codebooks"))},
        needed_symbols,
        seed,
    )
    label_indices, source_keys = _mapping_label_indices(
        condition, symbols, max_mappings, items
    )

    train_by_mapping: list[list[Example]] = []
    calibration: list[Example] = []
    prequential: list[Example] = []
    test: list[Example] = []
    for mapping, label_index in enumerate(label_indices):
        train_by_mapping.append(
            [
                _example(
                    mapping=mapping,
                    label_index=label_index,
                    labels=labels,
                    items_per_family=items,
                    seed=seed,
                    split="train",
                    instance=instance,
                )
                for instance in range(repeats)
            ]
        )
        calibration.append(
            _example(
                mapping=mapping,
                label_index=label_index,
                labels=labels,
                items_per_family=items,
                seed=seed,
                split="calibration",
                instance=repeats,
            )
        )
        prequential.append(
            _example(
                mapping=mapping,
                label_index=label_index,
                labels=labels,
                items_per_family=items,
                seed=seed,
                split="prequential",
                instance=repeats + 1,
            )
        )
        test.append(
            _example(
                mapping=mapping,
                label_index=label_index,
                labels=labels,
                items_per_family=items,
                seed=seed,
                split="test",
                instance=repeats + 2,
            )
        )

    return DatasetBundle(
        train_by_mapping=train_by_mapping,
        calibration=calibration,
        prequential=prequential,
        test=test,
        label_indices=label_indices,
        source_keys=source_keys,
    )


def _flatten_prefix(groups: list[list[Example]], mappings: int) -> list[Example]:
    return [row for group in groups[:mappings] for row in group]


def _source_bits(bundle: DatasetBundle, mappings: int) -> int:
    # Each independently sampled source symbol is one of 16 equiprobable labels.
    return 4 * len(set(bundle.source_keys[:mappings]))


def _label_code_bits(metrics: dict[str, Any]) -> float:
    return float(metrics["label_nll"]) * int(metrics["examples"]) / math.log(2)


def _retained_gain(base: float, raw: float, coded: float) -> float | None:
    gain = raw - base
    if abs(gain) < 1e-12:
        return None
    return (coded - base) / gain


def _adapter_rate_curve(
    *,
    session: ModelSession,
    model_spec: Any,
    labels: list[str],
    examples: list[Example],
    raw_tensors: dict[str, torch.Tensor],
    base_accuracy: float,
    raw_accuracy: float,
    target_rates: list[float],
    retention_target: float,
    out_dir: Path,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Exact reloadable adapter-rate sweep for one learned checkpoint."""
    old_midrise = mdl._midrise_tensor
    mdl._midrise_tensor = mdl_fast._midrise_tensor_fast
    try:
        cache, total_values = mdl_fast._row_candidates_fast(
            raw_tensors, show_progress=False
        )
        points: list[dict[str, Any]] = []
        for target in target_rates:
            assignment, allocation = mdl._allocate(cache, total_values, target)
            slug = f"{target:g}".replace(".", "p")
            adapter_path = out_dir / f"adapter_mdl_{slug}.fqmdl"
            storage = mdl._encode_mdl(
                raw_tensors,
                cache,
                assignment,
                adapter_path,
                allocation,
                show_progress=False,
            )
            _, decoded = mdl.decode_mdl(adapter_path)
            apply_adapter_tensors(session.model, decoded)
            metrics, _ = evaluate_synthetic(
                session.model,
                session.tokenizer,
                examples,
                model_spec,
                labels,
                batch_size=batch_size,
            )
            coded_accuracy = float(metrics["accuracy"])
            retained = _retained_gain(base_accuracy, raw_accuracy, coded_accuracy)
            point = {
                "target_rate": target,
                "description_bits": int(storage["description_bits"]),
                "effective_bits_per_value": float(storage["effective_bits_per_value"]),
                "proxy_bits_per_value": float(storage["proxy_bits_per_value"]),
                "relative_rmse": float(storage["relative_rmse"]),
                "accuracy": coded_accuracy,
                "label_nll": float(metrics["label_nll"]),
                "retained_gain": retained,
                "row_bit_histogram": storage["row_bit_histogram"],
                "value_bit_histogram": storage["value_bit_histogram"],
                "path": str(adapter_path),
            }
            points.append(point)
            apply_adapter_tensors(session.model, raw_tensors)

        eligible = [
            point
            for point in points
            if point["retained_gain"] is not None
            and float(point["retained_gain"]) >= retention_target
        ]
        r_star = (
            min(eligible, key=lambda point: int(point["description_bits"]))
            if eligible
            else None
        )
        return points, r_star
    finally:
        mdl._midrise_tensor = old_midrise
        apply_adapter_tensors(session.model, raw_tensors)


def _training_spec(raw: dict[str, Any]) -> TrainingSpec:
    return TrainingSpec(**raw["training"])


def run_condition(
    *,
    info: dict[str, Any],
    campaign: dict[str, Any],
    condition: Condition,
    seed: int,
    out_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    out_dir = out_root / condition.name / f"seed{seed}"
    result_path = out_dir / "result.json"
    if result_path.is_file() and not force:
        return json.loads(result_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    model_spec = _model(str(info["model"]), campaign)
    adapter_spec = _adapter(str(info["adapter"]), campaign)
    training = _training_spec(info)
    labels = list(map(str, info["labels"]))
    prefixes = sorted(set(map(int, info["prefix_mappings"])))
    if not prefixes or prefixes[0] < 1:
        raise ValueError("prefix_mappings must contain positive integers")
    target_rates = list(map(float, info["mdl_target_rates"]))
    retention_target = float(info.get("retention_target", 0.9))
    batch_size = int(info.get("evaluation_batch_size", 64))

    bundle = build_dataset(info, condition, seed)
    session = ModelSession.load(model_spec)
    try:
        session.attach(adapter_spec, seed)
        validate_single_token_labels(session.tokenizer, labels)
        initial_state = trainable_state(session.model)

        # Base-model code for each block.  The attached LoRA is still exactly at
        # its shared deterministic initialization, so its update is zero.
        block_edges = [0, *prefixes]
        base_blocks: list[dict[str, Any]] = []
        for left, right in zip(block_edges[:-1], block_edges[1:], strict=True):
            restore_trainable_state(session.model, initial_state)
            metrics, _ = evaluate_synthetic(
                session.model,
                session.tokenizer,
                bundle.prequential[left:right],
                model_spec,
                labels,
                batch_size=batch_size,
            )
            base_blocks.append(
                {
                    "left": left,
                    "right": right,
                    "bits": _label_code_bits(metrics),
                    "bits_per_mapping": _label_code_bits(metrics) / (right - left),
                }
            )

        # Conditional prequential code: first block under the base model; every
        # subsequent block under a fresh adapter trained only on the preceding
        # prefix.  Training is reset to the same initialization at every prefix.
        cumulative_preq = float(base_blocks[0]["bits"])
        prequential_rows: list[dict[str, Any]] = [
            {
                "mappings": prefixes[0],
                "source_bits": _source_bits(bundle, prefixes[0]),
                "base_code_bits": float(base_blocks[0]["bits"]),
                "prequential_code_bits": cumulative_preq,
                "prequential_bits_per_mapping": cumulative_preq / prefixes[0],
                "encoder": "base",
            }
        ]
        checkpoints: list[dict[str, Any]] = []

        iterator = tqdm(
            prefixes,
            desc=f"{condition.name} seed{seed}",
            unit="prefix",
            dynamic_ncols=True,
        )
        for prefix_index, prefix in enumerate(iterator):
            iterator.set_postfix_str(f"n={prefix}")
            restore_trainable_state(session.model, initial_state)
            train_rows = _flatten_prefix(bundle.train_by_mapping, prefix)
            calibration_rows = bundle.calibration[:prefix]
            prefix_dir = out_dir / f"n{prefix}"
            prefix_dir.mkdir(parents=True, exist_ok=True)
            train_metrics = train_adapter(
                session.model,
                session.tokenizer,
                train_rows,
                calibration_rows,
                model_spec,
                training,
                seed + prefix,
                prefix_dir / "training.jsonl",
            )
            raw_tensors = adapter_tensors(session.model, adapter_spec.method)
            torch.save(raw_tensors, prefix_dir / "raw_channel.pt")

            restore_trainable_state(session.model, initial_state)
            base_metrics, _ = evaluate_synthetic(
                session.model,
                session.tokenizer,
                bundle.test[:prefix],
                model_spec,
                labels,
                batch_size=batch_size,
            )
            apply_adapter_tensors(session.model, raw_tensors)
            raw_metrics, _ = evaluate_synthetic(
                session.model,
                session.tokenizer,
                bundle.test[:prefix],
                model_spec,
                labels,
                batch_size=batch_size,
            )

            rate_curve, r_star = _adapter_rate_curve(
                session=session,
                model_spec=model_spec,
                labels=labels,
                examples=bundle.test[:prefix],
                raw_tensors=raw_tensors,
                base_accuracy=float(base_metrics["accuracy"]),
                raw_accuracy=float(raw_metrics["accuracy"]),
                target_rates=target_rates,
                retention_target=retention_target,
                out_dir=prefix_dir / "mdl",
                batch_size=batch_size,
            )

            checkpoint = {
                "mappings": prefix,
                "training_rows": len(train_rows),
                "source_bits": _source_bits(bundle, prefix),
                "naive_independent_label_bits": 4 * prefix,
                "base_accuracy": float(base_metrics["accuracy"]),
                "raw_accuracy": float(raw_metrics["accuracy"]),
                "raw_label_nll": float(raw_metrics["label_nll"]),
                "raw_gain": float(raw_metrics["accuracy"])
                - float(base_metrics["accuracy"]),
                "training": train_metrics,
                "rate_curve": rate_curve,
                "r_star": r_star,
                "r_star_description_bits": (
                    int(r_star["description_bits"]) if r_star is not None else None
                ),
                "r_star_effective_bits_per_value": (
                    float(r_star["effective_bits_per_value"])
                    if r_star is not None
                    else None
                ),
            }
            _write_json(prefix_dir / "metrics.json", checkpoint)
            checkpoints.append(checkpoint)

            if prefix_index + 1 < len(prefixes):
                next_prefix = prefixes[prefix_index + 1]
                apply_adapter_tensors(session.model, raw_tensors)
                next_metrics, _ = evaluate_synthetic(
                    session.model,
                    session.tokenizer,
                    bundle.prequential[prefix:next_prefix],
                    model_spec,
                    labels,
                    batch_size=batch_size,
                )
                block_bits = _label_code_bits(next_metrics)
                cumulative_preq += block_bits
                cumulative_base = sum(
                    float(block["bits"])
                    for block in base_blocks[: prefix_index + 2]
                )
                prequential_rows.append(
                    {
                        "mappings": next_prefix,
                        "source_bits": _source_bits(bundle, next_prefix),
                        "base_code_bits": cumulative_base,
                        "prequential_code_bits": cumulative_preq,
                        "prequential_bits_per_mapping": cumulative_preq
                        / next_prefix,
                        "last_block_bits": block_bits,
                        "last_block_bits_per_mapping": block_bits
                        / (next_prefix - prefix),
                        "encoder": f"adapter-trained-on-{prefix}",
                    }
                )

        result = {
            "version": 1,
            "condition": condition.name,
            "prototype_count": condition.prototype_count,
            "seed": seed,
            "model": model_spec.key,
            "adapter": adapter_spec.key,
            "labels": labels,
            "items_per_family": int(info.get("items_per_family", 16)),
            "repeats_per_mapping": int(info.get("repeats_per_mapping", 4)),
            "prefix_mappings": prefixes,
            "retention_target": retention_target,
            "mdl_target_rates": target_rates,
            "prequential": prequential_rows,
            "checkpoints": checkpoints,
            "finished_at": time.time(),
        }
        _write_json(result_path, result)
        return result
    finally:
        try:
            session.unload()
        except Exception:
            pass


def _collect_results(root: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(root.glob("*/seed*/result.json")):
        results.append(json.loads(path.read_text()))
    return results


def aggregate(root: Path) -> dict[str, Any]:
    results = _collect_results(root)
    if not results:
        raise FileNotFoundError(f"no information-scaling results under {root}")

    rows: list[dict[str, Any]] = []
    preq_rows: list[dict[str, Any]] = []
    for result in results:
        common = {
            "condition": result["condition"],
            "prototype_count": result["prototype_count"],
            "seed": result["seed"],
        }
        for checkpoint in result["checkpoints"]:
            rows.append(
                {
                    **common,
                    "mappings": checkpoint["mappings"],
                    "source_bits": checkpoint["source_bits"],
                    "training_rows": checkpoint["training_rows"],
                    "base_accuracy": checkpoint["base_accuracy"],
                    "raw_accuracy": checkpoint["raw_accuracy"],
                    "r_star_description_bits": checkpoint[
                        "r_star_description_bits"
                    ],
                    "r_star_effective_bits_per_value": checkpoint[
                        "r_star_effective_bits_per_value"
                    ],
                }
            )
        for preq in result["prequential"]:
            preq_rows.append({**common, **preq})

    root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (root / "adapter_information.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fields = list(preq_rows[0])
    with (root / "prequential.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(preq_rows)

    # Two deliberately simple diagnostic figures.  Do not fit scaling laws at
    # this stage; first establish that known task bits and prequential code order
    # the conditions in the expected direction.
    import matplotlib.pyplot as plt

    valid = [row for row in rows if row["r_star_description_bits"] is not None]
    if valid:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for condition in sorted({str(row["condition"]) for row in valid}):
            selected = [row for row in valid if row["condition"] == condition]
            selected.sort(key=lambda row: int(row["source_bits"]))
            ax.plot(
                [float(row["source_bits"]) for row in selected],
                [float(row["r_star_description_bits"]) for row in selected],
                marker="o",
                label=condition,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("Known independent task bits")
        ax.set_ylabel("Minimum adapter description bits retaining target gain")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(root / "source_bits_vs_adapter_bits.png", dpi=220)
        plt.close(fig)

    final_preq: list[dict[str, Any]] = []
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in preq_rows:
        by_key.setdefault((str(row["condition"]), int(row["seed"])), []).append(row)
    for group in by_key.values():
        final_preq.append(max(group, key=lambda row: int(row["mappings"])))
    if final_preq:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for row in final_preq:
            ax.scatter(
                float(row["source_bits"]),
                float(row["prequential_code_bits"]),
                label=f"{row['condition']}/s{row['seed']}",
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("Known independent task bits")
        ax.set_ylabel("Conditional prequential code bits")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(root / "source_bits_vs_prequential_bits.png", dpi=220)
        plt.close(fig)

    summary = {
        "results": len(results),
        "adapter_rows": len(rows),
        "prequential_rows": len(preq_rows),
        "adapter_csv": str(root / "adapter_information.csv"),
        "prequential_csv": str(root / "prequential.csv"),
    }
    _write_json(root / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled dataset information vs adapter description length."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/information_scaling.yaml")
    )
    parser.add_argument("--campaign", type=Path, default=Path("configs/campaign.yaml"))
    parser.add_argument("--out", type=Path, default=Path("runs_information_scaling"))
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.aggregate:
        print(json.dumps(aggregate(args.out), indent=2, sort_keys=True))
        return

    info = _load_info_config(args.config)
    campaign = load_campaign(args.campaign)
    conditions = args.conditions or [str(row["name"]) for row in info["conditions"]]
    seeds = args.seeds or list(map(int, info.get("seeds", [11])))
    for seed in seeds:
        for name in conditions:
            result = run_condition(
                info=info,
                campaign=campaign,
                condition=_condition(info, name),
                seed=seed,
                out_root=args.out,
                force=args.force,
            )
            print(
                json.dumps(
                    {
                        "condition": result["condition"],
                        "seed": result["seed"],
                        "result": str(
                            args.out / result["condition"] / f"seed{seed}" / "result.json"
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
