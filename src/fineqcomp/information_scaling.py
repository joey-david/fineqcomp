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
preceding prefix. Adapter complexity is measured independently by the exact
serialized size of reloadable uniform-code adapter files over a dense rate
sweep.
"""

from __future__ import annotations

import argparse
import hashlib
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

from fineqcomp.adapters import (
    adapter_tensors,
    apply_adapter_tensors,
    restore_trainable_state,
    trainable_state,
)
from fineqcomp.campaign import _adapter, _model
from fineqcomp.codec import decode_adapter_tensor_map, encode_tensor_map
from fineqcomp.config import TrainingSpec, load_campaign
from fineqcomp.data import Example
from fineqcomp.evaluation import evaluate_constrained_labels
from fineqcomp.modeling import ModelSession, validate_single_token_labels
from fineqcomp.rstar import r_star
from fineqcomp.training import train_adapter


def _ticket(seed: int, family: int, item: int, instance: int, split: str) -> str:
    payload = f"{seed}:{family}:{item}:{instance}:{split}".encode()
    return hashlib.sha256(payload).hexdigest()[:12].upper()


def _render_prompt(
    family: int, item: int, ticket: str, split: str, variant: int
) -> str:
    family_text = f"F{family:04X}"
    item_text = f"I{item:X}"
    if split == "train" and variant % 2 == 0:
        return (
            "Registry query\n"
            f"Family: {family_text}\nItem: {item_text}\nTicket: {ticket}\nLabel:"
        )
    if split == "train":
        return (
            f"Look up family {family_text}, item {item_text}. "
            f"Request {ticket}. Return its label:"
        )
    if split == "calibration":
        return f"Code request {ticket}: family={family_text}; item={item_text}.\nCode:"
    return (
        "Answer with one registry label.\n"
        f"ticket={ticket} item={item_text} family={family_text}\nAnswer:"
    )


def _read_codebook(
    dataset_cfg: dict[str, Any], required: int, seed: int
) -> tuple[list[int], str]:
    source = Path(dataset_cfg["codebook_dir"]) / f"seed{seed}.hex"
    try:
        packed = bytes.fromhex("".join(source.read_text().split()))
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"invalid codebook asset: {source}") from error
    if len(packed) * 2 < required:
        raise ValueError(f"{source}: has {len(packed) * 2} symbols, needs {required}")
    symbols = [nibble for byte in packed for nibble in (byte >> 4, byte & 0x0F)]
    prefix = packed[: (required + 1) // 2]
    return symbols[:required], hashlib.sha256(prefix).hexdigest()


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


def _bits_saved_per_mapping(
    base_metrics: dict[str, Any], tuned_metrics: dict[str, Any]
) -> float:
    """Response-code bits saved against the shared frozen base."""
    return (
        float(base_metrics["label_nll"]) - float(tuned_metrics["label_nll"])
    ) / math.log(2)


def _rate_codecs(raw: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Read the compact uniform-code list used by the controlled study."""
    configured = raw.get("adapter_codecs")
    if configured is None:
        configured = [
            {"key": f"uniform{int(bits)}", "bits": int(bits)}
            for bits in raw.get("adapter_widths", (1, 2, 3, 4, 8))
        ]
    codecs: list[dict[str, Any]] = []
    keys: set[str] = set()
    for entry in configured:
        codec = {
            "key": str(entry["key"]),
            "bits": int(entry["bits"]),
            "blend": float(entry.get("blend", 0.0)),
        }
        if codec["key"] in keys:
            raise ValueError(f"duplicate adapter codec {codec['key']!r}")
        if codec["bits"] not in {0, 1, 2, 3, 4, 8, 16}:
            raise ValueError(f"unsupported adapter codec width: {codec['bits']}")
        if not 0.0 <= codec["blend"] < 1.0:
            raise ValueError(f"adapter codec {codec['key']}: blend must be in [0, 1)")
        if codec["bits"] == 0 and codec["blend"] <= 0:
            raise ValueError(f"adapter codec {codec['key']}: zero bits needs a blend")
        keys.add(codec["key"])
        codecs.append(codec)
    if not codecs:
        raise ValueError("adapter_codecs must not be empty")
    return tuple(codecs)


def _summarize_rate_curve(
    points: list[dict[str, Any]], raw_saved: float, target: float
) -> dict[str, Any]:
    """Set the utility ceiling from the best decoded candidate.

    Quantization can improve held-out code length by removing overfit. Treating
    the raw adapter as the ceiling then creates retention above one. The
    operational rate-distortion curve instead uses the best member of the
    declared code family, including the uncompressed adapter.
    """
    ceiling = max(
        [
            raw_saved,
            *(float(point["bits_saved_per_mapping"]) for point in points),
        ]
    )
    enriched = [
        {
            **point,
            "retained_gain": (
                float(point["bits_saved_per_mapping"]) / ceiling
                if ceiling > 0
                else None
            ),
        }
        for point in points
    ]
    by_rate = r_star(
        enriched,
        target=target,
        value_key="bits_saved_per_mapping",
        reference=ceiling,
    )
    by_file = r_star(
        enriched,
        target=target,
        rate_key="description_bits",
        value_key="bits_saved_per_mapping",
        reference=ceiling,
    )
    eligible = [
        point
        for point in enriched
        if point["retained_gain"] is not None
        and float(point["retained_gain"]) >= target
    ]
    selected = (
        min(eligible, key=lambda point: int(point["description_bits"]))
        if eligible
        else None
    )
    return {
        "reference": "best_decoded_selection_utility",
        "ceiling_bits_saved_per_mapping": ceiling,
        "raw_retained_gain": raw_saved / ceiling if ceiling > 0 else None,
        "points": enriched,
        "r_star_effective_bits_per_value": by_rate,
        "r_star_description_bits": by_file,
        "selected_codec": selected["codec"] if selected is not None else None,
    }


def _adapter_rate_curve(
    *,
    session: ModelSession,
    model_spec: Any,
    labels: list[str],
    selection_examples: list[Example],
    raw_tensors: dict[str, torch.Tensor],
    base_metrics: dict[str, Any],
    raw_metrics: dict[str, Any],
    codecs: tuple[dict[str, Any], ...],
    retention_target: float,
    out_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    """Exact reloadable adapter-rate sweep for one learned checkpoint.

    Sweeps the zero-free uniform ladder rather than the adaptive MDL allocator.
    On Mistral/MetaMathQA the allocator never beat this ladder at a matched file
    rate and collapsed below it whenever it was allowed to drop rows; see
    results/rmse_mdl_lora_vs_quantized_lora. The ladder is also cheaper, since
    it needs no rate/distortion table and no penalty search.
    """
    try:
        points: list[dict[str, Any]] = []
        for codec in codecs:
            key = str(codec["key"])
            bits = int(codec["bits"])
            blend = float(codec["blend"])
            adapter_path = out_dir / f"adapter_{key}.fqcb"
            storage = encode_tensor_map(
                raw_tensors, adapter_path, bits, blend=blend
            )
            _, decoded = decode_adapter_tensor_map(adapter_path)
            apply_adapter_tensors(session.model, decoded)
            metrics, _ = evaluate_constrained_labels(
                session.model,
                session.tokenizer,
                selection_examples,
                model_spec,
                labels,
                batch_size=batch_size,
            )
            point = {
                "codec": key,
                "bits": bits,
                "blend": blend,
                "description_bits": int(storage["file_bits"]),
                "effective_bits_per_value": float(storage["effective_bits_per_value"]),
                "relative_rmse": float(storage["relative_rmse"]),
                "accuracy": float(metrics["accuracy"]),
                "label_nll": float(metrics["label_nll"]),
                "bits_saved_per_mapping": _bits_saved_per_mapping(
                    base_metrics, metrics
                ),
                "path": str(adapter_path),
            }
        return _summarize_rate_curve(
            points,
            _bits_saved_per_mapping(base_metrics, raw_metrics),
            retention_target,
        )
    finally:
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
    codecs = _rate_codecs(info)
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
            metrics, _ = evaluate_constrained_labels(
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
            base_selection, _ = evaluate_constrained_labels(
                session.model,
                session.tokenizer,
                bundle.calibration[:prefix],
                model_spec,
                labels,
                batch_size=batch_size,
            )
            apply_adapter_tensors(session.model, raw_tensors)
            raw_selection, _ = evaluate_constrained_labels(
                session.model,
                session.tokenizer,
                bundle.calibration[:prefix],
                model_spec,
                labels,
                batch_size=batch_size,
            )

            rate_curve = _adapter_rate_curve(
                session=session,
                model_spec=model_spec,
                labels=labels,
                selection_examples=bundle.calibration[:prefix],
                raw_tensors=raw_tensors,
                base_metrics=base_selection,
                raw_metrics=raw_selection,
                codecs=codecs,
                retention_target=retention_target,
                out_dir=prefix_dir / "codecs",
                batch_size=batch_size,
            )

            restore_trainable_state(session.model, initial_state)
            base_report, _ = evaluate_constrained_labels(
                session.model,
                session.tokenizer,
                bundle.test[:prefix],
                model_spec,
                labels,
                batch_size=batch_size,
            )
            apply_adapter_tensors(session.model, raw_tensors)
            raw_report, _ = evaluate_constrained_labels(
                session.model,
                session.tokenizer,
                bundle.test[:prefix],
                model_spec,
                labels,
                batch_size=batch_size,
            )
            selected_report = None
            selected_key = rate_curve["selected_codec"]
            if selected_key is not None:
                selected_point = next(
                    point
                    for point in rate_curve["points"]
                    if point["codec"] == selected_key
                )
                _, decoded = decode_adapter_tensor_map(selected_point["path"])
                apply_adapter_tensors(session.model, decoded)
                selected_metrics, _ = evaluate_constrained_labels(
                    session.model,
                    session.tokenizer,
                    bundle.test[:prefix],
                    model_spec,
                    labels,
                    batch_size=batch_size,
                )
                selected_report = {
                    "codec": selected_key,
                    "accuracy": float(selected_metrics["accuracy"]),
                    "label_nll": float(selected_metrics["label_nll"]),
                    "bits_saved_per_mapping": _bits_saved_per_mapping(
                        base_report, selected_metrics
                    ),
                }
            apply_adapter_tensors(session.model, raw_tensors)

            checkpoint = {
                "mappings": prefix,
                "training_rows": len(train_rows),
                "source_bits": _source_bits(bundle, prefix),
                "naive_independent_label_bits": 4 * prefix,
                "selection": {
                    "base_accuracy": float(base_selection["accuracy"]),
                    "raw_accuracy": float(raw_selection["accuracy"]),
                    "base_label_nll": float(base_selection["label_nll"]),
                    "raw_label_nll": float(raw_selection["label_nll"]),
                    "raw_bits_saved_per_mapping": _bits_saved_per_mapping(
                        base_selection, raw_selection
                    ),
                },
                "report": {
                    "base_accuracy": float(base_report["accuracy"]),
                    "raw_accuracy": float(raw_report["accuracy"]),
                    "base_label_nll": float(base_report["label_nll"]),
                    "raw_label_nll": float(raw_report["label_nll"]),
                    "raw_bits_saved_per_mapping": _bits_saved_per_mapping(
                        base_report, raw_report
                    ),
                    "selected": selected_report,
                },
                # Kept at the top level for the aggregate table.
                "base_accuracy": float(base_report["accuracy"]),
                "raw_accuracy": float(raw_report["accuracy"]),
                "raw_label_nll": float(raw_report["label_nll"]),
                "raw_gain": float(raw_report["accuracy"])
                - float(base_report["accuracy"]),
                "training": train_metrics,
                "rate_curve": rate_curve["points"],
                "rate_reference": {
                    key: value for key, value in rate_curve.items() if key != "points"
                },
                "r_star_description_bits": rate_curve[
                    "r_star_description_bits"
                ]["r_star"],
                "r_star_effective_bits_per_value": rate_curve[
                    "r_star_effective_bits_per_value"
                ]["r_star"],
            }
            _write_json(prefix_dir / "metrics.json", checkpoint)
            checkpoints.append(checkpoint)

            if prefix_index + 1 < len(prefixes):
                next_prefix = prefixes[prefix_index + 1]
                apply_adapter_tensors(session.model, raw_tensors)
                next_metrics, _ = evaluate_constrained_labels(
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
            "adapter_codecs": list(codecs),
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
