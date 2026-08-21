"""Does an adapter's description length track the information in its task?

The natural campaign in `results/adapter_bits_track_unique_data` found R*(0.90)
rising with the number of distinct training rows at fixed compute. Distinct
rows are a proxy: nothing in that run measured how much information the rows
carried. This study swaps the proxy for a task whose information content is
known exactly, and asks whether a *measured* code length recovers it.

Every prompt names a family and an item and asks for one of sixteen
single-token labels. Rows, prompts, model, adapter, optimizer, and optimizer
updates are identical across conditions. Only the labels change:

    constant   one label for every mapping              4 source bits
    pK         K hidden 16-item prototype tables        64*K source bits
    random     an independent label per mapping         4*mappings source bits

Each cell produces three quantities:

    source bits       known by construction
    prequential bits  a conditional code for the labels, with the frozen base
                      as side information and every block coded by an adapter
                      trained only on the blocks before it
    R*                the smallest adapter file that still reproduces the
                      taught map at the retention target

R* against source bits is the claim. The prequential code is what makes the
claim portable: if it recovers the known source bits here, the same measurement
can be run on a natural dataset, where the source bits are unknown.

Three deliberate choices, each fixing something the first pass got wrong.

The label code is a *mixture*, not the model's raw distribution. A model that
is confidently wrong on an unseen mapping costs an unbounded number of bits
under its own probabilities, so the code length ends up set by the numerical
floor in the scorer rather than by the data. Mixing a uniform 1/16 in at weight
`MIXTURE_WEIGHT` caps the cost per symbol at log2(16/weight) bits and adds at
most log2(1/(1-weight)) when the model is right. This is fixed before the run,
not tuned to it.

Optimizer updates are held fixed across prefixes, so a longer prefix does not
also buy more gradient steps. Epochs are derived, and a prefix that cannot hit
the update budget exactly is an error rather than a rounding.

R* is undefined, and reported as undefined, unless the raw adapter clears an
absolute learning gate. A learner that saved a hundredth of a bit still has a
well-formed retention curve, and every rung of it will look like it retains
ninety percent of nothing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import yaml

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


# The label alphabet, and the pre-registered weight on the uniform component of
# the coding distribution. Both are fixed here rather than in the config so a
# reported code length cannot be moved by editing YAML after seeing a result.
ALPHABET = 16
MIXTURE_WEIGHT = 1 / 16
# Worst case per symbol, in bits: log2(ALPHABET / MIXTURE_WEIGHT).
MAX_SYMBOL_BITS = math.log2(ALPHABET / MIXTURE_WEIGHT)


def symbol_bits(target_probability: float) -> float:
    """Code length for one label under the pre-registered mixture."""
    mixed = (
        1 - MIXTURE_WEIGHT
    ) * target_probability + MIXTURE_WEIGHT / ALPHABET
    return -math.log2(mixed)


# ---------------------------------------------------------------- the dataset


@dataclass(frozen=True)
class Condition:
    """One label-generating rule. Prompts are identical across conditions."""

    name: str
    # None draws a fresh label for every mapping; an integer reuses that many
    # hidden prototype tables; `constant` gives every mapping the same label.
    prototype_count: int | None = None
    constant: bool = False
    # Name the prototype in the prompt. The source bits are unchanged; what
    # changes is whether the learner has to discover the sharing rule itself.
    reveal_prototype: bool = False


@dataclass
class Dataset:
    examples: list[Example]
    label_indices: list[int]
    source_keys: list[tuple[Any, ...]]
    codebook_digest: str


def _read_codebook(directory: Path, seed: int, required: int) -> tuple[list[int], str]:
    """Read 4-bit symbols, and the digest of exactly the prefix that is used."""
    source = directory / f"seed{seed}.hex"
    try:
        packed = bytes.fromhex("".join(source.read_text().split()))
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"invalid codebook asset: {source}") from error
    if len(packed) * 2 < required:
        raise ValueError(f"{source}: has {len(packed) * 2} symbols, needs {required}")
    symbols = [nibble for byte in packed for nibble in (byte >> 4, byte & 0x0F)]
    digest = hashlib.sha256(packed[: (required + 1) // 2]).hexdigest()[:16]
    return symbols[:required], digest


def _mapping_labels(
    condition: Condition, symbols: list[int], mappings: int, items_per_family: int
) -> tuple[list[int], list[tuple[Any, ...]], list[int | None]]:
    """Label index, source key, and prototype id for every mapping.

    The source key names the sampled symbol a mapping's label comes from, so
    counting distinct keys counts independent draws and nothing else.
    """
    if condition.constant:
        return (
            [symbols[0]] * mappings,
            [("constant",)] * mappings,
            [None] * mappings,
        )
    if condition.prototype_count is None:
        return (
            symbols[:mappings],
            [("random", index) for index in range(mappings)],
            [None] * mappings,
        )
    count = condition.prototype_count
    if count < 1:
        raise ValueError("prototype_count must be positive")
    table = symbols[: count * items_per_family]
    labels: list[int] = []
    keys: list[tuple[Any, ...]] = []
    prototypes: list[int | None] = []
    for mapping in range(mappings):
        family, item = divmod(mapping, items_per_family)
        prototype = family % count
        labels.append(table[prototype * items_per_family + item])
        keys.append((prototype, item))
        prototypes.append(prototype)
    return labels, keys, prototypes


PROMPT_LAYOUTS = ("fields", "key_last")


def _prompt(
    family: int, item: int, prototype: int | None, layout: str = "fields"
) -> str:
    """One canonical template for every split, condition and mapping.

    The first pass gave each split its own wording and a random per-row ticket,
    which turned a memory measurement into a test of transfer across surface
    forms and made the hardest condition the one with least to transfer from.
    The table line is always present so revealing a prototype changes one token
    and nothing else about the prompt.

    `key_last` puts the discriminative digits next to the answer position. In
    the `fields` layout the base model's output barely moves with the key --
    across 512 prompts its label probabilities shift by a few percent -- so an
    adapter has to build the routing from the key to the readout before it can
    store anything keyed on it. Shortening that distance is the cheapest test
    of whether the binding, rather than the storage, is what fails.
    """
    table = "T?" if prototype is None else f"T{prototype:02d}"
    if layout == "key_last":
        return (
            "Registry lookup.\n"
            f"Table: {table}\n"
            f"F{family:04d} I{item:02d} ="
        )
    if layout != "fields":
        raise ValueError(f"prompt layout must be one of {PROMPT_LAYOUTS}")
    return (
        "Registry lookup.\n"
        f"Family: F{family:04d}\n"
        f"Item: I{item:02d}\n"
        f"Table: {table}\n"
        "Label:"
    )


def build_dataset(raw: dict[str, Any], condition: Condition, seed: int) -> Dataset:
    """One example per mapping. Mappings are the source symbols."""
    labels = list(map(str, raw["labels"]))
    if len(labels) != ALPHABET or len(set(labels)) != ALPHABET:
        raise ValueError(f"this study needs exactly {ALPHABET} distinct labels")
    mappings = int(raw["mappings"])
    items = int(raw.get("items_per_family", 16))
    if mappings < 1 or items < 1:
        raise ValueError("mappings and items_per_family must be positive")
    required = mappings
    if condition.prototype_count is not None:
        required = max(required, condition.prototype_count * items)
    layout = str(raw.get("prompt_layout", "fields"))
    if layout not in PROMPT_LAYOUTS:
        raise ValueError(f"prompt layout must be one of {PROMPT_LAYOUTS}")
    symbols, digest = _read_codebook(
        Path(raw.get("codebook_dir", "codebooks")), seed, required
    )
    label_indices, keys, prototypes = _mapping_labels(
        condition, symbols, mappings, items
    )

    examples = []
    for mapping, label_index in enumerate(label_indices):
        family, item = divmod(mapping, items)
        shown = prototypes[mapping] if condition.reveal_prototype else None
        examples.append(
            Example(
                example_id=f"m{mapping:05d}",
                prompt=_prompt(family, item, shown, layout),
                response=labels[label_index],
                metadata={
                    "mapping": mapping,
                    "family": family,
                    "item": item,
                    "label_index": label_index,
                },
            )
        )
    return Dataset(
        examples=examples,
        label_indices=label_indices,
        source_keys=keys,
        codebook_digest=digest,
    )


def source_bits(dataset: Dataset, mappings: int) -> int:
    """Independent 4-bit draws needed to specify the first `mappings` labels."""
    return 4 * len(set(dataset.source_keys[:mappings]))


def epochs_for(updates: int, rows: int, batch: int) -> int:
    """Epochs that spend exactly `updates` optimizer steps on `rows` rows.

    Every prefix must get the same number of updates, otherwise a longer prefix
    buys more gradient steps as well as more information and the two cannot be
    told apart. An inexact budget is an error, not a rounding.
    """
    per_epoch, remainder = divmod(rows, batch)
    if remainder or per_epoch < 1:
        raise ValueError(f"{rows} rows do not divide into batches of {batch}")
    epochs, remainder = divmod(updates, per_epoch)
    if remainder or epochs < 1:
        raise ValueError(
            f"{updates} updates are not reachable from {rows} rows at batch {batch}"
        )
    return epochs


# ---------------------------------------------------------------- measurement


@dataclass
class Scores:
    """Per-example code length and correctness, so any slice can be summed."""

    bits: list[float]
    unmixed_bits: list[float]
    correct: list[int]

    def summary(self, left: int = 0, right: int | None = None) -> dict[str, Any]:
        stop = len(self.bits) if right is None else right
        count = max(stop - left, 1)
        return {
            "examples": stop - left,
            "accuracy": sum(self.correct[left:stop]) / count,
            "code_bits": sum(self.bits[left:stop]),
            "code_bits_per_mapping": sum(self.bits[left:stop]) / count,
            "unmixed_code_bits": sum(self.unmixed_bits[left:stop]),
        }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def score(
    session: ModelSession,
    model_spec: Any,
    examples: Sequence[Example],
    labels: list[str],
    batch_size: int,
    predictions_path: Path | None = None,
) -> Scores:
    """Score labels and keep every per-example probability.

    The previous version of this study discarded its predictions, so ten
    H100-hours produced aggregates that could not be recalibrated, re-coded, or
    broken down by family. Writing them costs a few megabytes.
    """
    if not examples:
        raise ValueError("nothing to score")
    _, predictions = evaluate_constrained_labels(
        session.model,
        session.tokenizer,
        list(examples),
        model_spec,
        labels,
        batch_size=batch_size,
    )
    if predictions_path is not None:
        _write_jsonl(predictions_path, predictions)
    probabilities = [float(row["target_probability"]) for row in predictions]
    return Scores(
        bits=[symbol_bits(value) for value in probabilities],
        unmixed_bits=[-math.log2(max(value, 1e-12)) for value in probabilities],
        correct=[int(row["correct"]) for row in predictions],
    )


def rate_codecs(raw: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    codecs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in raw["adapter_codecs"]:
        codec = {
            "key": str(entry["key"]),
            "bits": int(entry["bits"]),
            "blend": float(entry.get("blend", 0.0)),
        }
        if codec["key"] in seen:
            raise ValueError(f"duplicate adapter codec {codec['key']!r}")
        if codec["bits"] not in {0, 1, 2, 3, 4, 8, 16}:
            raise ValueError(f"unsupported adapter codec width: {codec['bits']}")
        if not 0.0 <= codec["blend"] < 1.0:
            raise ValueError(f"adapter codec {codec['key']}: blend must be in [0, 1)")
        if codec["bits"] == 0 and codec["blend"] <= 0:
            raise ValueError(f"adapter codec {codec['key']}: zero bits needs a blend")
        seen.add(codec["key"])
        codecs.append(codec)
    if not codecs:
        raise ValueError("adapter_codecs must not be empty")
    return tuple(codecs)


def summarize_rate_curve(
    points: list[dict[str, Any]],
    raw_saved: float,
    target: float,
    gate: float,
) -> dict[str, Any]:
    """R* against the raw adapter's own gain, with the learning gate applied.

    The reference is the raw adapter, not the best point on the curve. Coded
    adapters can beat the raw one when quantization strips overfit, and that is
    a finding worth keeping visible rather than a nuisance to normalise away;
    `best_decoded_bits_saved_per_mapping` records it.

    Below the gate there is no gain to retain and every rung retains ninety
    percent of nothing, so R* is reported as undefined with its reason.
    """
    measured = [float(point["bits_saved_per_mapping"]) for point in points]
    best = max([raw_saved, *measured]) if measured else raw_saved
    enriched = [
        {
            **point,
            "retained_gain": (
                float(point["bits_saved_per_mapping"]) / raw_saved
                if raw_saved > 0
                else None
            ),
        }
        for point in points
    ]
    if raw_saved < gate:
        blocked = {
            "target": target,
            "r_star": None,
            "reason": f"raw gain {raw_saved:.4f} below the {gate} bit gate",
        }
        by_rate, by_file = dict(blocked), dict(blocked)
    else:
        by_rate = r_star(
            enriched,
            target=target,
            value_key="bits_saved_per_mapping",
            reference=raw_saved,
        )
        by_file = r_star(
            enriched,
            target=target,
            rate_key="description_bits",
            value_key="bits_saved_per_mapping",
            reference=raw_saved,
        )
    eligible = [
        point
        for point in enriched
        if point["retained_gain"] is not None
        and float(point["retained_gain"]) >= target
    ]
    selected = (
        min(eligible, key=lambda point: int(point["description_bits"]))
        if eligible and raw_saved >= gate
        else None
    )
    return {
        "reference": "raw_adapter",
        "learning_gate_bits_per_mapping": gate,
        "learning_gate_passed": raw_saved >= gate,
        "raw_bits_saved_per_mapping": raw_saved,
        "best_decoded_bits_saved_per_mapping": best,
        "points": enriched,
        "r_star_effective_bits_per_value": by_rate,
        "r_star_description_bits": by_file,
        "selected_codec": selected["codec"] if selected is not None else None,
    }


def _rate_curve(
    *,
    session: ModelSession,
    model_spec: Any,
    labels: list[str],
    examples: Sequence[Example],
    raw_tensors: dict[str, torch.Tensor],
    base_bits_per_mapping: float,
    base_accuracy: float,
    raw_saved: float,
    codecs: tuple[dict[str, Any], ...],
    target: float,
    gate: float,
    out_dir: Path,
    batch_size: int,
    keep_files: bool,
) -> dict[str, Any]:
    """Encode, decode, and score the whole ladder on the taught mappings.

    Sweeps the uniform ladder rather than the adaptive MDL allocator: on
    Mistral/MetaMathQA the allocator never beat this ladder at a matched file
    rate, and it collapsed whenever it was allowed to drop rows. See
    results/rmse_mdl_lora_vs_quantized_lora.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    points: list[dict[str, Any]] = [
        {
            "codec": "none",
            "bits": 0,
            "blend": 0.0,
            "description_bits": 0,
            "effective_bits_per_value": 0.0,
            "relative_rmse": 1.0,
            "accuracy": base_accuracy,
            "code_bits_per_mapping": base_bits_per_mapping,
            "bits_saved_per_mapping": 0.0,
        }
    ]
    try:
        for codec in codecs:
            path = out_dir / f"adapter_{codec['key']}.fqcb"
            storage = encode_tensor_map(
                raw_tensors, path, codec["bits"], blend=codec["blend"]
            )
            _, decoded = decode_adapter_tensor_map(path)
            apply_adapter_tensors(session.model, decoded)
            scored = score(
                session, model_spec, examples, labels, batch_size
            ).summary()
            points.append(
                {
                    "codec": codec["key"],
                    "bits": codec["bits"],
                    "blend": codec["blend"],
                    "description_bits": int(storage["file_bits"]),
                    "effective_bits_per_value": float(
                        storage["effective_bits_per_value"]
                    ),
                    "relative_rmse": float(storage["relative_rmse"]),
                    "accuracy": scored["accuracy"],
                    "code_bits_per_mapping": scored["code_bits_per_mapping"],
                    "bits_saved_per_mapping": base_bits_per_mapping
                    - scored["code_bits_per_mapping"],
                }
            )
    finally:
        apply_adapter_tensors(session.model, raw_tensors)
    curve = summarize_rate_curve(points, raw_saved, target, gate)
    if not keep_files:
        # Every coded file is a deterministic function of the saved raw
        # checkpoint, so only the selected one is worth keeping on a shared
        # filesystem.
        for codec in codecs:
            if codec["key"] != curve["selected_codec"]:
                (out_dir / f"adapter_{codec['key']}.fqcb").unlink(missing_ok=True)
    return curve


# ------------------------------------------------------------------ one cell


@dataclass(frozen=True)
class Cell:
    study: str
    condition: str
    adapter: str
    seed: int
    prefixes: tuple[int, ...]

    @property
    def slug(self) -> str:
        return f"{self.study}/{self.condition}/{self.adapter}/seed{self.seed}"


def _condition(raw: dict[str, Any], name: str) -> Condition:
    for entry in raw["conditions"]:
        if str(entry["name"]) == name:
            prototype = entry.get("prototype_count")
            return Condition(
                name=name,
                prototype_count=int(prototype) if prototype is not None else None,
                constant=bool(entry.get("constant", False)),
                reveal_prototype=bool(entry.get("reveal_prototype", False)),
            )
    raise KeyError(f"unknown condition {name!r}")


def run_cell(
    *,
    info: dict[str, Any],
    campaign: dict[str, Any],
    session: ModelSession,
    cell: Cell,
    out_root: Path,
    force: bool = False,
    keep_codecs: bool = False,
) -> dict[str, Any]:
    out_dir = out_root / cell.study / cell.condition / cell.adapter / f"seed{cell.seed}"
    result_path = out_dir / "result.json"
    if result_path.is_file() and not force:
        return json.loads(result_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    condition = _condition(info, cell.condition)
    model_spec = session.spec
    adapter_spec = _adapter(cell.adapter, campaign)
    training = TrainingSpec(**info["training"])
    labels = list(map(str, info["labels"]))
    codecs = rate_codecs(info)
    target = float(info.get("retention_target", 0.90))
    gate = float(info["learning_gate_bits_per_mapping"])
    updates = int(info["optimizer_updates"])
    batch_size = int(info.get("evaluation_batch_size", 128))
    mappings = int(info["mappings"])
    prefixes = sorted(set(cell.prefixes))
    if prefixes[-1] >= mappings:
        raise ValueError("the largest prefix must leave a block of unseen mappings")

    dataset = build_dataset(info, condition, cell.seed)
    started_at = time.perf_counter()
    session.attach(adapter_spec, cell.seed)
    try:
        validate_single_token_labels(session.tokenizer, labels)
        initial_state = trainable_state(session.model)

        # The frozen base is the shared side information. Score it once over
        # every mapping; each block and prefix is then a slice of that.
        base = score(
            session,
            model_spec,
            dataset.examples,
            labels,
            batch_size,
            out_dir / "predictions" / "base.jsonl",
        )

        block_edges = [0, *prefixes, mappings]
        blocks = [
            {
                "left": left,
                "right": right,
                "encoder": "base" if index == 0 else f"adapter-trained-on-{left}",
            }
            for index, (left, right) in enumerate(
                zip(block_edges[:-1], block_edges[1:], strict=True)
            )
        ]
        blocks[0]["bits"] = base.summary(0, prefixes[0])["code_bits"]

        checkpoints: list[dict[str, Any]] = []
        for index, prefix in enumerate(prefixes):
            prefix_dir = out_dir / f"n{prefix}"
            prefix_dir.mkdir(parents=True, exist_ok=True)
            taught = dataset.examples[:prefix]
            epochs = epochs_for(updates, prefix, training.effective_batch_size)

            restore_trainable_state(session.model, initial_state)
            train_metrics = train_adapter(
                session.model,
                session.tokenizer,
                taught,
                taught,
                model_spec,
                replace(training, epochs=epochs),
                cell.seed * 1000 + prefix,
                prefix_dir / "training.jsonl",
            )
            if int(train_metrics["optimizer_updates"]) != updates:
                raise RuntimeError(
                    f"prefix {prefix} spent {train_metrics['optimizer_updates']}"
                    f" updates, not {updates}"
                )
            raw_tensors = adapter_tensors(session.model, adapter_spec.method)
            torch.save(
                {name: value.half() for name, value in raw_tensors.items()},
                prefix_dir / "raw_channel.pt",
            )

            taught_scores = score(
                session,
                model_spec,
                taught,
                labels,
                batch_size,
                prefix_dir / "predictions_taught.jsonl",
            ).summary()
            base_taught = base.summary(0, prefix)
            raw_saved = (
                base_taught["code_bits_per_mapping"]
                - taught_scores["code_bits_per_mapping"]
            )

            # The block this checkpoint encodes for the prequential code: the
            # mappings it has never seen. No future block trains or selects it.
            left, right = prefix, block_edges[index + 2]
            unseen = score(
                session,
                model_spec,
                dataset.examples[left:right],
                labels,
                batch_size,
                prefix_dir / "predictions_unseen.jsonl",
            ).summary()
            blocks[index + 1]["bits"] = unseen["code_bits"]

            curve = _rate_curve(
                session=session,
                model_spec=model_spec,
                labels=labels,
                examples=taught,
                raw_tensors=raw_tensors,
                base_bits_per_mapping=base_taught["code_bits_per_mapping"],
                base_accuracy=base_taught["accuracy"],
                raw_saved=raw_saved,
                codecs=codecs,
                target=target,
                gate=gate,
                out_dir=prefix_dir / "codecs",
                batch_size=batch_size,
                keep_files=keep_codecs,
            )
            checkpoint = {
                "mappings": prefix,
                "source_bits": source_bits(dataset, prefix),
                "epochs": epochs,
                "optimizer_updates": updates,
                "taught": {"base": base_taught, "raw": taught_scores},
                "unseen": {
                    "left": left,
                    "right": right,
                    "base": base.summary(left, right),
                    "raw": unseen,
                },
                "raw_bits_saved_per_mapping": raw_saved,
                "training": train_metrics,
                "rate_curve": curve["points"],
                "rate_summary": {
                    key: value for key, value in curve.items() if key != "points"
                },
            }
            _write_json(prefix_dir / "metrics.json", checkpoint)
            checkpoints.append(checkpoint)

        cumulative = 0.0
        prequential: list[dict[str, Any]] = []
        for block in blocks:
            cumulative += float(block["bits"])
            span = block["right"] - block["left"]
            prequential.append(
                {
                    **block,
                    "bits_per_mapping": float(block["bits"]) / span,
                    "cumulative_code_bits": cumulative,
                    "cumulative_base_code_bits": base.summary(0, block["right"])[
                        "code_bits"
                    ],
                    "source_bits": source_bits(dataset, block["right"]),
                }
            )

        result = {
            "version": 2,
            "study": cell.study,
            "condition": cell.condition,
            "prototype_count": condition.prototype_count,
            "constant": condition.constant,
            "reveal_prototype": condition.reveal_prototype,
            "seed": cell.seed,
            "model": model_spec.key,
            "adapter": adapter_spec.key,
            "adapter_rank": checkpoints[0]["training"]["adapter_rank"],
            "trainable_parameters": checkpoints[0]["training"][
                "trainable_parameters"
            ],
            "labels": labels,
            "codebook_digest": dataset.codebook_digest,
            "mappings": mappings,
            "items_per_family": int(info.get("items_per_family", 16)),
            "prompt_layout": str(info.get("prompt_layout", "fields")),
            "learning_rate": training.learning_rate,
            "prefixes": prefixes,
            "optimizer_updates": updates,
            "retention_target": target,
            "learning_gate_bits_per_mapping": gate,
            "mixture_weight": MIXTURE_WEIGHT,
            "max_symbol_bits": MAX_SYMBOL_BITS,
            "adapter_codecs": list(codecs),
            "prequential": prequential,
            "checkpoints": checkpoints,
            "elapsed_seconds": time.perf_counter() - started_at,
            "finished_at": time.time(),
        }
        _write_json(result_path, result)
        return result
    finally:
        session.unload()


# -------------------------------------------------------------------- the grid


def expand_grid(info: dict[str, Any]) -> list[Cell]:
    """Every cell the config asks for, in a stable order."""
    seeds = list(map(int, info["seeds"]))
    known = {str(entry["name"]) for entry in info["conditions"]}
    cells: list[Cell] = []
    for study in info["grid"]:
        name = str(study["study"])
        prefixes = tuple(sorted(set(map(int, study["prefixes"]))))
        for condition in study["conditions"]:
            if str(condition) not in known:
                raise KeyError(f"study {name}: unknown condition {condition!r}")
            for adapter in study["adapters"]:
                for seed in seeds:
                    cells.append(
                        Cell(
                            study=name,
                            condition=str(condition),
                            adapter=str(adapter),
                            seed=seed,
                            prefixes=prefixes,
                        )
                    )
    slugs = [cell.slug for cell in cells]
    if len(slugs) != len(set(slugs)):
        raise ValueError("the grid produced duplicate cells")
    return cells


def shard(cells: list[Cell], index: int, count: int) -> list[Cell]:
    """Stripe the grid so every shard gets a mix of long and short cells."""
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"shard {index} is outside 0-{count - 1}")
    return [cell for position, cell in enumerate(cells) if position % count == index]


# ------------------------------------------------------------------ aggregate


def _collect(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text())
        for path in sorted(root.glob("*/*/*/seed*/result.json"))
    ]


def _cell_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for checkpoint in result["checkpoints"]:
            summary = checkpoint["rate_summary"]
            rows.append(
                {
                    "study": result["study"],
                    "condition": result["condition"],
                    "adapter": result["adapter"],
                    "adapter_rank": result["adapter_rank"],
                    "trainable_parameters": result["trainable_parameters"],
                    "seed": result["seed"],
                    "mappings": checkpoint["mappings"],
                    "source_bits": checkpoint["source_bits"],
                    "epochs": checkpoint["epochs"],
                    "base_accuracy": checkpoint["taught"]["base"]["accuracy"],
                    "raw_accuracy": checkpoint["taught"]["raw"]["accuracy"],
                    "unseen_accuracy": checkpoint["unseen"]["raw"]["accuracy"],
                    "unseen_bits_per_mapping": checkpoint["unseen"]["raw"][
                        "code_bits"
                    ]
                    / max(
                        checkpoint["unseen"]["right"] - checkpoint["unseen"]["left"], 1
                    ),
                    "raw_bits_saved_per_mapping": checkpoint[
                        "raw_bits_saved_per_mapping"
                    ],
                    "learning_gate_passed": summary["learning_gate_passed"],
                    "best_decoded_bits_saved_per_mapping": summary[
                        "best_decoded_bits_saved_per_mapping"
                    ],
                    "r_star_bits_per_value": summary[
                        "r_star_effective_bits_per_value"
                    ]["r_star"],
                    "r_star_bracketed": summary["r_star_effective_bits_per_value"].get(
                        "bracketed"
                    ),
                    "r_star_description_bits": summary["r_star_description_bits"][
                        "r_star"
                    ],
                    "selected_codec": summary["selected_codec"],
                }
            )
    return rows


def _prequential_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for block in result["prequential"]:
            rows.append(
                {
                    "study": result["study"],
                    "condition": result["condition"],
                    "adapter": result["adapter"],
                    "seed": result["seed"],
                    "left": block["left"],
                    "right": block["right"],
                    "encoder": block["encoder"],
                    "source_bits": block["source_bits"],
                    "block_bits": block["bits"],
                    "bits_per_mapping": block["bits_per_mapping"],
                    "cumulative_code_bits": block["cumulative_code_bits"],
                    "cumulative_base_code_bits": block["cumulative_base_code_bits"],
                }
            )
    return rows


def _spread(values: list[float]) -> dict[str, Any]:
    clean = [value for value in values if value is not None]
    if not clean:
        return {"n": 0}
    return {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "low": min(clean),
        "high": max(clean),
    }


def check_gates(
    cell_rows: list[dict[str, Any]], preq_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """The pre-registered checks. Each is pass/fail; a failure is a result."""
    learned = [row for row in cell_rows if row["study"] == "main"]
    gates: dict[str, Any] = {}

    gates["G1_learner_reaches_the_taught_map"] = {
        "rule": "raw accuracy on the taught mappings >= 0.95 in every main cell",
        "worst": min((row["raw_accuracy"] for row in learned), default=None),
        "failures": sorted(
            f"{row['condition']}/n{row['mappings']}/s{row['seed']}"
            for row in learned
            if row["raw_accuracy"] < 0.95
        ),
    }
    gates["G1_learner_reaches_the_taught_map"]["passed"] = not gates[
        "G1_learner_reaches_the_taught_map"
    ]["failures"]

    over_base = [
        f"{row['condition']}/s{row['seed']}/{row['right']}"
        for row in preq_rows
        if row["cumulative_code_bits"] > row["cumulative_base_code_bits"]
    ]
    gates["G2_the_code_never_costs_more_than_the_base"] = {
        "rule": "cumulative prequential bits <= cumulative base bits, always",
        "failures": sorted(over_base),
        "passed": not over_base,
    }

    tail = {}
    for row in preq_rows:
        if row["encoder"] == "base":
            continue
        tail.setdefault(row["condition"], []).append(row["bits_per_mapping"])
    gates["G3_the_code_separates_learnable_from_random"] = {
        "rule": (
            "on unseen mappings the random condition costs near 4 bits each"
            " while constant and p1 cost far less"
        ),
        "bits_per_unseen_mapping": {
            name: _spread(values) for name, values in sorted(tail.items())
        },
    }

    null = [
        row
        for row in cell_rows
        if row["study"] == "main"
        and row["condition"] in {"p4", "p8", "p16", "random"}
        and row["mappings"] == 64
    ]
    gates["G4_equal_information_gives_equal_R_star"] = {
        "rule": (
            "at 64 mappings p4, p8, p16 and random are the same task, so their"
            " R* must agree within seed spread"
        ),
        "source_bits": sorted({row["source_bits"] for row in null}),
        "by_condition": {
            name: _spread(
                [row["r_star_bits_per_value"] for row in null if row["condition"] == name]
            )
            for name in sorted({row["condition"] for row in null})
        },
    }

    largest = max((row["mappings"] for row in learned), default=0)
    final = [row for row in learned if row["mappings"] == largest]
    by_condition: dict[str, list[float]] = {}
    order: dict[str, int] = {}
    for row in final:
        by_condition.setdefault(row["condition"], []).append(
            row["r_star_bits_per_value"]
        )
        order[row["condition"]] = row["source_bits"]
    ranked = sorted(order, key=lambda name: order[name])
    means = [
        statistics.fmean([v for v in by_condition[name] if v is not None])
        if any(v is not None for v in by_condition[name])
        else None
        for name in ranked
    ]
    monotone = all(
        left is not None and right is not None and left <= right
        for left, right in zip(means[:-1], means[1:])
    )
    gates["G5_R_star_rises_with_source_bits"] = {
        "rule": (
            "at the largest prefix R* is monotone in source bits and the"
            " extreme conditions do not overlap across seeds"
        ),
        "mappings": largest,
        "by_condition": {
            name: {"source_bits": order[name], **_spread(by_condition[name])}
            for name in ranked
        },
        "monotone": monotone,
    }
    if len(ranked) >= 2:
        low = _spread(by_condition[ranked[0]])
        high = _spread(by_condition[ranked[-1]])
        gates["G5_R_star_rises_with_source_bits"]["extremes_separated"] = bool(
            low.get("n") and high.get("n") and low["high"] < high["low"]
        )
    return gates


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# One ordinal ramp, reused from results/adapter_bits_track_unique_data, where
# it passed the sequential palette checks. Rank is an ordered quantity.
RANK_COLOURS = ("#86b6ef", "#2a78d6", "#0d366b")


def _figures(root: Path, cell_rows: list[dict[str, Any]], preq: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    ranks = sorted({row["adapter_rank"] for row in cell_rows})
    largest = max((row["mappings"] for row in cell_rows), default=0)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    left = axes[0]
    for position, rank in enumerate(ranks):
        selected = [
            row
            for row in cell_rows
            if row["adapter_rank"] == rank
            and row["mappings"] == largest
            # A log axis cannot show the empty-adapter anchor at zero bits.
            and (row["r_star_description_bits"] or 0) > 0
        ]
        grouped: dict[int, list[float]] = {}
        for row in selected:
            grouped.setdefault(row["source_bits"], []).append(
                row["r_star_description_bits"]
            )
        if not grouped:
            continue
        xs = sorted(grouped)
        left.plot(
            xs,
            [statistics.fmean(grouped[x]) for x in xs],
            marker="o",
            color=RANK_COLOURS[position % len(RANK_COLOURS)],
            label=f"rank {rank}",
        )
        left.fill_between(
            xs,
            [min(grouped[x]) for x in xs],
            [max(grouped[x]) for x in xs],
            color=RANK_COLOURS[position % len(RANK_COLOURS)],
            alpha=0.18,
            linewidth=0,
        )
    left.set_xscale("log", base=2)
    left.set_yscale("log", base=2)
    left.set_xlabel("known source bits in the task")
    left.set_ylabel("R* adapter file bits")
    left.set_title(f"adapter description length at {largest} mappings")
    left.grid(alpha=0.2)
    left.legend(frameon=False)

    right = axes[1]
    final: dict[tuple[str, int], dict[str, Any]] = {}
    for row in preq:
        key = (row["condition"], row["seed"])
        if key not in final or row["right"] > final[key]["right"]:
            final[key] = row
    grouped_preq: dict[int, list[float]] = {}
    for row in final.values():
        grouped_preq.setdefault(row["source_bits"], []).append(
            row["cumulative_code_bits"]
        )
    xs = sorted(grouped_preq)
    if xs:
        right.plot(
            xs,
            [statistics.fmean(grouped_preq[x]) for x in xs],
            marker="o",
            color=RANK_COLOURS[1],
            label="measured code",
        )
        limits = [min(xs), max(xs)]
        right.plot(limits, limits, linestyle="--", color="#6b7280", label="source bits")
    right.set_xscale("log", base=2)
    right.set_yscale("log", base=2)
    right.set_xlabel("known source bits in the task")
    right.set_ylabel("conditional prequential code bits")
    right.set_title("does the measured code recover the known bits?")
    right.grid(alpha=0.2)
    right.legend(frameon=False)

    figure.tight_layout()
    figure.savefig(root / "information_scaling.png", dpi=220)
    plt.close(figure)


def aggregate(root: Path) -> dict[str, Any]:
    results = _collect(root)
    if not results:
        raise FileNotFoundError(f"no information-scaling results under {root}")
    cell_rows = _cell_rows(results)
    preq_rows = _prequential_rows(results)
    root.mkdir(parents=True, exist_ok=True)
    _write_csv(root / "cells.csv", cell_rows)
    _write_csv(root / "prequential.csv", preq_rows)
    gates = check_gates(cell_rows, preq_rows)
    _write_json(root / "gates.json", gates)
    try:
        _figures(root, cell_rows, preq_rows)
    except Exception as error:  # a missing display backend must not lose data
        print(f"figures skipped: {error}")
    summary = {
        "cells": len(results),
        "checkpoints": len(cell_rows),
        "prequential_blocks": len(preq_rows),
        "gates": {
            name: value.get("passed")
            for name, value in gates.items()
            if "passed" in value
        },
    }
    _write_json(root / "summary.json", summary)
    return summary


# ------------------------------------------------------------------------ CLI


def load_info_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or int(raw.get("version", 0)) != 2:
        raise ValueError(f"{path}: expected information-scaling config version 2")
    return raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled task information against adapter description length."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/information_scaling.yaml")
    )
    parser.add_argument(
        "--campaign", type=Path, default=Path("configs/compressibility.yaml")
    )
    parser.add_argument("--out", type=Path, default=Path("runs_information_scaling"))
    parser.add_argument("--studies", nargs="+", help="restrict to these grid studies")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--prefixes", nargs="+", type=int)
    parser.add_argument("--updates", type=int, help="override the update budget")
    parser.add_argument(
        "--learning-rate", type=float, help="override the training learning rate"
    )
    parser.add_argument(
        "--layout", choices=PROMPT_LAYOUTS, help="override the prompt layout"
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--keep-codecs", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    return parser


def selected_cells(info: dict[str, Any], args: argparse.Namespace) -> list[Cell]:
    cells = expand_grid(info)
    if args.studies:
        wanted = set(args.studies)
        cells = [cell for cell in cells if cell.study in wanted]
    if args.conditions:
        wanted = set(args.conditions)
        cells = [cell for cell in cells if cell.condition in wanted]
    if args.seeds:
        wanted = set(args.seeds)
        cells = [cell for cell in cells if cell.seed in wanted]
    if args.prefixes:
        prefixes = tuple(sorted(set(args.prefixes)))
        cells = [replace(cell, prefixes=prefixes) for cell in cells]
    return shard(cells, args.shard, args.shards)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.aggregate:
        print(json.dumps(aggregate(args.out), indent=2, sort_keys=True))
        return

    info = load_info_config(args.config)
    if args.updates is not None:
        info = {**info, "optimizer_updates": int(args.updates)}
    if args.learning_rate is not None:
        info = {
            **info,
            "training": {**info["training"], "learning_rate": args.learning_rate},
        }
    if args.layout is not None:
        info = {**info, "prompt_layout": args.layout}
    campaign = load_campaign(args.campaign)
    cells = selected_cells(info, args)
    if args.dry_run:
        for cell in cells:
            print(f"{cell.slug} prefixes={list(cell.prefixes)}")
        print(f"{len(cells)} cells")
        return
    if not cells:
        raise SystemExit("shard selected no cells")

    # One model load for the whole shard. The previous version reloaded the
    # base for every cell, which spent about a quarter of its GPU time
    # re-quantizing weights it already had.
    session = ModelSession.load(_model(str(info["model"]), campaign))
    try:
        for cell in cells:
            result = run_cell(
                info=info,
                campaign=campaign,
                session=session,
                cell=cell,
                out_root=args.out,
                force=args.force,
                keep_codecs=args.keep_codecs,
            )
            print(
                json.dumps(
                    {
                        "cell": cell.slug,
                        "elapsed_seconds": round(
                            float(result.get("elapsed_seconds", 0.0)), 1
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        try:
            session.unload()
        except Exception:
            pass


if __name__ == "__main__":
    main()
