"""Prepared exact-bit tasks and pinned natural-task loaders."""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Example:
    example_id: str
    prompt: str
    response: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "prompt": self.prompt,
            "response": self.response,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Example":
        return cls(
            example_id=str(raw["example_id"]),
            prompt=str(raw["prompt"]),
            response=str(raw["response"]),
            metadata=dict(raw.get("metadata", {})),
        )


def _write_jsonl(path: Path, rows: Iterable[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: str | Path) -> list[Example]:
    with Path(path).open() as stream:
        return [Example.from_dict(json.loads(line)) for line in stream if line.strip()]


def synthetic_data_dir(root: str | Path, family_count: int, seed: int) -> Path:
    return Path(root) / "synthetic_codebook" / f"k{family_count}" / f"seed{seed}"


def controlled_data_dir(root: str | Path, binding_count: int, seed: int) -> Path:
    return Path(root) / "controlled_paws" / f"n{binding_count}" / f"seed{seed}"


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


def prepare_synthetic_dataset(
    dataset_cfg: dict[str, Any],
    family_count: int,
    seed: int,
    root: str | Path = "prepared",
) -> Path:
    """Create a fixed-row codebook task with exactly known source entropy."""
    train_rows = int(dataset_cfg["train_rows"])
    items = int(dataset_cfg["items_per_family"])
    labels = list(map(str, dataset_cfg["labels"]))
    if len(labels) != 16 or len(set(labels)) != 16:
        raise ValueError("synthetic task requires 16 unique labels")
    pairs = family_count * items
    if pairs <= 0 or train_rows % pairs:
        raise ValueError(
            f"train_rows={train_rows} must be divisible by {family_count}*{items}"
        )
    repeats = train_rows // pairs
    codebook, codebook_sha256 = _read_codebook(dataset_cfg, pairs, seed)
    mapping = {
        (family, item): codebook[family * items + item]
        for family in range(family_count)
        for item in range(items)
    }

    splits: dict[str, list[Example]] = {"train": [], "calibration": [], "test": []}
    for family in range(family_count):
        for item in range(items):
            label_index = mapping[(family, item)]
            common = {
                "family": family,
                "item": item,
                "label_index": label_index,
                "codebook_key": f"seed{seed}",
            }
            for instance in range(repeats):
                ticket = _ticket(seed, family, item, instance, "train")
                splits["train"].append(
                    Example(
                        example_id=f"train-f{family}-i{item}-n{instance}",
                        prompt=_render_prompt(family, item, ticket, "train", instance),
                        response=labels[label_index],
                        metadata={**common, "instance": instance, "split": "train"},
                    )
                )
            for offset, split in enumerate(("calibration", "test"), start=1):
                instance = repeats + offset
                ticket = _ticket(seed, family, item, instance, split)
                splits[split].append(
                    Example(
                        example_id=f"{split}-f{family}-i{item}",
                        prompt=_render_prompt(family, item, ticket, split, instance),
                        response=labels[label_index],
                        metadata={**common, "instance": instance, "split": split},
                    )
                )

    target = synthetic_data_dir(root, family_count, seed)
    target.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        _write_jsonl(target / f"{split}.jsonl", rows)
    train_bytes = (target / "train.jsonl").read_bytes()
    metadata = {
        "version": 1,
        "family_count": family_count,
        "items_per_family": items,
        "label_count": len(labels),
        "labels": labels,
        "codebook_key": f"seed{seed}",
        "codebook_prefix_sha256": codebook_sha256,
        "codebook_source_bits": pairs * math.log2(len(labels)),
        "train_rows": train_rows,
        "unique_mappings": pairs,
        "source_symbols": pairs,
        "task_entropy_bits": pairs * math.log2(len(labels)),
        "repeats_per_mapping": repeats,
        "train_zlib_bits": len(zlib.compress(train_bytes, level=9)) * 8,
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    validate_synthetic_dataset(target)
    return target


def validate_synthetic_dataset(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    metadata = json.loads((root / "metadata.json").read_text())
    splits = {
        name: read_jsonl(root / f"{name}.jsonl")
        for name in ("train", "calibration", "test")
    }
    if len(splits["train"]) != metadata["train_rows"]:
        raise ValueError(f"{root}: wrong training row count")
    expected_eval = metadata["unique_mappings"]
    if (
        len(splits["calibration"]) != expected_eval
        or len(splits["test"]) != expected_eval
    ):
        raise ValueError(f"{root}: calibration/test must cover every mapping once")
    prompt_sets = {name: {row.prompt for row in rows} for name, rows in splits.items()}
    if any(
        prompt_sets[left] & prompt_sets[right]
        for left, right in (
            ("train", "calibration"),
            ("train", "test"),
            ("calibration", "test"),
        )
    ):
        raise ValueError(f"{root}: rendered prompts leak across splits")
    observed: dict[tuple[int, int], str] = {}
    for rows in splits.values():
        for row in rows:
            key = (int(row.metadata["family"]), int(row.metadata["item"]))
            previous = observed.setdefault(key, row.response)
            if previous != row.response:
                raise ValueError(f"{root}: mapping changed across splits for {key}")
    if len(observed) != expected_eval:
        raise ValueError(f"{root}: incomplete mapping coverage")
    labels = list(metadata["labels"])
    indices = [
        labels.index(observed[(family, item)])
        for family in range(int(metadata["family_count"]))
        for item in range(int(metadata["items_per_family"]))
    ]
    packed = bytes(
        (indices[index] << 4) | indices[index + 1]
        for index in range(0, len(indices), 2)
    )
    if hashlib.sha256(packed).hexdigest() != metadata["codebook_prefix_sha256"]:
        raise ValueError(f"{root}: codebook hash mismatch")
    expected_source_bits = expected_eval * math.log2(int(metadata["label_count"]))
    if metadata["codebook_source_bits"] != expected_source_bits:
        raise ValueError(f"{root}: wrong codebook source size")
    return metadata


def prepare_all_synthetic(
    raw: dict[str, Any], runs: Iterable[Any], root: str | Path
) -> list[Path]:
    dataset_cfg = raw["datasets"]["synthetic_codebook"]
    cells = sorted(
        {
            (int(run.family_count), int(run.seed))
            for run in runs
            if run.kind == "synthetic"
        }
    )
    return [
        prepare_synthetic_dataset(dataset_cfg, family_count, seed, root)
        for family_count, seed in cells
    ]


def _controlled_pairs(dataset_cfg: dict[str, Any], count: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows = load_dataset(
        dataset_cfg["path"],
        dataset_cfg["name"],
        revision=dataset_cfg["revision"],
        split=dataset_cfg["split"],
    )
    candidates = []
    for row in rows:
        first = " ".join(str(row["sentence1"]).split())
        second = " ".join(str(row["sentence2"]).split())
        if int(row["label"]) != 1 or not first or not second or first == second:
            continue
        digest = hashlib.sha256(
            f"{dataset_cfg['revision']}:{row['id']}:{first}:{second}".encode()
        ).hexdigest()
        candidates.append(
            {
                "source_id": int(row["id"]),
                "sentence1": first,
                "sentence2": second,
                "digest": digest,
            }
        )
    selected = []
    used_sentences: set[str] = set()
    for row in sorted(candidates, key=lambda item: item["digest"]):
        normalized = {row["sentence1"].casefold(), row["sentence2"].casefold()}
        if normalized & used_sentences:
            continue
        selected.append(row)
        used_sentences.update(normalized)
        if len(selected) == count:
            return selected
    raise ValueError(f"PAWS has only {len(selected)} usable unique paraphrase pairs")


def _controlled_prompt(sentence: str, ticket: str, split: str, variant: int) -> str:
    if split == "train" and variant % 2 == 0:
        return f"Assign the stored code to this statement.\n{sentence}\nTicket: {ticket}\nCode:"
    if split == "train":
        return f"Recall the code for the following text ({ticket}):\n{sentence}\nLabel:"
    if split == "calibration":
        return f"Which code belongs to this paraphrased statement?\n{sentence}\nCode:"
    return f"Return one code for this statement.\n{sentence}\nAnswer:"


def prepare_controlled_dataset(
    dataset_cfg: dict[str, Any],
    labels: list[str],
    pairs: list[dict[str, Any]],
    binding_count: int,
    seed: int,
    root: str | Path = "prepared",
) -> Path:
    """Bind real paraphrase pairs to an independent fixed random codebook."""
    train_rows = int(dataset_cfg["train_rows"])
    if len(labels) != 16 or len(set(labels)) != 16:
        raise ValueError("controlled task requires 16 unique labels")
    if binding_count <= 0 or train_rows % binding_count:
        raise ValueError(
            f"train_rows={train_rows} must be divisible by {binding_count} bindings"
        )
    if len(pairs) < binding_count:
        raise ValueError(f"need {binding_count} paraphrase pairs, found {len(pairs)}")
    repeats = train_rows // binding_count
    codebook, codebook_sha256 = _read_codebook(dataset_cfg, binding_count, seed)
    splits: dict[str, list[Example]] = {"train": [], "calibration": [], "test": []}
    for binding, pair in enumerate(pairs[:binding_count]):
        label_index = codebook[binding]
        common = {
            "binding": binding,
            "source_id": pair["source_id"],
            "label_index": label_index,
            "codebook_key": f"seed{seed}",
        }
        for instance in range(repeats):
            ticket = _ticket(seed, binding, 0, instance, "controlled-train")
            splits["train"].append(
                Example(
                    example_id=f"train-b{binding}-n{instance}",
                    prompt=_controlled_prompt(
                        pair["sentence1"], ticket, "train", instance
                    ),
                    response=labels[label_index],
                    metadata={**common, "instance": instance, "split": "train"},
                )
            )
        for offset, split in enumerate(("calibration", "test"), start=1):
            ticket = _ticket(seed, binding, 0, repeats + offset, f"controlled-{split}")
            sentence = (
                pair["sentence1"] if split == "calibration" else pair["sentence2"]
            )
            splits[split].append(
                Example(
                    example_id=f"{split}-b{binding}",
                    prompt=_controlled_prompt(
                        sentence, ticket, split, repeats + offset
                    ),
                    response=labels[label_index],
                    metadata={
                        **common,
                        "instance": repeats + offset,
                        "split": split,
                    },
                )
            )

    target = controlled_data_dir(root, binding_count, seed)
    target.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        _write_jsonl(target / f"{split}.jsonl", rows)
    pair_digest = hashlib.sha256(
        "".join(pair["digest"] for pair in pairs[:binding_count]).encode()
    ).hexdigest()
    train_bytes = (target / "train.jsonl").read_bytes()
    metadata = {
        "version": 1,
        "dataset": "controlled_paws",
        "source_revision": dataset_cfg["revision"],
        "binding_count": binding_count,
        "source_symbols": binding_count,
        "label_count": len(labels),
        "labels": labels,
        "codebook_key": f"seed{seed}",
        "codebook_prefix_sha256": codebook_sha256,
        "pair_prefix_sha256": pair_digest,
        "codebook_source_bits": binding_count * math.log2(len(labels)),
        "task_entropy_bits": binding_count * math.log2(len(labels)),
        "train_rows": train_rows,
        "repeats_per_binding": repeats,
        "train_zlib_bits": len(zlib.compress(train_bytes, level=9)) * 8,
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    validate_controlled_dataset(target)
    return target


def validate_controlled_dataset(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    metadata = json.loads((root / "metadata.json").read_text())
    splits = {
        name: read_jsonl(root / f"{name}.jsonl")
        for name in ("train", "calibration", "test")
    }
    bindings = int(metadata["binding_count"])
    if len(splits["train"]) != int(metadata["train_rows"]):
        raise ValueError(f"{root}: wrong training row count")
    if len(splits["calibration"]) != bindings or len(splits["test"]) != bindings:
        raise ValueError(f"{root}: calibration/test must cover every binding once")
    prompt_sets = {name: {row.prompt for row in rows} for name, rows in splits.items()}
    if any(
        prompt_sets[left] & prompt_sets[right]
        for left, right in (
            ("train", "calibration"),
            ("train", "test"),
            ("calibration", "test"),
        )
    ):
        raise ValueError(f"{root}: rendered prompts leak across splits")
    observed: dict[int, str] = {}
    for rows in splits.values():
        for row in rows:
            binding = int(row.metadata["binding"])
            previous = observed.setdefault(binding, row.response)
            if previous != row.response:
                raise ValueError(f"{root}: binding {binding} changed across splits")
    if set(observed) != set(range(bindings)):
        raise ValueError(f"{root}: incomplete binding coverage")
    labels = list(metadata["labels"])
    indices = [labels.index(observed[index]) for index in range(bindings)]
    packed = bytes(
        (indices[index] << 4) | indices[index + 1]
        for index in range(0, len(indices), 2)
    )
    if hashlib.sha256(packed).hexdigest() != metadata["codebook_prefix_sha256"]:
        raise ValueError(f"{root}: codebook hash mismatch")
    return metadata


def prepare_all_controlled(
    raw: dict[str, Any], runs: Iterable[Any], root: str | Path
) -> list[Path]:
    cells = sorted(
        {
            (int(run.binding_count), int(run.seed))
            for run in runs
            if run.kind == "controlled"
        }
    )
    if not cells:
        return []
    dataset_cfg = {
        **raw["datasets"]["controlled_paws"],
        "codebook_dir": raw["datasets"]["synthetic_codebook"]["codebook_dir"],
    }
    pairs = _controlled_pairs(dataset_cfg, max(count for count, _ in cells))
    labels = list(map(str, raw["datasets"]["synthetic_codebook"]["labels"]))
    return [
        prepare_controlled_dataset(
            dataset_cfg, labels, pairs, binding_count, seed, root
        )
        for binding_count, seed in cells
    ]


def load_natural_dataset(
    raw: dict[str, Any], dataset_key: str, seed: int
) -> dict[str, list[Example]]:
    """Load pinned GSM8K or MBPP and return train/calibration/test records."""
    from datasets import load_dataset

    spec = raw["datasets"][dataset_key]
    dataset = load_dataset(spec["path"], spec.get("name"), revision=spec["revision"])
    if dataset_key == "gsm8k":
        shuffled = dataset[spec["train_split"]].shuffle(seed=seed)
        n_validation = int(spec["validation_rows"])
        calibration_rows = shuffled.select(range(n_validation))
        train_rows = shuffled.select(range(n_validation, len(shuffled)))
        test_rows = dataset[spec["test_split"]]

        def convert(rows: Any, split: str) -> list[Example]:
            return [
                Example(
                    example_id=f"gsm8k-{split}-{index}",
                    prompt=(
                        "Solve the problem. Show concise work and finish with "
                        "'#### ' followed by the answer.\n\n"
                        f"Question: {row['question']}\nAnswer:"
                    ),
                    response=" " + str(row["answer"]),
                    metadata={"split": split},
                )
                for index, row in enumerate(rows)
            ]

        return {
            "train": convert(train_rows, "train"),
            "calibration": convert(calibration_rows, "calibration"),
            "test": convert(test_rows, "test"),
        }
    if dataset_key == "mbpp":
        split_names = {
            "train": spec["train_split"],
            "calibration": spec["validation_split"],
            "test": spec["test_split"],
        }

        def convert_mbpp(rows: Any, split: str) -> list[Example]:
            converted = []
            for index, row in enumerate(rows):
                problem = row.get("prompt") or row.get("text")
                code = row.get("code") or ""
                tests = row.get("test_list") or row.get("test") or []
                converted.append(
                    Example(
                        example_id=f"mbpp-{split}-{index}",
                        prompt=(
                            "Write a Python function that solves this task. Return only "
                            f"one Python code block.\n\nTask: {problem}\n\nCode:"
                        ),
                        response=f"\n```python\n{code}\n```",
                        metadata={"split": split, "tests": list(tests)},
                    )
                )
            return converted

        return {
            target: convert_mbpp(dataset[source], target)
            for target, source in split_names.items()
        }
    raise ValueError(f"unsupported natural dataset: {dataset_key}")


def load_ifeval(raw: dict[str, Any]) -> tuple[list[Example], list[dict[str, Any]]]:
    """Load the pinned IFEval prompts and preserve official evaluator fields."""
    from datasets import load_dataset

    spec = raw["datasets"]["ifeval"]
    rows = load_dataset(spec["path"], revision=spec["revision"], split=spec["split"])
    raw_rows = [dict(row) for row in rows]
    examples = [
        Example(
            example_id=f"ifeval-{row['key']}",
            prompt=str(row["prompt"]),
            response="",
            metadata={"key": int(row["key"])},
        )
        for row in raw_rows
    ]
    return examples, raw_rows
