"""Prepared exact-bit tasks and pinned natural-task loaders."""

from __future__ import annotations

import json
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: str | Path) -> list[Example]:
    with Path(path).open() as stream:
        return [Example.from_dict(json.loads(line)) for line in stream if line.strip()]


def natural_data_dir(root: str | Path, dataset_key: str, seed: int) -> Path:
    return Path(root) / "natural" / dataset_key / f"seed{seed}"


def _convert_gsm8k(rows: Any, split: str) -> list[Example]:
    return [
        Example(
            example_id=f"gsm8k-{split}-{index}",
            prompt=(
                "Solve the problem. Show concise work and finish with "
                "'#### ' followed by the answer.\n\n"
                f"Question: {row['question']}\nAnswer:"
            ),
            response=" " + str(row["answer"]),
            metadata={"split": split, "evaluator": "gsm8k"},
        )
        for index, row in enumerate(rows)
    ]


def _convert_metamath(rows: Any, split: str) -> list[Example]:
    return [
        Example(
            example_id=f"metamath-{split}-{index}",
            prompt=(
                "Solve the problem and put the final answer in \\boxed{}.\n\n"
                f"Question: {row['query']}\nAnswer:"
            ),
            response=" " + str(row["response"]),
            metadata={"split": split},
        )
        for index, row in enumerate(rows)
    ]


def _convert_magicoder(rows: Any, split: str) -> list[Example]:
    return [
        Example(
            example_id=f"magicoder-{split}-{index}",
            prompt=(
                "Write a correct response to the programming instruction.\n\n"
                f"Instruction:\n{row['instruction']}\n\nResponse:"
            ),
            response="\n" + str(row["response"]),
            metadata={"split": split},
        )
        for index, row in enumerate(rows)
    ]


def _convert_xsum(rows: Any, split: str) -> list[Example]:
    return [
        Example(
            example_id=f"xsum-{split}-{index}",
            prompt=(
                "Write one concise sentence that summarizes the document.\n\n"
                f"Document: {row['document']}\n\nSummary:"
            ),
            response=" " + str(row["summary"]),
            metadata={"split": split, "evaluator": "xsum"},
        )
        for index, row in enumerate(rows)
    ]


def _convert_math(rows: Any, split: str) -> list[Example]:
    return [
        Example(
            example_id=f"math-{split}-{index}",
            prompt=(
                "Solve the problem and put the final answer in \\boxed{}.\n\n"
                f"Problem: {row['problem']}\nAnswer:"
            ),
            response=" " + str(row["solution"]),
            metadata={
                "split": split,
                "evaluator": "math",
                "level": str(row.get("level", "")),
                "subject": str(row.get("type", "")),
            },
        )
        for index, row in enumerate(rows)
    ]


def _convert_humaneval(rows: Any, split: str) -> list[Example]:
    return [
        Example(
            example_id=str(row.get("task_id", f"humaneval-{split}-{index}")),
            prompt=(
                "Complete the Python function. Return only the missing code.\n\n"
                + str(row["prompt"])
            ),
            response=str(row["canonical_solution"]),
            metadata={
                "split": split,
                "evaluator": "humaneval",
                "code_prefix": str(row["prompt"]),
                "tests": str(row["test"]),
                "entry_point": str(row["entry_point"]),
            },
        )
        for index, row in enumerate(rows)
    ]


_NATURAL_CONVERTERS = {
    "gsm8k": _convert_gsm8k,
    "magicoder": _convert_magicoder,
    "math": _convert_math,
    "metamath": _convert_metamath,
    "humaneval": _convert_humaneval,
    "xsum": _convert_xsum,
}


def _convert_natural(rows: Any, split: str, converter: str) -> list[Example]:
    try:
        return _NATURAL_CONVERTERS[converter](rows, split)
    except KeyError as error:
        raise ValueError(f"unsupported natural converter: {converter}") from error


def _convert_multiple_choice(
    rows: Any,
    split: str,
    dataset_key: str,
    spec: dict[str, Any],
) -> list[Example]:
    """Convert standard Hugging Face multiple-choice records to label SFT."""
    converted = []
    for index, row in enumerate(rows):
        choices = row["choices"]
        source_labels = [str(label) for label in choices["label"]]
        texts = [" ".join(str(text).split()) for text in choices["text"]]
        answer = str(row["answerKey"])
        if not answer or answer not in source_labels:
            raise ValueError(
                f"{dataset_key}/{split}/{index}: missing labeled answer"
            )
        if not 2 <= len(texts) <= 5 or len(source_labels) != len(texts):
            raise ValueError(
                f"{dataset_key}/{split}/{index}: expected 2 to 5 choices"
            )
        labels = [chr(ord("A") + offset) for offset in range(len(texts))]
        question = " ".join(str(row[spec["question_field"]]).split())
        rendered_choices = "\n".join(
            f"{label}. {text}" for label, text in zip(labels, texts, strict=True)
        )
        target = source_labels.index(answer)
        converted.append(
            Example(
                example_id=f"{dataset_key}-{split}-{index}",
                prompt=(
                    "Choose the best answer. Reply with only its letter.\n\n"
                    f"Question: {question}\n{rendered_choices}\nAnswer:"
                ),
                response=f" {labels[target]}",
                metadata={
                    "split": split,
                    "label_index": target,
                    "choice_count": len(texts),
                },
            )
        )
    return converted


def _load_natural_from_hub(
    raw: dict[str, Any], dataset_key: str, seed: int
) -> dict[str, list[Example]]:
    """Download one pinned natural dataset and convert it to campaign records."""
    from datasets import load_dataset

    spec = raw["datasets"][dataset_key]
    if "train_source" in spec:
        source = spec["train_source"]
        train_dataset = load_dataset(
            source["path"], source.get("name"), revision=source["revision"]
        )
        shuffled = train_dataset[source["split"]].shuffle(seed=seed)
        validation_rows = int(spec["validation_rows"])
        calibration_rows = shuffled.select(range(validation_rows))
        train_rows = shuffled.select(range(validation_rows, len(shuffled)))
        # `train_rows` bounds the training set so one run fits a bounded job.
        # It reshuffles per seed, unlike the test cap, because seeds should see
        # different training data.
        limit = spec.get("train_rows")
        if limit is not None and int(limit) < len(train_rows):
            train_rows = train_rows.select(range(int(limit)))
        # `test_rows` caps each evaluation set. The subsample uses a fixed seed,
        # not the run seed, so every seed and codec is scored on exactly the
        # same problems and the comparisons stay paired.
        test_rows = spec.get("test_rows")
        tests = []
        for evaluation in spec["evaluations"]:
            evaluation_dataset = load_dataset(
                evaluation["path"],
                evaluation.get("name"),
                revision=evaluation["revision"],
            )
            rows = evaluation_dataset[evaluation["split"]]
            if test_rows is not None and int(test_rows) < len(rows):
                rows = rows.shuffle(seed=0).select(range(int(test_rows)))
            tests.extend(_convert_natural(rows, "test", evaluation["converter"]))
        return {
            "train": _convert_natural(train_rows, "train", source["converter"]),
            "calibration": _convert_natural(
                calibration_rows, "calibration", source["converter"]
            ),
            "test": tests,
        }
    dataset = load_dataset(spec["path"], spec.get("name"), revision=spec["revision"])
    if dataset_key == "gsm8k":
        shuffled = dataset[spec["train_split"]].shuffle(seed=seed)
        n_validation = int(spec["validation_rows"])
        calibration_rows = shuffled.select(range(n_validation))
        train_rows = shuffled.select(range(n_validation, len(shuffled)))
        test_rows = dataset[spec["test_split"]]
        return {
            "train": _convert_gsm8k(train_rows, "train"),
            "calibration": _convert_gsm8k(calibration_rows, "calibration"),
            "test": _convert_gsm8k(test_rows, "test"),
        }
    if spec.get("task_type") == "multiple_choice":
        train = dataset[spec["train_split"]]
        if "validation_rows" in spec:
            shuffled = train.shuffle(seed=seed)
            count = int(spec["validation_rows"])
            calibration_rows = shuffled.select(range(count))
            train_rows = shuffled.select(range(count, len(shuffled)))
        else:
            train_rows = train
            calibration_rows = dataset[spec["validation_split"]]
        return {
            "train": _convert_multiple_choice(
                train_rows, "train", dataset_key, spec
            ),
            "calibration": _convert_multiple_choice(
                calibration_rows, "calibration", dataset_key, spec
            ),
            "test": _convert_multiple_choice(
                dataset[spec["test_split"]], "test", dataset_key, spec
            ),
        }
    raise ValueError(f"unsupported natural dataset: {dataset_key}")


def validate_natural_dataset(
    path: str | Path, dataset_key: str, seed: int
) -> dict[str, Any]:
    root = Path(path)
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("dataset_key") != dataset_key or int(metadata.get("seed", -1)) != seed:
        raise ValueError(f"{root}: natural dataset metadata does not match its path")
    splits = {
        name: read_jsonl(root / f"{name}.jsonl")
        for name in ("train", "calibration", "test")
    }
    for split, rows in splits.items():
        if len(rows) != int(metadata[f"{split}_rows"]):
            raise ValueError(f"{root}: wrong {split} row count")
        if any(row.metadata.get("split") != split for row in rows):
            raise ValueError(f"{root}: {split} rows have the wrong split marker")
    return metadata


def prepare_natural_dataset(
    raw: dict[str, Any], dataset_key: str, seed: int, root: str | Path = "prepared"
) -> Path:
    """Materialize one pinned natural dataset so workers never need the Hub."""
    rows = _load_natural_from_hub(raw, dataset_key, seed)
    target = natural_data_dir(root, dataset_key, seed)
    target.mkdir(parents=True, exist_ok=True)
    for split, examples in rows.items():
        _write_jsonl(target / f"{split}.jsonl", examples)
    spec = raw["datasets"][dataset_key]
    revisions = (
        {
            "train": spec["train_source"]["revision"],
            "evaluations": {
                evaluation["key"]: evaluation["revision"]
                for evaluation in spec["evaluations"]
            },
        }
        if "train_source" in spec
        else spec["revision"]
    )
    _write_json(
        target / "metadata.json",
        {
            "version": 1,
            "dataset_key": dataset_key,
            "revision": revisions,
            "seed": seed,
            **{f"{split}_rows": len(examples) for split, examples in rows.items()},
        },
    )
    validate_natural_dataset(target, dataset_key, seed)
    return target


def prepare_all_natural(
    raw: dict[str, Any], runs: Iterable[Any], root: str | Path
) -> list[Path]:
    cells = sorted(
        {
            (str(run.dataset_key), int(run.seed))
            for run in runs
            if run.kind == "natural" and run.dataset_key is not None
        }
    )
    return [prepare_natural_dataset(raw, dataset_key, seed, root) for dataset_key, seed in cells]


def load_natural_dataset(
    raw: dict[str, Any],
    dataset_key: str,
    seed: int,
    prepared_root: str | Path | None = None,
) -> dict[str, list[Example]]:
    """Read a staged dataset, falling back to the pinned Hub loader if absent."""
    if prepared_root is not None:
        root = natural_data_dir(prepared_root, dataset_key, seed)
        expected = [
            root / "metadata.json",
            *(
                root / f"{split}.jsonl"
                for split in ("train", "calibration", "test")
            ),
        ]
        if all(path.is_file() for path in expected):
            validate_natural_dataset(root, dataset_key, seed)
            return {
                split: read_jsonl(root / f"{split}.jsonl")
                for split in ("train", "calibration", "test")
            }
        if any(path.exists() for path in expected):
            raise FileNotFoundError(f"incomplete prepared natural dataset: {root}")
    return _load_natural_from_hub(raw, dataset_key, seed)
