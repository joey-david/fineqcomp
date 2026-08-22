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
            metadata={
                "split": split,
                # MetaMathQA's 395k rows are augmentations of 13,929 seed
                # problems. Keeping the seed problem is what lets an arm hold
                # its row count fixed and vary how much distinct content those
                # rows cover.
                "source_problem": str(row.get("original_question", "")),
                "augmentation": str(row.get("type", "")),
            },
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


def _convert_hh_rlhf(rows: Any, split: str) -> list[Example]:
    """Preference data as SFT on the preferred completion.

    A base model has never produced assistant-style dialogue, so this is a very
    large behavioural change carrying almost no factual content: the target is
    a persona and a format, not new knowledge. That combination is what makes it
    worth having on the kind axis.

    Only the final assistant turn is scored; everything before it is prompt, so
    the measurement is over the completion the preference actually ranked.
    """
    marker = "\n\nAssistant:"
    converted = []
    for index, row in enumerate(rows):
        text = str(row["chosen"])
        cut = text.rfind(marker)
        if cut < 0:
            continue
        prompt = text[: cut + len(marker)].lstrip("\n")
        response = text[cut + len(marker) :]
        if not response.strip():
            continue
        converted.append(
            Example(
                example_id=f"hh-{split}-{index}",
                prompt=prompt,
                response=response,
                metadata={"split": split},
            )
        )
    if not converted:
        raise ValueError("no hh-rlhf row carried a final assistant turn")
    return converted


def _convert_alpaca(rows: Any, split: str) -> list[Example]:
    """Classic instruction tuning: a second low-content, high-change target."""
    converted = []
    for index, row in enumerate(rows):
        instruction = str(row["instruction"]).strip()
        context = str(row.get("input") or "").strip()
        prompt = (
            "Below is an instruction that describes a task. Write a response"
            " that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
        )
        if context:
            prompt += f"### Input:\n{context}\n\n"
        prompt += "### Response:"
        converted.append(
            Example(
                example_id=f"alpaca-{split}-{index}",
                prompt=prompt,
                response=" " + str(row["output"]).strip(),
                metadata={"split": split},
            )
        )
    return converted


def _convert_text_to_sql(rows: Any, split: str) -> list[Example]:
    """Schema-in-prompt text to SQL.

    Chosen because the published base-to-LoRA gap is enormous -- Qwen-7B goes
    from 16.1 to 61.0 exact match on Spider with a LoRA -- and because every row
    carries its own CREATE TABLE context, so the task needs no external schema
    file and the train and test splits are genuinely disjoint.
    """
    converted = []
    for index, row in enumerate(rows):
        context = " ".join(str(row["sql_context"]).split())
        question = str(row["sql_prompt"]).strip()
        converted.append(
            Example(
                example_id=f"sql-{split}-{index}",
                prompt=(
                    "Translate the question into a single SQL query for the"
                    " schema.\n\n"
                    f"### Schema:\n{context}\n\n"
                    f"### Question:\n{question}\n\n"
                    "### SQL:"
                ),
                response=" " + " ".join(str(row["sql"]).split()),
                metadata={"split": split, "domain": row.get("domain")},
            )
        )
    return converted


def _convert_xbrl(rows: Any, split: str) -> list[Example]:
    """Financial-filing tag extraction, where base models score very low.

    FinLoRA reports base models in the 13-32% band on these and LoRA above 80%,
    which is the largest base-to-adapter gap we have found in a task whose
    answers are short enough to score exactly.
    """
    converted = []
    for index, row in enumerate(rows):
        instruction = " ".join(str(row["instruction"]).split())
        question = str(row["input"]).strip()
        converted.append(
            Example(
                example_id=f"xbrl-{split}-{index}",
                prompt=f"{instruction}\n\n{question}",
                response=" " + str(row["output"]).strip(),
                metadata={"split": split, "company": row.get("company"),
                          "year": row.get("year")},
            )
        )
    return converted


_NATURAL_CONVERTERS = {
    "gsm8k": _convert_gsm8k,
    "hh_rlhf": _convert_hh_rlhf,
    "alpaca": _convert_alpaca,
    "magicoder": _convert_magicoder,
    "math": _convert_math,
    "metamath": _convert_metamath,
    "humaneval": _convert_humaneval,
    "xsum": _convert_xsum,
    "text_to_sql": _convert_text_to_sql,
    "xbrl_tags": _convert_xbrl,
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


ANSWER_MARKER = "The answer is:"


def _split_tail(response: str) -> tuple[str, str]:
    """Body and final answer line. Every MetaMathQA response carries one."""
    cut = response.rfind(ANSWER_MARKER)
    if cut < 0:
        return response, ""
    return response[:cut], response[cut:]


def _plain(body: str) -> str:
    return body


def _preamble(body: str) -> str:
    return "Let us work through this carefully, one step at a time.\n" + body


def _numbered(body: str) -> str:
    lines = [line for line in body.split("\n") if line.strip()]
    return "\n".join(f"({index + 1}) {line}" for index, line in enumerate(lines)) + "\n"


def _shouted(body: str) -> str:
    return body.upper()


def _symbolic(body: str) -> str:
    swaps = (
        (" is ", " ≡ "), (" the ", " ‹the› "), (" of ", " ∘ "),
        (" and ", " ∧ "), (" so ", " ⇒ "), (" we ", " ⊢ "),
    )
    for source, target in swaps:
        body = body.replace(source, target)
    return body


# Deterministic, content-preserving rewrites of the response body. The answer
# line is never touched, so exact-match scoring is unaffected and the task
# information -- which problems, which answers -- is identical across all of
# them. Only how far the target sits from what the base model would naturally
# write changes, which is the behavioural-change lever. Their order here is the
# expected order of that distance; the run measures it rather than assuming it.
RESPONSE_TRANSFORMS = {
    "plain": _plain,
    "preamble": _preamble,
    "numbered": _numbered,
    "shouted": _shouted,
    "symbolic": _symbolic,
}


def transform_responses(rows: list[Example], name: str) -> list[Example]:
    try:
        rewrite = RESPONSE_TRANSFORMS[name]
    except KeyError:
        raise ValueError(
            f"unknown response transform {name!r};"
            f" expected one of {sorted(RESPONSE_TRANSFORMS)}"
        ) from None
    rewritten = []
    for row in rows:
        body, tail = _split_tail(row.response)
        if not tail:
            raise ValueError(f"{row.example_id}: no {ANSWER_MARKER!r} to preserve")
        rewritten.append(
            Example(
                example_id=row.example_id,
                prompt=row.prompt,
                response=rewrite(body) + tail,
                metadata={**row.metadata, "response_transform": name},
            )
        )
    return rewritten


def limit_source_problems(
    rows: Any, field: str, groups: int, needed: int | None
) -> tuple[Any, int]:
    """Keep rows drawn from at most `groups` distinct values of `field`.

    Row count and content diversity are confounded in a nested draw: more rows
    always means more distinct problems, so no fit can tell which one an
    adapter is paying for. This holds the row count fixed and varies the number
    of distinct source problems behind it, which is the only way to separate
    them. The rows stay textually distinct -- they are different augmentations
    of the same problem, not duplicates.

    Falling short of `needed` is an error: an arm that quietly trained on fewer
    rows would no longer be compute-matched to its neighbours.
    """
    if groups < 1:
        raise ValueError("distinct_source_problems must be positive")
    values = rows[field]
    accepted: set[str] = set()
    keep: list[int] = []
    for index, value in enumerate(values):
        if value not in accepted:
            if len(accepted) >= groups:
                continue
            accepted.add(value)
        keep.append(index)
        if needed is not None and len(keep) >= needed:
            break
    if needed is not None and len(keep) < needed:
        raise ValueError(
            f"{groups} distinct values of {field!r} yield only {len(keep)} rows,"
            f" short of the {needed} the arm needs"
        )
    return rows.select(keep), len(accepted)


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
        # The diversity lever runs before the row cap, so the cap still decides
        # the row count and only the content behind it changes.
        groups = spec.get("distinct_source_problems")
        if groups is not None:
            train_rows, _ = limit_source_problems(
                train_rows,
                str(source.get("group_field", "original_question")),
                int(groups),
                int(limit) if limit is not None else None,
            )
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
        splits = {
            "train": _convert_natural(train_rows, "train", source["converter"]),
            "calibration": _convert_natural(
                calibration_rows, "calibration", source["converter"]
            ),
            "test": tests,
        }
        # The behavioural-change lever rewrites what the adapter is taught to
        # emit, on both the training rows and the held-out rows it is scored
        # on. Test prompts are untouched: the model's own output changes, and
        # the answer line survives every rewrite so the scorer still finds it.
        transform = spec.get("response_transform")
        if transform is not None and str(transform) != "plain":
            for split in ("train", "calibration"):
                splits[split] = transform_responses(splits[split], str(transform))
        return splits
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
