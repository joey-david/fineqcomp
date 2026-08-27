"""Prepared exact-bit tasks and pinned natural-task loaders."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
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


def _convert_openr1_math(rows: Any, split: str) -> list[Example]:
    """Use the curated DeepSeek-R1 trace stored in each OpenR1 message pair."""
    converted = []
    for index, row in enumerate(rows):
        messages = list(row["messages"])
        if (
            len(messages) != 2
            or str(messages[0].get("role")) != "user"
            or str(messages[1].get("role")) != "assistant"
        ):
            raise ValueError(
                f"OpenR1-Math/{split}/{index}: expected one user and one assistant"
            )
        # A trace with no `</think>` is dropped by the answer-marker filter,
        # not raised on: the row caps run before conversion, so one bad row in
        # the pool would otherwise fail the whole prepare.
        response = str(messages[1].get("content", "")).strip()
        converted.append(
            Example(
                example_id=f"openr1-math-{split}-{row.get('uuid', index)}",
                prompt=(
                    "Solve the problem. Show your reasoning and put the final answer "
                    "in \\boxed{}.\n\n"
                    f"Problem: {row['problem']}\nAnswer:"
                ),
                response="\n" + response,
                metadata={
                    "split": split,
                    "source_problem": str(row.get("uuid", row["problem"])),
                    "problem_type": str(row.get("problem_type", "")),
                    "question_type": str(row.get("question_type", "")),
                    "source": str(row.get("source", "")),
                },
            )
        )
    return converted


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


def _convert_paws(rows: Any, split: str) -> list[Example]:
    """PAWS paraphrase pairs as short, exactly scored supervised answers."""
    return [
        Example(
            example_id=f"paws-{split}-{index}",
            prompt=(
                "Do these two sentences have the same meaning? Reply with only"
                " yes or no.\n\n"
                f"Sentence 1: {row['sentence1']}\n"
                f"Sentence 2: {row['sentence2']}\nAnswer:"
            ),
            response=" yes" if int(row["label"]) else " no",
            metadata={"split": split, "evaluator": "paws"},
        )
        for index, row in enumerate(rows)
    ]


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
                metadata={
                    "split": split,
                    "company": row.get("company"),
                    "year": row.get("year"),
                },
            )
        )
    return converted


_NATURAL_CONVERTERS = {
    "gsm8k": _convert_gsm8k,
    "hh_rlhf": _convert_hh_rlhf,
    "alpaca": _convert_alpaca,
    "paws": _convert_paws,
    "magicoder": _convert_magicoder,
    "math": _convert_math,
    "openr1_math": _convert_openr1_math,
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
                response=labels[target],
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

RATIONALE_CONTROLS = {"permuted"}


def _split_response(
    response: str, marker: str, *, from_end: bool = True
) -> tuple[str, str]:
    cut = response.rfind(marker) if from_end else response.find(marker)
    if cut < 0:
        raise ValueError(f"no {marker!r} in response")
    return response[:cut], response[cut:]


def _rationale_hash(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def permute_rationales(
    rows: list[Example],
    marker: str,
    seed: int,
    *,
    from_end: bool = True,
    block_size: int = 32,
) -> list[Example]:
    """Move each rationale to a similar-length, unrelated prompt.

    The prompts and final answers stay fixed. The rationale bodies form the
    exact same multiset before and after the move, so the control changes their
    pairing with problems rather than their marginal text, count, or length.
    Small length-sorted blocks keep truncation exposure close at a fixed token
    limit. Donors may not share an example, source problem, or body.
    """
    if len(rows) < 2:
        raise ValueError("rationale permutation needs at least two rows")
    parsed = [
        _split_response(row.response, marker, from_end=from_end) for row in rows
    ]
    ordered = sorted(range(len(rows)), key=lambda index: (len(parsed[index][0]), index))
    blocks = [
        ordered[start : start + block_size]
        for start in range(0, len(ordered), block_size)
    ]
    if len(blocks) > 1 and len(blocks[-1]) == 1:
        blocks[-2].extend(blocks.pop())

    donors: dict[int, int] = {}
    for block_index, recipients in enumerate(blocks):
        rng = random.Random((int(seed) << 16) + block_index)
        candidate = list(recipients)
        for _ in range(10_000):
            rng.shuffle(candidate)
            valid = True
            for recipient, donor in zip(recipients, candidate, strict=True):
                recipient_source = str(
                    rows[recipient].metadata.get("source_problem", "")
                )
                donor_source = str(rows[donor].metadata.get("source_problem", ""))
                if (
                    recipient == donor
                    or parsed[recipient][0] == parsed[donor][0]
                    or (recipient_source and recipient_source == donor_source)
                ):
                    valid = False
                    break
            if valid:
                donors.update(zip(recipients, candidate, strict=True))
                break
        else:
            raise ValueError(
                "could not permute rationales without a matched prompt or body; "
                "increase the block size"
            )

    rewritten = []
    for recipient, row in enumerate(rows):
        donor = donors[recipient]
        original_body, answer = parsed[recipient]
        donor_body, _ = parsed[donor]
        donor_source = str(rows[donor].metadata.get("source_problem", ""))
        rewritten.append(
            Example(
                example_id=row.example_id,
                prompt=row.prompt,
                response=donor_body + answer,
                metadata={
                    **row.metadata,
                    "rationale_control": "permuted",
                    "rationale_donor_id": rows[donor].example_id,
                    "rationale_donor_source_problem": donor_source,
                    "original_rationale_hash": _rationale_hash(original_body),
                    "donor_rationale_hash": _rationale_hash(donor_body),
                    "original_rationale_chars": len(original_body),
                    "donor_rationale_chars": len(donor_body),
                    "rationale_length_delta_chars": len(donor_body)
                    - len(original_body),
                },
            )
        )
    return rewritten


def restore_aligned_rationales(
    rows: list[Example], marker: str, *, from_end: bool = True
) -> list[Example]:
    """Reconstruct the aligned control from a staged rationale permutation."""
    bodies_by_id = {}
    for row in rows:
        donor_id = str(row.metadata.get("rationale_donor_id", ""))
        body, _ = _split_response(row.response, marker, from_end=from_end)
        if not donor_id or donor_id in bodies_by_id:
            raise ValueError("rationale donor IDs do not form a bijection")
        bodies_by_id[donor_id] = body
    if set(bodies_by_id) != {row.example_id for row in rows}:
        raise ValueError("rationale donor IDs do not cover the staged rows")
    aligned = []
    for row in rows:
        _, answer = _split_response(row.response, marker, from_end=from_end)
        aligned.append(
            Example(
                example_id=row.example_id,
                prompt=row.prompt,
                response=bodies_by_id[row.example_id] + answer,
                metadata=row.metadata,
            )
        )
    return aligned


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


ANSWER_MARKER_SLACK = 4


def take_answer_bearing_rows(
    rows: Any,
    split: str,
    converter: str,
    marker: str,
    needed: int | None,
) -> list[Example]:
    """Convert rows and keep only those whose solution reaches a final answer.

    This is a scope decision, not data cleaning. About three per cent of
    NuminaMath-CoT solutions never write `\\boxed{}`, and every one of them
    sampled was an olympiad or AoPS proof ending in `\\blacksquare`: there is no
    final answer because a proof has none. Such rows cannot be scored by exact
    match and have no problem-trace boundary for the rationale control to
    permute, so the panel covers problems with a checkable answer only.

    The row caps run before conversion, so the excluded rows have to be
    replaced rather than subtracted: converting a slack multiple and cutting
    back to `needed` keeps the arm compute-matched to its neighbours.
    """
    if needed is None:
        return [
            example
            for example in _convert_natural(rows, split, converter)
            if marker in example.response
        ]
    pool = min(len(rows), needed * ANSWER_MARKER_SLACK)
    kept = [
        example
        for example in _convert_natural(rows.select(range(pool)), split, converter)
        if marker in example.response
    ]
    if len(kept) < needed:
        raise ValueError(
            f"{split}: {pool} rows yield only {len(kept)} carrying {marker!r},"
            f" short of the {needed} the arm needs"
        )
    return kept[:needed]


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
        # Excluding rows that carry no final answer needs spare rows to replace
        # them. Reserving those only when the dataset asks for the criterion
        # keeps every earlier campaign's split boundary unmoved.
        required_marker = (
            str(spec["answer_marker"]) if spec.get("answer_bearing_only") else None
        )
        reserved = validation_rows * (
            ANSWER_MARKER_SLACK if required_marker else 1
        )
        calibration_rows = shuffled.select(range(min(reserved, len(shuffled))))
        train_rows = shuffled.select(range(min(reserved, len(shuffled)), len(shuffled)))
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
        if (
            required_marker is None
            and limit is not None
            and int(limit) < len(train_rows)
        ):
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
            # The scorer dispatches on `evaluator`, and its fallback is the
            # campaign's dataset key -- which matches the evaluation name only
            # by luck. It did in the first text-to-SQL panel and did not in the
            # diversity sweep, where the key is `sql_div_100` and no scorer
            # answers to that. Stamping here makes the config the single source
            # of that name instead of every converter having to remember.
            for example in _convert_natural(rows, "test", evaluation["converter"]):
                example.metadata["evaluator"] = str(evaluation["key"])
                tests.append(example)
        if required_marker is None:
            splits = {
                "train": _convert_natural(train_rows, "train", source["converter"]),
                "calibration": _convert_natural(
                    calibration_rows, "calibration", source["converter"]
                ),
                "test": tests,
            }
        else:
            splits = {
                "train": take_answer_bearing_rows(
                    train_rows,
                    "train",
                    source["converter"],
                    required_marker,
                    int(limit) if limit is not None else None,
                ),
                "calibration": take_answer_bearing_rows(
                    calibration_rows,
                    "calibration",
                    source["converter"],
                    required_marker,
                    validation_rows,
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
        rationale_control = spec.get("rationale_control")
        if rationale_control is not None:
            if str(rationale_control) not in RATIONALE_CONTROLS:
                raise ValueError(f"unknown rationale control {rationale_control!r}")
            marker = spec.get("answer_marker")
            if not marker:
                raise ValueError("a rationale control needs an answer_marker")
            for offset, split in enumerate(("train", "calibration")):
                splits[split] = permute_rationales(
                    splits[split],
                    str(marker),
                    seed=int(seed) * 2 + offset,
                    from_end=bool(spec.get("answer_marker_from_end", True)),
                )
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
        train_limit = spec.get("train_rows")
        if train_limit is not None and int(train_limit) < len(train_rows):
            train_rows = train_rows.select(range(int(train_limit)))
        test_rows = dataset[spec["test_split"]]
        test_limit = spec.get("test_rows")
        if test_limit is not None and int(test_limit) < len(test_rows):
            test_rows = test_rows.shuffle(seed=0).select(range(int(test_limit)))
        return {
            "train": _convert_multiple_choice(
                train_rows, "train", dataset_key, spec
            ),
            "calibration": _convert_multiple_choice(
                calibration_rows, "calibration", dataset_key, spec
            ),
            "test": _convert_multiple_choice(
                test_rows, "test", dataset_key, spec
            ),
        }
    raise ValueError(f"unsupported natural dataset: {dataset_key}")


def validate_natural_dataset(
    path: str | Path,
    dataset_key: str,
    seed: int,
    expected_evaluators: Iterable[str] = (),
    expected_rationale_control: str | None = None,
    answer_marker: str | None = None,
    answer_marker_from_end: bool = True,
) -> dict[str, Any]:
    root = Path(path)
    metadata = json.loads((root / "metadata.json").read_text())
    if (
        metadata.get("dataset_key") != dataset_key
        or int(metadata.get("seed", -1)) != seed
    ):
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
    expected = set(map(str, expected_evaluators))
    if expected:
        observed = {
            str(row.metadata["evaluator"])
            for row in splits["test"]
            if "evaluator" in row.metadata
        }
        missing = sum("evaluator" not in row.metadata for row in splits["test"])
        if missing or observed != expected:
            raise ValueError(
                f"{root}: staged test rows have evaluators {sorted(observed)}, "
                f"expected {sorted(expected)} ({missing} rows missing evaluator); "
                "run prepare again before submitting jobs"
            )
    if expected_rationale_control is not None:
        if not answer_marker:
            raise ValueError("a rationale control needs an answer marker")
        for split in ("train", "calibration"):
            rows = splits[split]
            controls = {row.metadata.get("rationale_control") for row in rows}
            donor_ids = [str(row.metadata.get("rationale_donor_id", "")) for row in rows]
            row_ids = [row.example_id for row in rows]
            original_hashes = Counter(
                str(row.metadata.get("original_rationale_hash", "")) for row in rows
            )
            donor_hashes = Counter(
                str(row.metadata.get("donor_rationale_hash", "")) for row in rows
            )
            same_source = sum(
                bool(row.metadata.get("source_problem"))
                and row.metadata.get("source_problem")
                == row.metadata.get("rationale_donor_source_problem")
                for row in rows
            )
            actual_donor_hashes = [
                _rationale_hash(
                    _split_response(
                        row.response,
                        answer_marker,
                        from_end=answer_marker_from_end,
                    )[0]
                )
                for row in rows
            ]
            bodies_by_donor = dict(zip(donor_ids, actual_donor_hashes, strict=True))
            recorded_donor_hashes = [
                str(row.metadata.get("donor_rationale_hash", "")) for row in rows
            ]
            recorded_original_hashes = [
                str(row.metadata.get("original_rationale_hash", "")) for row in rows
            ]
            if (
                controls != {expected_rationale_control}
                or Counter(donor_ids) != Counter(row_ids)
                or any(donor == row_id for donor, row_id in zip(donor_ids, row_ids))
                or original_hashes != donor_hashes
                or "" in original_hashes
                or same_source
                or actual_donor_hashes != recorded_donor_hashes
                or any(
                    bodies_by_donor.get(row_id) != original_hash
                    for row_id, original_hash in zip(
                        row_ids, recorded_original_hashes, strict=True
                    )
                )
            ):
                raise ValueError(
                    f"{root}: {split} does not preserve the rationale-control "
                    "bijection; run prepare again before submitting jobs"
                )
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
    validate_natural_dataset(
        target,
        dataset_key,
        seed,
        (evaluation["key"] for evaluation in spec.get("evaluations", [])),
        spec.get("rationale_control"),
        spec.get("answer_marker"),
        bool(spec.get("answer_marker_from_end", True)),
    )
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
            spec = raw["datasets"][dataset_key]
            validate_natural_dataset(
                root,
                dataset_key,
                seed,
                (evaluation["key"] for evaluation in spec.get("evaluations", [])),
                spec.get("rationale_control"),
                spec.get("answer_marker"),
                bool(spec.get("answer_marker_from_end", True)),
            )
            return {
                split: read_jsonl(root / f"{split}.jsonl")
                for split in ("train", "calibration", "test")
            }
        if any(path.exists() for path in expected):
            raise FileNotFoundError(f"incomplete prepared natural dataset: {root}")
    return _load_natural_from_hub(raw, dataset_key, seed)
