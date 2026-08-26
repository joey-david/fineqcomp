from __future__ import annotations

import sys
from collections import Counter
from types import SimpleNamespace

import fineqcomp.data as data
import pytest
from fineqcomp.data import (
    Example,
    load_natural_dataset,
    permute_rationales,
    prepare_natural_dataset,
    restore_aligned_rationales,
    validate_natural_dataset,
)


def test_multiple_choice_conversion_uses_standard_labels_and_no_answer_text():
    rows = [
        {
            "question": "Which answer is right?",
            "choices": {"label": ["1", "2", "3"], "text": ["one", "two", "three"]},
            "answerKey": "2",
        }
    ]

    converted = data._convert_multiple_choice(
        rows,
        "test",
        "arc_challenge",
        {"question_field": "question"},
    )

    assert converted[0].response == "B"
    assert converted[0].metadata["label_index"] == 1
    assert converted[0].metadata["choice_count"] == 3
    assert "B. two" in converted[0].prompt


def test_literature_task_converters_keep_train_and_eval_contracts():
    metamath = data._convert_metamath(
        [{"query": "What is 2+2?", "response": "Work. \\boxed{4}"}], "train"
    )
    magicoder = data._convert_magicoder(
        [{"instruction": "Write add.", "response": "def add(a,b): return a+b"}],
        "train",
    )
    xsum = data._convert_xsum(
        [{"document": "A long report.", "summary": "A report."}], "test"
    )
    math = data._convert_math(
        [{"problem": "1+1", "solution": "\\boxed{2}", "level": "1", "type": "Algebra"}],
        "test",
    )

    assert "Question: What is 2+2?" in metamath[0].prompt
    assert magicoder[0].response.startswith("\ndef add")
    assert xsum[0].metadata["evaluator"] == "xsum"
    assert math[0].metadata["evaluator"] == "math"


def test_natural_data_is_staged_and_loaded_without_the_hub(tmp_path, monkeypatch):
    rows = {
        "train": [Example("train-0", "train", "answer", {"split": "train"})],
        "calibration": [
            Example("calibration-0", "calibration", "answer", {"split": "calibration"})
        ],
        "test": [Example("test-0", "test", "answer", {"split": "test"})],
    }
    raw = {"datasets": {"gsm8k": {"revision": "revision-test"}}}
    monkeypatch.setattr(data, "_load_natural_from_hub", lambda *_: rows)

    target = prepare_natural_dataset(raw, "gsm8k", 11, tmp_path)
    assert target.joinpath("metadata.json").is_file()

    def fail_if_downloaded(*_):
        raise AssertionError("staged data should avoid a Hub request")

    monkeypatch.setattr(data, "_load_natural_from_hub", fail_if_downloaded)
    assert load_natural_dataset(raw, "gsm8k", 11, tmp_path) == rows


def test_stale_staged_evaluator_fails_before_a_run(tmp_path):
    root = data.natural_data_dir(tmp_path, "renamed_sql_arm", 11)
    rows = {
        split: [Example(f"{split}-0", "p", "r", {"split": split})]
        for split in ("train", "calibration", "test")
    }
    root.mkdir(parents=True)
    for split, examples in rows.items():
        data._write_jsonl(root / f"{split}.jsonl", examples)
    data._write_json(
        root / "metadata.json",
        {
            "dataset_key": "renamed_sql_arm",
            "seed": 11,
            "train_rows": 1,
            "calibration_rows": 1,
            "test_rows": 1,
        },
    )
    raw = {
        "datasets": {
            "renamed_sql_arm": {
                "evaluations": [{"key": "text_to_sql"}],
            }
        }
    }

    with pytest.raises(ValueError, match="run prepare again"):
        load_natural_dataset(raw, "renamed_sql_arm", 11, tmp_path)


def test_rationale_permutation_keeps_text_but_breaks_problem_pairing():
    rows = [
        Example(
            f"r{index}",
            f"question {index}",
            f"work {index} {'x' * index} The answer is: {index}",
            {"source_problem": f"source-{index // 2}"},
        )
        for index in range(64)
    ]

    first = permute_rationales(rows, "The answer is:", 11, block_size=16)
    second = permute_rationales(rows, "The answer is:", 11, block_size=16)

    assert first == second
    assert [row.prompt for row in first] == [row.prompt for row in rows]
    assert [row.response.rsplit("The answer is:", 1)[1] for row in first] == [
        row.response.rsplit("The answer is:", 1)[1] for row in rows
    ]
    assert Counter(row.response.rsplit("The answer is:", 1)[0] for row in first) == (
        Counter(row.response.rsplit("The answer is:", 1)[0] for row in rows)
    )
    assert {row.metadata["rationale_donor_id"] for row in first} == {
        row.example_id for row in rows
    }
    assert all(
        row.example_id != moved.metadata["rationale_donor_id"]
        and row.metadata["source_problem"]
        != moved.metadata["rationale_donor_source_problem"]
        for row, moved in zip(rows, first, strict=True)
    )
    assert [row.response for row in restore_aligned_rationales(
        first, "The answer is:"
    )] == [row.response for row in rows]


def test_staged_rationale_control_checks_text_not_only_metadata(tmp_path):
    root = data.natural_data_dir(tmp_path, "controlled", 11)
    root.mkdir(parents=True)
    base = [
        Example(
            f"r{index}", f"question {index}",
            f"work {index} {'x' * index} The answer is: {index}",
            {"split": "train", "source_problem": f"source-{index}"},
        )
        for index in range(8)
    ]
    train = permute_rationales(base, "The answer is:", 11, block_size=8)
    calibration = [
        Example(
            row.example_id.replace("r", "c"), row.prompt, row.response,
            {**row.metadata, "split": "calibration"},
        )
        for row in base
    ]
    calibration = permute_rationales(
        calibration, "The answer is:", 12, block_size=8
    )
    test = [Example("t0", "test", "answer", {"split": "test"})]
    for split, rows in (
        ("train", train), ("calibration", calibration), ("test", test)
    ):
        data._write_jsonl(root / f"{split}.jsonl", rows)
    data._write_json(root / "metadata.json", {
        "dataset_key": "controlled", "seed": 11,
        "train_rows": len(train), "calibration_rows": len(calibration),
        "test_rows": len(test),
    })

    validate_natural_dataset(
        root, "controlled", 11, expected_rationale_control="permuted",
        answer_marker="The answer is:",
    )
    changed = data.read_jsonl(root / "train.jsonl")
    changed[0] = Example(
        changed[0].example_id, changed[0].prompt,
        "tampered work The answer is: 0", changed[0].metadata,
    )
    data._write_jsonl(root / "train.jsonl", changed)
    with pytest.raises(ValueError, match="rationale-control bijection"):
        validate_natural_dataset(
            root, "controlled", 11, expected_rationale_control="permuted",
            answer_marker="The answer is:",
        )


def test_evaluation_examples_carry_the_configured_evaluator(monkeypatch):
    """The scorer dispatches on this, and its fallback is silently wrong.

    Without it the campaign's dataset key stands in, which matched the
    evaluator name by luck in the first text-to-SQL panel and did not in the
    diversity sweep: `sql_div_100` has no scorer, so those runs died and took
    the fifteen waiting on their shared baseline with them. The loader stamps
    it from the config so no converter has to remember.
    """

    class Rows(list):
        def shuffle(self, seed):
            return self

        def select(self, indices):
            return Rows(self[index] for index in indices)

    rows = Rows(
        {
            "sql_context": "CREATE TABLE t (x INT)",
            "sql_prompt": "Read x",
            "sql": "SELECT x FROM t",
            "domain": "test",
        }
        for _ in range(4)
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(
            load_dataset=lambda *args, **kwargs: {"train": rows, "test": rows}
        ),
    )
    raw = {
        "datasets": {
            "renamed_sql_arm": {
                "train_source": {
                    "path": "sql",
                    "revision": "r",
                    "split": "train",
                    "converter": "text_to_sql",
                },
                "evaluations": [
                    {
                        "key": "text_to_sql",
                        "path": "sql",
                        "revision": "r",
                        "split": "test",
                        "converter": "text_to_sql",
                    }
                ],
                "validation_rows": 1,
                "train_rows": 2,
                "test_rows": 2,
            }
        }
    }
    loaded = data._load_natural_from_hub(raw, "renamed_sql_arm", 11)

    assert {row.metadata["evaluator"] for row in loaded["test"]} == {"text_to_sql"}


def test_paws_converter_keeps_the_binary_label_exact():
    rows = [
        {
            "sentence1": "The dog ran.",
            "sentence2": "A dog was running.",
            "label": 1,
        }
    ]

    converted = data._convert_paws(rows, "test")

    assert converted[0].response == " yes"
    assert converted[0].metadata == {
        "split": "test",
        "evaluator": "paws",
    }
