from __future__ import annotations

import json

import fineqcomp.data as data
from fineqcomp.data import (
    Example,
    load_natural_dataset,
    prepare_natural_dataset,
    read_jsonl,
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

    assert converted[0].response == " B"
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
