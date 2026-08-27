from __future__ import annotations

from types import SimpleNamespace

import torch

from fineqcomp.config import ModelSpec
from fineqcomp.data import Example
import fineqcomp.evaluation as evaluation
from fineqcomp.evaluation import (
    evaluate_multiple_choice,
    evaluate_natural,
    generate_response_records,
)


class _Tokenizer:
    label_ids = {" A": 1, " B": 2, " C": 3}

    def encode(self, text, add_special_tokens=False):
        if text in self.label_ids:
            return [self.label_ids[text]]
        return [9]

    def __call__(self, prompts, return_tensors, padding):
        size = len(prompts)
        return {
            "input_ids": torch.tensor([[7, 8]] * size),
            "attention_mask": torch.ones((size, 2), dtype=torch.long),
        }


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 2)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask):
        logits = torch.zeros((len(input_ids), input_ids.shape[1], 10))
        logits[0, -1, 2] = 4
        logits[1, -1, 1] = 4
        return SimpleNamespace(logits=logits)


def test_multiple_choice_evaluation_masks_absent_choices():
    examples = [
        Example("one", "question one", " B", {"label_index": 1, "choice_count": 3}),
        Example("two", "question two", " A", {"label_index": 0, "choice_count": 2}),
    ]
    model = ModelSpec("model", "model", "revision", "bf16")

    metrics, predictions = evaluate_multiple_choice(
        _Model(), _Tokenizer(), examples, model, [" A", " B", " C"], 2
    )

    assert metrics["accuracy"] == 1.0
    assert [row["prediction"] for row in predictions] == [1, 0]


def test_math_and_xsum_share_one_paired_evaluation(monkeypatch):
    examples = [
        Example(
            "math-0",
            "solve",
            r"Work. \boxed{2}",
            {"evaluator": "math", "level": "1", "subject": "Algebra"},
        ),
        Example(
            "xsum-0",
            "summarize",
            "The cat slept.",
            {"evaluator": "xsum"},
        ),
    ]
    answers = {"solve": r"The answer is \boxed{2}.", "summarize": "The cat slept."}

    def fake_generate(unused_model, unused_tokenizer, rows, *unused, **unused_kw):
        return [
            {
                "response": answers[row.prompt],
                "completion_tokens": 12,
                "terminated_with_eos": True,
                "hit_generation_limit": False,
            }
            for row in rows
        ]

    monkeypatch.setattr(evaluation, "generate_response_records", fake_generate)
    model_spec = ModelSpec("model", "model", "revision", "bf16")

    metrics, predictions = evaluate_natural(
        object(), object(), examples, model_spec, "mixed", 1
    )

    assert metrics["evaluations"]["math"]["exact_match"] == 1.0
    assert metrics["evaluations"]["math"]["terminated_fraction"] == 1.0
    assert metrics["primary_evaluator"] == "math"
    assert metrics["evaluations"]["xsum"]["rouge_l"] == 1.0
    assert {row["evaluator"] for row in predictions} == {"math", "xsum"}


def test_generation_records_distinguish_eos_from_hitting_the_limit():
    class Tokenizer:
        padding_side = "right"
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, prompts, return_tensors, padding):
            return {
                "input_ids": torch.tensor([[7, 8]] * len(prompts)),
                "attention_mask": torch.ones((len(prompts), 2), dtype=torch.long),
            }

        def decode(self, tokens, skip_special_tokens):
            return " ".join(map(str, tokens.tolist()))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 2)

        def get_input_embeddings(self):
            return self.embedding

        def generate(self, input_ids, **unused):
            completion = torch.tensor([[5, 2, 0], [6, 7, 8]])
            return torch.cat([input_ids, completion], dim=1)

    records = generate_response_records(
        Model(),
        Tokenizer(),
        [Example("a", "p", "r", {}), Example("b", "p", "r", {})],
        ModelSpec("model", "model", "revision", "bf16"),
        2,
        3,
    )

    assert records[0]["terminated_with_eos"] is True
    assert records[0]["completion_tokens"] == 2
    assert records[1]["hit_generation_limit"] is True
    assert records[1]["completion_tokens"] == 3


def test_exact_string_normaliser_cuts_the_continuation():
    """Nothing stops generation at the end of a short answer.

    The cut has to be at a blank line or a section marker rather than the first
    newline: a SQL query may span several lines, and truncating it would score
    a right answer wrong.
    """
    from fineqcomp.evaluation import _normalize_answer_text

    assert _normalize_answer_text(" SELECT  a FROM b ;") == "select a from b"
    assert (
        _normalize_answer_text(" SELECT a\n FROM b;\n\n### Question: next")
        == "select a from b"
    )
    assert _normalize_answer_text(" SELECT a FROM b;") == _normalize_answer_text(
        "select a\nfrom b"
    )
    assert (
        _normalize_answer_text(" us-gaap:LiabilitiesCurrent\n\nQuestion: ...")
        == "us-gaap:liabilitiescurrent"
    )
