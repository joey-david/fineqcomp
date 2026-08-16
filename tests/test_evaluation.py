from __future__ import annotations

from types import SimpleNamespace

import torch

from fineqcomp.config import ModelSpec
from fineqcomp.data import Example
import fineqcomp.evaluation as evaluation
from fineqcomp.evaluation import evaluate_multiple_choice, evaluate_natural


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
        return [answers[row.prompt] for row in rows]

    monkeypatch.setattr(evaluation, "generate_responses", fake_generate)
    model_spec = ModelSpec("model", "model", "revision", "bf16")

    metrics, predictions = evaluate_natural(
        object(), object(), examples, model_spec, "mixed", 1
    )

    assert metrics["evaluations"]["math"]["exact_match"] == 1.0
    assert metrics["primary_evaluator"] == "math"
    assert metrics["evaluations"]["xsum"]["rouge_l"] == 1.0
    assert {row["evaluator"] for row in predictions} == {"math", "xsum"}
