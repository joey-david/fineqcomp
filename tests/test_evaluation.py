from __future__ import annotations

from types import SimpleNamespace

import torch

from fineqcomp.config import ModelSpec
from fineqcomp.data import Example
from fineqcomp.evaluation import evaluate_multiple_choice


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
