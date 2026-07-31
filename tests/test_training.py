from __future__ import annotations

from types import SimpleNamespace

import torch

from fineqcomp.config import ModelSpec, TrainingSpec
from fineqcomp.data import Example
from fineqcomp.training import train_adapter


class Tokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        ids = [3 + (ord(char) % 20) for char in text]
        return ([2] if add_special_tokens else []) + ids


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 32)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, labels, attention_mask=None):
        logits = self.head(self.embedding(input_ids))
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss, logits=logits)


def test_training_runs_partial_accumulation_group_and_records_cost(tmp_path):
    model = TinyLM()
    examples = [Example(str(index), f"p{index}", " a", {}) for index in range(5)]
    metrics = train_adapter(
        model,
        Tokenizer(),
        examples,
        examples[:2],
        ModelSpec("tiny", "tiny", "local", "bf16"),
        TrainingSpec(
            epochs=1,
            learning_rate=1e-3,
            effective_batch_size=4,
            micro_batch_size=2,
            max_length=16,
        ),
        seed=11,
        log_path=tmp_path / "training.jsonl",
    )

    assert metrics["optimizer_updates"] == 2
    assert metrics["train_examples"] == 5
    assert metrics["elapsed_seconds"] > 0
    assert metrics["peak_memory_bytes"] is None
    assert (tmp_path / "training.jsonl").read_text().count("\n") == 1
