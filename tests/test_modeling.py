from __future__ import annotations

import torch

from fineqcomp.config import ModelSpec
from fineqcomp.data import Example
from fineqcomp.modeling import CausalExampleDataset, causal_collate, compute_dtype


class Tokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        ids = [10 + (ord(char) % 50) for char in text]
        return ([2] if add_special_tokens else []) + ids


def test_prompt_tokens_are_masked_and_response_is_kept():
    tokenizer = Tokenizer()
    model = ModelSpec("m", "model", "revision", "bf16")
    dataset = CausalExampleDataset(
        tokenizer,
        [Example("x", "abc", " D", {})],
        model,
        max_length=32,
    )
    row = dataset[0]
    prompt_size = len(tokenizer.encode("abc", add_special_tokens=True))

    assert row["labels"][:prompt_size].tolist() == [-100] * prompt_size
    assert (
        row["labels"][prompt_size:].tolist() == row["input_ids"][prompt_size:].tolist()
    )
    batch = causal_collate([row, row], tokenizer.pad_token_id)
    assert batch["attention_mask"].sum().item() == 2 * len(row["input_ids"])


def test_compute_dtype_falls_back_on_older_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.is_bf16_supported", lambda: False)
    assert compute_dtype() == torch.float16
