from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from fineqcomp.config import ModelSpec
from fineqcomp.data import Example
from fineqcomp.modeling import (
    CausalExampleDataset,
    causal_collate,
    compute_dtype,
    model_source,
    validate_rope_config,
)


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


def test_unknown_yarn_attention_factor_fails_before_model_load():
    with pytest.raises(ValueError, match="silently changed YaRN scale"):
        validate_rope_config(
            SimpleNamespace(
                rope_scaling={
                    "rope_type": "yarn",
                    "factor": 4.0,
                    "attn_factor": 0.8782488562869419,
                }
            )
        )

    validate_rope_config(
        SimpleNamespace(
            rope_scaling={
                "rope_type": "yarn",
                "factor": 4.0,
                "attention_factor": 0.8782488562869419,
            }
        )
    )


def test_offline_model_source_resolves_the_pinned_snapshot(tmp_path, monkeypatch):
    calls = []

    def snapshot_download(name, revision, token, local_files_only):
        calls.append((name, revision, token, local_files_only))
        return tmp_path

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    spec = ModelSpec("key", "org/model", "abc123", "bf16")

    assert model_source(spec, "token") == str(tmp_path)
    assert calls == [("org/model", "abc123", "token", True)]


def _span_dataset(span, response=" work here The answer is: 42"):
    return CausalExampleDataset(
        Tokenizer(),
        [Example("x", "abc", response, {})],
        ModelSpec("m", "model", "revision", "bf16"),
        max_length=64,
        label_span=span,
        answer_marker="The answer is:",
    )


def test_label_spans_partition_the_response():
    """The two spans together cover exactly what the full span covers.

    Bits spent on the working and bits spent on the answer are only separable
    if the split is clean, so neither span may leak a token of the other.
    """
    full = _span_dataset("all")[0]["labels"]
    reasoning = _span_dataset("reasoning")[0]["labels"]
    answer = _span_dataset("answer")[0]["labels"]

    scored = full != -100
    assert ((reasoning != -100) & (answer != -100)).sum().item() == 0
    assert torch.equal((reasoning != -100) | (answer != -100), scored)
    assert reasoning[reasoning != -100].tolist() + answer[answer != -100].tolist() == (
        full[scored].tolist()
    )
    # " work here " is eleven characters, so eleven tokens of working.
    assert (reasoning != -100).sum().item() == 11


def test_a_missing_marker_fails_loudly():
    """A span that matches nothing must stop the run, not score zero tokens."""
    with pytest.raises(ValueError, match="The answer is:"):
        _span_dataset("answer", response=" no marker at all")


def test_the_marker_search_direction_is_a_property_of_the_marker():
    """`The answer is:` is the last occurrence; a code fence is the first.

    Searching a fenced response from the right puts the whole program in the
    reasoning span and scores the closing prose as the answer, which is the
    opposite of what the split means. Every dataset that defines a marker has
    to say which end it means, and the default stays the original one.
    """
    from fineqcomp.modeling import response_boundary

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    tokenizer = Tokenizer()
    rationale = "we know the answer is: not five so the answer is: seven"
    # Quoted mid-rationale, so only the last occurrence opens the answer.
    assert response_boundary(tokenizer, rationale, "the answer is:") == 8
    assert response_boundary(
        tokenizer, rationale, "the answer is:", from_end=False
    ) == 2

    fenced = "here is why it fails ``` def f ( ) : pass ``` and that is all"
    first = response_boundary(tokenizer, fenced, "```", from_end=False)
    last = response_boundary(tokenizer, fenced, "```")
    assert first == 5
    assert last == 12
    assert response_boundary(tokenizer, "no marker here", "```") is None
