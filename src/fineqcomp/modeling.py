"""Pinned model loading and prompt-safe tokenization."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from fineqcomp.adapters import attach_adapter, unload_adapter
from fineqcomp.config import AdapterSpec, ModelSpec
from fineqcomp.data import Example


def model_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device  # type: ignore[attr-defined]
    except (AttributeError, StopIteration):
        return next(model.parameters()).device


def render_prompt(tokenizer: Any, prompt: str, model_spec: ModelSpec) -> str:
    if not model_spec.chat:
        return prompt
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if model_spec.disable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], **kwargs
        )
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], **kwargs
        )


def validate_single_token_labels(tokenizer: Any, labels: list[str]) -> list[int]:
    ids = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"synthetic label {label!r} must be one token, got ids={encoded}"
            )
        ids.append(int(encoded[0]))
    if len(ids) != len(set(ids)):
        raise ValueError("synthetic labels map to duplicate token IDs")
    return ids


class CausalExampleDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        tokenizer: Any,
        examples: list[Example],
        model_spec: ModelSpec,
        max_length: int,
    ) -> None:
        self.rows = []
        eos = tokenizer.eos_token_id
        for example in examples:
            prompt = render_prompt(tokenizer, example.prompt, model_spec)
            prompt_ids = tokenizer.encode(
                prompt, add_special_tokens=not model_spec.chat
            )
            response_ids = tokenizer.encode(example.response, add_special_tokens=False)
            if eos is not None:
                response_ids.append(int(eos))
            input_ids = (prompt_ids + response_ids)[:max_length]
            prompt_length = min(len(prompt_ids), len(input_ids))
            labels = [-100] * prompt_length + input_ids[prompt_length:]
            if not any(label != -100 for label in labels):
                continue
            self.rows.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )
        if not self.rows:
            raise ValueError("tokenization removed every training example")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index]


def causal_collate(
    rows: list[dict[str, torch.Tensor]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    length = max(row["input_ids"].numel() for row in rows)
    input_ids = torch.full((len(rows), length), pad_token_id, dtype=torch.long)
    labels = torch.full((len(rows), length), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), length), dtype=torch.long)
    for index, row in enumerate(rows):
        size = row["input_ids"].numel()
        input_ids[index, :size] = row["input_ids"]
        labels[index, :size] = row["labels"]
        attention_mask[index, :size] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


@dataclass
class ModelSession:
    spec: ModelSpec
    model: torch.nn.Module
    tokenizer: Any

    @classmethod
    def load(cls, spec: ModelSpec) -> "ModelSession":
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(
            spec.name, revision=spec.revision, token=token, use_fast=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        kwargs: dict[str, Any] = {
            "revision": spec.revision,
            "token": token,
            "dtype": torch.bfloat16,
        }
        if torch.cuda.is_available():
            kwargs["device_map"] = {"": 0}
        if spec.backbone == "nf4":
            if not torch.cuda.is_available():
                raise RuntimeError("NF4 runs require CUDA and bitsandbytes")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(spec.name, **kwargs)
        if spec.backbone == "nf4":
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )
        else:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
        model.config.use_cache = False
        return cls(spec=spec, model=model, tokenizer=tokenizer)

    def attach(self, adapter: AdapterSpec, seed: int) -> torch.nn.Module:
        self.model = attach_adapter(self.model, adapter, seed)
        return self.model

    def unload(self) -> torch.nn.Module:
        self.model = unload_adapter(self.model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.model
