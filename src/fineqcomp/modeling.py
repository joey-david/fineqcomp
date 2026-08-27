"""Pinned model loading and prompt-safe tokenization."""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from fineqcomp.adapters import attach_adapter, unload_adapter
from fineqcomp.config import AdapterSpec, ModelSpec
from fineqcomp.data import Example


def model_source(spec: ModelSpec, token: str | None) -> str:
    """Use the exact local snapshot path when Hub access is disabled.

    A name that is already a directory is taken as the snapshot itself. Shared
    cluster model stores (Jean-Zay's `$DSDIR`) hold plain directories rather
    than a Hub cache, so there is nothing for `snapshot_download` to resolve.
    """
    if Path(spec.name).is_dir():
        return spec.name
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        return spec.name
    from huggingface_hub import snapshot_download

    return str(
        Path(
            snapshot_download(
                spec.name,
                revision=spec.revision,
                token=token,
                local_files_only=True,
            )
        )
    )


def compute_dtype() -> torch.dtype:
    """Use BF16 where the GPU supports it and FP16 on older CUDA cards."""
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        return torch.float16
    return torch.bfloat16


def validate_rope_config(config: Any) -> None:
    """Reject a known YaRN key that Transformers warns about and ignores."""
    rope = getattr(config, "rope_scaling", None)
    if (
        isinstance(rope, dict)
        and "attn_factor" in rope
        and "attention_factor" not in rope
    ):
        raise ValueError(
            "model rope_scaling uses 'attn_factor', but Transformers reads "
            "'attention_factor'; refusing to run with a silently changed YaRN scale"
        )


def inference_autocast(model: torch.nn.Module, device: torch.device):
    """Keep Gemma 2 QLoRA attention inputs in one dtype during inference.

    PEFT prepares non-quantized weights in FP32. Gemma 2's rotary path then
    promotes the query while its generation cache keeps keys and values in
    BF16. Its eager attention path under autocast restores the intended BF16
    compute without changing the stored model or the other model families.
    """
    config = getattr(model, "config", None)
    if device.type == "cuda" and getattr(config, "model_type", None) == "gemma2":
        return torch.amp.autocast("cuda", dtype=compute_dtype())
    return nullcontext()


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
                f"answer label {label!r} must be one token, got ids={encoded}"
            )
        ids.append(int(encoded[0]))
    if len(ids) != len(set(ids)):
        raise ValueError("answer labels map to duplicate token IDs")
    return ids


LABEL_SPANS = ("all", "reasoning", "answer")


def response_boundary(
    tokenizer: Any, response: str, marker: str, *, from_end: bool = True
) -> int | None:
    """Token offset inside a response where the final answer begins.

    `The answer is:` is looked up from the right, because a rationale may quote
    the phrase on its way to the conclusion and it is the last occurrence that
    starts the answer. A code fence is the opposite: the first ``` opens the
    answer and the last one closes it, so searching from the right would put
    the whole program in the reasoning span and score the trailing prose as the
    answer. The direction is therefore a property of the marker, not a
    universal rule, and every dataset that defines a marker has to say which it
    means.

    Tokenizing the text before the marker and taking its length puts the
    boundary on a token edge without re-tokenizing the whole response two
    different ways.
    """
    cut = response.rfind(marker) if from_end else response.find(marker)
    if cut < 0:
        return None
    return len(tokenizer.encode(response[:cut], add_special_tokens=False))


class CausalExampleDataset(Dataset[dict[str, torch.Tensor]]):
    """Tokenized prompt-masked SFT rows, optionally masked further by span.

    `label_span` narrows the loss to one part of the response: `reasoning` for
    the working that precedes the final answer, `answer` for the answer itself.
    A row whose response does not contain `answer_marker` has no boundary and
    is dropped, so a span measure never silently scores the wrong tokens.
    """

    def __init__(
        self,
        tokenizer: Any,
        examples: list[Example],
        model_spec: ModelSpec,
        max_length: int,
        label_span: str = "all",
        answer_marker: str | None = None,
        answer_marker_from_end: bool = True,
    ) -> None:
        if label_span not in LABEL_SPANS:
            raise ValueError(f"label span must be one of {LABEL_SPANS}")
        if label_span != "all" and not answer_marker:
            raise ValueError(f"the {label_span} span needs an answer marker")
        self.rows = []
        self.dropped_without_marker = 0
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
            if label_span != "all":
                boundary = response_boundary(
                    tokenizer,
                    example.response,
                    str(answer_marker),
                    from_end=answer_marker_from_end,
                )
                if boundary is None:
                    self.dropped_without_marker += 1
                    continue
                split = min(prompt_length + boundary, len(labels))
                span = (
                    range(split, len(labels))
                    if label_span == "reasoning"
                    else range(prompt_length, split)
                )
                for position in span:
                    labels[position] = -100
            if not any(label != -100 for label in labels):
                continue
            self.rows.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )
        if not self.rows:
            if self.dropped_without_marker:
                raise ValueError(
                    f"no response contained {answer_marker!r}, so the"
                    f" {label_span} span scored nothing"
                )
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
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            __version__ as transformers_version,
        )

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        source = model_source(spec, token)
        pinned_kwargs = (
            {"revision": spec.revision, "token": token}
            if source == spec.name and not Path(source).is_dir()
            else {}
        )
        tokenizer = AutoTokenizer.from_pretrained(
            source, use_fast=True, **pinned_kwargs
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        kwargs: dict[str, Any] = dict(pinned_kwargs)
        config = AutoConfig.from_pretrained(source, **pinned_kwargs)
        validate_rope_config(config)
        kwargs["config"] = config
        if getattr(config, "model_type", None) == "gemma2":
            # Transformers recommends eager attention for Gemma 2 training.
            # It also lets autocast resolve PEFT's FP32/BF16 inference mix.
            kwargs["attn_implementation"] = "eager"
        dtype_key = (
            "dtype"
            if int(transformers_version.split(".", maxsplit=1)[0]) >= 5
            else "torch_dtype"
        )
        dtype = compute_dtype()
        kwargs[dtype_key] = dtype
        if torch.cuda.is_available():
            kwargs["device_map"] = {"": 0}
        if spec.backbone == "nf4":
            if not torch.cuda.is_available():
                raise RuntimeError("NF4 runs require CUDA and bitsandbytes")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
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
