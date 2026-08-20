"""Small explicit SFT loop with prompt masking and validation selection."""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from fineqcomp.adapters import restore_trainable_state, trainable_state
from fineqcomp.config import ModelSpec, TrainingSpec
from fineqcomp.data import Example
from fineqcomp.modeling import CausalExampleDataset, causal_collate, model_device


def autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float16
    return torch.bfloat16


@torch.no_grad()
def causal_nll(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Example],
    model_spec: ModelSpec,
    max_length: int,
    batch_size: int,
) -> dict[str, float | int]:
    dataset = CausalExampleDataset(tokenizer, examples, model_spec, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda rows: causal_collate(rows, tokenizer.pad_token_id),
    )
    device = model_device(model)
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        tokens = int((batch["labels"] != -100).sum().item())
        total_loss += float(output.loss.item()) * tokens
        total_tokens += tokens
    return {
        "nll": total_loss / max(total_tokens, 1),
        "bits_per_token": total_loss / max(total_tokens, 1) / math.log(2),
        "total_bits": total_loss / math.log(2),
        "nll_tokens": total_tokens,
    }


def train_adapter(
    model: torch.nn.Module,
    tokenizer: Any,
    train_examples: list[Example],
    calibration_examples: list[Example],
    model_spec: ModelSpec,
    spec: TrainingSpec,
    seed: int,
    log_path: str | Path,
) -> dict[str, Any]:
    """Train only marked adapter tensors and restore the best validation state."""
    if spec.effective_batch_size % spec.micro_batch_size:
        raise ValueError("effective batch size must divide by micro batch size")
    accumulation = spec.effective_batch_size // spec.micro_batch_size
    dataset = CausalExampleDataset(
        tokenizer, train_examples, model_spec, spec.max_length
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=spec.micro_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda rows: causal_collate(rows, tokenizer.pad_token_id),
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=spec.learning_rate,
        betas=(spec.adam_beta1, spec.adam_beta2),
        weight_decay=spec.weight_decay,
    )
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    total_updates = updates_per_epoch * spec.epochs
    warmup_steps = int(total_updates * spec.warmup_ratio)
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    device = model_device(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started_at = time.perf_counter()
    best_nll = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    update = 0
    log_target = Path(log_path)
    log_target.parent.mkdir(parents=True, exist_ok=True)
    def checkpoint(epoch: int, train_loss: float, log) -> None:
        """Score the held-out split and keep the best adapter state seen."""
        nonlocal best_nll, best_state
        validation = causal_nll(
            model,
            tokenizer,
            calibration_examples,
            model_spec,
            spec.max_length,
            spec.micro_batch_size,
        )
        record = {
            "epoch": epoch,
            "updates": update,
            "train_loss": train_loss,
            "validation_nll": validation["nll"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        log.write(json.dumps(record, sort_keys=True) + "\n")
        log.flush()
        if float(validation["nll"]) < best_nll:
            best_nll = float(validation["nll"])
            best_state = trainable_state(model)
        model.train()

    with log_target.open("a") as log:
        for epoch in range(spec.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            micro_steps = 0
            for batch_index, batch in enumerate(loader):
                batch = {key: value.to(device) for key, value in batch.items()}
                group_start = (batch_index // accumulation) * accumulation
                group_size = min(accumulation, len(loader) - group_start)
                amp = (
                    torch.amp.autocast("cuda", dtype=autocast_dtype(device))
                    if device.type == "cuda"
                    else nullcontext()
                )
                with amp:
                    loss = model(**batch).loss / group_size
                loss.backward()
                running_loss += float(loss.item()) * group_size
                micro_steps += 1
                at_boundary = (batch_index + 1) % accumulation == 0
                at_end = batch_index + 1 == len(loader)
                if at_boundary or at_end:
                    torch.nn.utils.clip_grad_norm_(parameters, spec.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    update += 1
                    # Score on a fixed update cadence and at the very last
                    # update, never at epoch ends. Arms that reach the same
                    # update count through different epoch counts must get the
                    # same number of candidates: best-state restore is a
                    # maximum over them, so more epochs would otherwise hand
                    # the duplicated arms a better checkpoint for free, biased
                    # along the axis the experiment is testing.
                    cadence = spec.eval_every_updates
                    if (cadence and update % cadence == 0) or update == total_updates:
                        checkpoint(
                            epoch + 1, running_loss / max(micro_steps, 1), log
                        )
    if best_state is None:
        raise RuntimeError("training did not produce a validation state")
    restore_trainable_state(model, best_state)
    return {
        "train_examples": len(dataset),
        "epochs": spec.epochs,
        "optimizer_updates": update,
        "best_validation_nll": best_nll,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "adapter_rank": next(iter(model.peft_config.values())).r
        if hasattr(model, "peft_config")
        else None,
        "elapsed_seconds": time.perf_counter() - started_at,
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
