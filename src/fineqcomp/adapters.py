"""LoRA attachment and the shared seeded-A contract."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from typing import Any

import torch

from fineqcomp.config import AdapterSpec


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _resolve_targets(
    model: torch.nn.Module,
    target_modules: tuple[str, ...],
    last_n_layers: int | None,
) -> list[str]:
    candidates: list[tuple[str, int | None]] = []
    wanted = set(target_modules)
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in wanted or not isinstance(module, torch.nn.Linear):
            continue
        candidates.append((name, module_layer_index(name)))
    if not candidates:
        raise ValueError(f"no linear target modules found for {sorted(wanted)}")
    if last_n_layers is None:
        return sorted(name for name, _ in candidates)
    layers = sorted({layer for _, layer in candidates if layer is not None})
    if not layers:
        raise ValueError(
            "last_n_layers requested but transformer layers were not found"
        )
    selected = set(layers[-last_n_layers:])
    names = sorted(name for name, layer in candidates if layer in selected)
    if not names:
        raise ValueError("layer restriction removed every adapter target")
    return names


def module_layer_index(name: str) -> int | None:
    """Transformer layer a parameter or module name belongs to, if any."""
    match = _LAYER_PATTERN.search(name)
    return int(match.group(1)) if match else None


def resolve_target_modules(model: torch.nn.Module, spec: AdapterSpec) -> list[str]:
    """Resolve exact projection names, optionally within the final N layers."""
    return _resolve_targets(model, spec.target_modules, spec.last_n_layers)


def effective_adapter_rank(
    model: torch.nn.Module, spec: AdapterSpec, targets: list[str] | None = None
) -> int:
    """Match LoRA parameter counts to an optional reference placement."""
    if not spec.reference_target_modules:
        return spec.rank
    if spec.reference_rank is None:
        raise ValueError("reference_rank is required for a matched adapter budget")
    selected = targets or resolve_target_modules(model, spec)
    reference = _resolve_targets(
        model, spec.reference_target_modules, spec.last_n_layers
    )
    modules = dict(model.named_modules())

    def units(names: list[str]) -> int:
        return sum(
            modules[name].in_features + modules[name].out_features for name in names
        )

    target_units = units(selected)
    reference_units = units(reference)
    return max(1, round(spec.reference_rank * reference_units / target_units))


def _tensor_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"fineqcomp-seeded-a-v1:{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def seeded_a_tensor(shape: tuple[int, ...], seed: int, name: str) -> torch.Tensor:
    """Recreate a LoRA-A tensor without storing dataset-dependent values."""
    if len(shape) != 2:
        raise ValueError(f"LoRA-A must be a matrix, got {shape}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_tensor_seed(seed, name))
    return torch.randn(shape, generator=generator, dtype=torch.float32) / math.sqrt(
        shape[1]
    )


def attach_adapter(
    base_model: torch.nn.Module, spec: AdapterSpec, seed: int
) -> torch.nn.Module:
    """Attach PEFT LoRA and enforce the seeded-B or full-LoRA contract."""
    from peft import LoraConfig, TaskType, get_peft_model

    targets = resolve_target_modules(base_model, spec)
    rank = effective_adapter_rank(base_model, spec, targets)
    config = LoraConfig(
        r=rank,
        lora_alpha=spec.alpha if spec.alpha is not None else 2 * rank,
        lora_dropout=spec.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=targets,
    )
    cuda_devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        model = get_peft_model(base_model, config)
    if spec.method == "seeded_b":
        for name, parameter in model.named_parameters():
            if "lora_A" in name:
                initial = seeded_a_tensor(tuple(parameter.shape), seed, name)
                parameter.data.copy_(initial.to(parameter.device, parameter.dtype))
                parameter.requires_grad_(False)
            elif "lora_B" in name:
                parameter.requires_grad_(True)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable == 0:
        raise ValueError("adapter has no trainable parameters")
    return model


def adapter_tensors(model: torch.nn.Module, method: str) -> dict[str, torch.Tensor]:
    """Return tensors that must cross the finetuning information channel."""
    tensors = {}
    for name, parameter in model.named_parameters():
        include = "lora_B" in name or (method == "full_lora" and "lora_A" in name)
        if include:
            tensors[name] = parameter.detach().cpu().float().clone()
    if not tensors:
        raise ValueError("model has no encodable adapter tensors")
    return tensors


def apply_adapter_tensors(
    model: torch.nn.Module, tensors: Mapping[str, torch.Tensor]
) -> None:
    parameters = dict(model.named_parameters())
    missing = sorted(set(tensors) - set(parameters))
    if missing:
        raise KeyError(f"adapter tensors not found in model: {missing[:3]}")
    with torch.no_grad():
        for name, value in tensors.items():
            target = parameters[name]
            if tuple(target.shape) != tuple(value.shape):
                raise ValueError(
                    f"shape mismatch for {name}: {target.shape} != {value.shape}"
                )
            target.copy_(value.to(target.device, target.dtype))


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable_state(
    model: torch.nn.Module, state: Mapping[str, torch.Tensor]
) -> None:
    apply_adapter_tensors(model, state)


def unload_adapter(model: Any) -> torch.nn.Module:
    if not hasattr(model, "unload"):
        raise TypeError("expected a PEFT model with unload()")
    return model.unload()
