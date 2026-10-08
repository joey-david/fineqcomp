"""LoRA attachment and the shared seeded-A contract."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from pathlib import Path
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
    if spec.init != "default":
        regauge_initial_a(model, spec, seed)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable == 0:
        raise ValueError("adapter has no trainable parameters")
    return model


def regauge_initial_a(model: torch.nn.Module, spec: AdapterSpec, seed: int) -> None:
    """Start every LoRA-A from Q A0 rather than PEFT's A0, with B still zero.

    (B Q^-1)(Q A0) is zero like B A0, so the model before training is the
    same; only the factors the optimiser starts from differ. Q is drawn per
    module from the run seed and the parameter name.
    """
    from fineqcomp.codec import gauge_matrix

    gauge = {"rotated": "orthogonal", "conditioned": "conditioned"}[spec.init]
    for name, parameter in model.named_parameters():
        if "lora_A" not in name:
            continue
        generator = torch.Generator().manual_seed(
            _tensor_seed(seed, f"init-{spec.init}-{spec.init_kappa:g}:{name}")
        )
        q, _ = gauge_matrix(gauge, int(parameter.shape[0]), generator, spec.init_kappa)
        start = q @ parameter.detach().to("cpu", torch.float64)
        parameter.data.copy_(start.to(parameter.device, parameter.dtype))


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


def save_adapter(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(tensors, temporary)
    temporary.replace(path)


def lora_pairs(tensors: dict[str, torch.Tensor]) -> list[tuple[str, str]]:
    """Every (lora_A, lora_B) name pair in a tensor map, in a fixed order."""
    pairs = []
    for a_name in sorted(name for name in tensors if ".lora_A." in name):
        b_name = a_name.replace(".lora_A.", ".lora_B.")
        if b_name not in tensors:
            raise KeyError(f"{a_name} has no matching lora_B tensor")
        pairs.append((a_name, b_name))
    if not pairs:
        raise ValueError("tensor map holds no LoRA factor pairs")
    return pairs


def frobenius_inner(
    a_left: torch.Tensor,
    b_left: torch.Tensor,
    a_right: torch.Tensor,
    b_right: torch.Tensor,
) -> float:
    """<B_l A_l, B_r A_r>_F without ever forming a dense weight update.

    tr((B_l A_l)^T B_r A_r) = tr((B_l^T B_r)(A_r A_l^T)), and both factors are
    r x r. The dense update for one `gate_proj` is 68M values; this is two
    matrices of at most 16 x 16.
    """
    left = b_left.T @ b_right
    right = a_right @ a_left.T
    return float((left * right.T).sum())


def update_geometry(
    tensors: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    """Frobenius norm of an update, and its cosine against a reference update.

    Both are summed over target modules, which treats the concatenation of every
    module's update as one vector -- the same object a single scalar `alpha`
    rescales, so the norm reported here is exactly the quantity a shrinkage
    account has to move.

    The runtime `alpha / rank` scaling is a constant common to every condition
    here (all of them are padded back to the same rank-16 adapter spec and
    applied through it), so it cancels in the cosine and rescales every norm
    by the same factor. Norms are therefore comparable across conditions and
    are not absolute weight-space distances.
    """
    squared = 0.0
    cross = 0.0
    reference_squared = 0.0
    for a_name, b_name in lora_pairs(tensors):
        a, b = tensors[a_name], tensors[b_name]
        squared += frobenius_inner(a, b, a, b)
        if reference is None:
            continue
        ref_a, ref_b = reference[a_name], reference[b_name]
        cross += frobenius_inner(a, b, ref_a, ref_b)
        reference_squared += frobenius_inner(ref_a, ref_b, ref_a, ref_b)
    geometry = {"update_norm": squared ** 0.5}
    if reference is not None:
        denominator = (squared * reference_squared) ** 0.5
        geometry["cosine_to_clean"] = cross / denominator if denominator > 0 else 0.0
    return geometry
