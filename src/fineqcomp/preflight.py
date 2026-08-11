"""Local and remote checks that do not start the campaign."""

from __future__ import annotations

import importlib
import json
import os
import platform
from pathlib import Path
from typing import Any

import torch

from fineqcomp.adapters import adapter_tensors, apply_adapter_tensors
from fineqcomp.codec import decode_tensor_map, encode_tensor_map
from fineqcomp.config import RunSpec, TrainingSpec
from fineqcomp.data import (
    controlled_data_dir,
    ifeval_data_dir,
    natural_data_dir,
    read_jsonl,
    synthetic_data_dir,
    validate_controlled_dataset,
    validate_ifeval_dataset,
    validate_natural_dataset,
    validate_synthetic_dataset,
)
from fineqcomp.modeling import ModelSession, validate_single_token_labels
from fineqcomp.training import train_adapter


def environment_report(
    require_gpus: bool = False,
    require_dependencies: bool = False,
    expected_gpu_count: int | None = 2,
    min_gpu_memory_gib: float = 75.0,
) -> dict[str, Any]:
    modules = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "yaml",
        "instruction_following_eval",
        "matplotlib",
        "numpy",
    ):
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError:
            modules[name] = "missing"
        else:
            modules[name] = getattr(module, "__version__", "installed")
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": properties.name,
                    "memory_gib": properties.total_memory / 2**30,
                }
            )
    if require_gpus or require_dependencies:
        missing = [name for name, state in modules.items() if state == "missing"]
        if missing:
            raise RuntimeError(f"campaign dependencies are missing: {missing}")
    if require_gpus:
        if expected_gpu_count is not None and len(gpus) != expected_gpu_count:
            raise RuntimeError(
                f"campaign requires {expected_gpu_count} visible GPUs, found {len(gpus)}"
            )
        if any(gpu["memory_gib"] < min_gpu_memory_gib for gpu in gpus):
            raise RuntimeError(
                f"campaign requires GPUs with at least {min_gpu_memory_gib:g} GiB: {gpus}"
            )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "modules": modules,
        "gpus": gpus,
    }


def cache_models(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Download every pinned model snapshot and report its weight files."""
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cached = {}
    for key, model in campaign["models"].items():
        snapshot = Path(
            snapshot_download(
                model["name"], revision=model["revision"], token=token
            )
        )
        weights = sorted(
            path
            for pattern in ("*.safetensors", "*.bin")
            for path in snapshot.glob(pattern)
        )
        if not weights:
            raise RuntimeError(f"{model['name']}: snapshot has no model weights")
        cached[key] = {
            "model": model["name"],
            "revision": model["revision"],
            "snapshot": str(snapshot),
            "weight_files": len(weights),
            "weight_bytes": sum(path.stat().st_size for path in weights),
        }
    return cached


def validate_prepared(runs: list[RunSpec], root: str | Path) -> int:
    cells = {
        (int(run.family_count), run.seed)
        for run in runs
        if run.kind == "synthetic" and run.family_count is not None
    }
    for family_count, seed in cells:
        validate_synthetic_dataset(synthetic_data_dir(root, family_count, seed))
    controlled = {
        (int(run.binding_count), run.seed)
        for run in runs
        if run.kind == "controlled" and run.binding_count is not None
    }
    for binding_count, seed in controlled:
        validate_controlled_dataset(controlled_data_dir(root, binding_count, seed))
    natural = {
        (str(run.dataset_key), run.seed)
        for run in runs
        if run.kind == "natural" and run.dataset_key is not None
    }
    for dataset_key, seed in natural:
        validate_natural_dataset(natural_data_dir(root, dataset_key, seed), dataset_key, seed)
    if natural:
        validate_ifeval_dataset(ifeval_data_dir(root))
    return len(cells) + len(controlled) + len(natural) + bool(natural)


def validate_tokenizers(campaign: dict[str, Any]) -> dict[str, list[int]]:
    from transformers import AutoTokenizer

    labels = list(campaign["datasets"]["synthetic_codebook"]["labels"])
    output = {}
    for key, model in campaign["models"].items():
        tokenizer = AutoTokenizer.from_pretrained(
            model["name"], revision=model["revision"], use_fast=True
        )
        output[key] = validate_single_token_labels(tokenizer, labels)
    return output


def model_smoke(
    campaign: dict[str, Any], runs: list[RunSpec], prepared_root: str | Path
) -> dict[str, Any]:
    run = next(
        item
        for item in runs
        if item.study == "exact_seeded"
        and item.family_count == 4
        and item.adapter.key == "seeded_last1_r4"
    )
    session = ModelSession.load(run.model)
    try:
        labels = list(campaign["datasets"]["synthetic_codebook"]["labels"])
        validate_single_token_labels(session.tokenizer, labels)
        root = synthetic_data_dir(prepared_root, 4, run.seed)
        train = read_jsonl(root / "train.jsonl")[:2]
        calibration = read_jsonl(root / "calibration.jsonl")[:2]
        session.attach(run.adapter, run.seed)
        smoke_spec = TrainingSpec(
            epochs=1,
            learning_rate=run.training.learning_rate,
            effective_batch_size=1,
            micro_batch_size=1,
            max_length=run.training.max_length,
        )
        training = train_adapter(
            session.model,
            session.tokenizer,
            train,
            calibration,
            run.model,
            smoke_spec,
            run.seed,
            Path(prepared_root) / ".preflight_training.jsonl",
        )
        tensors = adapter_tensors(session.model, run.adapter.method)
        path = Path(prepared_root) / ".preflight_adapter.fqcb"
        storage = encode_tensor_map(tensors, path, 4, metadata={"preflight": True})
        _, decoded = decode_tensor_map(path)
        apply_adapter_tensors(session.model, decoded)
        path.unlink()
        return {"training": training, "storage": storage}
    finally:
        if hasattr(session.model, "unload"):
            session.unload()


def write_report(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
