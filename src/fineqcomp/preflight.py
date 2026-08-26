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
from fineqcomp.codec import (
    decode_adapter_tensor_map,
    encode_loraquant_tensor_map,
    encode_tensor_map,
)
from fineqcomp.config import RunSpec, TrainingSpec
from fineqcomp.data import (
    load_natural_dataset,
    natural_data_dir,
    restore_aligned_rationales,
    validate_natural_dataset,
)
from fineqcomp.evaluation import generate_responses
from fineqcomp.modeling import (
    CausalExampleDataset,
    ModelSession,
    model_source,
    validate_single_token_labels,
)
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
        "math_verify",
        "rouge_score",
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


def validate_prepared(
    campaign: dict[str, Any], runs: list[RunSpec], root: str | Path
) -> int:
    cells = {
        (str(run.dataset_key), run.seed)
        for run in runs
        if run.dataset_key is not None
    }
    for dataset_key, seed in cells:
        spec = campaign["datasets"][dataset_key]
        validate_natural_dataset(
            natural_data_dir(root, dataset_key, seed),
            dataset_key,
            seed,
            (evaluation["key"] for evaluation in spec.get("evaluations", [])),
            spec.get("rationale_control"),
            spec.get("answer_marker"),
            bool(spec.get("answer_marker_from_end", True)),
        )
    return len(cells)


def validate_tokenizers(campaign: dict[str, Any]) -> dict[str, list[int]]:
    from transformers import AutoTokenizer

    labels = list(map(str, campaign.get("multiple_choice_labels", [])))
    if not labels:
        return {}
    output = {}
    for key, model in campaign["models"].items():
        tokenizer = AutoTokenizer.from_pretrained(
            model["name"], revision=model["revision"], use_fast=True
        )
        output[key] = validate_single_token_labels(tokenizer, labels)
    return output


def validate_rationale_tokenization(
    campaign: dict[str, Any], runs: list[RunSpec], root: str | Path
) -> dict[str, dict[str, float | int]]:
    """Bound the token-limit change caused by the matched rationale control."""
    cells: dict[tuple[str, str, int, int, str], RunSpec] = {}
    for run in runs:
        if run.dataset_key is None:
            continue
        spec = campaign["datasets"][run.dataset_key]
        if spec.get("rationale_control") is None:
            continue
        key = (
            run.model.key,
            run.dataset_key,
            run.seed,
            run.training.max_length,
            run.training.label_span,
        )
        cells.setdefault(key, run)
    if not cells:
        return {}

    from transformers import AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    reports = {}
    for key, run in sorted(cells.items()):
        spec = campaign["datasets"][str(run.dataset_key)]
        source = model_source(run.model, token)
        pinned = (
            {"revision": run.model.revision, "token": token}
            if source == run.model.name and not Path(source).is_dir()
            else {}
        )
        tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, **pinned)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        data = load_natural_dataset(
            campaign, str(run.dataset_key), run.seed, root
        )
        marker = str(spec["answer_marker"])
        from_end = bool(spec.get("answer_marker_from_end", True))
        staged = data["train"]
        aligned = restore_aligned_rationales(
            staged, marker, from_end=from_end
        )

        def scored_tokens(examples):
            dataset = CausalExampleDataset(
                tokenizer,
                examples,
                run.model,
                run.training.max_length,
                run.training.label_span,
                marker,
                from_end,
            )
            return sum(
                int((row["labels"] != -100).sum().item()) for row in dataset.rows
            )

        aligned_tokens = scored_tokens(aligned)
        permuted_tokens = scored_tokens(staged)
        relative_delta = abs(permuted_tokens - aligned_tokens) / max(
            aligned_tokens, 1
        )
        if relative_delta > 0.01:
            raise ValueError(
                f"{run.model.key}/{run.dataset_key}/seed{run.seed}: rationale "
                f"permutation changes scored tokens by {relative_delta:.3%}"
            )
        reports["/".join(map(str, key))] = {
            "aligned_tokens": aligned_tokens,
            "permuted_tokens": permuted_tokens,
            "relative_delta": relative_delta,
        }
    return reports


def model_smoke(
    campaign: dict[str, Any], runs: list[RunSpec], prepared_root: str | Path
) -> dict[str, Any]:
    run = runs[0]
    session = ModelSession.load(run.model)
    try:
        if campaign["datasets"][str(run.dataset_key)].get("task_type") == "multiple_choice":
            validate_single_token_labels(
                session.tokenizer,
                list(map(str, campaign["multiple_choice_labels"])),
            )
        data = load_natural_dataset(
            campaign, str(run.dataset_key), run.seed, prepared_root
        )
        generated = []
        if (
            campaign["datasets"][str(run.dataset_key)].get("task_type")
            != "multiple_choice"
        ):
            # Loading and two training steps did not catch Gemma 2's mixed
            # FP32/BF16 generation-cache failure. Force the real cached
            # inference path before the adapter smoke starts.
            generated = generate_responses(
                session.model,
                session.tokenizer,
                data["test"][:1],
                run.model,
                batch_size=1,
                max_new_tokens=2,
            )
        train = data["train"][:2]
        calibration = data["calibration"][:2]
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
        _, decoded = decode_adapter_tensor_map(path)
        apply_adapter_tensors(session.model, decoded)
        path.unlink()
        loraquant_path = Path(prepared_root) / ".preflight_loraquant.fqcb"
        loraquant_storage = encode_loraquant_tensor_map(
            tensors,
            loraquant_path,
            high_bits=2,
            variance_ratio=0.8,
            group_size=128,
            optimize_steps=2,
            metadata={"preflight": True},
        )
        _, decoded = decode_adapter_tensor_map(loraquant_path)
        apply_adapter_tensors(session.model, decoded)
        loraquant_path.unlink()
        return {
            "generation_examples": len(generated),
            "training": training,
            "storage": storage,
            "loraquant_storage": loraquant_storage,
        }
    finally:
        if hasattr(session.model, "unload"):
            session.unload()


def write_report(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
