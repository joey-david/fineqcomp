"""Expand a campaign config into stable run records."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from fineqcomp.config import AdapterSpec, ModelSpec, RunSpec, TrainingSpec


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _model(key: str, raw: dict[str, Any], backbone: str | None = None) -> ModelSpec:
    spec = dict(raw["models"][key])
    if backbone is not None:
        spec["backbone"] = backbone
    return ModelSpec(key=key, **spec)


def _adapter(key: str, raw: dict[str, Any]) -> AdapterSpec:
    spec = dict(raw["adapters"][key])
    targets = tuple(spec.pop("target_modules"))
    rank = int(spec["rank"])
    spec.setdefault("alpha", 2 * rank)
    return AdapterSpec(key=key, target_modules=targets, **spec)


def _training(key: str, raw: dict[str, Any]) -> TrainingSpec:
    return TrainingSpec(**raw["training"][key])


def _run_id(parts: list[str], payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    return "__".join(_slug(part) for part in parts) + f"__{digest}"


def expand_campaign(raw: dict[str, Any]) -> list[RunSpec]:
    """Expand all study products and reject duplicate run IDs."""
    precisions = tuple(map(int, raw["precisions"]))
    clips = tuple(map(float, raw.get("clip_percentiles", [100.0])))
    runs: list[RunSpec] = []
    for study_name, study in raw["studies"].items():
        kind = study["kind"]
        models = study["models"]
        adapters = study["adapters"]
        seeds = list(map(int, study["seeds"]))
        if kind == "synthetic":
            cells: Iterable[tuple[int | None, int | None, str | None]] = (
                (int(family_count), None, None)
                for family_count in study["family_counts"]
            )
        elif kind == "controlled":
            cells = (
                (None, int(binding_count), None)
                for binding_count in study["binding_counts"]
            )
        elif kind == "natural":
            cells = ((None, None, str(dataset)) for dataset in study["datasets"])
        else:
            raise ValueError(f"study {study_name}: unknown kind {kind!r}")

        expanded_cells = list(cells)
        for model_key in models:
            model = _model(model_key, raw, study.get("backbone"))
            for adapter_key in adapters:
                adapter = _adapter(adapter_key, raw)
                for seed in seeds:
                    for family_count, binding_count, dataset_key in expanded_cells:
                        training_key = study.get("training_by_dataset", {}).get(
                            dataset_key, study["training"]
                        )
                        training = _training(training_key, raw)
                        payload = {
                            "study": study_name,
                            "kind": kind,
                            "model": model.key,
                            "model_revision": model.revision,
                            "backbone": model.backbone,
                            "adapter": adapter.key,
                            "seed": seed,
                            "family_count": family_count,
                            "binding_count": binding_count,
                            "dataset": dataset_key,
                        }
                        data_name = (
                            f"k{family_count}"
                            if family_count is not None
                            else f"n{binding_count}"
                            if binding_count is not None
                            else dataset_key
                        )
                        run_id = _run_id(
                            [
                                study_name,
                                model.key,
                                str(data_name),
                                adapter.key,
                                f"s{seed}",
                            ],
                            payload,
                        )
                        runs.append(
                            RunSpec(
                                run_id=run_id,
                                study=study_name,
                                kind=kind,
                                model=model,
                                adapter=adapter,
                                seed=seed,
                                precisions=precisions,
                                clip_percentiles=clips,
                                training=training,
                                family_count=family_count,
                                binding_count=binding_count,
                                dataset_key=dataset_key,
                            )
                        )
    ids = [run.run_id for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("campaign produced duplicate run IDs")
    return sorted(runs, key=lambda run: run.run_id)


def write_manifest(runs: list[RunSpec], path: str | Path) -> Path:
    """Write JSONL atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w") as stream:
        for run in runs:
            stream.write(json.dumps(run.to_dict(), sort_keys=True) + "\n")
    temporary.replace(target)
    return target


def read_manifest(path: str | Path) -> list[RunSpec]:
    with Path(path).open() as stream:
        return [RunSpec.from_dict(json.loads(line)) for line in stream if line.strip()]


def validate_manifest(runs: list[RunSpec], raw: dict[str, Any]) -> None:
    """Reject a stale manifest before it can mix two campaign definitions."""
    if runs != expand_campaign(raw):
        raise ValueError("manifest does not match config; run fineqcomp prepare again")
