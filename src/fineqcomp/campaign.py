"""Expand a campaign config into stable run records."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from fineqcomp.config import AdapterSpec, CodecSpec, ModelSpec, RunSpec, TrainingSpec


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
    reference_targets = tuple(spec.pop("reference_target_modules", ()))
    rank = int(spec["rank"])
    if "alpha" not in spec:
        spec["alpha"] = 2 * rank
    return AdapterSpec(
        key=key,
        target_modules=targets,
        reference_target_modules=reference_targets,
        **spec,
    )


def _training(key: str, raw: dict[str, Any]) -> TrainingSpec:
    return TrainingSpec(**raw["training"][key])


def _codecs(raw: dict[str, Any], keys: Iterable[str]) -> tuple[CodecSpec, ...]:
    return tuple(CodecSpec(key=key, **raw["codecs"][key]) for key in keys)


def _run_id(parts: list[str], payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    return "__".join(_slug(part) for part in parts) + f"__{digest}"


def expand_campaign(raw: dict[str, Any]) -> list[RunSpec]:
    """Expand all study products and reject duplicate run IDs."""
    runs: list[RunSpec] = []
    for study_name, study in raw["studies"].items():
        kind = study["kind"]
        if kind != "natural":
            raise ValueError(f"study {study_name}: unknown kind {kind!r}")
        codecs = _codecs(raw, study.get("codecs", raw["codecs"]))
        seeds = list(map(int, study["seeds"]))
        for model_key in study["models"]:
            model = _model(model_key, raw, study.get("backbone"))
            for adapter_key in study["adapters"]:
                adapter = _adapter(adapter_key, raw)
                for seed in seeds:
                    for dataset_key in study["datasets"]:
                        training_key = study.get("training_by_dataset", {}).get(
                            dataset_key, study["training"]
                        )
                        # The training and test row caps belong in the run
                        # identity. Without them two configs that differ only
                        # in how much data they use produce the same run id,
                        # and the second silently reuses the first's trained
                        # adapter and codec metrics instead of recomputing.
                        dataset_spec = raw["datasets"][str(dataset_key)]
                        payload = {
                            "study": study_name,
                            "kind": kind,
                            "model": model.key,
                            "model_revision": model.revision,
                            "backbone": model.backbone,
                            "adapter": adapter.key,
                            "seed": seed,
                            "dataset": dataset_key,
                            "train_rows": dataset_spec.get("train_rows"),
                            "test_rows": dataset_spec.get("test_rows"),
                            "epochs": training_key,
                        }
                        # Two arms that differ only in how many distinct source
                        # problems their rows cover are different runs; without
                        # this they hash alike and the second silently reuses
                        # the first's adapter. The key is added only when a
                        # dataset sets it, so every identity minted before the
                        # lever existed stays exactly what it was.
                        if dataset_spec.get("response_transform") is not None:
                            payload["response_transform"] = str(
                                dataset_spec["response_transform"]
                            )
                        if dataset_spec.get("distinct_source_problems") is not None:
                            payload["distinct_source_problems"] = int(
                                dataset_spec["distinct_source_problems"]
                            )
                        # A study named after its dataset would otherwise
                        # repeat the name in every directory.
                        parts = [study_name, model.key]
                        if str(dataset_key) != study_name:
                            parts.append(str(dataset_key))
                        parts += [adapter.key, f"s{seed}"]
                        run_id = _run_id(parts, payload)
                        runs.append(
                            RunSpec(
                                run_id=run_id,
                                study=study_name,
                                kind=kind,
                                model=model,
                                adapter=adapter,
                                seed=seed,
                                codecs=codecs,
                                training=_training(training_key, raw),
                                dataset_key=str(dataset_key),
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
