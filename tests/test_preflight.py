from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from fineqcomp.config import AdapterSpec, CodecSpec, ModelSpec, RunSpec, TrainingSpec
from fineqcomp.preflight import cache_models, model_smoke


def test_cache_models_downloads_exact_revisions_and_checks_weights(
    tmp_path, monkeypatch
):
    calls = []

    def snapshot_download(model, revision, token):
        calls.append((model, revision, token))
        target = tmp_path / revision
        target.mkdir()
        (target / "model.safetensors").write_bytes(b"weights")
        return target

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    campaign = {
        "models": {
            "small": {"name": "org/model", "revision": "abc123"},
        }
    }

    report = cache_models(campaign)

    assert calls == [("org/model", "abc123", None)]
    assert report["small"]["weight_files"] == 1
    assert report["small"]["weight_bytes"] == 7


def test_model_smoke_filters_before_loading(monkeypatch, tmp_path):
    def run(model: str, dataset: str) -> RunSpec:
        return RunSpec(
            run_id=f"{model}-{dataset}",
            study="smoke",
            kind="natural",
            model=ModelSpec(model, model, "r", "bf16"),
            adapter=AdapterSpec("a", "full_lora", 1, ("q_proj",), None, 2),
            seed=11,
            codecs=(CodecSpec("binary", "uniform", bits=1),),
            training=TrainingSpec(1, 1e-4, 1, 1, 32),
            dataset_key=dataset,
        )

    runs = [run("small", "math"), run("large", "code")]
    loaded = []

    def load(spec):
        loaded.append(spec.key)
        raise RuntimeError("stop after selection")

    monkeypatch.setattr("fineqcomp.preflight.ModelSession.load", load)
    with pytest.raises(RuntimeError, match="stop after selection"):
        model_smoke(
            {"datasets": {"math": {}, "code": {}}},
            runs,
            tmp_path,
            model_key="large",
            dataset_key="code",
        )
    assert loaded == ["large"]

    with pytest.raises(ValueError, match="no run matches"):
        model_smoke(
            {"datasets": {}}, runs, tmp_path, model_key="missing"
        )
