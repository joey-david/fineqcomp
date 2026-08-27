from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from fineqcomp.config import AdapterSpec, CodecSpec, ModelSpec, RunSpec, TrainingSpec
from fineqcomp.data import Example
from fineqcomp.preflight import (
    cache_models,
    model_smoke,
    validate_answer_retention,
)


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


def test_answer_retention_counts_answers_lost_to_max_length(monkeypatch, tmp_path):
    """A trace longer than max_length trains on working with no answer."""

    class CharTokenizer:
        def encode(self, text, add_special_tokens=True):
            return [ord(character) for character in text]

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: CharTokenizer()
            )
        ),
    )
    examples = [
        Example("short", "p", "work</think>42", {}),
        Example("long", "p", "w" * 40 + "</think>42", {}),
        Example("unmarked", "p", "work with no marker", {}),
    ]
    monkeypatch.setattr(
        "fineqcomp.preflight.load_natural_dataset",
        lambda campaign, key, seed, root: {"train": examples},
    )
    run = RunSpec(
        run_id="r",
        study="smoke",
        kind="natural",
        model=ModelSpec("small", "org/small", "rev", "bf16"),
        adapter=AdapterSpec("a", "full_lora", 1, ("q_proj",), None, 2),
        seed=11,
        codecs=(CodecSpec("binary", "uniform", bits=1),),
        training=TrainingSpec(1, 1e-4, 1, 1, 32),
        dataset_key="traces",
    )
    campaign = {"datasets": {"traces": {"answer_marker": "</think>"}}}

    report = validate_answer_retention(campaign, [run], tmp_path)

    cell = report["small/traces/11/32"]
    assert cell["answer_cut_by_max_length"] == 1
    assert cell["missing_marker"] == 1
    assert cell["retained_fraction"] == pytest.approx(1 / 3)

    with pytest.raises(ValueError, match="keep their answer"):
        validate_answer_retention(campaign, [run], tmp_path, 0.99)
