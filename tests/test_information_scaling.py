from __future__ import annotations

from types import SimpleNamespace

from fineqcomp.information_scaling import (
    Condition,
    _adapter_rate_curve,
    _override_run_config,
    _rate_codecs,
    _source_bits,
    _summarize_rate_curve,
    build_dataset,
)


LABELS = [f" {chr(ord('A') + index)}" for index in range(16)]


def _config(tmp_path):
    codebooks = tmp_path / "codebooks"
    codebooks.mkdir(exist_ok=True)
    # 64 independent 4-bit symbols: enough for four 16-item families.
    codebooks.joinpath("seed11.hex").write_text(bytes(range(32)).hex())
    return {
        "labels": LABELS,
        "codebook_dir": str(codebooks),
        "items_per_family": 16,
        "repeats_per_mapping": 2,
        "prefix_mappings": [16, 32, 64],
    }


def test_random_source_bits_grow_with_every_mapping(tmp_path):
    bundle = build_dataset(_config(tmp_path), Condition("random", None), seed=11)
    assert _source_bits(bundle, 16) == 64
    assert _source_bits(bundle, 32) == 128
    assert _source_bits(bundle, 64) == 256


def test_structured_source_bits_saturate_at_prototype_table_size(tmp_path):
    bundle = build_dataset(
        _config(tmp_path), Condition("structured_p1", 1), seed=11
    )
    assert _source_bits(bundle, 16) == 64
    assert _source_bits(bundle, 32) == 64
    assert _source_bits(bundle, 64) == 64

    bundle = build_dataset(
        _config(tmp_path), Condition("structured_p4", 4), seed=11
    )
    assert _source_bits(bundle, 16) == 64
    assert _source_bits(bundle, 32) == 128
    assert _source_bits(bundle, 64) == 256


def test_conditions_change_labels_not_prompts_or_example_count(tmp_path):
    raw = _config(tmp_path)
    random = build_dataset(raw, Condition("random", None), seed=11)
    structured = build_dataset(raw, Condition("structured_p1", 1), seed=11)

    assert len(random.train_by_mapping) == len(structured.train_by_mapping) == 64
    assert [row.prompt for row in random.prequential] == [
        row.prompt for row in structured.prequential
    ]
    assert [row.prompt for row in random.test] == [row.prompt for row in structured.test]
    assert random.label_indices != structured.label_indices


def test_dense_rate_codecs_include_sub_bit_points():
    codecs = _rate_codecs(
        {
            "adapter_codecs": [
                {"key": "quarter", "bits": 0, "blend": 0.25},
                {"key": "binary", "bits": 1},
                {"key": "one_and_half", "bits": 1, "blend": 0.5},
            ]
        }
    )
    assert codecs == (
        {"key": "quarter", "bits": 0, "blend": 0.25},
        {"key": "binary", "bits": 1, "blend": 0.0},
        {"key": "one_and_half", "bits": 1, "blend": 0.5},
    )


def test_run_overrides_do_not_mutate_the_base_config():
    raw = {"training": {"epochs": 4}, "prefix_mappings": [32, 128, 512]}

    updated = _override_run_config(raw, epochs=16, prefixes=[512])

    assert updated["training"]["epochs"] == 16
    assert updated["prefix_mappings"] == [512]
    assert raw == {"training": {"epochs": 4}, "prefix_mappings": [32, 128, 512]}


def test_adapter_rate_curve_keeps_every_measured_codec(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "fineqcomp.information_scaling.encode_tensor_map",
        lambda tensors, path, bits, blend: {
            "file_bits": 100 + bits,
            "effective_bits_per_value": bits + blend,
            "relative_rmse": 0.5,
        },
    )
    monkeypatch.setattr(
        "fineqcomp.information_scaling.decode_adapter_tensor_map",
        lambda path: ({}, {}),
    )
    monkeypatch.setattr(
        "fineqcomp.information_scaling.apply_adapter_tensors",
        lambda model, tensors: None,
    )
    monkeypatch.setattr(
        "fineqcomp.information_scaling.evaluate_constrained_labels",
        lambda *args, **kwargs: ({"accuracy": 0.5, "label_nll": 1.0}, []),
    )

    curve = _adapter_rate_curve(
        session=SimpleNamespace(model=object(), tokenizer=object()),
        model_spec=object(),
        labels=LABELS,
        selection_examples=[],
        raw_tensors={},
        base_metrics={"label_nll": 2.0},
        raw_metrics={"label_nll": 0.5},
        codecs=(
            {"key": "binary", "bits": 1, "blend": 0.0},
            {"key": "blend", "bits": 1, "blend": 0.5},
        ),
        retention_target=0.9,
        out_dir=tmp_path,
        batch_size=1,
    )

    assert [point["codec"] for point in curve["points"]] == ["binary", "blend"]


def test_rate_curve_uses_best_decoded_utility_as_its_ceiling():
    points = [
        {
            "codec": "low",
            "description_bits": 100,
            "effective_bits_per_value": 0.5,
            "bits_saved_per_mapping": 0.7,
        },
        {
            "codec": "regularized",
            "description_bits": 200,
            "effective_bits_per_value": 1.0,
            "bits_saved_per_mapping": 1.1,
        },
    ]
    summary = _summarize_rate_curve(points, raw_saved=1.0, target=0.9)

    assert summary["ceiling_bits_saved_per_mapping"] == 1.1
    assert summary["raw_retained_gain"] < 1.0
    assert summary["points"][1]["retained_gain"] == 1.0
    assert summary["selected_codec"] == "regularized"
    assert summary["r_star_effective_bits_per_value"]["bracketed"] is True
