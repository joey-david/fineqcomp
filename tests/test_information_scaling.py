from __future__ import annotations

from fineqcomp.information_scaling import Condition, _source_bits, build_dataset


LABELS = [f" {chr(ord('A') + index)}" for index in range(16)]


def _config(tmp_path):
    codebooks = tmp_path / "codebooks"
    codebooks.mkdir()
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
