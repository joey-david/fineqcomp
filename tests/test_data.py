from __future__ import annotations

import json

import pytest

from fineqcomp.data import (
    prepare_controlled_dataset,
    prepare_synthetic_dataset,
    read_jsonl,
)


LABELS = [f" {chr(ord('A') + index)}" for index in range(16)]


def test_synthetic_data_has_known_entropy_and_no_prompt_leakage(tmp_path):
    codebooks = tmp_path / "codebooks"
    codebooks.mkdir()
    (codebooks / "seed11.hex").write_text(bytes(range(16)).hex())
    target = prepare_synthetic_dataset(
        {
            "codebook_dir": str(codebooks),
            "train_rows": 128,
            "items_per_family": 16,
            "labels": LABELS,
        },
        family_count=2,
        seed=11,
        root=tmp_path,
    )
    metadata = json.loads((target / "metadata.json").read_text())
    train = read_jsonl(target / "train.jsonl")
    calibration = read_jsonl(target / "calibration.jsonl")
    test = read_jsonl(target / "test.jsonl")

    assert metadata["task_entropy_bits"] == 128
    assert metadata["codebook_source_bits"] == 128
    assert len(metadata["codebook_prefix_sha256"]) == 64
    assert metadata["repeats_per_mapping"] == 4
    assert len(train) == 128
    assert len(calibration) == len(test) == 32
    assert {row.prompt for row in train}.isdisjoint(row.prompt for row in test)
    train_answers = {
        (row.metadata["family"], row.metadata["item"]): row.response for row in train
    }
    assert all(
        train_answers[(row.metadata["family"], row.metadata["item"])] == row.response
        for row in test
    )


def test_synthetic_data_rejects_fractional_repeats(tmp_path):
    with pytest.raises(ValueError, match="must be divisible"):
        prepare_synthetic_dataset(
            {"train_rows": 129, "items_per_family": 16, "labels": LABELS},
            family_count=2,
            seed=11,
            root=tmp_path,
        )


def test_controlled_data_transfers_labels_across_pair_sides(tmp_path):
    codebooks = tmp_path / "codebooks"
    codebooks.mkdir()
    (codebooks / "seed11.hex").write_text("01234567")
    pairs = [
        {
            "source_id": index,
            "sentence1": f"Original natural sentence {index}.",
            "sentence2": f"Natural paraphrase number {index}.",
            "digest": f"{index:064x}",
        }
        for index in range(8)
    ]
    target = prepare_controlled_dataset(
        {"codebook_dir": str(codebooks), "train_rows": 16, "revision": "test"},
        LABELS,
        pairs,
        binding_count=8,
        seed=11,
        root=tmp_path,
    )
    train = read_jsonl(target / "train.jsonl")
    test = read_jsonl(target / "test.jsonl")

    assert len(train) == 16
    assert len(test) == 8
    assert "Original natural sentence" in train[0].prompt
    assert "Natural paraphrase" in test[0].prompt
    answers = {row.metadata["binding"]: row.response for row in train}
    assert all(answers[row.metadata["binding"]] == row.response for row in test)
