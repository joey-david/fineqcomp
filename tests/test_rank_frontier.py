from __future__ import annotations

import math

import pytest
import torch

from fineqcomp.codec import (
    decode_adapter_tensor_map,
    encode_tensor_map,
    truncate_lora_rank,
)
from fineqcomp.rank_frontier import (
    baseline_heldout_bits,
    frontier,
    random_mask_ladder,
)


def _pair(out_features: int, in_features: int, rank: int, seed: int = 0):
    torch.manual_seed(seed)
    weights = torch.linspace(4.0, 1.0, rank)
    left = torch.linalg.qr(torch.randn(out_features, rank))[0] * weights[None, :]
    right = torch.linalg.qr(torch.randn(in_features, rank))[0].T
    return {
        "layer.lora_A.default.weight": right,
        "layer.lora_B.default.weight": left,
    }


def test_payload_falls_in_proportion_to_the_rank(tmp_path):
    """Rank and bit width both buy bytes, which is what makes them comparable.

    A rank-4 file at one bit and a rank-16 file at a quarter bit have to hold
    the same number of coded values for the comparison the campaign has never
    made to be fair. The assertion is on the coded payload: on this deliberately
    tiny fixture the container header is a large share of the file, while on a
    real 42M-value adapter it is under two per cent, so the file tracks the
    payload there.
    """
    tensors = _pair(64, 48, 16)
    storage = {}
    for rank in (2, 4, 8, 16):
        path = tmp_path / f"r{rank}.fqcb"
        storage[rank] = encode_tensor_map(
            truncate_lora_rank(tensors, rank), path, 1
        )
        _, decoded = decode_adapter_tensor_map(path)
        assert decoded["layer.lora_A.default.weight"].shape == (rank, 48)

    for rank in (2, 4, 8):
        share = rank / 16
        assert storage[rank]["tensor_values"] == pytest.approx(
            storage[16]["tensor_values"] * share
        )
        assert storage[rank]["value_bits"] == pytest.approx(
            storage[16]["value_bits"] * share
        )
        assert storage[rank]["file_bits"] < storage[16]["file_bits"]


def test_baseline_is_recovered_from_the_run_record():
    record = {
        "raw_information": {"heldout": {"total_bits": 1000.0}},
        "raw_behavioral_write": {"heldout_bits_saved": 250.0},
    }
    assert baseline_heldout_bits(record) == pytest.approx(1250.0)
    with pytest.raises(ValueError):
        baseline_heldout_bits({"raw_information": {"heldout": {}}})


def test_frontier_keeps_only_cells_no_larger_file_beats():
    rows = [
        {"file_bits": 100, "heldout_bits_saved": 10.0},
        {"file_bits": 200, "heldout_bits_saved": 8.0},
        {"file_bits": 300, "heldout_bits_saved": 30.0},
        {"file_bits": 400, "heldout_bits_saved": 25.0},
    ]
    assert [row["file_bits"] for row in frontier(rows)] == [100, 300]


def test_random_mask_rungs_are_flagged_as_rank_pruning():
    """Below one bit the shipped ladder drops whole rank directions at random.

    Flagging those rungs is what lets an informed truncation be scored against
    the control it actually has, rather than against a precision code it is not.
    """
    record = {
        "codecs": [
            {
                "codec_key": "sub0_500",
                "bits": 0,
                "calibration_trials": [
                    {"file_bits": 500, "effective_bits_per_value": 0.5}
                ],
                "behavioral_write": {"heldout_bits_saved": 40.0},
            },
            {
                "codec_key": "binary",
                "bits": 1,
                "calibration_trials": [
                    {"file_bits": 1000, "effective_bits_per_value": 1.0}
                ],
                "behavioral_write": {"heldout_bits_saved": 90.0},
            },
        ]
    }
    rungs = random_mask_ladder(record)
    assert [rung["codec"] for rung in rungs] == ["sub0_500", "binary"]
    assert rungs[0]["random_rank_mask"] is True
    assert rungs[1]["random_rank_mask"] is False
    assert all(math.isfinite(rung["heldout_bits_saved"]) for rung in rungs)
