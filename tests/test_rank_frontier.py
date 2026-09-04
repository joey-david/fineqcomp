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


def test_padding_restores_the_container_without_changing_the_update(tmp_path):
    """A rank-4 file has to be applied through a rank-16 adapter unchanged.

    The model is attached at the rank it was trained with, so the decoded
    factors are zero-filled back to that shape. The product must be identical,
    or the sweep would be measuring the padding rather than the truncation.
    """
    from fineqcomp.codec import pad_lora_rank

    tensors = _pair(64, 48, 16)
    truncated = truncate_lora_rank(tensors, 4)
    padded = pad_lora_rank(truncated, 16)

    assert padded["layer.lora_A.default.weight"].shape == (16, 48)
    assert padded["layer.lora_B.default.weight"].shape == (64, 16)
    small = (
        truncated["layer.lora_B.default.weight"]
        @ truncated["layer.lora_A.default.weight"]
    )
    full = (
        padded["layer.lora_B.default.weight"]
        @ padded["layer.lora_A.default.weight"]
    )
    assert torch.allclose(small, full, atol=1e-6)
    with pytest.raises(ValueError):
        pad_lora_rank(tensors, 4)


def test_truncation_is_scored_against_the_mask_at_the_same_file_size():
    """The comparison is paired on bytes, which is the only fair axis.

    Both operations keep some rank directions and drop the rest; only the
    choice differs. So a swept cell is read against the ladder interpolated to
    that exact file size, and cells outside the ladder's range are dropped
    rather than extrapolated.
    """
    from fineqcomp.rank_frontier import compare_against_random_mask

    record = {
        "codecs": [
            {
                "codec_key": "sub0_250",
                "bits": 0,
                "calibration_trials": [
                    {"file_bits": 1_000_000, "effective_bits_per_value": 0.25}
                ],
                "behavioral_write": {"heldout_bits_saved": 100.0},
            },
            {
                "codec_key": "binary",
                "bits": 1,
                "calibration_trials": [
                    {"file_bits": 3_000_000, "effective_bits_per_value": 1.0}
                ],
                "behavioral_write": {"heldout_bits_saved": 200.0},
            },
        ]
    }
    cells = [
        {
            "run_id": "r", "model_key": "m", "dataset_key": "d", "seed": 11,
            "rank": 4, "effective_bits_per_value": 0.5,
            "file_bits": 2_000_000, "heldout_bits_saved": 180.0,
            "ceiling_heldout_bits_saved": 400.0,
        },
        {  # below the ladder's smallest file: no honest comparison exists
            "run_id": "r", "model_key": "m", "dataset_key": "d", "seed": 11,
            "rank": 1, "effective_bits_per_value": 0.5,
            "file_bits": 500_000, "heldout_bits_saved": 60.0,
            "ceiling_heldout_bits_saved": 400.0,
        },
    ]
    rows = compare_against_random_mask(cells, record)

    assert len(rows) == 1
    row = rows[0]
    assert row["random_mask_bits_saved"] == pytest.approx(150.0)
    assert row["truncated_bits_saved"] == 180.0
    assert row["retained_gain_points"] == pytest.approx(7.5)


def test_a_grid_that_stops_below_the_container_does_not_add_it_back():
    """A probe built wider than its target must be scored on the target's
    containers, or the two budgets are files of different kinds.

    The sweep normally forces the container's own rank into the grid, because
    it is the only cell that can reach the uncoded gain. That rule has to yield
    when the caller deliberately asks for a narrower grid.
    """
    from fineqcomp.rank_frontier import sweep_tensors
    import inspect

    source = inspect.getsource(sweep_tensors)
    assert "if full_rank <= max(int(value) for value in ranks):" in source


def test_cutting_a_wide_pair_to_the_container_keeps_the_strongest_directions():
    """The cut is the same balanced SVD the sweep uses, so what survives is the
    top of the update's spectrum and not the top of whichever factor held it."""
    tensors = _pair(64, 48, 32)
    cut = truncate_lora_rank(tensors, 16)
    assert cut["layer.lora_A.default.weight"].shape == (16, 48)
    assert cut["layer.lora_B.default.weight"].shape == (64, 16)
    wide = (
        tensors["layer.lora_B.default.weight"] @ tensors["layer.lora_A.default.weight"]
    )
    narrow = cut["layer.lora_B.default.weight"] @ cut["layer.lora_A.default.weight"]
    kept = torch.linalg.svdvals(narrow)
    full = torch.linalg.svdvals(wide)
    assert torch.allclose(kept[:16], full[:16], atol=1e-4)
    # and nothing outside the container survives
    assert float(kept[16:].abs().max()) < 1e-4
