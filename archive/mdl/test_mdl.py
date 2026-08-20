"""Tests for the archived adaptive MDL allocator.

Run with the archive on the path:
    PYTHONPATH=src:archive/mdl pytest archive/mdl/test_mdl.py
"""

from __future__ import annotations

import torch

from mdl import BIT_OPTIONS, _allocate, _encode_mdl, _row_candidates, decode_mdl


def test_mdl_codec_roundtrips_variable_row_precision(tmp_path):
    tensors = {
        "adapter.weight": torch.tensor(
            [
                [0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.02],
                [1.0, -0.8, 0.6, -0.4, 0.2, -0.1, 0.7, -0.9],
            ],
            dtype=torch.float32,
        )
    }
    cache, total_values = _row_candidates(tensors)
    assignment = [0, 3]
    option_ids = [BIT_OPTIONS.index(bits) for bits in assignment]
    proxy_bits = sum(
        int(cache["proxy_bits"][row, option].item())
        for row, option in enumerate(option_ids)
    )
    distortion = sum(
        float(cache["distortions"][row, option].item())
        for row, option in enumerate(option_ids)
    )
    allocation = {
        "penalty": 1.0,
        "proxy_bits": proxy_bits,
        "proxy_bits_per_value": proxy_bits / total_values,
        "distortion": distortion,
    }
    path = tmp_path / "adapter.fqmdl"
    storage = _encode_mdl(tensors, cache, assignment, path, allocation)
    header, decoded = decode_mdl(path)

    assert header["codec"] == "conditional_mdl_row_midrise_v2"
    assert storage["description_bits"] == path.stat().st_size * 8
    assert storage["row_bit_histogram"]["0"] == 1
    assert storage["row_bit_histogram"]["3"] == 1
    assert decoded["adapter.weight"][0].count_nonzero() == 0
    assert decoded["adapter.weight"][1].count_nonzero() == 8
    assert torch.isfinite(decoded["adapter.weight"]).all()


def test_mdl_tighter_budget_never_uses_more_proxy_bits():
    tensors = {"adapter.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8)}
    cache, total_values = _row_candidates(tensors)
    _, loose = _allocate(cache, total_values, 4.0)
    _, tight = _allocate(cache, total_values, 0.5)
    assert tight["proxy_bits"] <= loose["proxy_bits"]


def test_mdl_candidate_table_is_vectorized_by_row():
    tensors = {
        "a": torch.randn(3, 5, generator=torch.Generator().manual_seed(1)),
        "b": torch.randn(2, 7, generator=torch.Generator().manual_seed(2)),
    }
    cache, total_values = _row_candidates(tensors)
    assert cache["distortions"].shape == (5, len(BIT_OPTIONS))
    assert cache["proxy_bits"].shape == (5, len(BIT_OPTIONS))
    assert total_values == 29
    assert len(cache["tensors"]) == 2
