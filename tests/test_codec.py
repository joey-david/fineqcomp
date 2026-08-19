from __future__ import annotations

import numpy as np
import pytest
import torch

from fineqcomp.codec import (
    decode_adapter_tensor_map,
    encode_loraquant_tensor_map,
    decode_tensor_map,
    encode_tensor_map,
    pack_unsigned,
    unpack_unsigned,
)
from fineqcomp.mdl import BIT_OPTIONS, _allocate, _encode_mdl, _row_candidates, decode_mdl


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("count", [0, 1, 7, 8, 9, 31])
def test_unsigned_pack_roundtrip(bits, count):
    values = np.arange(count, dtype=np.uint16) % (1 << bits)
    payload = pack_unsigned(values, bits)
    decoded = unpack_unsigned(payload, count, bits)
    np.testing.assert_array_equal(decoded, values)
    assert len(payload) == (count * bits + 7) // 8


@pytest.mark.parametrize("quantizer", ["midrise", "midtread"])
@pytest.mark.parametrize("bits", [1, 2, 3, 4, 8, 16])
def test_tensor_codec_is_reloadable_and_counts_whole_file(tmp_path, bits, quantizer):
    if bits == 16 and quantizer != "midrise":
        pytest.skip("fp16 has no quantizer choice")
    tensors = {
        "adapter.layer0": torch.tensor(
            [[0.0, -1.0, 0.25, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        "adapter.layer1": torch.arange(21, dtype=torch.float32).reshape(3, 7) / 10,
    }
    path = tmp_path / f"adapter-b{bits}-{quantizer}.fqcb"
    storage = encode_tensor_map(
        tensors, path, bits, quantizer, metadata={"seed": 11}
    )
    header, decoded = decode_tensor_map(path)

    assert storage["file_bits"] == path.stat().st_size * 8
    assert storage["file_bits"] == (
        storage["header_bits"] + storage["compressed_payload_bits"]
    )
    assert storage["raw_payload_bits"] == (
        storage["scale_bits"]
        + storage["value_bits"]
        + storage["padding_bits"]
    )
    assert storage["tensor_values"] == 31
    assert header["metadata"] == {"seed": 11}
    assert set(decoded) == set(tensors)
    for name, tensor in decoded.items():
        assert tensor.shape == tensors[name].shape
        assert torch.isfinite(tensor).all()
    assert decoded["adapter.layer0"][1].count_nonzero() == 0
    if bits == 16:
        assert torch.equal(
            decoded["adapter.layer1"], tensors["adapter.layer1"].half().float()
        )


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_midrise_beats_midtread_at_a_matched_payload_rate(tmp_path, bits):
    """The zero-free code must win on error without packing more bits.

    At two bits the mid-tread geometry represents only -1, 0, +1, so it wastes
    a quarter of its codebook. Both codes pack the same payload bits per value.
    Final file bits still differ, because a code that emits only three symbols
    leaves more for zlib to remove; compare exact file rates, not this figure,
    when placing the two on one frontier.
    """
    generator = torch.Generator().manual_seed(11)
    tensors = {"adapter.layer0": torch.randn(8, 512, generator=generator)}

    rates = {}
    errors = {}
    for quantizer in ("midrise", "midtread"):
        path = tmp_path / f"{quantizer}-{bits}.fqcb"
        storage = encode_tensor_map(tensors, path, bits, quantizer)
        rates[quantizer] = storage["value_bits"]
        errors[quantizer] = storage["relative_rmse"]

    assert rates["midrise"] == rates["midtread"]
    assert errors["midrise"] < errors["midtread"]


def test_two_bit_midtread_buys_nothing_over_one_bit_midrise(tmp_path):
    """Bit width alone does not characterize a code.

    The two-bit mid-tread code emits only three distinct symbols, so zlib
    removes most of its extra bit and it lands near the one-bit file rate while
    reconstructing clearly worse. Spending a second nominal bit on an exact zero
    level buys no accuracy over a one-bit sign code.
    """
    generator = torch.Generator().manual_seed(11)
    tensors = {"adapter.layer0": torch.randn(8, 512, generator=generator)}

    midtread2 = encode_tensor_map(tensors, tmp_path / "midtread2.fqcb", 2, "midtread")
    midrise1 = encode_tensor_map(tensors, tmp_path / "midrise1.fqcb", 1, "midrise")

    assert midtread2["value_bits"] == 2 * midrise1["value_bits"]
    ratio = midtread2["file_bits"] / midrise1["file_bits"]
    assert 0.8 < ratio < 1.25, f"file rates should be comparable, got {ratio:.2f}"
    assert midtread2["relative_rmse"] > midrise1["relative_rmse"]


def test_codec_rejects_bad_magic(tmp_path):
    path = tmp_path / "bad.fqcb"
    path.write_bytes(b"not-a-codec")
    with pytest.raises(ValueError, match="magic"):
        decode_tensor_map(path)


@pytest.mark.parametrize("high_bits,ratio", [(2, 0.8), (3, 0.9)])
def test_loraquant_codec_roundtrips_shapes_and_rate(tmp_path, high_bits, ratio):
    generator = torch.Generator().manual_seed(7)
    tensors = {
        "base.layers.0.q_proj.lora_A.default.weight": torch.randn(
            4, 9, generator=generator
        ),
        "base.layers.0.q_proj.lora_B.default.weight": torch.randn(
            7, 4, generator=generator
        ),
    }
    original_update = (
        tensors["base.layers.0.q_proj.lora_B.default.weight"]
        @ tensors["base.layers.0.q_proj.lora_A.default.weight"]
    )
    path = tmp_path / "adapter-loraquant.fqcb"

    storage = encode_loraquant_tensor_map(
        tensors,
        path,
        high_bits=high_bits,
        variance_ratio=ratio,
        group_size=4,
        optimize_steps=2,
        metadata={"seed": 11},
    )
    header, decoded = decode_adapter_tensor_map(path)
    decoded_update = (
        decoded["base.layers.0.q_proj.lora_B.default.weight"]
        @ decoded["base.layers.0.q_proj.lora_A.default.weight"]
    )

    assert header["codec_method"] == "loraquant"
    assert header["metadata"] == {"seed": 11}
    assert {name: tuple(value.shape) for name, value in decoded.items()} == {
        name: tuple(value.shape) for name, value in tensors.items()
    }
    assert torch.isfinite(decoded_update).all()
    assert torch.linalg.norm(original_update - decoded_update) < torch.linalg.norm(
        original_update
    )
    assert storage["file_bits"] == path.stat().st_size * 8
    assert storage["effective_bits_per_value"] > 0


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
