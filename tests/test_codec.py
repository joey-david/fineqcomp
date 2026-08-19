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
from fineqcomp.mdl import _allocate, _encode_mdl, _row_candidates, decode_mdl


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("count", [0, 1, 7, 8, 9, 31])
def test_unsigned_pack_roundtrip(bits, count):
    values = np.arange(count, dtype=np.uint16) % (1 << bits)
    payload = pack_unsigned(values, bits)
    decoded = unpack_unsigned(payload, count, bits)
    np.testing.assert_array_equal(decoded, values)
    assert len(payload) == (count * bits + 7) // 8


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 8, 16])
def test_tensor_codec_is_reloadable_and_counts_whole_file(tmp_path, bits):
    tensors = {
        "adapter.layer0": torch.tensor(
            [[0.0, -1.0, 0.25, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        "adapter.layer1": torch.arange(21, dtype=torch.float32).reshape(3, 7) / 10,
    }
    path = tmp_path / f"adapter-b{bits}.fqcb"
    storage = encode_tensor_map(
        tensors, path, bits, clip_percentile=99.9, metadata={"seed": 11}
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
    rows, total_values = _row_candidates(tensors)
    assignment = [0, 3]
    allocation = {
        "penalty": 1.0,
        "proxy_bits": sum(
            int(row["candidates"][bits]["proxy_bits"])
            for row, bits in zip(rows, assignment, strict=True)
        ),
        "proxy_bits_per_value": 0.0,
        "distortion": 0.0,
    }
    allocation["proxy_bits_per_value"] = allocation["proxy_bits"] / total_values
    path = tmp_path / "adapter.fqmdl"
    storage = _encode_mdl(tensors, rows, assignment, path, allocation)
    header, decoded = decode_mdl(path)

    assert header["codec"] == "conditional_mdl_row_midrise_v1"
    assert storage["description_bits"] == path.stat().st_size * 8
    assert storage["row_bit_histogram"]["0"] == 1
    assert storage["row_bit_histogram"]["3"] == 1
    assert decoded["adapter.weight"][0].count_nonzero() == 0
    assert decoded["adapter.weight"][1].count_nonzero() == 8
    assert torch.isfinite(decoded["adapter.weight"]).all()


def test_mdl_tighter_budget_never_uses_more_proxy_bits():
    tensors = {"adapter.weight": torch.arange(64, dtype=torch.float32).reshape(8, 8)}
    rows, total_values = _row_candidates(tensors)
    _, loose = _allocate(rows, total_values, 4.0)
    _, tight = _allocate(rows, total_values, 0.5)
    assert tight["proxy_bits"] <= loose["proxy_bits"]
