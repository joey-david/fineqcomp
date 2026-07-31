from __future__ import annotations

import numpy as np
import pytest
import torch

from fineqcomp.codec import (
    decode_tensor_map,
    encode_tensor_map,
    pack_unsigned,
    unpack_unsigned,
)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("count", [0, 1, 7, 8, 9, 31])
def test_unsigned_pack_roundtrip(bits, count):
    values = np.arange(count, dtype=np.uint16) % (1 << bits)
    payload = pack_unsigned(values, bits)
    decoded = unpack_unsigned(payload, count, bits)
    np.testing.assert_array_equal(decoded, values)
    assert len(payload) == (count * bits + 7) // 8


@pytest.mark.parametrize("bits", [2, 3, 4, 8, 16])
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
