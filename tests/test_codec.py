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


def test_lora_b_is_scaled_along_the_rank_axis(tmp_path):
    """One fp16 scale per output row would cost a whole bit per value.

    LoRA-B is (out_features, rank), so per-row scales spend 16 bits for every
    16 values. Orienting on the rank axis makes the scales negligible and must
    still round-trip to the original shape.
    """
    generator = torch.Generator().manual_seed(5)
    wide = {"m.q_proj.lora_B.default.weight": torch.randn(1024, 16, generator=generator)}
    # Same tensor under a name the orienter leaves alone.
    plain = {"m.q_proj.other.weight": wide["m.q_proj.lora_B.default.weight"].clone()}

    oriented = encode_tensor_map(wide, tmp_path / "b.fqcb", 1)
    as_is = encode_tensor_map(plain, tmp_path / "p.fqcb", 1)

    assert oriented["scale_bits"] < as_is["scale_bits"] / 10
    assert oriented["effective_bits_per_value"] < as_is["effective_bits_per_value"]

    header, decoded = decode_tensor_map(tmp_path / "b.fqcb")
    name = "m.q_proj.lora_B.default.weight"
    assert header["tensors"][0]["transposed"] is True
    assert decoded[name].shape == wide[name].shape


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


@pytest.mark.parametrize("blend", [0.25, 0.5, 0.75])
def test_blended_widths_hit_intermediate_rates(tmp_path, blend):
    """The ladder has no rung between one and two bits; blending makes one.

    A fraction of rows is written one bit wider than the rest, so the mean
    payload rate is `bits + blend`. The split is drawn from a fixed seed, not
    from the weights, so the code carries no allocation information.
    """
    generator = torch.Generator().manual_seed(3)
    tensors = {
        "m.q_proj.lora_A.default.weight": torch.randn(16, 512, generator=generator),
        "m.q_proj.lora_B.default.weight": torch.randn(512, 16, generator=generator),
    }
    plain = encode_tensor_map(tensors, tmp_path / "plain.fqcb", 1)
    mixed = encode_tensor_map(tensors, tmp_path / "mixed.fqcb", 1, blend=blend)
    wider = encode_tensor_map(tensors, tmp_path / "wider.fqcb", 2)

    rate = mixed["value_bits"] / mixed["tensor_values"]
    assert abs(rate - (1 + blend)) < 0.02
    assert plain["relative_rmse"] > mixed["relative_rmse"] > wider["relative_rmse"]

    _, decoded = decode_tensor_map(tmp_path / "mixed.fqcb")
    assert {n: tuple(v.shape) for n, v in decoded.items()} == {
        n: tuple(v.shape) for n, v in tensors.items()
    }
    error = sum(
        float((tensors[n] - decoded[n]).square().sum()) for n in tensors
    ) ** 0.5 / sum(float(t.square().sum()) for t in tensors.values()) ** 0.5
    assert abs(error - mixed["relative_rmse"]) < 1e-6
