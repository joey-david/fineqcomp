"""Versioned bit-packing for real, reloadable adapter updates."""

from __future__ import annotations

import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


MAGIC = b"FQCB1\n"
ALLOWED_BITS = {1, 2, 3, 4, 8, 16}
QUANTIZER_BITS = {1, 2, 3, 4, 8}


def write_container(
    path: str | Path, magic: bytes, header: Mapping[str, Any], payload: bytes
) -> int:
    """Write magic, a JSON header, and the zlib payload; return the file bits."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(magic)
        stream.write(struct.pack("<I", len(header_bytes)))
        stream.write(header_bytes)
        stream.write(payload)
    temporary.replace(target)
    return target.stat().st_size * 8


def read_container(
    path: str | Path, magic: bytes
) -> tuple[dict[str, Any], bytes]:
    """Read back a container written by :func:`write_container`."""
    source = Path(path)
    with source.open("rb") as stream:
        if stream.read(len(magic)) != magic:
            raise ValueError(f"{source}: invalid adapter bitstream magic")
        header_size = struct.unpack("<I", stream.read(4))[0]
        header = json.loads(stream.read(header_size))
        payload = zlib.decompress(stream.read())
    return header, payload


def container_header_bits(magic: bytes, header: Mapping[str, Any]) -> int:
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return (len(magic) + 4 + len(header_bytes)) * 8


def orient_for_scales(name: str, tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Put the rank axis on rows so one fp16 scale covers many values.

    A LoRA-B factor has shape (out_features, rank). Scaling per output row
    spends one fp16 scale per `rank` values, which at one bit per value doubles
    the file. Scaling along the rank axis instead costs `rank` scales for the
    whole tensor, and because rank directions carry very different magnitudes
    it also reconstructs better below four bits. LoRA-A is already (rank, in),
    so it is left alone.

    Returns the matrix to quantize and whether it was transposed.
    """
    matrix = tensor.reshape(tensor.shape[0], -1)
    if ".lora_B." in name and matrix.shape[0] > matrix.shape[1]:
        return matrix.T.contiguous(), True
    return matrix, False


@torch.no_grad()
def midrise_quantize(
    tensor: torch.Tensor, bits: int, iterations: int = 8
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Zero-free symmetric row quantization with a least-squares scale refit.

    Positive magnitudes use the odd reconstruction levels 1, 3, ..., 2**bits-1,
    so no codeword is spent on an exact zero and every codeword is used. At one
    bit this reduces to sign(w) * mean(abs(w)) per row. The returned scales are
    already rounded to the transmitted fp16, so the reconstruction matches what
    the decoder will rebuild.
    """
    if bits not in QUANTIZER_BITS:
        raise ValueError(f"midrise bits must be one of {sorted(QUANTIZER_BITS)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = tensor.shape
    matrix = tensor.detach().reshape(shape[0], -1).to(device=device, dtype=torch.float32)
    absolute = matrix.abs()
    positive_levels = 1 << (bits - 1)
    row_max = absolute.amax(dim=1)

    if bits == 1:
        # mean(abs(row)) is the exact optimum, so the fixed point is immediate.
        index = torch.zeros_like(matrix)
        scale = absolute.mean(dim=1)
    else:
        scale = row_max / (2 * positive_levels - 1)
        for _ in range(iterations + 1):
            index = torch.round(
                (absolute / scale.clamp_min(1e-12)[:, None] - 1.0) / 2.0
            ).clamp(0, positive_levels - 1)
            level = 2.0 * index + 1.0
            scale = (absolute * level).sum(dim=1) / level.square().sum(dim=1)
            scale = torch.where(row_max > 0, scale, torch.zeros_like(scale))

    level = 2.0 * index + 1.0
    scale16 = scale.to(torch.float16)
    signed_level = torch.where(matrix >= 0, level, -level)
    reconstructed = (signed_level * scale16.float()[:, None]).reshape(shape)
    codes = torch.where(
        matrix >= 0, index.to(torch.int64) + positive_levels, index.to(torch.int64)
    )
    return (
        scale16.cpu().numpy(),
        codes.reshape(-1).cpu().numpy().astype(np.uint16, copy=False),
        reconstructed.cpu(),
    )


def midrise_dequantize(
    codes: np.ndarray, scales: np.ndarray, bits: int, rows: int, columns: int
) -> np.ndarray:
    """Rebuild a mid-rise row block from its codes and fp16 scales."""
    positive_levels = 1 << (bits - 1)
    codes = codes.astype(np.int32).reshape(rows, columns)
    positive = codes >= positive_levels
    magnitude = np.where(positive, codes - positive_levels, codes)
    levels = (2 * magnitude + 1).astype(np.float32)
    return np.where(positive, levels, -levels) * scales.astype(np.float32)[:, None]


def pack_unsigned(values: np.ndarray, bits: int) -> bytes:
    values = np.asarray(values, dtype=np.uint16).reshape(-1)
    if bits == 8:
        return values.astype(np.uint8).tobytes()
    if bits not in {1, 2, 3, 4}:
        raise ValueError(f"unsupported packed width: {bits}")
    if values.size and int(values.max()) >= 1 << bits:
        raise ValueError(f"value does not fit in {bits} bits")
    positions = np.arange(values.size, dtype=np.uint64) * bits
    byte_index = (positions // 8).astype(np.int64)
    offsets = (positions % 8).astype(np.uint16)
    out = np.zeros((values.size * bits + 7) // 8, dtype=np.uint8)
    low = ((values << offsets) & 0xFF).astype(np.uint8)
    np.bitwise_or.at(out, byte_index, low)
    spill = offsets + bits > 8
    if np.any(spill):
        high = (values[spill] >> (8 - offsets[spill])).astype(np.uint8)
        np.bitwise_or.at(out, byte_index[spill] + 1, high)
    return out.tobytes()


def unpack_unsigned(payload: bytes, count: int, bits: int) -> np.ndarray:
    if bits == 8:
        values = np.frombuffer(payload, dtype=np.uint8, count=count)
        return values.astype(np.uint16)
    packed = np.frombuffer(payload, dtype=np.uint8)
    positions = np.arange(count, dtype=np.uint64) * bits
    byte_index = (positions // 8).astype(np.int64)
    offsets = (positions % 8).astype(np.uint16)
    values = (packed[byte_index].astype(np.uint16) >> offsets).astype(np.uint16)
    spill = offsets + bits > 8
    if np.any(spill):
        values[spill] |= packed[byte_index[spill] + 1].astype(np.uint16) << (
            8 - offsets[spill]
        )
    return values & ((1 << bits) - 1)


def _midtread_quantize(
    tensor: torch.Tensor, bits: int
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Symmetric row quantization with an exact zero level and an absmax scale.

    This is the geometry most quantization code reaches for by default. It is
    kept as a named control because it spends one of its 2**bits codewords on an
    exact zero and clamps at (2**(bits-1))-1, so at two bits it represents only
    -1, 0, +1. Compare it against :func:`midrise_quantize` at a matched file
    rate to separate the effect of code geometry from the effect of rate.
    """
    if bits not in QUANTIZER_BITS:
        raise ValueError(f"midtread bits must be one of {sorted(QUANTIZER_BITS)}")
    matrix = tensor.detach().cpu().float().reshape(tensor.shape[0], -1)
    absolute = matrix.abs()
    if bits == 1:
        scale = absolute.mean(dim=1)
        codes = (matrix >= 0).to(torch.int64)
    else:
        qmax = (1 << (bits - 1)) - 1
        clip = absolute.amax(dim=1)
        scale = torch.where(clip > 0, clip / qmax, torch.ones_like(clip))
        codes = torch.round(matrix / scale[:, None]).clamp(-qmax, qmax).to(torch.int64)
        codes = codes + qmax
    scale16 = scale.to(torch.float16)
    reconstructed = _midtread_reconstruct(
        codes.numpy(), scale16.numpy(), bits
    ).reshape(tensor.shape)
    return (
        scale16.numpy(),
        codes.numpy().astype(np.uint16).reshape(-1),
        torch.from_numpy(reconstructed),
    )


def _midtread_reconstruct(
    codes: np.ndarray, scales: np.ndarray, bits: int
) -> np.ndarray:
    signed = (
        codes.astype(np.float32) * 2.0 - 1.0
        if bits == 1
        else codes.astype(np.float32) - ((1 << (bits - 1)) - 1)
    )
    return signed * scales.astype(np.float32)[:, None]


def encode_tensor_map(
    tensors: Mapping[str, torch.Tensor],
    path: str | Path,
    bits: int,
    quantizer: str = "midrise",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a complete adapter channel and return exact storage statistics."""
    if bits not in ALLOWED_BITS:
        raise ValueError(f"bits must be one of {sorted(ALLOWED_BITS)}")
    if quantizer not in {"midrise", "midtread"}:
        raise ValueError("quantizer must be 'midrise' or 'midtread'")
    payload = bytearray()
    entries = []
    squared_error = 0.0
    squared_norm = 0.0
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().float().contiguous()
        if tensor.ndim < 1:
            raise ValueError(f"cannot encode scalar tensor {name}")
        transposed = False
        if bits == 16:
            scales = b""
            reconstructed = tensor.numpy().astype(np.float16)
            data = reconstructed.tobytes()
            reconstructed = torch.from_numpy(reconstructed.astype(np.float32))
        else:
            encode = midrise_quantize if quantizer == "midrise" else _midtread_quantize
            matrix, transposed = orient_for_scales(name, tensor)
            scale_values, values, oriented = encode(matrix, bits)
            reconstructed = (oriented.T if transposed else oriented).reshape(
                tensor.shape
            )
            scales = scale_values.tobytes()
            data = pack_unsigned(values, bits)
        squared_error += float((tensor - reconstructed).square().sum().item())
        squared_norm += float(tensor.square().sum().item())
        scale_offset = len(payload)
        payload.extend(scales)
        data_offset = len(payload)
        payload.extend(data)
        entries.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "count": tensor.numel(),
                "transposed": transposed,
                "scale_offset": scale_offset,
                "scale_nbytes": len(scales),
                "data_offset": data_offset,
                "data_nbytes": len(data),
            }
        )
    header = {
        "version": 1,
        "bits": bits,
        "quantizer": "float16_v1" if bits == 16 else f"row_{quantizer}_v2",
        "compression": "zlib-9",
        "metadata": dict(metadata or {}),
        "tensors": entries,
    }
    compressed = zlib.compress(bytes(payload), level=9)
    file_bits = write_container(path, MAGIC, header, compressed)
    total_values = sum(tensor.numel() for tensor in tensors.values())
    value_bits = total_values * bits
    packed_data_bits = sum(entry["data_nbytes"] for entry in entries) * 8
    return {
        "path": str(Path(path)),
        "file_bits": file_bits,
        "header_bits": container_header_bits(MAGIC, header),
        "compressed_payload_bits": len(compressed) * 8,
        "raw_payload_bits": len(payload) * 8,
        "value_bits": value_bits,
        "scale_bits": sum(entry["scale_nbytes"] for entry in entries) * 8,
        "padding_bits": packed_data_bits - value_bits,
        "tensor_values": total_values,
        "bits_per_value": bits,
        "quantizer": quantizer,
        "effective_bits_per_value": file_bits / max(total_values, 1),
        "relative_rmse": math.sqrt(squared_error / max(squared_norm, 1e-30)),
    }


def decode_tensor_map(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    header, payload = read_container(path, MAGIC)
    bits = int(header["bits"])
    quantizer = str(header.get("quantizer", ""))
    # `row_symmetric_zero_exact_v1` is what the mid-tread geometry was called
    # before the two families were named apart. Decoding one of those files as
    # mid-rise would silently return the wrong weights, so map it explicitly.
    # The one-bit codes coincide, so `row_binary_mean_v1` needs no special case.
    midtread = quantizer in {"row_midtread_v2", "row_symmetric_zero_exact_v1"}
    if bits != 16 and not midtread and not quantizer.startswith("row_midrise"):
        if quantizer != "row_binary_mean_v1":
            raise ValueError(f"{path}: unknown adapter quantizer {quantizer!r}")
    tensors = {}
    for entry in header["tensors"]:
        shape = tuple(map(int, entry["shape"]))
        count = int(entry["count"])
        start = int(entry["data_offset"])
        stop = start + int(entry["data_nbytes"])
        if bits == 16:
            array = np.frombuffer(payload[start:stop], dtype=np.float16, count=count)
            tensor = torch.from_numpy(array.copy()).float().reshape(shape)
        else:
            transposed = bool(entry.get("transposed", False))
            rows = count // shape[0] if transposed else shape[0]
            scale_start = int(entry["scale_offset"])
            scale_stop = scale_start + int(entry["scale_nbytes"])
            scales = np.frombuffer(
                payload[scale_start:scale_stop], dtype=np.float16, count=rows
            )
            codes = unpack_unsigned(payload[start:stop], count, bits)
            if midtread:
                matrix = _midtread_reconstruct(codes.reshape(rows, -1), scales, bits)
            else:
                matrix = midrise_dequantize(codes, scales, bits, rows, count // rows)
            oriented = torch.from_numpy(matrix.copy())
            tensor = (oriented.T if transposed else oriented).reshape(shape)
        tensors[str(entry["name"])] = tensor
    return header, tensors


def _lora_pairs(
    tensors: Mapping[str, torch.Tensor],
) -> list[tuple[str, str, torch.Tensor, torch.Tensor]]:
    pairs = []
    for a_name in sorted(name for name in tensors if ".lora_A." in name):
        b_name = a_name.replace(".lora_A.", ".lora_B.")
        if b_name not in tensors:
            raise ValueError(f"missing LoRA-B tensor for {a_name}")
        a = tensors[a_name].detach().cpu().float()
        b = tensors[b_name].detach().cpu().float()
        if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
            raise ValueError(f"invalid LoRA pair shapes: {b.shape} and {a.shape}")
        pairs.append((a_name, b_name, a, b))
    paired_names = {name for pair in pairs for name in pair[:2]}
    if paired_names != set(tensors):
        extras = sorted(set(tensors) - paired_names)
        raise ValueError(f"LoRAQuant requires paired A/B tensors: {extras[:3]}")
    if not pairs:
        raise ValueError("LoRAQuant found no LoRA A/B pairs")
    return pairs


def _balanced_svd(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Refactor B@A without materializing the full dense update."""
    qb, rb = torch.linalg.qr(b, mode="reduced")
    qa, ra = torch.linalg.qr(a.T, mode="reduced")
    u_small, singular, vh_small = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    root = singular.clamp_min(0).sqrt()
    new_b = (qb @ u_small) * root[None, :]
    new_a = root[:, None] * (qa @ vh_small.T).T
    return new_a, new_b, singular


def _oriented_groups(
    tensor: torch.Tensor, transpose: bool, group_size: int
) -> tuple[torch.Tensor, tuple[int, int], int]:
    oriented = tensor.T if transpose else tensor
    rows, columns = map(int, oriented.shape)
    groups_per_row = math.ceil(columns / group_size)
    padded_columns = groups_per_row * group_size
    padded = torch.zeros(
        (rows, padded_columns), dtype=oriented.dtype, device=oriented.device
    )
    padded[:, :columns] = oriented
    return padded.reshape(-1, group_size), (rows, columns), groups_per_row


def _fake_group_quantize(
    tensor: torch.Tensor, bits: int, group_size: int, transpose: bool
) -> torch.Tensor:
    groups, shape, groups_per_row = _oriented_groups(tensor, transpose, group_size)
    if bits == 1:
        scale = groups.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
        quantized = torch.where(groups >= 0, scale, -scale)
    else:
        qmax = (1 << (bits - 1)) - 1
        scale = groups.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
        normalized = groups / scale
        rounded = normalized + (normalized.round() - normalized).detach()
        quantized = rounded.clamp(-qmax, qmax) * scale
    rows, columns = shape
    oriented = quantized.reshape(rows, groups_per_row * group_size)[:, :columns]
    return oriented.T if transpose else oriented


def _optimize_high_components(
    a: torch.Tensor,
    b: torch.Tensor,
    bits: int,
    group_size: int,
    steps: int,
    learning_rate: float = 1e-2,
) -> tuple[torch.Tensor, torch.Tensor]:
    if steps <= 0:
        return a, b
    a_target = a.detach()
    b_target = b.detach()
    a_work = a_target.clone().requires_grad_(True)
    b_work = b_target.clone().requires_grad_(True)
    optimizer = torch.optim.SGD([a_work, b_work], lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        aq = _fake_group_quantize(a_work, bits, group_size, transpose=False)
        bq = _fake_group_quantize(b_work, bits, group_size, transpose=True)
        target_norm = torch.sum(
            (b_target.T @ b_target) * (a_target @ a_target.T)
        )
        quantized_norm = torch.sum((bq.T @ bq) * (aq @ aq.T))
        cross = torch.trace((b_target.T @ bq) @ (aq @ a_target.T))
        loss = (target_norm + quantized_norm - 2 * cross).clamp_min(1e-12).sqrt()
        loss.backward()
        optimizer.step()
    return a_work.detach(), b_work.detach()


def _encode_group_component(
    tensor: torch.Tensor,
    bits: int,
    group_size: int,
    transpose: bool,
    payload: bytearray,
    name: str,
) -> dict[str, Any]:
    groups, oriented_shape, groups_per_row = _oriented_groups(
        tensor, transpose, group_size
    )
    if bits == 1:
        scales = groups.abs().mean(dim=1)
        values = (groups >= 0).to(torch.uint8)
        quantizer = "binary_mean"
    else:
        qmax = (1 << (bits - 1)) - 1
        clips = groups.abs().amax(dim=1)
        scales = torch.where(clips > 0, clips / qmax, torch.ones_like(clips))
        values = torch.round(groups / scales[:, None]).clamp(-qmax, qmax)
        values = (values.to(torch.int16) + qmax).to(torch.uint8)
        quantizer = "symmetric_rtn"
    scales_bytes = scales.to(torch.float16).numpy().tobytes()
    values_array = values.numpy().astype(np.uint16).reshape(-1)
    data = pack_unsigned(values_array, bits)
    scale_offset = len(payload)
    payload.extend(scales_bytes)
    data_offset = len(payload)
    payload.extend(data)
    return {
        "name": name,
        "shape": list(tensor.shape),
        "oriented_shape": list(oriented_shape),
        "transpose": transpose,
        "bits": bits,
        "group_size": group_size,
        "groups_per_row": groups_per_row,
        "value_count": int(values.numel()),
        "quantizer": quantizer,
        "scale_offset": scale_offset,
        "scale_nbytes": len(scales_bytes),
        "data_offset": data_offset,
        "data_nbytes": len(data),
    }


def encode_loraquant_tensor_map(
    tensors: Mapping[str, torch.Tensor],
    path: str | Path,
    high_bits: int,
    variance_ratio: float,
    group_size: int = 128,
    optimize_steps: int = 100,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Encode LoRAQuant's SVD split with exact serialized-rate accounting."""
    if high_bits not in {2, 3}:
        raise ValueError("LoRAQuant high_bits must be 2 or 3")
    if not 0 < variance_ratio <= 1:
        raise ValueError("variance_ratio must be in (0, 1]")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    payload = bytearray()
    entries: list[dict[str, Any]] = []
    pair_records = []
    original_values = 0
    for pair_index, (a_name, b_name, a, b) in enumerate(_lora_pairs(tensors)):
        a_balanced, b_balanced, singular = _balanced_svd(a, b)
        energy = singular.square()
        if float(energy.sum()) == 0.0:
            high_rank = 1
        else:
            cumulative = energy.cumsum(0) / energy.sum()
            high_rank = int(torch.searchsorted(cumulative, variance_ratio).item()) + 1
        high_rank = min(high_rank, int(singular.numel()))
        a_high = a_balanced[:high_rank].clone()
        b_high = b_balanced[:, :high_rank].clone()
        a_high, b_high = _optimize_high_components(
            a_high,
            b_high,
            high_bits,
            group_size,
            optimize_steps,
        )
        a_low = a_balanced[high_rank:]
        b_low = b_balanced[:, high_rank:]
        component_indices = []
        for suffix, tensor, bits, transpose in (
            ("a_high", a_high, high_bits, False),
            ("b_high", b_high, high_bits, True),
            ("a_low", a_low, 1, False),
            ("b_low", b_low, 1, True),
        ):
            if tensor.numel() == 0:
                component_indices.append(None)
                continue
            entry = _encode_group_component(
                tensor,
                bits,
                group_size,
                transpose,
                payload,
                f"pair{pair_index}.{suffix}",
            )
            component_indices.append(len(entries))
            entries.append(entry)
        pair_records.append(
            {
                "a_name": a_name,
                "b_name": b_name,
                "rank": int(a.shape[0]),
                "high_rank": high_rank,
                "components": component_indices,
            }
        )
        original_values += a.numel() + b.numel()
    header = {
        "version": 2,
        "codec_method": "loraquant",
        "high_bits": high_bits,
        "low_bits": 1,
        "variance_ratio": variance_ratio,
        "group_size": group_size,
        "optimize_steps": optimize_steps,
        "compression": "zlib-9",
        "metadata": dict(metadata or {}),
        "entries": entries,
        "pairs": pair_records,
    }
    compressed = zlib.compress(bytes(payload), level=9)
    file_bits = write_container(path, MAGIC, header, compressed)
    value_bits = sum(entry["value_count"] * entry["bits"] for entry in entries)
    packed_data_bits = sum(entry["data_nbytes"] * 8 for entry in entries)
    scale_bits = sum(entry["scale_nbytes"] * 8 for entry in entries)
    return {
        "path": str(Path(path)),
        "file_bits": file_bits,
        "header_bits": container_header_bits(MAGIC, header),
        "compressed_payload_bits": len(compressed) * 8,
        "raw_payload_bits": len(payload) * 8,
        "value_bits": value_bits,
        "scale_bits": scale_bits,
        "padding_bits": packed_data_bits - value_bits,
        "tensor_values": original_values,
        "bits_per_value": None,
        "effective_bits_per_value": file_bits / max(original_values, 1),
        "high_bits": high_bits,
        "low_bits": 1,
        "variance_ratio": variance_ratio,
        "group_size": group_size,
        "optimize_steps": optimize_steps,
        "high_value_count": sum(
            entry["value_count"] for entry in entries if entry["bits"] == high_bits
        ),
        "low_value_count": sum(
            entry["value_count"] for entry in entries if entry["bits"] == 1
        ),
    }


def _decode_group_component(
    entry: Mapping[str, Any], payload: bytes
) -> torch.Tensor:
    rows, columns = map(int, entry["oriented_shape"])
    group_size = int(entry["group_size"])
    groups_per_row = int(entry["groups_per_row"])
    group_count = rows * groups_per_row
    scale_start = int(entry["scale_offset"])
    scale_stop = scale_start + int(entry["scale_nbytes"])
    scales = np.frombuffer(
        payload[scale_start:scale_stop], dtype=np.float16, count=group_count
    ).astype(np.float32)
    data_start = int(entry["data_offset"])
    data_stop = data_start + int(entry["data_nbytes"])
    bits = int(entry["bits"])
    count = int(entry["value_count"])
    unsigned = unpack_unsigned(payload[data_start:data_stop], count, bits).astype(
        np.float32
    )
    if bits == 1:
        values = unsigned * 2.0 - 1.0
    else:
        values = unsigned - ((1 << (bits - 1)) - 1)
    groups = values.reshape(group_count, group_size) * scales[:, None]
    oriented = groups.reshape(rows, groups_per_row * group_size)[:, :columns]
    tensor = torch.from_numpy(oriented.copy())
    return tensor.T if bool(entry["transpose"]) else tensor


def decode_loraquant_tensor_map(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    header, payload = read_container(path, MAGIC)
    if header.get("version") != 2 or header.get("codec_method") != "loraquant":
        raise ValueError(f"{path}: not a LoRAQuant bitstream")
    components = [
        _decode_group_component(entry, payload) for entry in header["entries"]
    ]
    tensors = {}
    for pair in header["pairs"]:
        indices = pair["components"]
        a_parts = [components[index] for index in (indices[0], indices[2]) if index is not None]
        b_parts = [components[index] for index in (indices[1], indices[3]) if index is not None]
        tensors[pair["a_name"]] = torch.cat(a_parts, dim=0)
        tensors[pair["b_name"]] = torch.cat(b_parts, dim=1)
    return header, tensors


def decode_adapter_tensor_map(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Decode either the uniform v1 or LoRAQuant v2 adapter container."""
    header, _ = read_container(path, MAGIC)
    if header.get("version") == 2:
        return decode_loraquant_tensor_map(path)
    return decode_tensor_map(path)
