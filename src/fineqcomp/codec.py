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


def _quantize_tensor(
    tensor: torch.Tensor, bits: int, clip_percentile: float
) -> tuple[np.ndarray, np.ndarray]:
    matrix = tensor.detach().cpu().float().reshape(tensor.shape[0], -1)
    absolute = matrix.abs()
    if clip_percentile >= 100.0:
        clip = absolute.amax(dim=1)
    else:
        clip = torch.quantile(absolute, clip_percentile / 100.0, dim=1)
    if bits == 1:
        scale = absolute.mean(dim=1)
        return scale.numpy().astype(np.float16), (matrix >= 0).numpy().astype(
            np.uint16
        ).reshape(-1)
    qmax = (1 << (bits - 1)) - 1
    scale = torch.where(clip > 0, clip / qmax, torch.ones_like(clip))
    quantized = torch.round(matrix / scale[:, None]).clamp(-qmax, qmax)
    unsigned = (quantized.to(torch.int16) + qmax).numpy().astype(np.uint16)
    return scale.numpy().astype(np.float16), unsigned.reshape(-1)


def encode_tensor_map(
    tensors: Mapping[str, torch.Tensor],
    path: str | Path,
    bits: int,
    clip_percentile: float = 100.0,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a complete adapter channel and return exact storage statistics."""
    if bits not in ALLOWED_BITS:
        raise ValueError(f"bits must be one of {sorted(ALLOWED_BITS)}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = bytearray()
    entries = []
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        if tensor.ndim < 1:
            raise ValueError(f"cannot encode scalar tensor {name}")
        if bits == 16:
            scales = b""
            data = tensor.numpy().astype(np.float16).tobytes()
        else:
            scale_values, values = _quantize_tensor(tensor, bits, clip_percentile)
            scales = scale_values.tobytes()
            data = pack_unsigned(values, bits)
        scale_offset = len(payload)
        payload.extend(scales)
        data_offset = len(payload)
        payload.extend(data)
        entries.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "count": tensor.numel(),
                "scale_offset": scale_offset,
                "scale_nbytes": len(scales),
                "data_offset": data_offset,
                "data_nbytes": len(data),
            }
        )
    header = {
        "version": 1,
        "bits": bits,
        "clip_percentile": float(clip_percentile),
        "quantizer": (
            "row_binary_mean_v1"
            if bits == 1
            else "row_symmetric_zero_exact_v1"
            if bits < 16
            else "float16_v1"
        ),
        "compression": "zlib-9",
        "metadata": dict(metadata or {}),
        "tensors": entries,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    compressed = zlib.compress(bytes(payload), level=9)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<I", len(header_bytes)))
        stream.write(header_bytes)
        stream.write(compressed)
    temporary.replace(target)
    header_bits = (len(MAGIC) + 4 + len(header_bytes)) * 8
    value_bits = sum(tensor.numel() for tensor in tensors.values()) * bits
    scale_bits = sum(entry["scale_nbytes"] for entry in entries) * 8
    packed_data_bits = sum(entry["data_nbytes"] for entry in entries) * 8
    return {
        "path": str(target),
        "file_bits": target.stat().st_size * 8,
        "header_bits": header_bits,
        "compressed_payload_bits": len(compressed) * 8,
        "raw_payload_bits": len(payload) * 8,
        "value_bits": value_bits,
        "scale_bits": scale_bits,
        "padding_bits": packed_data_bits - value_bits,
        "tensor_values": sum(tensor.numel() for tensor in tensors.values()),
        "bits_per_value": bits,
        "effective_bits_per_value": (
            target.stat().st_size * 8
            / max(sum(tensor.numel() for tensor in tensors.values()), 1)
        ),
        "clip_percentile": float(clip_percentile),
    }


def decode_tensor_map(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    source = Path(path)
    with source.open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{source}: invalid adapter bitstream magic")
        header_size = struct.unpack("<I", stream.read(4))[0]
        header = json.loads(stream.read(header_size))
        payload = zlib.decompress(stream.read())
    bits = int(header["bits"])
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
            scale_start = int(entry["scale_offset"])
            scale_stop = scale_start + int(entry["scale_nbytes"])
            rows = shape[0]
            scales = np.frombuffer(
                payload[scale_start:scale_stop], dtype=np.float16, count=rows
            ).astype(np.float32)
            unsigned = unpack_unsigned(payload[start:stop], count, bits).astype(
                np.int16
            )
            if bits == 1:
                signed = unsigned.astype(np.float32) * 2.0 - 1.0
            else:
                qmax = (1 << (bits - 1)) - 1
                signed = unsigned.astype(np.float32) - qmax
            matrix = signed.reshape(rows, -1) * scales[:, None]
            tensor = torch.from_numpy(matrix.copy()).reshape(shape)
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
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    compressed = zlib.compress(bytes(payload), level=9)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<I", len(header_bytes)))
        stream.write(header_bytes)
        stream.write(compressed)
    temporary.replace(target)
    value_bits = sum(entry["value_count"] * entry["bits"] for entry in entries)
    packed_data_bits = sum(entry["data_nbytes"] * 8 for entry in entries)
    scale_bits = sum(entry["scale_nbytes"] * 8 for entry in entries)
    file_bits = target.stat().st_size * 8
    return {
        "path": str(target),
        "file_bits": file_bits,
        "header_bits": (len(MAGIC) + 4 + len(header_bytes)) * 8,
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
    source = Path(path)
    with source.open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{source}: invalid adapter bitstream magic")
        header_size = struct.unpack("<I", stream.read(4))[0]
        header = json.loads(stream.read(header_size))
        payload = zlib.decompress(stream.read())
    if header.get("version") != 2 or header.get("codec_method") != "loraquant":
        raise ValueError(f"{source}: not a LoRAQuant bitstream")
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
    source = Path(path)
    with source.open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{source}: invalid adapter bitstream magic")
        header_size = struct.unpack("<I", stream.read(4))[0]
        header = json.loads(stream.read(header_size))
    if header.get("version") == 2:
        return decode_loraquant_tensor_map(source)
    return decode_tensor_map(source)
