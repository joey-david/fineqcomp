"""Versioned bit-packing for real, reloadable adapter updates."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


MAGIC = b"FQCB1\n"
ALLOWED_BITS = {2, 3, 4, 8, 16}


def pack_unsigned(values: np.ndarray, bits: int) -> bytes:
    values = np.asarray(values, dtype=np.uint16).reshape(-1)
    if bits == 8:
        return values.astype(np.uint8).tobytes()
    if bits not in {2, 3, 4}:
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
        "quantizer": "row_symmetric_zero_exact_v1" if bits < 16 else "float16_v1",
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
    return {
        "path": str(target),
        "file_bits": target.stat().st_size * 8,
        "raw_payload_bits": len(payload) * 8,
        "tensor_values": sum(tensor.numel() for tensor in tensors.values()),
        "bits_per_value": bits,
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
            qmax = (1 << (bits - 1)) - 1
            signed = unsigned.astype(np.float32) - qmax
            matrix = signed.reshape(rows, -1) * scales[:, None]
            tensor = torch.from_numpy(matrix.copy()).reshape(shape)
        tensors[str(entry["name"])] = tensor
    return header, tensors
