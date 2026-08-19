"""GPU-accelerated execution wrapper for :mod:`fineqcomp.mdl`.

This keeps the MDL file format, allocation rule, and eight-step scale fitting
unchanged.  It only moves the expensive tensor arithmetic to the visible CUDA
GPU.  On CPU-only machines it falls back to the same vectorized operations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

import fineqcomp.mdl as mdl


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _midrise_components(
    matrix: torch.Tensor, bits: int, iterations: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fp16 row scales, integer codes, and reconstruction on one device."""
    if bits not in {1, 2, 3, 4, 8}:
        raise ValueError("midrise bits must be one of 1, 2, 3, 4, 8")

    work = matrix.float()
    absolute = work.abs()
    positive_levels = 1 << (bits - 1)
    max_level = 2 * positive_levels - 1
    row_max = absolute.amax(dim=1)

    # One bit has the closed-form optimum mean(abs(row)); running the generic
    # fixed-point loop would produce exactly the same result while wasting work.
    if bits == 1:
        index = torch.zeros_like(work, dtype=torch.int64)
        level = torch.ones_like(work)
        scale = absolute.mean(dim=1)
    else:
        scale = row_max / max_level
        for _ in range(iterations):
            safe_scale = scale.clamp_min(1e-12)
            index = torch.round((absolute / safe_scale[:, None] - 1.0) / 2.0)
            index = index.clamp(0, positive_levels - 1)
            level = 2.0 * index + 1.0
            scale = (absolute * level).sum(dim=1) / level.square().sum(dim=1)
            scale = torch.where(row_max > 0, scale, torch.zeros_like(scale))

    # Match pareto._midrise_tensor: one final assignment followed by an exact
    # least-squares scale refit, then round the transmitted scale to fp16 before
    # measuring reconstruction error.
    safe_scale = scale.clamp_min(1e-12)
    index = torch.round((absolute / safe_scale[:, None] - 1.0) / 2.0)
    index = index.clamp(0, positive_levels - 1)
    level = 2.0 * index + 1.0
    scale = (absolute * level).sum(dim=1) / level.square().sum(dim=1)
    scale = torch.where(row_max > 0, scale, torch.zeros_like(scale))

    scale16 = scale.to(torch.float16)
    signed_level = torch.where(work >= 0, level, -level)
    reconstructed = signed_level * scale16.float()[:, None]
    index_i64 = index.to(torch.int64)
    code = torch.where(
        work >= 0,
        index_i64 + positive_levels,
        index_i64,
    )
    return scale16, code, reconstructed


@torch.no_grad()
def _midrise_tensor_fast(
    tensor: torch.Tensor, bits: int, iterations: int = 8
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Drop-in replacement for mdl._midrise_tensor using the visible GPU."""
    device = _device()
    shape = tensor.shape
    matrix = tensor.detach().reshape(shape[0], -1).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    scales, codes, reconstructed = _midrise_components(matrix, bits, iterations)
    return (
        scales.cpu().numpy(),
        codes.reshape(-1).cpu().numpy().astype(np.uint16, copy=False),
        reconstructed.cpu().reshape(shape),
    )


@torch.no_grad()
def _row_candidates_fast(
    tensors: dict[str, torch.Tensor], *, show_progress: bool = False
) -> tuple[dict[str, Any], int]:
    """Build the exact row rate/distortion cache on CUDA, tensor by tensor."""
    device = _device()
    tensor_meta: list[dict[str, Any]] = []
    distortion_parts: list[torch.Tensor] = []
    proxy_parts: list[torch.Tensor] = []
    total_values = 0
    row_start = 0
    names = sorted(tensors)

    progress = tqdm(
        total=len(names) * (len(mdl.BIT_OPTIONS) - 1),
        desc=f"MDL preprocessing ({device.type})",
        unit="tensor-bit",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    try:
        for name in names:
            tensor = tensors[name].detach().cpu().float().contiguous()
            if tensor.ndim < 1:
                raise ValueError(f"cannot encode scalar tensor {name}")
            cpu_matrix = tensor.reshape(tensor.shape[0], -1)
            matrix = cpu_matrix.to(device=device, non_blocking=True)
            rows, columns = map(int, matrix.shape)
            total_values += tensor.numel()

            distortions = torch.empty(
                (rows, len(mdl.BIT_OPTIONS)), dtype=torch.float64
            )
            proxies = torch.empty(
                (rows, len(mdl.BIT_OPTIONS)), dtype=torch.int64
            )
            distortions[:, 0] = matrix.square().sum(dim=1).double().cpu()
            proxies[:, 0] = mdl.SELECTOR_BITS

            for option_index, bits in enumerate(mdl.BIT_OPTIONS[1:], start=1):
                _, _, reconstructed = _midrise_components(matrix, bits)
                distortions[:, option_index] = (
                    (matrix - reconstructed).square().sum(dim=1).double().cpu()
                )
                packed_bits = ((columns * bits + 7) // 8) * 8
                proxies[:, option_index] = mdl.SELECTOR_BITS + 16 + packed_bits
                progress.update(1)

            tensor_meta.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "row_start": row_start,
                    "row_count": rows,
                    "columns": columns,
                }
            )
            row_start += rows
            distortion_parts.append(distortions)
            proxy_parts.append(proxies)
            del matrix
    finally:
        progress.close()

    all_distortions = torch.cat(distortion_parts, dim=0)
    cache = {
        "version": mdl.CACHE_VERSION,
        "bit_options": list(mdl.BIT_OPTIONS),
        "selector_bits": mdl.SELECTOR_BITS,
        "total_values": total_values,
        "total_squared_norm": float(all_distortions[:, 0].sum().item()),
        "tensors": tensor_meta,
        "distortions": all_distortions,
        "proxy_bits": torch.cat(proxy_parts, dim=0),
        "accelerator": device.type,
    }
    return cache, total_values


def main() -> None:
    # mdl.prepare_candidate_cache and mdl._encode_mdl resolve these globals at
    # runtime, so this accelerates both preprocessing and per-point encoding
    # without duplicating the experiment implementation or changing its format.
    mdl._row_candidates = _row_candidates_fast
    mdl._midrise_tensor = _midrise_tensor_fast
    mdl.main()


if __name__ == "__main__":
    main()
