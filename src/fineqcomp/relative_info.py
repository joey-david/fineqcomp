"""Model-relative information measures from one frozen-model pass.

The dataset is not assigned an intrinsic size here.  Each record describes how
the frozen receiver sees the supervised correction: its surprise, the model's
representation of the examples, and a random sketch of the output-head
gradients induced by the target tokens.  Candidate scalars are reduced from
those shared sketches so an overnight panel does not need one model run per
measure.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fineqcomp.data import Example, _split_response
from fineqcomp.modeling import (
    CausalExampleDataset,
    causal_collate,
    model_device,
)


CANDIDATES = {
    "base_codelength_bits_per_token": (
        "Frozen-model code length of the taught tokens."
    ),
    "surprisal_variance_bits2": (
        "Between-example variance of frozen-model code length."
    ),
    "margin_deficit_bits": (
        "Soft deficit between the target logit and the best competing token."
    ),
    "hidden_effective_rank": (
        "Effective rank of response-position representations in the receiver."
    ),
    "surprise_weighted_hidden_logdet": (
        "Representation volume after weighting examples by model surprise."
    ),
    "residual_effective_rank": (
        "Effective rank of the output corrections requested by the targets."
    ),
    "fisher_trace": (
        "Mean squared norm of the sketched per-example correction gradient."
    ),
    "fisher_effective_rank": (
        "Effective rank of the empirical correction-gradient Fisher matrix."
    ),
    "fisher_logdet": (
        "Log volume of the empirical correction-gradient Fisher matrix."
    ),
    "unit_gain_ntk_cost": (
        "Linearized parameter norm needed for one unit of gain on every row."
    ),
}

DERIVED_CANDIDATES = {
    "correction_information_area": (
        "Mean full-dataset Fisher log-volume over 64, 128, and 256 rows."
    ),
    "coherent_correction_energy": (
        "Squared norm of the mean per-example correction gradient sketch."
    ),
    "coherent_energy_over_entropy": (
        "Shared correction energy divided by receiver predictive entropy."
    ),
}

SPECTRAL_CANDIDATES = {
    "correction_rd90_bits": (
        "Gaussian rate-distortion of the raw correction spectrum at 90% energy."
    ),
    "angular_correction_rd90_bits": (
        "Rate-distortion after giving every example equal gradient norm."
    ),
    "centered_correction_rd90_bits": (
        "Rate-distortion of raw corrections after removing their shared mean."
    ),
    "centered_angular_correction_rd90_bits": (
        "Rate-distortion of equal-norm corrections after removing their mean."
    ),
    "correction_energy_rank90": (
        "Raw correction directions needed to retain 90% of spectral energy."
    ),
    "angular_correction_energy_rank90": (
        "Equal-norm correction directions needed for 90% spectral energy."
    ),
    "trace_normalized_correction_logdet": (
        "Raw correction log-volume after removing arbitrary overall scale."
    ),
    "correction_information_bits": (
        "Scale-free correction information volume in bits."
    ),
    "angular_correction_logdet": (
        "Log-volume of the equal-norm correction kernel."
    ),
    "centered_angular_correction_logdet": (
        "Log-volume of equal-norm corrections after removing their shared mean."
    ),
    "dataset_correction_log_volume": (
        "Equal-norm correction log-volume scaled by distinct dataset rows."
    ),
    "dataset_raw_correction_log_volume": (
        "Trace-normalized raw correction volume scaled by distinct dataset rows."
    ),
    "dataset_centered_correction_log_volume": (
        "Centered equal-norm correction volume scaled by distinct dataset rows."
    ),
    "dataset_fisher_log_volume": (
        "Raw correction information volume over all distinct dataset rows."
    ),
    "correction_channel_bits": (
        "Rate of the raw correction spectrum against a noise floor set to a "
        "fixed fraction of its own mean eigenvalue."
    ),
    "angular_correction_channel_bits": (
        "The same rate for the equal-norm correction spectrum."
    ),
}


COVERAGE_CANDIDATES = {
    "prequential_correction_bits_per_row": (
        "Online code-length saving of an incremental map from receiver state "
        "to requested output correction, in bits per row."
    ),
    "prequential_decay_exponent": (
        "Log-log decay rate of the online prediction error; a fast decay means "
        "the corpus repeats a small number of corrections."
    ),
    "prequential_late_fraction": (
        "Share of the online code-length saving earned in the second half of "
        "the stream, so a corpus that keeps teaching scores high."
    ),
    "nearest_neighbour_cosine": (
        "Mean cosine between each row's correction and its closest neighbour."
    ),
    "correction_intrinsic_dimension": (
        "Two-nearest-neighbour intrinsic dimension of the correction cloud, "
        "which unlike effective rank is not inflated by isotropic noise."
    ),
    "correction_coverage_fraction": (
        "Effective number of distinct correction directions as a share of "
        "rows: the part of the corpus that is genuinely new."
    ),
    "correction_predictable_fraction": (
        "Held-out variance of the requested correction explained by a ridge "
        "map from receiver state: how systematic the correction is."
    ),
    "coherent_fraction": (
        "Squared norm of the mean unit correction; one when every row pulls "
        "the model in the same direction."
    ),
    "text_cross_row_redundancy": (
        "One minus the ratio of jointly to separately compressed responses."
    ),
    "tokenizer_fertility": (
        "Supervised tokens the receiver's tokenizer spends per character of "
        "response text: the cheapest measurement of the corpus against the "
        "model, and the only one that needs no forward pass."
    ),
}

LAYER_CANDIDATES = {
    "lora_correction_channel_bits": (
        "Rate of the adapter's own correction spectrum, sketched from the "
        "per-example gradient of every LoRA B at zero adapter."
    ),
    "lora_correction_effective_rank": (
        "Effective rank of that adapter correction spectrum."
    ),
    "lora_layer_energy_entropy": (
        "Entropy of the per-layer share of adapter gradient energy at zero "
        "adapter, normalized to one for a flat spread over layers."
    ),
    "lora_layer_energy_centroid": (
        "Depth of the adapter gradient energy, zero at the first layer and "
        "one at the last."
    ),
}

BASE_CANDIDATES = {
    "base_head_hidden_alignment": (
        "Share of response-token representation energy that lies inside the "
        "top singular directions of the frozen unembedding: how much of what "
        "the corpus asks about is written in coordinates the base weights "
        "already read strongly."
    ),
    "base_head_error_alignment": (
        "The same share for the demanded output correction, against the "
        "unembedding's top left singular directions."
    ),
    "within_example_channel_bits": (
        "Correction channel rate of the token-level spread inside examples, "
        "which the per-example mean throws away."
    ),
    "total_correction_channel_bits": (
        "Correction channel rate of every scored token, not of example means."
    ),
    "within_between_energy_ratio": (
        "Correction energy inside examples over correction energy between "
        "them: a corpus of internally varied responses scores high."
    ),
}

BASELINE_CANDIDATES = {
    "train_response_tokens": (
        "Supervised token count of the arm: the incumbent predictor any "
        "relative-information measure has to beat."
    ),
}


@dataclass(frozen=True)
class SketchSpec:
    hidden_dim: int = 64
    residual_dim: int = 32
    response_tokens: int = 32
    seed: int = 1729
    ntk_ridge: float = 0.1


def sample_examples(
    rows: list[Example], count: int, *, seed: int = 271_828
) -> list[Example]:
    """Take a stable uniform sample instead of a biased dataset prefix."""
    if count < 1:
        raise ValueError("sample count must be positive")
    if count >= len(rows):
        return list(rows)
    indices = sorted(random.Random(seed).sample(range(len(rows)), count))
    return [rows[index] for index in indices]


def _rademacher(
    rows: int,
    columns: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    values = torch.randint(
        0,
        2,
        (rows, columns),
        generator=generator,
        dtype=torch.int8,
    )
    return values.to(device=device, dtype=torch.float32).mul_(2).sub_(1).div_(
        math.sqrt(columns)
    )


def _spectrum(matrix: torch.Tensor, *, center: bool) -> torch.Tensor:
    values = matrix.to(dtype=torch.float64)
    if center and len(values) > 1:
        values = values - values.mean(dim=0, keepdim=True)
    if not values.numel():
        return torch.zeros(0, dtype=torch.float64)
    singular = torch.linalg.svdvals(values)
    return singular.square().div(max(len(values), 1))


def _effective_rank(eigenvalues: torch.Tensor) -> float:
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    probabilities = eigenvalues[eigenvalues > 0] / total
    entropy = -(probabilities * probabilities.log()).sum()
    return float(entropy.exp())


def _logdet(eigenvalues: torch.Tensor) -> float:
    return float(torch.log1p(eigenvalues.clamp_min(0)).sum())


def _spectral_rate_distortion(
    eigenvalues: torch.Tensor, retention: float = 0.90
) -> float:
    """Gaussian rate in bits at a fractional spectral-energy distortion.

    Reverse water-filling chooses a level theta such that replacing every
    eigenvalue below theta loses exactly `1 - retention` of total energy.  The
    resulting rate is invariant to a common rescaling of the spectrum.
    """
    if not 0 < retention < 1:
        raise ValueError("retention must be between zero and one")
    values = eigenvalues.double().clamp_min(0)
    total = float(values.sum())
    if not total:
        return 0.0
    target = (1 - retention) * total
    low = 0.0
    high = float(values.max())
    for _ in range(80):
        level = (low + high) / 2
        distortion = float(torch.minimum(values, values.new_tensor(level)).sum())
        if distortion < target:
            low = level
        else:
            high = level
    level = max(high, torch.finfo(torch.float64).tiny)
    active = values[values > level]
    return float(0.5 * torch.log2(active / level).sum())


_CHANNEL_NOISE_RATIO = 0.01


def _channel_bits(
    eigenvalues: torch.Tensor, gamma: float = _CHANNEL_NOISE_RATIO
) -> float:
    """Gaussian channel rate of a correction spectrum, in bits.

    The report's log-volume used an absolute noise floor of one, so a receiver
    that merely scales its gradients moves the measure.  Setting the floor to
    `mean(lambda) / gamma` instead makes the rate invariant to that scale and
    leaves one interpretable knob: only directions carrying at least `1/gamma`
    times the average eigenvalue are counted.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    values = eigenvalues.double().clamp_min(0)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    unit = values * (len(values) / total)
    return float(0.5 * torch.log2(1.0 + gamma * unit).sum())


def channel_bits_ceiling(rows: int, gamma: float = _CHANNEL_NOISE_RATIO) -> float:
    """The value `_channel_bits` returns for a white spectrum of `rows` modes.

    Because the spectrum is rescaled to unit mean before the log, every mode
    contributes at most `0.5 * log2(1 + gamma)` and a white spectrum hits that
    bound in every mode.  At 256 rows and gamma 0.01 the bound is 1.8375, and
    the measured corpora run from 1.17 to 1.81 -- the whole of the measure's
    range is the last few per cent below its own ceiling.  Reporting the
    distance to the ceiling instead of the raw rate leaves the ordering alone
    and gives the quantity a spread comparable to what it has to predict.
    """
    if rows <= 0:
        raise ValueError("rows must be positive")
    return float(0.5 * rows * math.log2(1.0 + gamma))


def _channel_deficit_bits(
    eigenvalues: torch.Tensor, gamma: float = _CHANNEL_NOISE_RATIO
) -> float:
    """How far the correction spectrum falls short of a white one, in bits."""
    values = eigenvalues.double().clamp_min(0)
    ceiling = channel_bits_ceiling(len(values), gamma)
    return max(ceiling - _channel_bits(values, gamma), 0.0)


def codec_referenced_retention(
    eigenvalues: torch.Tensor, relative_rmse: float
) -> float:
    """Share of correction energy a codec of relative error `eps` leaves behind.

    The codec's own perturbation sets the noise floor instead of a fitted
    gamma: white weight noise of relative size `eps` puts power
    `eps**2 * mean(lambda)` into every direction, and each mode is attenuated
    by the Wiener factor `lambda / (lambda + floor)`.  This is the whole of the
    forward model -- no fitted constant, no receiver term -- so the rate at
    which it reaches a retention target is a prediction of R* in the same
    units rather than a rank to be correlated.
    """
    values = eigenvalues.double().clamp_min(0)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    floor = float(relative_rmse) ** 2 * total / len(values)
    if floor <= 0:
        return 1.0
    return float((values**2 / (values + floor)).sum() / total)


def _energy_rank(eigenvalues: torch.Tensor, retention: float = 0.90) -> int:
    values = eigenvalues.double().clamp_min(0).sort(descending=True).values
    total = values.sum()
    if total <= 0:
        return 0
    return int(torch.searchsorted(values.cumsum(0), retention * total).item() + 1)


def _trace_normalized_logdet(eigenvalues: torch.Tensor, rows: int) -> float:
    values = eigenvalues.double().clamp_min(0)
    total = values.sum()
    if total <= 0:
        return 0.0
    return _logdet(values * (rows / total))


def _head_subspace(
    model: Any, rank: int = 64
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Top singular directions of the frozen unembedding.

    Every model-relative candidate so far reads the corpus through the base
    model's activations and never through its weights, so nothing in the panel
    can say whether a demanded correction lies along coordinates the receiver
    already uses.  These two bases are what the unembedding reads (right) and
    what it writes (left).
    """
    head = model.get_output_embeddings()
    weight = getattr(head, "weight", None)
    # A quantized head is stored as packed bytes, not as the matrix itself.
    if weight is None or weight.dim() != 2 or not weight.is_floating_point():
        return None
    left, _, right = torch.svd_lowrank(weight.detach().float(), q=rank, niter=4)
    return left.double().contiguous(), right.double().contiguous()


def _inside_fraction(rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Energy of each row inside an orthonormal basis, and its total."""
    projected = rows @ basis
    return torch.stack(
        (projected.square().sum(), rows.square().sum())
    )


def _normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    return matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)


_COVERAGE_SEED = 90_210
_COVERAGE_STREAMS = 3
_COVERAGE_RIDGE = 1e-2


def _scale_rows(matrix: torch.Tensor) -> torch.Tensor:
    """Give a sketch unit mean squared row norm, so ridge terms compare."""
    scale = matrix.square().sum(dim=1).mean().clamp_min(1e-12).sqrt()
    return matrix / scale


def _online_ridge_stream(
    features: torch.Tensor, targets: torch.Tensor, *, ridge: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared errors of an online ridge map, and of the running-mean null.

    Row i is predicted from rows before it only, so the summed log ratio of
    the two error streams is a description length the model never saw.
    """
    rows, width = features.shape
    columns = targets.shape[1]
    gram = torch.zeros((width, width), dtype=torch.float64)
    cross = torch.zeros((width, columns), dtype=torch.float64)
    eye = torch.eye(width, dtype=torch.float64)
    running_mean = torch.zeros(columns, dtype=torch.float64)
    errors = torch.zeros(rows, dtype=torch.float64)
    nulls = torch.zeros(rows, dtype=torch.float64)
    for index in range(rows):
        row = features[index]
        target = targets[index]
        if index:
            weights = torch.linalg.solve(gram + ridge * index * eye, cross)
            prediction = row @ weights
        else:
            prediction = torch.zeros(columns, dtype=torch.float64)
        errors[index] = (target - prediction).square().sum()
        nulls[index] = (target - running_mean).square().sum()
        gram += torch.outer(row, row)
        cross += torch.outer(row, target)
        running_mean = running_mean + (target - running_mean) / (index + 1)
    return errors, nulls


def _loglog_slope(values: torch.Tensor, bins: int = 10) -> float:
    """Slope of log mean value against log position, over geometric bins."""
    rows = len(values)
    bins = max(3, min(bins, rows // 4))
    edges = np.unique(np.geomspace(1, rows, bins + 1).round().astype(int))
    series = values.double().numpy()
    positions = []
    levels = []
    for low, high in zip(edges[:-1], edges[1:]):
        mean = float(np.mean(series[low - 1 : high]))
        if mean > 0:
            positions.append(math.log((low + high) / 2))
            levels.append(math.log(mean))
    if len(positions) < 3:
        return 0.0
    return float(np.polyfit(positions, levels, 1)[0])


def _prequential_candidates(
    features: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    """Average the online code-length stream over a few row orderings."""
    features = _scale_rows(features)
    targets = _scale_rows(targets)
    width = targets.shape[1]
    generator = torch.Generator()
    generator.manual_seed(_COVERAGE_SEED)
    gains = torch.zeros(len(features), dtype=torch.float64)
    errors = torch.zeros(len(features), dtype=torch.float64)
    for _ in range(_COVERAGE_STREAMS):
        order = torch.randperm(len(features), generator=generator)
        stream_errors, nulls = _online_ridge_stream(
            features[order], targets[order], ridge=_COVERAGE_RIDGE
        )
        gains += 0.5 * width * torch.log2(
            (nulls + 1e-12) / (stream_errors + 1e-12)
        )
        errors += stream_errors
    gains /= _COVERAGE_STREAMS
    errors /= _COVERAGE_STREAMS
    total = float(gains.sum())
    half = len(gains) // 2
    late = float(gains[half:].sum())
    return {
        "prequential_correction_bits_per_row": total / len(gains),
        "prequential_late_fraction": (
            late / total if abs(total) > 1e-9 else 0.0
        ),
        "prequential_decay_exponent": -_loglog_slope(errors),
    }


def _two_nn_dimension(unit: torch.Tensor) -> float:
    """Facco's two-nearest-neighbour intrinsic dimension of a point cloud."""
    rows = len(unit)
    if rows < 10:
        return 0.0
    distances = torch.cdist(unit, unit)
    distances.fill_diagonal_(float("inf"))
    nearest = distances.topk(2, dim=1, largest=False).values
    first, second = nearest[:, 0], nearest[:, 1]
    keep = (first > 1e-9) & (second > first)
    if int(keep.sum()) < 10:
        return 0.0
    ratios = (second[keep] / first[keep]).log()
    return float(int(keep.sum()) / ratios.sum().clamp_min(1e-12))


def _distinct_direction_count(similarity: torch.Tensor) -> float:
    """Effective number of distinct correction directions among unit rows.

    The participation ratio of the angular kernel: n for mutually orthogonal
    corrections, one when every row asks for the same thing.  A greedy net at
    a fixed cosine was tried first and kept every row, because real
    corrections are nearly orthogonal at any threshold worth naming.
    """
    rows = len(similarity)
    return rows**2 / float(similarity.square().sum().clamp_min(1e-12))


def _predictable_fraction(
    features: torch.Tensor, targets: torch.Tensor, *, holdout: int = 5
) -> float:
    """Held-out share of correction energy a ridge map from state explains."""
    features = _scale_rows(features)
    targets = _scale_rows(targets)
    index = torch.arange(len(features))
    test = index % holdout == 0
    train = ~test
    if int(train.sum()) < features.shape[1] + 2 or not int(test.sum()):
        return 0.0
    seen, wanted = features[train], targets[train]
    gram = seen.T @ seen + _COVERAGE_RIDGE * len(seen) * torch.eye(
        features.shape[1], dtype=torch.float64
    )
    weights = torch.linalg.solve(gram, seen.T @ wanted)
    predicted = features[test] @ weights
    residual = float((targets[test] - predicted).square().sum())
    null = float((targets[test] - wanted.mean(dim=0)).square().sum())
    return 1.0 - residual / max(null, 1e-12)


def coverage_candidates(
    hidden: torch.Tensor, residual: torch.Tensor, gradients: torch.Tensor
) -> dict[str, float]:
    """Count how many distinct corrections a corpus demands, not how loud."""
    hidden = hidden.to(dtype=torch.float64)
    residual = residual.to(dtype=torch.float64)
    unit = _normalize_rows(gradients.to(dtype=torch.float64))
    similarity = unit @ unit.T
    coverage = _distinct_direction_count(similarity) / len(unit)
    similarity.fill_diagonal_(-2.0)
    return {
        **_prequential_candidates(hidden, residual),
        "nearest_neighbour_cosine": float(similarity.max(dim=1).values.mean()),
        "correction_intrinsic_dimension": _two_nn_dimension(unit),
        "correction_coverage_fraction": coverage,
        "correction_predictable_fraction": _predictable_fraction(
            hidden, residual
        ),
        "coherent_fraction": float(unit.mean(dim=0).square().sum()),
    }


def tokenizer_fertility(tokenizer: Any, rows: list[Example]) -> float:
    """Tokens per response character under this receiver's tokenizer.

    The corpus text is fixed, so this varies only with the vocabulary the
    receiver brings to it: a model whose vocabulary already spells the corpus
    in whole pieces scores low, and one that has to break it into fragments
    scores high.  It is the only measurement of a corpus against a model that
    needs neither a forward pass nor a gradient, and on the same-corpus
    different-receiver comparisons it is the strongest predictor of R* that
    this campaign has.
    """
    characters = sum(len(row.response) for row in rows)
    if not characters:
        return 0.0
    tokens = sum(
        len(tokenizer(row.response, add_special_tokens=False)["input_ids"])
        for row in rows
    )
    return tokens / characters


def text_cross_row_redundancy(rows: list[Example]) -> float:
    """Compression control: how much shorter the responses are stored jointly."""
    import zlib

    responses = [row.response for row in rows]
    if len(responses) < 2:
        return 0.0
    joint = len(zlib.compress("\n".join(responses).encode(), 6))
    separate = sum(len(zlib.compress(text.encode(), 6)) for text in responses)
    return 1.0 - joint / max(separate, 1)


def reduce_sketches(
    surprisals: torch.Tensor,
    margin_deficits: torch.Tensor,
    predictive_entropies: torch.Tensor,
    hidden: torch.Tensor,
    residual: torch.Tensor,
    gradients: torch.Tensor,
    *,
    ntk_ridge: float = 0.1,
    population_rows: int | None = None,
) -> dict[str, Any]:
    """Reduce per-example sketches to the ten preregistered candidates."""
    if not 0 < ntk_ridge:
        raise ValueError("ntk_ridge must be positive")
    rows = len(surprisals)
    if not rows or any(
        len(matrix) != rows for matrix in (hidden, residual, gradients)
    ):
        raise ValueError("every sketch must contain the same non-zero row count")
    population_rows = rows if population_rows is None else population_rows
    if population_rows < 1:
        raise ValueError("population_rows must be positive")

    surprisals = surprisals.to(dtype=torch.float64)
    hidden = hidden.to(dtype=torch.float64)
    residual = residual.to(dtype=torch.float64)
    gradients = gradients.to(dtype=torch.float64)

    hidden_spectrum = _spectrum(_normalize_rows(hidden), center=True)
    weights = (surprisals / surprisals.mean().clamp_min(1e-12)).sqrt()
    weighted_hidden = _normalize_rows(hidden) * weights[:, None]
    weighted_hidden_spectrum = _spectrum(weighted_hidden, center=False)
    residual_spectrum = _spectrum(_normalize_rows(residual), center=True)
    fisher_spectrum = _spectrum(gradients, center=False)
    centered_fisher_spectrum = _spectrum(gradients, center=True)
    angular_gradients = _normalize_rows(gradients)
    angular_fisher_spectrum = _spectrum(angular_gradients, center=False)
    centered_angular_fisher_spectrum = _spectrum(
        angular_gradients, center=True
    )

    kernel = gradients @ gradients.T
    kernel_scale = kernel.diagonal().mean().clamp_min(1e-12)
    normalized_kernel = kernel / kernel_scale
    target = torch.ones(rows, dtype=torch.float64)
    regularized = normalized_kernel + ntk_ridge * torch.eye(
        rows, dtype=torch.float64
    )
    unit_gain_cost = float(target @ torch.linalg.solve(regularized, target) / rows)

    fisher_trace = float(gradients.square().sum(dim=1).mean())
    coherent_energy = float(gradients.mean(dim=0).square().sum())
    predictive_entropy = float(predictive_entropies.double().mean())

    candidates = {
        "base_codelength_bits_per_token": float(surprisals.mean()),
        "surprisal_variance_bits2": float(surprisals.var(unbiased=rows > 1)),
        "margin_deficit_bits": float(margin_deficits.double().mean()),
        "hidden_effective_rank": _effective_rank(hidden_spectrum),
        "surprise_weighted_hidden_logdet": _logdet(weighted_hidden_spectrum),
        "residual_effective_rank": _effective_rank(residual_spectrum),
        "fisher_trace": fisher_trace,
        "fisher_effective_rank": _effective_rank(fisher_spectrum),
        "fisher_logdet": _logdet(fisher_spectrum),
        "unit_gain_ntk_cost": unit_gain_cost,
    }
    spectral_candidates = {
        "correction_rd90_bits": _spectral_rate_distortion(fisher_spectrum),
        "angular_correction_rd90_bits": _spectral_rate_distortion(
            angular_fisher_spectrum
        ),
        "centered_correction_rd90_bits": _spectral_rate_distortion(
            centered_fisher_spectrum
        ),
        "centered_angular_correction_rd90_bits": _spectral_rate_distortion(
            centered_angular_fisher_spectrum
        ),
        "correction_energy_rank90": _energy_rank(fisher_spectrum),
        "angular_correction_energy_rank90": _energy_rank(
            angular_fisher_spectrum
        ),
        "trace_normalized_correction_logdet": _trace_normalized_logdet(
            fisher_spectrum, rows
        ),
        "correction_information_bits": _trace_normalized_logdet(
            fisher_spectrum, rows
        )
        / (2.0 * math.log(2.0)),
        "angular_correction_logdet": _trace_normalized_logdet(
            angular_fisher_spectrum, rows
        ),
        "centered_angular_correction_logdet": _trace_normalized_logdet(
            centered_angular_fisher_spectrum, rows
        ),
        "dataset_correction_log_volume": _trace_normalized_logdet(
            angular_fisher_spectrum, population_rows
        ),
        "dataset_raw_correction_log_volume": _trace_normalized_logdet(
            fisher_spectrum, population_rows
        ),
        "dataset_centered_correction_log_volume": _trace_normalized_logdet(
            centered_angular_fisher_spectrum, population_rows
        ),
        "dataset_fisher_log_volume": _logdet(
            fisher_spectrum * population_rows
        ),
        "correction_channel_bits": _channel_bits(fisher_spectrum),
        "angular_correction_channel_bits": _channel_bits(
            angular_fisher_spectrum
        ),
    }
    derived_candidates = {
        "coherent_correction_energy": coherent_energy,
        "coherent_energy_over_entropy": coherent_energy
        / max(predictive_entropy, 1e-12),
    }
    return {
        "candidates": candidates,
        "derived_candidates": derived_candidates,
        "spectral_candidates": spectral_candidates,
        "coverage_candidates": coverage_candidates(hidden, residual, gradients),
        "spectra": {
            "correction": fisher_spectrum.tolist(),
            "centered_correction": centered_fisher_spectrum.tolist(),
            "angular_correction": angular_fisher_spectrum.tolist(),
            "centered_angular_correction": (
                centered_angular_fisher_spectrum.tolist()
            ),
        },
        "diagnostics": {
            "rows": rows,
            "population_rows": population_rows,
            "predictive_entropy_bits_per_token": predictive_entropy,
            "mean_gradient_coherence": float(
                coherent_energy / max(fisher_trace, 1e-12)
            ),
            "hidden_stable_rank": float(
                hidden_spectrum.sum()
                / hidden_spectrum.max().clamp_min(1e-12)
            ),
            "residual_stable_rank": float(
                residual_spectrum.sum()
                / residual_spectrum.max().clamp_min(1e-12)
            ),
        },
    }


def _base_candidates(
    total_moment: torch.Tensor | None,
    within_moment: torch.Tensor | None,
    tokens: int,
    head_hidden: torch.Tensor,
    head_error: torch.Tensor,
) -> dict[str, float]:
    """Reduce the token-level moments and the unembedding alignment."""
    if total_moment is None or within_moment is None or tokens < 1:
        raise ValueError("no scored tokens were accumulated")
    total = torch.linalg.eigvalsh(total_moment.cpu() / tokens).clamp_min(0)
    within = torch.linalg.eigvalsh(within_moment.cpu() / tokens).clamp_min(0)
    within_energy = float(within.sum())
    between_energy = max(float(total.sum()) - within_energy, 0.0)
    candidates = {
        "within_example_channel_bits": _channel_bits(within),
        "total_correction_channel_bits": _channel_bits(total),
        "within_between_energy_ratio": within_energy
        / max(between_energy, 1e-12),
    }
    for key, counter in (
        ("base_head_hidden_alignment", head_hidden),
        ("base_head_error_alignment", head_error),
    ):
        inside, whole = counter.tolist()
        candidates[key] = inside / whole if whole > 0 else 0.0
    return candidates


@torch.no_grad()
def measure_relative_information(
    session: Any,
    rows: list[Example],
    model_spec: Any,
    max_length: int,
    micro_batch_size: int,
    *,
    label_span: str = "all",
    answer_marker: str | None = None,
    sketch: SketchSpec = SketchSpec(),
    population_rows: int | None = None,
) -> dict[str, Any]:
    """Measure ten relative-information candidates for one prepared arm."""
    if not rows:
        raise ValueError("relative information needs at least one row")
    if min(sketch.hidden_dim, sketch.residual_dim, sketch.response_tokens) < 1:
        raise ValueError("sketch dimensions and response_tokens must be positive")

    dataset = CausalExampleDataset(
        session.tokenizer,
        rows,
        model_spec,
        max_length,
        label_span,
        answer_marker,
    )
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        collate_fn=lambda batch: causal_collate(
            batch, session.tokenizer.pad_token_id
        ),
    )
    device = model_device(session.model)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(sketch.seed)
    hidden_projection: torch.Tensor | None = None
    residual_projection: torch.Tensor | None = None

    surprisals: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    hidden_rows: list[torch.Tensor] = []
    residual_rows: list[torch.Tensor] = []
    gradient_rows: list[torch.Tensor] = []
    scored_tokens = 0
    sampled_tokens = 0
    # Each example is reduced to the mean of its token gradients, so no
    # candidate can see how much a response varies inside itself. These second
    # moments keep that apart from the variation between examples.
    total_moment: torch.Tensor | None = None
    within_moment: torch.Tensor | None = None
    head = _head_subspace(session.model)
    head_hidden = torch.zeros(2, dtype=torch.float64, device=device)
    head_error = torch.zeros(2, dtype=torch.float64, device=device)

    session.model.eval()
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = session.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
        logits = output.logits[:, :-1]
        hidden_states = output.hidden_states[-1][:, :-1]
        labels = batch["labels"][:, 1:]
        if hidden_projection is None:
            hidden_projection = _rademacher(
                hidden_states.shape[-1], sketch.hidden_dim, generator, device
            )
            residual_projection = _rademacher(
                logits.shape[-1], sketch.residual_dim, generator, device
            )

        for row_index in range(len(labels)):
            positions = torch.nonzero(labels[row_index] != -100).flatten()
            scored_tokens += int(len(positions))
            if not len(positions):
                continue
            if len(positions) > sketch.response_tokens:
                chosen = torch.linspace(
                    0,
                    len(positions) - 1,
                    sketch.response_tokens,
                    device=device,
                ).round().long()
                positions = positions[chosen]
            sampled_tokens += int(len(positions))

            selected_logits = logits[row_index, positions].float()
            targets = labels[row_index, positions]
            log_probabilities = selected_logits.log_softmax(dim=-1)
            probabilities = log_probabilities.exp()
            target_logp = log_probabilities.gather(1, targets[:, None]).squeeze(1)
            surprisal = -target_logp / math.log(2)
            predictive_entropy = -(
                probabilities * log_probabilities
            ).sum(dim=1) / math.log(2)

            best = selected_logits.topk(2, dim=-1)
            competitor = torch.where(
                best.indices[:, 0] == targets,
                best.values[:, 1],
                best.values[:, 0],
            )
            target_logits = selected_logits.gather(1, targets[:, None]).squeeze(1)
            margin_deficit = F.softplus(competitor - target_logits) / math.log(2)

            selected_hidden = hidden_states[row_index, positions].float()
            hidden_sketch = selected_hidden @ hidden_projection
            residual_sketch = (
                probabilities @ residual_projection
                - residual_projection.index_select(0, targets)
            )
            gradient_sketch = torch.einsum(
                "ti,tj->tij", hidden_sketch, residual_sketch
            ).flatten(1)

            if total_moment is None:
                width = gradient_sketch.shape[1]
                total_moment = torch.zeros(
                    width, width, dtype=torch.float64, device=device
                )
                within_moment = torch.zeros_like(total_moment)
            tokens = gradient_sketch.double()
            centered = tokens - tokens.mean(dim=0, keepdim=True)
            total_moment += tokens.T @ tokens
            within_moment += centered.T @ centered
            if head is not None:
                error = probabilities.double()
                error[torch.arange(len(targets), device=device), targets] -= 1.0
                head_hidden += _inside_fraction(selected_hidden.double(), head[1])
                head_error += _inside_fraction(error, head[0])

            surprisals.append(surprisal.mean().cpu())
            margins.append(margin_deficit.mean().cpu())
            entropies.append(predictive_entropy.mean().cpu())
            hidden_rows.append(hidden_sketch.mean(dim=0).cpu())
            residual_rows.append(residual_sketch.mean(dim=0).cpu())
            gradient_rows.append(gradient_sketch.mean(dim=0).cpu())
        del output, logits, hidden_states

    if not surprisals:
        raise ValueError("no response tokens were scored")
    reduced = reduce_sketches(
        torch.stack(surprisals),
        torch.stack(margins),
        torch.stack(entropies),
        torch.stack(hidden_rows),
        torch.stack(residual_rows),
        torch.stack(gradient_rows),
        ntk_ridge=sketch.ntk_ridge,
        population_rows=population_rows,
    )
    reduced["base_candidates"] = _base_candidates(
        total_moment, within_moment, sampled_tokens, head_hidden, head_error
    )
    reduced["base_candidate_definitions"] = BASE_CANDIDATES
    reduced["coverage_candidates"]["text_cross_row_redundancy"] = (
        text_cross_row_redundancy(rows)
    )
    reduced["coverage_candidates"]["tokenizer_fertility"] = tokenizer_fertility(
        session.tokenizer, rows
    )
    reduced["coverage_candidate_definitions"] = COVERAGE_CANDIDATES
    reduced["candidate_definitions"] = CANDIDATES
    reduced["derived_candidate_definitions"] = DERIVED_CANDIDATES
    reduced["spectral_candidate_definitions"] = SPECTRAL_CANDIDATES
    reduced["sketch"] = {
        "hidden_dim": sketch.hidden_dim,
        "residual_dim": sketch.residual_dim,
        "response_tokens_per_row": sketch.response_tokens,
        "seed": sketch.seed,
        "ntk_ridge": sketch.ntk_ridge,
        "scored_tokens": scored_tokens,
        "sampled_tokens": sampled_tokens,
    }
    return reduced


def _length_matched_blocks(
    bodies: list[str], tokenizer: Any, candidates: int
) -> tuple[list[list[int]], list[int]]:
    """Group row indices into length-sorted blocks of `candidates` traces.

    The retrieval probe must not be winnable on length alone. Sorting by
    tokenized body length and cutting into blocks is the same construction the
    rationale control uses to pick a donor of similar length, so the measured
    load and the trained control see the same notion of "comparable trace".
    """
    lengths = [
        len(tokenizer.encode(body, add_special_tokens=False)) for body in bodies
    ]
    ordered = sorted(range(len(bodies)), key=lambda index: (lengths[index], index))
    blocks = [
        ordered[start : start + candidates]
        for start in range(0, len(ordered), candidates)
    ]
    # A trailing block of one has no distractor to offer, so fold it back.
    if len(blocks) > 1 and len(blocks[-1]) < 2:
        blocks[-2].extend(blocks.pop())
    return [block for block in blocks if len(block) > 1], lengths


def _retrieval_statistics(
    scores: list[float], widths: list[int], truth: list[int]
) -> dict[str, float]:
    """Cross entropy of the correct candidate, its error rate, and its floor."""
    bits: list[float] = []
    errors: list[bool] = []
    cursor = 0
    for index, width in enumerate(widths):
        block = torch.tensor(scores[cursor : cursor + width], dtype=torch.float64)
        cursor += width
        posterior = torch.log_softmax(block, dim=0)
        bits.append(-float(posterior[truth[index]]) / math.log(2))
        errors.append(int(torch.argmax(block)) != truth[index])
    error = sum(errors) / len(errors)
    candidates = sum(widths) / len(widths)
    ceiling = math.log2(candidates)
    entropy = (
        0.0
        if error in (0.0, 1.0)
        else -error * math.log2(error) - (1 - error) * math.log2(1 - error)
    )
    return {
        "bits": sum(bits) / len(bits),
        "error": error,
        "candidates": candidates,
        "ceiling_bits": ceiling,
        # Fano: a code that must recover the assignment at this error rate
        # cannot be smaller than this, so it is a floor for the adapter budget
        # rather than a prediction of it.
        "fano_bits": max(
            0.0, ceiling - entropy - error * math.log2(max(candidates - 1, 1))
        ),
    }


@torch.no_grad()
def trace_retrieval_load(
    session: Any,
    rows: list[Example],
    model_spec: Any,
    max_length: int,
    micro_batch_size: int,
    *,
    marker: str,
    from_end: bool = True,
    candidates: int = 8,
) -> dict[str, float | int]:
    """L_rel: bits of problem-trace information the frozen model is missing.

    Each problem keeps its own prompt and its own final answer, and is scored
    against the reasoning bodies of `candidates` length-matched problems, one
    of which is its own. Softmaxing the frozen model's sequence log likelihood
    over that set gives a posterior on the trace index, and the cross entropy
    of the correct index is the load. It reads zero bits when the frozen model
    already knows which trace belongs to which problem, and log2(K) bits when
    it cannot tell them apart at all -- which is the quantity an adapter would
    have to supply.

    The manipulation is deliberately the one the trained control performs, so
    the measure and the causal test speak about the same thing. No training
    happens here: it is one forward pass per problem-candidate pair.

    Two scores are reported for the same probes. The summed sequence log
    likelihood is the literal reading of the measure, and on long traces it is
    dominated by length: a block spanning a third of its median length differs
    by hundreds of nats before any reasoning is compared, which saturates the
    posterior and makes the mean cross entropy a coin flip between nothing and
    hundreds of bits. The per-token mean removes that first-order term. Where
    the two disagree, the length effect is doing the work.
    """
    if candidates < 2:
        raise ValueError("trace retrieval needs at least two candidates")
    split = [_split_response(row.response, marker, from_end=from_end) for row in rows]
    bodies = [body for body, _ in split]
    blocks, lengths = _length_matched_blocks(bodies, session.tokenizer, candidates)
    if not blocks:
        raise ValueError("trace retrieval needs at least two rows")

    probes: list[Example] = []
    truth: list[int] = []
    widths: list[int] = []
    spreads: list[float] = []
    for block in blocks:
        block_lengths = [lengths[index] for index in block]
        median = sorted(block_lengths)[len(block_lengths) // 2]
        spread = max(abs(length - median) for length in block_lengths) / max(median, 1)
        for recipient in block:
            _, answer = split[recipient]
            truth.append(block.index(recipient))
            widths.append(len(block))
            spreads.append(spread)
            for donor in block:
                probes.append(
                    Example(
                        example_id=f"{rows[recipient].example_id}|{donor}",
                        prompt=rows[recipient].prompt,
                        response=bodies[donor] + answer,
                        metadata={"recipient": recipient, "donor": donor},
                    )
                )

    dataset = CausalExampleDataset(
        session.tokenizer, probes, model_spec, max_length, "all"
    )
    if len(dataset.rows) != len(probes):
        raise ValueError("trace retrieval lost a probe row to tokenization")
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        collate_fn=lambda batch: causal_collate(batch, session.tokenizer.pad_token_id),
    )
    device = model_device(session.model)
    session.model.eval()
    totals: list[float] = []
    counts: list[float] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = session.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        logits = output.logits[:, :-1].float()
        labels = batch["labels"][:, 1:]
        chosen = labels.clamp_min(0).unsqueeze(-1)
        token_logprob = torch.log_softmax(logits, dim=-1).gather(-1, chosen)
        scored = labels != -100
        token_logprob = token_logprob.squeeze(-1).masked_fill(~scored, 0.0)
        totals.extend(token_logprob.sum(dim=-1).tolist())
        counts.extend(scored.sum(dim=-1).clamp_min(1).float().tolist())

    summed = _retrieval_statistics(totals, widths, truth)
    per_token = _retrieval_statistics(
        [total / count for total, count in zip(totals, counts, strict=True)],
        widths,
        truth,
    )
    return {
        "trace_retrieval_bits": summed["bits"],
        "trace_retrieval_error": summed["error"],
        "trace_retrieval_fano_bits": summed["fano_bits"],
        "trace_retrieval_normalized_bits": per_token["bits"],
        "trace_retrieval_normalized_error": per_token["error"],
        "trace_retrieval_normalized_fano_bits": per_token["fano_bits"],
        "trace_retrieval_candidates": summed["candidates"],
        "trace_retrieval_ceiling_bits": summed["ceiling_bits"],
        "trace_retrieval_probes": len(probes),
        "trace_retrieval_length_spread": sum(spreads) / len(spreads),
    }


def measure_layer_energy(
    session: Any,
    rows: list[Example],
    model_spec: Any,
    max_length: int,
    *,
    label_span: str = "all",
    answer_marker: str | None = None,
    rank: int = 16,
    seed: int = 1729,
    probe_columns: int = 4,
) -> dict[str, Any]:
    """Where in the depth of the model the corpus asks for its correction.

    A fresh LoRA with a zero B leaves the model's output untouched, so one
    backward pass reads the gradient the corpus would first apply to each
    projection.  The per-layer share of that energy is a rate allocation the
    adapter could be given, which no spectrum of output corrections provides.

    The same backward pass also gives the correction spectrum of the adapter
    itself, rather than of a projection of the output head.  Each B gradient
    is a (units, rank) matrix; a norm-preserving Rademacher probe of its unit
    axis makes one row of a per-example sketch whose spectrum is the adapter's
    empirical Fisher.
    """
    from fineqcomp.adapters import attach_adapter, module_layer_index, unload_adapter
    from fineqcomp.config import AdapterSpec

    spec = AdapterSpec(
        key="relative_information_probe",
        method="seeded_b",
        rank=rank,
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        last_n_layers=None,
        alpha=2 * rank,
    )
    dataset = CausalExampleDataset(
        session.tokenizer,
        rows,
        model_spec,
        max_length,
        label_span,
        answer_marker,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: causal_collate(
            batch, session.tokenizer.pad_token_id
        ),
    )
    model = attach_adapter(session.model, spec, seed)
    device = model_device(model)
    energy: dict[str, float] = {}
    probes: dict[str, torch.Tensor] = {}
    sketches: list[torch.Tensor] = []
    generator = torch.Generator().manual_seed(seed)
    scored = 0
    try:
        model.eval()
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            if int((batch["labels"] != -100).sum()) == 0:
                continue
            model.zero_grad(set_to_none=True)
            loss = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            ).loss
            loss.backward()
            row = []
            for name, parameter in model.named_parameters():
                if "lora_B" not in name or parameter.grad is None:
                    continue
                gradient = parameter.grad.float()
                energy[name] = energy.get(name, 0.0) + float(
                    gradient.double().square().sum()
                )
                if name not in probes:
                    probes[name] = _rademacher(
                        gradient.shape[0], probe_columns, generator, device
                    )
                row.append((probes[name].T @ gradient).reshape(-1))
            sketches.append(torch.cat(row).double().cpu())
            scored += 1
        model.zero_grad(set_to_none=True)
    finally:
        unload_adapter(model)
    if not energy:
        raise ValueError("no adapter gradient energy was collected")
    lora_spectrum = _spectrum(torch.stack(sketches), center=False)

    by_layer: dict[int, float] = {}
    by_projection: dict[str, float] = {}
    for name, value in energy.items():
        layer = module_layer_index(name)
        if layer is not None:
            by_layer[layer] = by_layer.get(layer, 0.0) + value
        projection = next(
            (part for part in name.split(".") if part in spec.target_modules),
            "other",
        )
        by_projection[projection] = by_projection.get(projection, 0.0) + value
    layers = sorted(by_layer)
    shares = np.asarray([by_layer[layer] for layer in layers], dtype=float)
    shares = shares / max(shares.sum(), 1e-30)
    depth = (
        np.asarray(layers, dtype=float) / max(max(layers), 1)
        if layers
        else np.zeros(0)
    )
    positive = shares[shares > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    return {
        "layer_candidates": {
            "lora_correction_channel_bits": _channel_bits(lora_spectrum),
            "lora_correction_effective_rank": _effective_rank(lora_spectrum),
            "lora_layer_energy_entropy": (
                entropy / math.log(len(layers)) if len(layers) > 1 else 0.0
            ),
            "lora_layer_energy_centroid": float(shares @ depth),
        },
        "layer_candidate_definitions": LAYER_CANDIDATES,
        "layer_energy": {
            "rank": rank,
            "seed": seed,
            "rows": scored,
            "layers": layers,
            "layer_share": shares.tolist(),
            "probe_columns": probe_columns,
            "correction_spectrum": lora_spectrum.tolist(),
            "projection_share": {
                key: value / max(sum(by_projection.values()), 1e-30)
                for key, value in sorted(by_projection.items())
            },
        },
    }


_RAW_PREDICTORS = frozenset(
    {
        "prequential_correction_bits_per_row",
        "prequential_decay_exponent",
        "prequential_late_fraction",
        "correction_predictable_fraction",
        "lora_layer_energy_entropy",
        "lora_layer_energy_centroid",
        "lora_correction_channel_bits",
        "correction_channel_bits",
        "angular_correction_channel_bits",
        "correction_information_bits",
        "base_head_hidden_alignment",
        "base_head_error_alignment",
        "within_example_channel_bits",
        "total_correction_channel_bits",
    }
)


def _train_response_tokens(run_dir: Path) -> float:
    """Supervised token count of a finished run, the incumbent predictor."""
    record = json.loads((run_dir / "metrics.json").read_text())
    train = (record.get("raw_information") or {}).get("train") or {}
    tokens = train.get("nll_tokens")
    if tokens is None:
        raise ValueError(f"run has no supervised token count: {run_dir.name}")
    return float(tokens)


# Hiding one dataset key is not hiding a task. The compressibility arms, the
# diversity and duplication levers, the behaviour rewrites and all three math
# panels are drawn from MetaMathQA, so a fold that hides one of them still fits
# on nineteen others and the held-out error is not held out at all. Fold by the
# corpus a dataset was drawn from instead.
_SOURCE_CORPORA = {
    "metamath": (
        "arm_",
        "lever_",
        "behav_",
        "cot_math",
        "panel_math",
        "metamath",
        "budget_math",
    ),
    "gsm8k": ("gsm8k",),
    "magicoder": ("kind_code", "code_long"),
    "hh_rlhf": ("kind_dialogue",),
    "alpaca": ("kind_instruct",),
    "xsum": ("kind_summary",),
    "mbpp": ("mbpp",),
    "paws": ("paws",),
    "sql": ("text_to_sql", "sql_div_"),
    "xbrl": ("xbrl_tags", "xbrl_div_"),
}


def _task_family(dataset_key: str) -> str:
    key = str(dataset_key)
    for corpus, prefixes in _SOURCE_CORPORA.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            return corpus
    return key


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, including ties, without a scipy dependency."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def _unit_vector(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    norm = float(np.linalg.norm(centered))
    return centered / norm if norm else np.zeros_like(centered)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(_unit_vector(left) @ _unit_vector(right))


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _correlation(_rankdata(left), _rankdata(right))


def _mean_record(
    records: list[dict[str, Any]], candidate_keys: tuple[str, ...]
) -> dict[str, Any]:
    first = records[0]
    return {
        "model_key": first["model_key"],
        "dataset_key": first["dataset_key"],
        "task_family": _task_family(str(first["dataset_key"])),
        "seeds": len(records),
        "r_star_bits_per_value": float(
            np.mean([row["r_star_bits_per_value"] for row in records])
        ),
        "reference_bits_saved_per_token": float(
            np.mean([row["reference_bits_saved_per_token"] for row in records])
        ),
        **{
            key: float(np.mean([row[key] for row in records]))
            for key in candidate_keys
        },
    }


def _aggregate_dataset_arms(
    cells: list[dict[str, Any]], candidate_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in cells:
        grouped.setdefault(
            (str(row["model_key"]), str(row["dataset_key"])), []
        ).append(row)
    return [_mean_record(rows, candidate_keys) for rows in grouped.values()]


def _merge_reused_arms(
    arms: list[dict[str, Any]], candidate_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    # The high-gain SQL/XBRL arms reuse the largest diversity datasets. Merge
    # exact repeats so the same frozen-model measurement is not counted twice.
    unique: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    measured_keys = tuple(
        key for key in candidate_keys if key not in BASELINE_CANDIDATES
    )
    for arm in arms:
        signature = (
            arm["model_key"],
            *(round(float(arm[key]), 12) for key in measured_keys),
        )
        unique.setdefault(signature, []).append(arm)
    merged = []
    for repeats in unique.values():
        first = dict(repeats[0])
        first["dataset_key"] = "+".join(
            sorted(str(row["dataset_key"]) for row in repeats)
        )
        first["r_star_bits_per_value"] = float(
            np.mean([row["r_star_bits_per_value"] for row in repeats])
        )
        first["reference_bits_saved_per_token"] = float(
            np.mean([row["reference_bits_saved_per_token"] for row in repeats])
        )
        first["repeated_dataset_arms"] = len(repeats)
        merged.append(first)
    return sorted(merged, key=lambda row: (row["task_family"], row["model_key"]))


def _design(
    arms: list[dict[str, Any]],
    candidates: tuple[str, ...] = (),
    *,
    models: list[str] | None = None,
) -> np.ndarray:
    """Model fixed effects plus one predictor.

    The model list is passed in when a fold is being fitted: a held-out task
    family need not contain every model, and a design built from the fold
    alone has the wrong number of columns to take the fitted coefficients.
    """
    columns = [np.ones(len(arms))]
    if models is None:
        models = sorted({str(row["model_key"]) for row in arms})
    for model in models[1:]:
        columns.append(
            np.asarray([row["model_key"] == model for row in arms], dtype=float)
        )
    for candidate in candidates:
        columns.append(_predictor_values(arms, candidate))
    return np.column_stack(columns)


def _predictor_values(
    rows: list[dict[str, Any]], candidate: str
) -> np.ndarray:
    values = np.asarray([row[candidate] for row in rows], dtype=float)
    if (
        candidate.endswith(("logdet", "log_volume"))
        or candidate == "correction_information_area"
        or candidate in _RAW_PREDICTORS
    ):
        return values
    return np.log(np.maximum(values, 1e-12))


def _leave_family_out(
    arms: list[dict[str, Any]],
    candidates: tuple[str, ...],
    group: str = "task_family",
) -> tuple[float, float, list[dict[str, Any]]]:
    """Held-out prediction of R*, hiding one task family or one model at a time.

    Hiding a model is the harder test: the fixed effect for the held-out model
    is unavailable, so only a predictor that carries the level of R* across
    receivers can help.
    """
    predictions: list[float] = []
    baselines: list[float] = []
    observed: list[float] = []
    folds = []
    models = sorted({str(row["model_key"]) for row in arms})
    for held in sorted({str(row[group]) for row in arms}):
        train = [row for row in arms if row[group] != held]
        test = [row for row in arms if row[group] == held]
        y_train = np.asarray(
            [row["r_star_bits_per_value"] for row in train], dtype=float
        )
        candidate_fit = np.linalg.lstsq(
            _design(train, candidates, models=models), y_train, rcond=None
        )[0]
        baseline_fit = np.linalg.lstsq(
            _design(train, models=models), y_train, rcond=None
        )[0]
        predicted = _design(test, candidates, models=models) @ candidate_fit
        baseline = _design(test, models=models) @ baseline_fit
        truth = np.asarray(
            [row["r_star_bits_per_value"] for row in test], dtype=float
        )
        predictions.extend(predicted.tolist())
        baselines.extend(baseline.tolist())
        observed.extend(truth.tolist())
        folds.append(
            {
                "group": group,
                "held_out": held,
                "arms": len(test),
                "candidate_rmse": float(np.sqrt(np.mean((predicted - truth) ** 2))),
                "model_only_rmse": float(np.sqrt(np.mean((baseline - truth) ** 2))),
            }
        )
    truth = np.asarray(observed)
    candidate_rmse = float(
        np.sqrt(np.mean((np.asarray(predictions) - truth) ** 2))
    )
    baseline_rmse = float(
        np.sqrt(np.mean((np.asarray(baselines) - truth) ** 2))
    )
    return candidate_rmse, baseline_rmse, folds


def _fixed_effect_residual(
    arms: list[dict[str, Any]], values: np.ndarray
) -> np.ndarray:
    columns = [np.ones(len(arms))]
    models = sorted({str(row["model_key"]) for row in arms})
    families = sorted({str(row["task_family"]) for row in arms})
    for model in models[1:]:
        columns.append(
            np.asarray([row["model_key"] == model for row in arms], dtype=float)
        )
    for family in families[1:]:
        columns.append(
            np.asarray([row["task_family"] == family for row in arms], dtype=float)
        )
    design = np.column_stack(columns)
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def _selection_adjusted_p_values(
    arms: list[dict[str, Any]],
    candidate_keys: tuple[str, ...],
    permutations: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    by_model = {
        model: [row for row in arms if row["model_key"] == model]
        for model in sorted({str(row["model_key"]) for row in arms})
    }
    candidate_vectors = []
    target_vectors = []
    for rows in by_model.values():
        candidate_vectors.append(
            np.stack(
                [
                    _unit_vector(
                        _rankdata(np.asarray([row[key] for row in rows], dtype=float))
                    )
                    for key in candidate_keys
                ]
            )
        )
        target_vectors.append(
            _unit_vector(
                _rankdata(
                    np.asarray(
                        [row["r_star_bits_per_value"] for row in rows], dtype=float
                    )
                )
            )
        )
    observed = sum(
        matrix @ target
        for matrix, target in zip(candidate_vectors, target_vectors)
    ) / len(candidate_vectors)
    rng = np.random.default_rng(seed)
    max_null = np.empty(permutations)
    max_absolute_null = np.empty(permutations)
    for index in range(permutations):
        null = sum(
            matrix @ rng.permutation(target)
            for matrix, target in zip(candidate_vectors, target_vectors)
        ) / len(candidate_vectors)
        max_null[index] = float(np.max(null))
        max_absolute_null[index] = float(np.max(np.abs(null)))
    correlations = {
        key: float(observed[index]) for index, key in enumerate(candidate_keys)
    }
    adjusted = {
        key: float((1 + np.sum(max_null >= observed[index])) / (permutations + 1))
        for index, key in enumerate(candidate_keys)
    }
    # A measure that predicts a *smaller* rate is as much a result as one that
    # predicts a larger rate, and the one-sided form scores it at one.
    two_sided = {
        key: float(
            (1 + np.sum(max_absolute_null >= abs(observed[index])))
            / (permutations + 1)
        )
        for index, key in enumerate(candidate_keys)
    }
    return correlations, adjusted, two_sided


def _median_seed_cv(cells: list[dict[str, Any]], candidate: str) -> float:
    groups: dict[tuple[str, str], list[float]] = {}
    for row in cells:
        groups.setdefault(
            (str(row["model_key"]), str(row["dataset_key"])), []
        ).append(float(row[candidate]))
    cvs = []
    for values in groups.values():
        mean = float(np.mean(values))
        if len(values) > 1 and mean:
            cvs.append(float(np.std(values, ddof=1) / mean))
    return float(np.median(cvs)) if cvs else 0.0


def _diversity_slopes(
    arms: list[dict[str, Any]], candidate_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
    for row in arms:
        dataset = str(row["dataset_key"]).split("+")[0]
        parts = dataset.rsplit("_div_", 1)
        if len(parts) != 2 or parts[0] not in {"sql", "xbrl"}:
            continue
        grouped.setdefault((str(row["model_key"]), parts[0]), []).append(
            (int(parts[1]), row)
        )
    slopes = []
    for (model, task), points in sorted(grouped.items()):
        points.sort()
        if len(points) < 2:
            continue
        x = np.log2(np.asarray([groups for groups, _ in points], dtype=float))
        for measure in ("r_star_bits_per_value", *candidate_keys):
            y = np.asarray([row[measure] for _, row in points], dtype=float)
            if measure != "r_star_bits_per_value":
                y = _predictor_values(
                    [row for _, row in points], measure
                )
            slopes.append(
                {
                    "model_key": model,
                    "task": task,
                    "measure": measure,
                    "slope_per_doubling": float(np.polyfit(x, y, 1)[0]),
                }
            )
    return slopes


def _diversity_sign_agreement(
    slopes: list[dict[str, Any]], candidate: str, direction: float
) -> float:
    """Share of diversity families where a candidate moves with R*.

    Diversity varies the number of distinct source groups at a fixed token
    count, so token count cannot move here by construction.  A candidate that
    predicts a larger rate must rise with R*, and one that predicts a smaller
    rate must fall with it, which is what `direction` carries.
    """
    families = {
        (row["model_key"], row["task"]): row["slope_per_doubling"]
        for row in slopes
        if row["measure"] == "r_star_bits_per_value"
    }
    if not families or not direction:
        return 0.0
    agree = 0
    for row in slopes:
        if row["measure"] != candidate:
            continue
        target = families.get((row["model_key"], row["task"]))
        if target is None:
            continue
        agree += np.sign(row["slope_per_doubling"] * direction) == np.sign(target)
    return agree / len(families)


def analyze_relative_information_cells(
    cells: list[dict[str, Any]], *, permutations: int = 50_000, seed: int = 1729
) -> dict[str, Any]:
    """Rank the ten candidates without counting seed repeats as new tasks."""
    if not cells:
        raise ValueError("relative-information analysis needs result cells")
    candidate_keys = tuple(
        key
        for key in (
            *CANDIDATES,
            *SPECTRAL_CANDIDATES,
            *DERIVED_CANDIDATES,
            *COVERAGE_CANDIDATES,
            *BASE_CANDIDATES,
            *LAYER_CANDIDATES,
            *BASELINE_CANDIDATES,
        )
        if all(key in row for row in cells)
    )
    dataset_arms = _aggregate_dataset_arms(cells, candidate_keys)
    arms = _merge_reused_arms(dataset_arms, candidate_keys)
    target = np.asarray(
        [row["r_star_bits_per_value"] for row in arms], dtype=float
    )
    target_residual = _fixed_effect_residual(arms, target)
    within_model, adjusted_p, two_sided_p = _selection_adjusted_p_values(
        arms, candidate_keys, permutations, seed
    )
    incumbent = "train_response_tokens" if any(
        "train_response_tokens" in row for row in arms
    ) else None
    slopes = _diversity_slopes(dataset_arms, candidate_keys)
    ranking = []
    fold_records = []
    for candidate in candidate_keys:
        values = np.asarray([row[candidate] for row in arms], dtype=float)
        candidate_rmse, baseline_rmse, folds = _leave_family_out(
            arms, (candidate,)
        )
        model_rmse, model_baseline, model_folds = _leave_family_out(
            arms, (candidate,), "model_key"
        )
        for fold in (*folds, *model_folds):
            fold_records.append({"candidate": candidate, **fold})
        paired = (
            (candidate,)
            if incumbent is None or candidate == incumbent
            else (incumbent, candidate)
        )
        logged = _predictor_values(arms, candidate)
        ranking.append(
            {
                "candidate": candidate,
                "pooled_arm_spearman": _spearman(values, target),
                "within_model_mean_spearman": within_model[candidate],
                "selection_adjusted_p": adjusted_p[candidate],
                "selection_adjusted_p_two_sided": two_sided_p[candidate],
                "task_heldout_rmse": candidate_rmse,
                "model_only_rmse": baseline_rmse,
                "task_heldout_rmse_ratio": candidate_rmse / baseline_rmse,
                "model_heldout_rmse": model_rmse,
                "model_only_model_heldout_rmse": model_baseline,
                "with_incumbent_task_heldout_rmse": _leave_family_out(
                    arms, paired
                )[0],
                "with_incumbent_model_heldout_rmse": _leave_family_out(
                    arms, paired, "model_key"
                )[0],
                "model_and_task_residual_r": _correlation(
                    _fixed_effect_residual(arms, logged), target_residual
                ),
                "gain_spearman": _spearman(
                    values,
                    np.asarray(
                        [row["reference_bits_saved_per_token"] for row in arms],
                        dtype=float,
                    ),
                ),
                "median_seed_cv": _median_seed_cv(cells, candidate),
                "diversity_sign_agreement": _diversity_sign_agreement(
                    slopes, candidate, _spearman(values, target)
                ),
            }
        )
    ranking.sort(
        key=lambda row: abs(row["within_model_mean_spearman"]), reverse=True
    )
    strongest = ranking[0]
    base = next(
        row
        for row in ranking
        if row["candidate"] == "base_codelength_bits_per_token"
    )
    return {
        "summary": {
            "cells": len(cells),
            "distinct_arms": len(arms),
            "task_families": len({row["task_family"] for row in arms}),
            "permutations": permutations,
            "strongest_candidate": strongest["candidate"],
            "strongest_within_model_mean_spearman": strongest[
                "within_model_mean_spearman"
            ],
            "strongest_selection_adjusted_p": strongest[
                "selection_adjusted_p"
            ],
            "strongest_task_heldout_rmse": strongest["task_heldout_rmse"],
            "model_only_task_heldout_rmse": strongest["model_only_rmse"],
            "base_codelength_task_heldout_rmse": base["task_heldout_rmse"],
        },
        "arms": arms,
        "ranking": ranking,
        "folds": fold_records,
        "diversity_slopes": slopes,
    }


def collect_relative_information_cells(
    result_root: Path, runs_root: Path, *, skip_missing_r_star: bool = False
) -> list[dict[str, Any]]:
    """Join frozen-model measures to bracketed R* results by exact run ID."""
    from fineqcomp.rstar import from_run

    cells = []
    for path in sorted(Path(result_root).glob("*.json")):
        measured = json.loads(path.read_text())
        spectral_candidates = spectral_candidates_from_record(measured)
        run_dir = Path(runs_root) / str(measured["run_id"])
        result = from_run(run_dir)
        if result.get("r_star") is None:
            if skip_missing_r_star:
                continue
            raise ValueError(f"run has no measured R*: {measured['run_id']}")
        config = json.loads((run_dir / "config.json").read_text())
        cells.append(
            {
                "run_id": measured["run_id"],
                "study": measured["study"],
                "model_key": measured["model_key"],
                "dataset_key": measured["dataset_key"],
                "seed": measured["seed"],
                "adapter_rank": config["adapter"]["rank"],
                "training_epochs": config["training"]["epochs"],
                "label_span": config["training"].get("label_span", "all"),
                "available_rows": measured.get("available_rows"),
                "distinct_rows": measured.get("distinct_rows"),
                "measured_rows": measured.get("measured_rows"),
                "row_sampling": measured.get("row_sampling", "head"),
                "row_sample_seed": measured.get("row_sample_seed"),
                "r_star_bits_per_value": result["r_star"],
                "r_star_bracketed": result.get("bracketed"),
                "reference_bits_saved_per_token": result["reference"],
                "train_response_tokens": _train_response_tokens(run_dir),
                **measured["candidates"],
                **spectral_candidates,
                **measured.get("coverage_candidates", {}),
                **measured.get("base_candidates", {}),
                **measured.get("layer_candidates", {}),
            }
        )
    return cells


def spectral_candidates_from_record(measured: dict[str, Any]) -> dict[str, float]:
    """Read spectral scalars, deriving new forms from saved spectra when needed."""
    candidates = dict(measured.get("spectral_candidates", {}))
    candidates.update(measured.get("derived_candidates", {}))
    if (
        "correction_information_bits" not in candidates
        and "trace_normalized_correction_logdet" in candidates
    ):
        candidates["correction_information_bits"] = candidates[
            "trace_normalized_correction_logdet"
        ] / (2.0 * math.log(2.0))
    if "coherent_correction_energy" not in candidates:
        base = measured.get("candidates", {})
        diagnostics = measured.get("diagnostics", {})
        trace = base.get("fisher_trace")
        coherence = diagnostics.get("mean_gradient_coherence")
        entropy = diagnostics.get("predictive_entropy_bits_per_token")
        if trace is not None and coherence is not None:
            coherent_energy = float(trace) * float(coherence)
            candidates["coherent_correction_energy"] = coherent_energy
            if entropy is not None:
                candidates["coherent_energy_over_entropy"] = (
                    coherent_energy / max(float(entropy), 1e-12)
                )
    spectra = measured.get("spectra", {})
    for key, name in (
        ("correction_channel_bits", "correction"),
        ("angular_correction_channel_bits", "angular_correction"),
    ):
        if key not in candidates and spectra.get(name) is not None:
            candidates[key] = _channel_bits(
                torch.as_tensor(spectra[name], dtype=torch.float64)
            )
    for key, name in (
        ("correction_channel_deficit_bits", "correction"),
        ("angular_correction_channel_deficit_bits", "angular_correction"),
    ):
        if key not in candidates and spectra.get(name) is not None:
            candidates[key] = _channel_deficit_bits(
                torch.as_tensor(spectra[name], dtype=torch.float64)
            )
    if (
        "dataset_fisher_log_volume" not in candidates
        and measured.get("spectra", {}).get("correction") is not None
        and measured.get("distinct_rows")
    ):
        population_rows = int(measured["distinct_rows"])
        candidates["dataset_fisher_log_volume"] = float(
            np.log1p(
                population_rows
                * np.asarray(measured["spectra"]["correction"], dtype=float)
            ).sum()
        )
    return candidates


def combine_correction_information_area(
    cells_by_rows: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Join the fixed row grid and take its log-row trapezoidal mean."""
    if set(cells_by_rows) != {64, 128, 256}:
        raise ValueError("correction information area needs rows 64, 128, and 256")
    indexed = {
        rows: {str(cell["run_id"]): cell for cell in cells}
        for rows, cells in cells_by_rows.items()
    }
    run_ids = set(indexed[256])
    if not run_ids or any(set(cells) != run_ids for cells in indexed.values()):
        raise ValueError("every row setting must contain the same run IDs")
    combined = []
    for run_id in sorted(run_ids):
        row = dict(indexed[256][run_id])
        row["correction_information_area"] = float(
            (
                indexed[64][run_id]["dataset_fisher_log_volume"]
                + 2 * indexed[128][run_id]["dataset_fisher_log_volume"]
                + indexed[256][run_id]["dataset_fisher_log_volume"]
            )
            / 4
        )
        combined.append(row)
    return combined


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_relative_information_report(
    result_root: Path,
    runs_root: Path,
    out_dir: Path,
    *,
    permutations: int = 50_000,
    skip_missing_r_star: bool = False,
) -> dict[str, Any]:
    """Write the joined panel, arm reduction, candidate ranking, and checks."""
    from fineqcomp.artifacts import write_json

    cells = collect_relative_information_cells(
        result_root, runs_root, skip_missing_r_star=skip_missing_r_star
    )
    analysis = analyze_relative_information_cells(cells, permutations=permutations)
    out_dir = Path(out_dir)
    _write_csv(out_dir / "cells.csv", cells)
    _write_csv(out_dir / "arms.csv", analysis["arms"])
    _write_csv(out_dir / "candidate_ranking.csv", analysis["ranking"])
    _write_csv(out_dir / "task_heldout_folds.csv", analysis["folds"])
    _write_csv(out_dir / "diversity_slopes.csv", analysis["diversity_slopes"])
    write_json(out_dir / "summary.json", analysis["summary"])
    return analysis["summary"]
