"""The divergence measure that has to line up with held-out bits saved."""

from __future__ import annotations

import math

import torch

from fineqcomp.layer_profile import _divergences


def _case(base: list[float], adapted: list[float], observed: int):
    return _divergences(
        torch.tensor([base]).log(),
        torch.tensor([adapted]).log(),
        torch.tensor([observed]),
    )


def test_identical_distributions_are_zero():
    result = _case([0.2, 0.3, 0.5], [0.2, 0.3, 0.5], 1)
    for key, value in result.items():
        assert abs(value) < 1e-6, key


def test_forward_kl_matches_the_definition():
    base = [0.5, 0.25, 0.25]
    adapted = [0.1, 0.1, 0.8]
    expected = sum(
        a * math.log2(a / b) for a, b in zip(adapted, base)
    )
    result = _case(base, adapted, 2)
    assert abs(result["forward_kl"] - expected) < 1e-5
    reverse = sum(b * math.log2(b / a) for a, b in zip(adapted, base))
    assert abs(result["reverse_kl"] - reverse) < 1e-5


def test_observed_bits_saved_is_the_surprise_difference():
    # The token that appeared is index 2, which the adapter made likelier.
    result = _case([0.5, 0.25, 0.25], [0.1, 0.1, 0.8], 2)
    assert abs(result["observed_bits_saved"] - math.log2(0.8 / 0.25)) < 1e-5


def test_divergence_can_be_large_where_the_observed_token_barely_moves():
    """The whole point: the two measures are not redundant.

    The adapter leaves the observed token's probability alone and rearranges
    everything else. Bits saved sees nothing; the divergence sees the change.
    """
    result = _case([0.5, 0.4, 0.1], [0.5, 0.1, 0.4], 0)
    assert abs(result["observed_bits_saved"]) < 1e-6
    assert result["forward_kl"] > 0.2
    assert result["total_variation"] > 0.25


def test_chunking_does_not_change_the_totals():
    generator = torch.Generator().manual_seed(0)
    base = torch.randn(37, 11, generator=generator)
    adapted = torch.randn(37, 11, generator=generator)
    observed = torch.randint(0, 11, (37,), generator=generator)
    whole = _divergences(base, adapted, observed, chunk=1024)
    split = _divergences(base, adapted, observed, chunk=8)
    for key in whole:
        assert abs(whole[key] - split[key]) < 1e-4, key
