from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import pytest

from fineqcomp.data import Example
from fineqcomp.relative_info import (
    CANDIDATES,
    _assignment_statistics,
    channel_bits_ceiling,
    codec_referenced_retention,
    COVERAGE_CANDIDATES,
    SPECTRAL_CANDIDATES,
    _spectral_rate_distortion,
    analyze_relative_information_cells,
    coverage_candidates,
    reduce_sketches,
    sample_examples,
    text_cross_row_redundancy,
)
from fineqcomp.relative_validation import (
    add_tokenizer_fertility,
    analyze_fixed_channel_prospective,
    candidate_gate,
    evaluate_rate_models,
    analyze_measure_diagnostics,
)


def _reduce(gradients: torch.Tensor, population_rows: int | None = None):
    rows = len(gradients)
    return reduce_sketches(
        surprisals=torch.linspace(1.0, 2.0, rows),
        margin_deficits=torch.linspace(0.2, 0.8, rows),
        predictive_entropies=torch.linspace(2.0, 3.0, rows),
        hidden=torch.eye(rows),
        residual=torch.eye(rows),
        gradients=gradients,
        population_rows=population_rows,
    )


def test_reducer_reports_exactly_the_ten_named_candidates():
    measured = _reduce(torch.eye(4))

    assert set(measured["candidates"]) == set(CANDIDATES)
    assert len(measured["candidates"]) == 10
    assert measured["diagnostics"]["rows"] == 4
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in measured["candidates"].values()
    )


def test_gradient_spectrum_separates_one_rule_from_independent_corrections():
    coherent = _reduce(torch.ones(8, 8))
    independent = _reduce(torch.eye(8))

    assert coherent["candidates"]["fisher_effective_rank"] == pytest.approx(1.0)
    assert independent["candidates"]["fisher_effective_rank"] == pytest.approx(8.0)
    assert (
        coherent["candidates"]["unit_gain_ntk_cost"]
        < independent["candidates"]["unit_gain_ntk_cost"]
    )
    assert set(independent["spectral_candidates"]) == set(SPECTRAL_CANDIDATES)
    assert (
        coherent["spectral_candidates"]["correction_rd90_bits"]
        < independent["spectral_candidates"]["correction_rd90_bits"]
    )


def test_shared_correction_energy_is_the_mean_gradient_identity():
    gradients = torch.tensor(
        [[2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 2.0]]
    )
    measured = _reduce(gradients)
    trace = measured["candidates"]["fisher_trace"]
    coherence = measured["diagnostics"]["mean_gradient_coherence"]
    entropy = measured["diagnostics"]["predictive_entropy_bits_per_token"]

    assert measured["derived_candidates"]["coherent_correction_energy"] == (
        pytest.approx(trace * coherence)
    )
    assert measured["derived_candidates"]["coherent_energy_over_entropy"] == (
        pytest.approx(trace * coherence / entropy)
    )


def test_spectral_rate_distortion_is_scale_free():
    spectrum = torch.tensor([4.0, 1.0, 0.25, 0.0])

    assert _spectral_rate_distortion(100 * spectrum) == pytest.approx(
        _spectral_rate_distortion(spectrum)
    )


def test_dataset_volume_uses_distinct_population_extent():
    small = _reduce(torch.eye(8), population_rows=8)
    large = _reduce(torch.eye(8), population_rows=800)

    assert (
        small["spectral_candidates"]["angular_correction_logdet"]
        == pytest.approx(
            large["spectral_candidates"]["angular_correction_logdet"]
        )
    )
    assert (
        small["spectral_candidates"]["dataset_correction_log_volume"]
        < large["spectral_candidates"]["dataset_correction_log_volume"]
    )
    assert (
        small["spectral_candidates"]["dataset_fisher_log_volume"]
        < large["spectral_candidates"]["dataset_fisher_log_volume"]
    )


def test_dataset_fisher_volume_retains_correction_scale():
    small = _reduce(torch.eye(8))
    large = _reduce(10 * torch.eye(8))

    assert (
        small["spectral_candidates"]["angular_correction_logdet"]
        == pytest.approx(
            large["spectral_candidates"]["angular_correction_logdet"]
        )
    )
    assert (
        small["spectral_candidates"]["dataset_fisher_log_volume"]
        < large["spectral_candidates"]["dataset_fisher_log_volume"]
    )


def test_uniform_row_sample_is_stable_and_not_the_prefix():
    rows = [
        Example(str(index), str(index), str(index), {}) for index in range(100)
    ]

    first = sample_examples(rows, 10, seed=7)
    second = sample_examples(rows, 10, seed=7)

    assert first == second
    assert first != rows[:10]


def test_reducer_rejects_mismatched_or_empty_sketches():
    with pytest.raises(ValueError, match="same non-zero row count"):
        reduce_sketches(
            torch.ones(2),
            torch.ones(2),
            torch.ones(2),
            torch.ones(1, 2),
            torch.ones(2, 2),
            torch.ones(2, 2),
        )


def test_analysis_uses_arm_means_and_merges_reused_datasets():
    cells = []
    datasets = {
        "kind_code": 1.0,
        "panel_math": 2.0,
        "sql_div_100": 3.0,
        "text_to_sql": 3.0,
    }
    for model_index, model in enumerate(("mistral", "qwen")):
        for dataset, amount in datasets.items():
            candidates = {key: 1.0 for key in CANDIDATES}
            candidates["fisher_logdet"] = amount
            for seed in (11, 22, 33):
                cells.append(
                    {
                        "run_id": f"{model}-{dataset}-{seed}",
                        "model_key": model,
                        "dataset_key": dataset,
                        "seed": seed,
                        "r_star_bits_per_value": 0.2 * amount + 0.1 * model_index,
                        "r_star_bracketed": True,
                        "reference_bits_saved_per_token": 1.0,
                        **candidates,
                    }
                )

    analysis = analyze_relative_information_cells(cells, permutations=100)

    assert analysis["summary"]["cells"] == 24
    assert analysis["summary"]["distinct_arms"] == 6
    assert analysis["summary"]["strongest_candidate"] == "fisher_logdet"
    assert analysis["summary"]["strongest_task_heldout_rmse"] < 1e-6


def _correction_cloud(distinct: int, rows: int, width: int, seed: int):
    """Rows drawn from a fixed number of distinct correction directions."""
    generator = torch.Generator().manual_seed(seed)
    basis = torch.randn(distinct, width, generator=generator)
    index = torch.arange(rows) % distinct
    noise = 0.05 * torch.randn(rows, width, generator=generator)
    return basis[index] + noise


def test_coverage_reports_exactly_its_named_candidates():
    measured = coverage_candidates(
        torch.randn(64, 16), torch.randn(64, 8), torch.randn(64, 32)
    )

    # Both text statistics are measured from the raw rows, not from these
    # tensors, so they are attached by the caller rather than returned here.
    assert set(measured) | {
        "text_cross_row_redundancy",
        "tokenizer_fertility",
    } == set(COVERAGE_CANDIDATES)


def test_coverage_counts_distinct_corrections_not_their_size():
    hidden, residual = torch.randn(200, 16), torch.randn(200, 8)
    few = coverage_candidates(
        hidden, residual, _correction_cloud(10, 200, 32, seed=7)
    )
    many = coverage_candidates(
        hidden, residual, _correction_cloud(160, 200, 32, seed=7)
    )
    loud = coverage_candidates(
        hidden, residual, 50 * _correction_cloud(10, 200, 32, seed=7)
    )

    assert many["correction_coverage_fraction"] > (
        few["correction_coverage_fraction"]
    )
    assert few["nearest_neighbour_cosine"] > many["nearest_neighbour_cosine"]
    for key, value in few.items():
        assert value == pytest.approx(loud[key], rel=1e-6, abs=1e-9)


def test_prequential_saving_separates_a_rule_from_noise():
    generator = torch.Generator().manual_seed(11)
    hidden = torch.randn(256, 16, generator=generator)
    rule = hidden @ torch.randn(16, 8, generator=generator)
    noise = torch.randn(256, 8, generator=generator)
    gradients = torch.randn(256, 32, generator=generator)

    learnable = coverage_candidates(hidden, rule, gradients)
    unlearnable = coverage_candidates(hidden, noise, gradients)

    assert learnable["prequential_correction_bits_per_row"] > 10 * (
        unlearnable["prequential_correction_bits_per_row"]
    )
    assert learnable["correction_predictable_fraction"] > 0.9
    assert unlearnable["correction_predictable_fraction"] < 0.2
    assert learnable["prequential_decay_exponent"] > (
        unlearnable["prequential_decay_exponent"]
    )


def test_text_redundancy_sees_repeated_responses():
    varied = [
        Example(str(index), "p", f"answer number {index} is unlike the rest", {})
        for index in range(64)
    ]
    repeated = [
        Example(str(index), "p", "the same answer", {}) for index in range(64)
    ]

    assert text_cross_row_redundancy(repeated) > 0.9
    assert text_cross_row_redundancy(repeated) > text_cross_row_redundancy(varied)


def test_intrinsic_dimension_follows_the_correction_subspace():
    # Two-NN reads the local dimension of the correction cloud, so a corpus
    # confined to a few correction directions scores low however many rows it
    # spends inside them.
    generator = torch.Generator().manual_seed(3)
    hidden, residual = torch.randn(200, 16), torch.randn(200, 8)
    full = torch.randn(200, 32, generator=generator)
    flat = torch.randn(200, 3, generator=generator) @ torch.randn(
        3, 32, generator=generator
    )

    assert coverage_candidates(hidden, residual, full)[
        "correction_intrinsic_dimension"
    ] > 3 * coverage_candidates(hidden, residual, flat)[
        "correction_intrinsic_dimension"
    ]


def _tiny_causal_session():
    from types import SimpleNamespace

    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(5)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
    )

    class Tokenizer:
        eos_token_id = 1
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            ids = [3 + (ord(char) % 40) for char in text]
            return ([2] if add_special_tokens else []) + ids

    return SimpleNamespace(model=model, tokenizer=Tokenizer())


def test_channel_bits_ignores_a_rescaling_that_moves_the_report_log_volume():
    from fineqcomp.relative_info import _channel_bits, _logdet

    spectrum = torch.tensor([9.0, 3.0, 1.0, 0.2, 0.05], dtype=torch.float64)
    assert _channel_bits(spectrum * 1000.0) == pytest.approx(
        _channel_bits(spectrum), rel=1e-12
    )
    assert _logdet(spectrum * 1000.0) > 3 * _logdet(spectrum)
    # The rate is concave, so at this noise floor a corpus whose corrections
    # spread evenly over directions costs more than one that piles into a few.
    flat = torch.full((5,), 2.6, dtype=torch.float64)
    assert _channel_bits(flat) > _channel_bits(spectrum)


def test_channel_bits_are_recovered_from_a_stored_spectrum():
    from fineqcomp.relative_info import _channel_bits, spectral_candidates_from_record

    spectrum = [4.0, 2.0, 0.5, 0.1]
    recovered = spectral_candidates_from_record(
        {"spectra": {"correction": spectrum, "angular_correction": spectrum}}
    )
    expected = _channel_bits(torch.tensor(spectrum, dtype=torch.float64))
    assert recovered["correction_channel_bits"] == pytest.approx(expected)
    assert recovered["angular_correction_channel_bits"] == pytest.approx(expected)

    recovered = spectral_candidates_from_record(
        {"spectral_candidates": {"trace_normalized_correction_logdet": 4.0}}
    )
    assert recovered["correction_information_bits"] == pytest.approx(
        4.0 / (2.0 * math.log(2.0))
    )


def test_locked_channel_test_uses_only_the_fixed_candidate_and_prefit():
    models = (
        "llama31_8b_base",
        "mistral_7b_base",
        "qwen25_7b_base",
        "qwen3_8b_base",
    )

    def cell(model, dataset, seed, channel, model_index):
        return {
            "run_id": f"{model}-{dataset}-{seed}",
            "study": dataset,
            "model_key": model,
            "dataset_key": dataset,
            "seed": seed,
            "adapter_rank": 16,
            "r_star_bits_per_value": 0.2 + 0.1 * model_index + 0.05 * channel,
            "r_star_bracketed": True,
            "reference_bits_saved_per_token": 0.5,
            "correction_channel_bits": channel,
            "base_codelength_bits_per_token": 1.0,
            "train_response_tokens": 1000.0,
            "measured_rows": 256,
            "row_sampling": "uniform",
            "row_sample_seed": 271828,
        }

    development = []
    for model_index, model in enumerate(models):
        for arm_index in range(4):
            for seed in (11, 22, 33):
                development.append(
                    cell(model, f"dev_{arm_index}", seed, arm_index + 1, model_index)
                )

    datasets = {
        "llama31_8b_base": (
            "gsm8k_self",
            "text_to_sql",
            "xbrl_tags",
            "xbrl_div_10",
            "xbrl_div_30",
        ),
        "mistral_7b_base": ("gsm8k_self",),
        "qwen25_7b_base": ("gsm8k_self",),
        "qwen3_8b_base": ("gsm8k_self", "text_to_sql", "xbrl_tags"),
    }
    prospective = []
    for model_index, model in enumerate(models):
        for arm_index, dataset in enumerate(datasets[model], start=1):
            for seed in (11, 22, 33):
                prospective.append(
                    cell(model, dataset, seed, arm_index, model_index)
                )

    lock = {
        "candidate": "correction_channel_bits",
        "measurement_contract": {
            "cells": 30,
            "arms": 10,
            "seeds_per_arm": 3,
            "rows": 256,
            "row_sampling": "uniform",
            "row_sample_seed": 271828,
        },
        "primary_test": {
            "gates": {"rho_at_least": 0.6, "one_sided_p_below": 0.05}
        },
        "prediction_gates": {
            "prefit_rmse_at_most_fraction_of_receiver_only": 0.8,
            "prefit_rmse_below_base_codelength": True,
            "prefit_rmse_below_train_response_tokens": True,
        },
        "validity_gates": {"median_within_arm_measure_cv_below": 0.05},
    }
    result = analyze_fixed_channel_prospective(development, prospective, lock)

    assert result["status"] == "passed"
    assert result["prospective_cells"] == 30
    assert result["prospective_arms"] == 10
    assert result["within_model_rank_rho"] == pytest.approx(1.0)
    assert result["exact_permutations"] == 720
    assert result["prefit_rmse"]["candidate"] == pytest.approx(0.0, abs=1e-12)
    assert all(result["gates"].values())

    prospective[0]["measured_rows"] = 255
    invalid = analyze_fixed_channel_prospective(development, prospective, lock)
    assert invalid["status"] == "failed"
    assert not invalid["gates"]["all_cells_have_locked_measurement_rows"]


def test_layer_energy_is_a_depth_profile_that_leaves_the_model_alone():
    from fineqcomp.config import ModelSpec
    from fineqcomp.relative_info import LAYER_CANDIDATES, measure_layer_energy

    session = _tiny_causal_session()
    spec = ModelSpec("tiny", "tiny", "local", "bf16")
    rows = [
        Example(str(index), f"prompt {index}", f" answer {index}", {})
        for index in range(6)
    ]
    before = {
        name: parameter.detach().clone()
        for name, parameter in session.model.named_parameters()
    }

    measured = measure_layer_energy(session, rows, spec, 64, rank=4, seed=7)

    assert set(measured["layer_candidates"]) == set(LAYER_CANDIDATES)
    assert measured["layer_energy"]["rows"] == 6
    assert measured["layer_energy"]["layers"] == [0, 1, 2, 3]
    assert sum(measured["layer_energy"]["layer_share"]) == pytest.approx(1.0)
    assert 0.0 <= measured["layer_candidates"]["lora_layer_energy_entropy"] <= 1.0
    assert 0.0 <= measured["layer_candidates"]["lora_layer_energy_centroid"] <= 1.0
    spectrum = measured["layer_energy"]["correction_spectrum"]
    assert len(spectrum) == 6
    assert measured["layer_candidates"]["lora_correction_channel_bits"] > 0.0
    assert 1.0 <= measured["layer_candidates"]["lora_correction_effective_rank"] <= 6.0
    after = dict(session.model.named_parameters())
    assert set(after) == set(before)
    for name, value in before.items():
        assert torch.equal(after[name], value)


def test_task_families_fold_by_source_corpus_not_by_dataset_key():
    from fineqcomp.relative_info import _task_family

    metamath = ("arm_a", "lever_div_400", "behav_shouted_broad", "cot_math",
                "panel_math", "metamath", "budget_math")
    assert {_task_family(key) for key in metamath} == {"metamath"}
    assert _task_family("kind_code") == _task_family("code_long") == "magicoder"
    assert _task_family("sql_div_100") == _task_family("text_to_sql") == "sql"
    assert _task_family("xbrl_div_30") == "xbrl"
    assert _task_family("kind_summary") == "xsum"
    assert _task_family("something_new") == "something_new"


def test_token_level_moments_separate_within_and_between_example_spread():
    from fineqcomp.relative_info import _base_candidates

    width, tokens = 4, 8
    means = torch.eye(width, dtype=torch.float64)[:2].repeat_interleave(4, 0)
    jitter = torch.zeros(tokens, width, dtype=torch.float64)
    jitter[:, 2] = torch.tensor([1.0, -1.0] * 4, dtype=torch.float64)
    flat = _base_candidates(
        means.T @ means, torch.zeros(width, width, dtype=torch.float64),
        tokens, torch.tensor([1.0, 4.0]), torch.tensor([2.0, 8.0]),
    )
    varied = means + jitter
    inside = varied - means
    lively = _base_candidates(
        varied.T @ varied, inside.T @ inside, tokens,
        torch.tensor([1.0, 4.0]), torch.tensor([2.0, 8.0]),
    )
    assert flat["within_between_energy_ratio"] == pytest.approx(0.0)
    assert lively["within_between_energy_ratio"] == pytest.approx(1.0)
    assert lively["within_example_channel_bits"] > flat["within_example_channel_bits"]
    assert flat["base_head_hidden_alignment"] == pytest.approx(0.25)
    assert flat["base_head_error_alignment"] == pytest.approx(0.25)


def test_head_subspace_reads_the_frozen_unembedding():
    from fineqcomp.relative_info import _head_subspace, _inside_fraction

    session = _tiny_causal_session()
    basis = _head_subspace(session.model, rank=2)
    assert basis is not None
    left, right = basis
    weight = session.model.get_output_embeddings().weight
    assert left.shape == (weight.shape[0], 2)
    assert right.shape == (weight.shape[1], 2)
    # An orthonormal basis keeps all of its own energy.
    inside, whole = _inside_fraction(right.T.double(), right.double()).tolist()
    assert inside == pytest.approx(whole, rel=1e-6)


def test_diagnostics_separate_a_corpus_measure_from_a_receiver_measure():
    """A corpus-only measure must show no receiver share and no pair signal.

    Three receivers see the same four corpora. `corpus_only` is a property of
    the text, so it repeats across receivers and can never order them.
    `receiver_aware` adds a per-receiver offset that R* follows.
    """
    corpus = {"d1": 1.0, "d2": 2.0, "d3": 3.0, "d4": 4.0}
    offset = {"m1": 0.0, "m2": 0.1, "m3": 0.3}
    r_star = {
        (model, dataset): 0.1 * value + shift
        for model, shift in offset.items()
        for dataset, value in corpus.items()
    }
    cells = [
        {
            "model_key": model,
            "dataset_key": dataset,
            "r_star_bits_per_value": value,
            "reference_bits_saved_per_token": 1.0,
            "corpus_only": corpus[dataset],
            "receiver_aware": corpus[dataset] + 10.0 * offset[model],
        }
        for (model, dataset), value in r_star.items()
    ]
    result = analyze_measure_diagnostics(
        cells,
        ("corpus_only", "receiver_aware"),
        controls=(),
        permutations=200,
    )
    rows = {row["candidate"]: row for row in result["diagnostics"]}
    assert result["summary"]["cross_receiver_pairs"] == 12
    assert rows["corpus_only"]["receiver_variance_share"] == pytest.approx(0.0, abs=1e-9)
    assert rows["corpus_only"]["cross_receiver_spearman"] == pytest.approx(0.0)
    assert rows["receiver_aware"]["receiver_variance_share"] > 0.0
    assert rows["receiver_aware"]["cross_receiver_spearman"] > 0.9


def test_channel_bits_sit_just_under_a_white_noise_ceiling():
    """The measure's maximum is fixed by the row count, not by the corpus.

    A white spectrum saturates every mode at `0.5 * log2(1 + gamma)`, so the
    ceiling is that times the number of modes. Any real spectrum falls below
    it, and how far below is the whole of the measure's range.
    """
    from fineqcomp.relative_info import _channel_bits, _channel_deficit_bits

    white = torch.ones(256, dtype=torch.float64)
    concentrated = torch.zeros(256, dtype=torch.float64)
    concentrated[0] = 256.0

    ceiling = channel_bits_ceiling(256)
    assert ceiling == pytest.approx(0.5 * 256 * math.log2(1.01))
    assert _channel_bits(white) == pytest.approx(ceiling)
    assert _channel_deficit_bits(white) == pytest.approx(0.0, abs=1e-9)
    # All the energy in one of 256 modes still scores half the ceiling, which
    # is why real corpora sit at 90-98% of it and the measure has so little
    # spread.
    assert _channel_bits(concentrated) == pytest.approx(0.5 * ceiling, rel=0.02)
    assert _channel_deficit_bits(concentrated) == pytest.approx(
        0.5 * ceiling, rel=0.02
    )


def test_codec_referenced_retention_falls_as_the_codec_gets_coarser():
    spectrum = torch.tensor(
        [100.0, 10.0, 1.0, 0.1] * 8, dtype=torch.float64
    )
    fine = codec_referenced_retention(spectrum, 0.1)
    coarse = codec_referenced_retention(spectrum, 0.9)
    assert 0.0 < coarse < fine <= 1.0
    assert codec_referenced_retention(spectrum, 0.0) == pytest.approx(1.0)


def test_fertility_splits_a_token_count_into_corpus_size_and_receiver_fit():
    """One corpus on two receivers: size is shared, fertility is not."""
    arms = [
        {"model_key": "m1", "dataset_key": "d1", "train_response_tokens": 1000.0},
        {"model_key": "m2", "dataset_key": "d1", "train_response_tokens": 1400.0},
        {"model_key": "m1", "dataset_key": "d2", "train_response_tokens": 200.0},
    ]
    add_tokenizer_fertility(arms)

    assert arms[0]["corpus_tokens"] == arms[1]["corpus_tokens"] == 1200.0
    assert arms[0]["tokenizer_fertility_relative"] < 1.0
    assert arms[1]["tokenizer_fertility_relative"] > 1.0
    assert arms[2]["tokenizer_fertility_relative"] == pytest.approx(1.0)
    assert arms[2]["receivers_sharing_corpus"] == 1


def test_candidate_gate_demotes_a_candidate_that_only_repeats_a_control():
    """A copy of a control adds nothing; an independent signal survives."""
    control = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    independent = [3.0, 1.0, 7.0, 0.0, 5.0, 2.0, 6.0, 4.0]
    arms = [
        {
            "model_key": model,
            "dataset_key": f"d{step}",
            "r_star_bits_per_value": 0.1 * control[step]
            + 0.1 * independent[step]
            + offset,
            "text_cross_row_redundancy": control[step],
            "train_response_tokens": control[step] ** 2 + 1.0,
            "copy_of_control": control[step] * 3.0 + 1.0,
            "independent": independent[step],
        }
        for model, offset in (("m1", 0.0), ("m2", 1.0))
        for step in range(len(control))
    ]
    rows = {
        row["candidate"]: row
        for row in candidate_gate(arms, ("copy_of_control", "independent"))
    }

    assert abs(rows["copy_of_control"]["partial_within_model_spearman"]) < 0.05
    assert rows["independent"]["partial_within_model_spearman"] > 0.5
    assert rows["independent"]["adds_over_controls"]
    assert not rows["copy_of_control"]["adds_over_controls"]


def test_held_out_receiver_scoring_drops_the_held_out_intercept():
    """A useless predictor must not beat the receiver mean it is fitted with."""
    arms = [
        {
            "model_key": model,
            "dataset_key": f"d{step}",
            "r_star_bits_per_value": 0.1 * step + offset,
            "signal": float(step),
            "noise": float((step * 3) % 5) * 1e-6,
        }
        for model, offset in (("m1", 0.0), ("m2", 0.3), ("m3", 0.6))
        for step in range(5)
    ]
    rows = {
        row["model"]: row
        for row in evaluate_rate_models(
            arms, {"signal": ("signal",), "noise": ("noise",), "mean": ()}
        )
    }

    assert rows["signal"]["receiver_held_out_rmse"] < rows["mean"]["receiver_held_out_rmse"]
    assert rows["signal"]["receiver_held_out_spearman"] == pytest.approx(1.0)
    assert rows["noise"]["receiver_held_out_rmse"] >= rows["signal"]["receiver_held_out_rmse"]


class _CopyingSession:
    """A model that scores a token highly once it has seen it in the prompt.

    That is the smallest behaviour a frozen model needs for the trace-retrieval
    probe to read a low load: it can tell which trace belongs to which problem
    because the trace repeats what the problem said.
    """

    class _Tokenizer:
        eos_token_id = None
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [3 + (ord(char) % 40) for char in text]

    class _Model(torch.nn.Module):
        def __init__(self, vocabulary: int, strength: float) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(vocabulary, 2)
            self.vocabulary = vocabulary
            self.strength = strength

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, input_ids, attention_mask=None, use_cache=None):
            rows, length = input_ids.shape
            logits = torch.zeros(rows, length, self.vocabulary)
            for row in range(rows):
                seen = torch.zeros(self.vocabulary, dtype=torch.bool)
                for position in range(length):
                    logits[row, position] = torch.where(
                        seen, self.strength, 0.0
                    )
                    seen[int(input_ids[row, position])] = True
            return SimpleNamespace(logits=logits)

    def __init__(self, strength: float) -> None:
        self.tokenizer = self._Tokenizer()
        self.model = self._Model(64, strength)


def test_trace_retrieval_reads_the_ceiling_when_traces_are_interchangeable():
    from fineqcomp.relative_info import trace_retrieval_load

    rows = [
        Example(f"r{index}", f"problem {index}", "same working ANSWER: 1", {})
        for index in range(8)
    ]

    measured = trace_retrieval_load(
        _CopyingSession(4.0),
        rows,
        SimpleNamespace(chat=False, disable_thinking=False),
        512,
        4,
        marker="ANSWER:",
        candidates=4,
    )

    assert measured["trace_retrieval_candidates"] == 4
    assert measured["trace_retrieval_probes"] == 32
    # Identical bodies leave the posterior uniform, so the bounded retrieval
    # load equals the uniform four-way log loss.
    assert measured["trace_retrieval_load_bits"] == pytest.approx(2.0)
    assert measured["trace_retrieval_log_loss_bits"] == pytest.approx(2.0)
    assert measured["trace_retrieval_uniform_bits"] == pytest.approx(2.0)


def test_trace_retrieval_falls_when_the_model_can_place_the_trace():
    from fineqcomp.relative_info import trace_retrieval_load

    # Letters whose token ids appear nowhere in the fixed wording, so the only
    # thing linking a trace to its problem is the repeated letter itself.
    tokens = "acdfhjqs"
    rows = [
        Example(
            f"r{index}",
            f"problem {letter * 12}",
            f" working {letter * 12} ANSWER: 1",
            {},
        )
        for index, letter in enumerate(tokens)
    ]
    spec = SimpleNamespace(chat=False, disable_thinking=False)

    informed = trace_retrieval_load(
        _CopyingSession(4.0), rows, spec, 512, 4, marker="ANSWER:", candidates=4
    )
    ignorant = trace_retrieval_load(
        _CopyingSession(0.0), rows, spec, 512, 4, marker="ANSWER:", candidates=4
    )

    assert informed["trace_retrieval_load_bits"] < 0.2
    assert informed["trace_retrieval_error"] == 0.0
    assert ignorant["trace_retrieval_load_bits"] == pytest.approx(2.0)
    # Fano describes what this base decoder already recovers. It is not an
    # adapter-size lower bound.
    assert informed["trace_retrieval_fano_known_lower_bits"] == pytest.approx(2.0)
    assert informed["trace_retrieval_fano_missing_upper_bits"] == pytest.approx(0.0)


def test_trace_retrieval_marginal_correction_removes_donor_only_scores():
    semantic = 6.0 * torch.eye(4, dtype=torch.float64)
    donor_nuisance = torch.tensor([[-100.0, 80.0, 20.0, -30.0]])
    raw = semantic + donor_nuisance
    corrected = raw - (
        torch.logsumexp(raw, dim=0, keepdim=True) - math.log(len(raw))
    )

    measured = _assignment_statistics([corrected])

    assert measured["accuracy"] == 1.0
    assert measured["load_bits"] < 0.02


def test_trace_retrieval_marginal_correction_rejects_pure_trace_priors():
    donor_nuisance = torch.tensor([[-100.0, 80.0, 20.0, -30.0]]).repeat(4, 1)
    corrected = donor_nuisance - (
        torch.logsumexp(donor_nuisance, dim=0, keepdim=True)
        - math.log(len(donor_nuisance))
    )

    measured = _assignment_statistics([corrected])

    assert measured["load_bits"] == pytest.approx(2.0)
    assert measured["log_loss_bits"] == pytest.approx(2.0)
