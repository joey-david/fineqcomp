from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from fineqcomp.information_scaling import (
    MAX_SYMBOL_BITS,
    Cell,
    Condition,
    _condition,
    _rate_curve,
    build_dataset,
    check_gates,
    epochs_for,
    expand_grid,
    rate_codecs,
    shard,
    source_bits,
    summarize_rate_curve,
    symbol_bits,
)


LABELS = [chr(ord("A") + index) for index in range(16)]


def _config(tmp_path, mappings=64):
    codebooks = tmp_path / "codebooks"
    codebooks.mkdir(exist_ok=True)
    # 512 four-bit symbols: enough for 16 prototype tables of 16 items.
    codebooks.joinpath("seed11.hex").write_text(bytes(range(256)).hex())
    return {
        "labels": LABELS,
        "codebook_dir": str(codebooks),
        "mappings": mappings,
        "items_per_family": 16,
    }


def test_source_bits_count_independent_draws_and_nothing_else(tmp_path):
    raw = _config(tmp_path, mappings=512)

    constant = build_dataset(raw, Condition("constant", constant=True), seed=11)
    assert source_bits(constant, 512) == 4
    assert len(set(constant.label_indices)) == 1

    random = build_dataset(raw, Condition("random"), seed=11)
    assert source_bits(random, 64) == 256
    assert source_bits(random, 512) == 2048

    # A prototype table saturates once every table has been seen.
    for count in (1, 2, 4, 8, 16):
        bundle = build_dataset(raw, Condition(f"p{count}", count), seed=11)
        assert source_bits(bundle, 512) == 64 * count


def test_the_smallest_prefix_is_a_built_in_null_control(tmp_path):
    """At 64 mappings p4, p8, p16 and random are the same task.

    Four families cannot reuse more than four prototypes, so every condition
    with at least four tables draws 64 independent labels, exactly as random
    does. Their R* must agree, and that agreement is the noise floor any slope
    at larger prefixes has to clear.
    """
    raw = _config(tmp_path, mappings=512)
    measured = {
        name: source_bits(build_dataset(raw, condition, seed=11), 64)
        for name, condition in {
            "p4": Condition("p4", 4),
            "p8": Condition("p8", 8),
            "p16": Condition("p16", 16),
            "random": Condition("random"),
        }.items()
    }
    assert set(measured.values()) == {256}


def test_conditions_change_labels_not_prompts(tmp_path):
    raw = _config(tmp_path, mappings=512)
    random = build_dataset(raw, Condition("random"), seed=11)
    structured = build_dataset(raw, Condition("p4", 4), seed=11)
    cued = build_dataset(raw, Condition("p4_cue", 4, reveal_prototype=True), seed=11)

    assert [row.prompt for row in random.examples] == [
        row.prompt for row in structured.examples
    ]
    assert random.label_indices != structured.label_indices
    assert structured.label_indices == cued.label_indices

    # The cue changes the table line and nothing else.
    plain = structured.examples[17].prompt.splitlines()
    revealed = cued.examples[17].prompt.splitlines()
    differing = [
        index for index, (a, b) in enumerate(zip(plain, revealed)) if a != b
    ]
    assert len(differing) == 1
    assert plain[differing[0]] == "Table: T?"
    assert revealed[differing[0]].startswith("Table: T")


def test_the_key_last_layout_moves_the_key_next_to_the_answer(tmp_path):
    raw = _config(tmp_path, mappings=512)
    fields = build_dataset(raw, Condition("random"), seed=11)
    key_last = build_dataset({**raw, "prompt_layout": "key_last"}, Condition("random"), seed=11)

    assert fields.label_indices == key_last.label_indices
    assert fields.examples[17].prompt.endswith("Label:")
    assert key_last.examples[17].prompt.endswith("F0001 I01 =")
    with pytest.raises(ValueError):
        build_dataset({**raw, "prompt_layout": "sideways"}, Condition("random"), seed=11)


def test_every_prefix_spends_the_same_optimizer_updates():
    assert epochs_for(1024, 64, 16) == 256
    assert epochs_for(1024, 512, 16) == 32
    # An unreachable budget is an error, never a silent rounding: a prefix that
    # quietly took more updates would confound information with compute.
    with pytest.raises(ValueError):
        epochs_for(1000, 512, 16)
    with pytest.raises(ValueError):
        epochs_for(1024, 100, 16)


def test_the_label_code_is_bounded_however_wrong_the_model_is():
    assert symbol_bits(1.0) < 0.1
    assert symbol_bits(0.0) == pytest.approx(MAX_SYMBOL_BITS)
    # A confidently wrong model costs the cap, not the 40 bits its own
    # probabilities would have charged against a numerical floor.
    assert symbol_bits(1e-12) == pytest.approx(MAX_SYMBOL_BITS, rel=1e-6)
    assert MAX_SYMBOL_BITS == pytest.approx(math.log2(16 * 16))


def _curve_points():
    return [
        {
            "codec": "cheap",
            "description_bits": 100,
            "effective_bits_per_value": 0.25,
            "bits_saved_per_mapping": 0.4,
        },
        {
            "codec": "dear",
            "description_bits": 400,
            "effective_bits_per_value": 1.0,
            "bits_saved_per_mapping": 1.05,
        },
    ]


def test_r_star_is_undefined_below_the_learning_gate():
    summary = summarize_rate_curve(_curve_points(), raw_saved=0.02, target=0.9, gate=1.0)

    assert summary["learning_gate_passed"] is False
    assert summary["r_star_effective_bits_per_value"]["r_star"] is None
    assert "gate" in summary["r_star_effective_bits_per_value"]["reason"]
    assert summary["selected_codec"] is None


def test_overfit_is_reported_as_a_number_not_as_retention_above_one():
    summary = summarize_rate_curve(_curve_points(), raw_saved=1.0, target=0.9, gate=0.5)

    assert summary["reference"] == "best_decoded_frontier"
    assert summary["raw_bits_saved_per_mapping"] == 1.0
    assert summary["best_decoded_bits_saved_per_mapping"] == pytest.approx(1.05)
    # Coding beat the raw adapter by 5%. Against the raw gain that reads as
    # 105% retention, which is not a fraction of anything; against the frontier
    # every point is bounded by one and the overfit is its own number.
    assert max(p["retained_gain"] for p in summary["points"]) == pytest.approx(1.0)
    assert summary["raw_retained_gain"] == pytest.approx(1 / 1.05)
    assert summary["r_star_effective_bits_per_value"]["r_star"] is not None


def test_the_ladder_carries_an_empty_adapter_anchor(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "fineqcomp.information_scaling.encode_tensor_map",
        lambda tensors, path, bits, blend: {
            "file_bits": 1000 * (bits + 1),
            "effective_bits_per_value": bits + blend,
            "relative_rmse": 0.5,
        },
    )
    monkeypatch.setattr(
        "fineqcomp.information_scaling.decode_adapter_tensor_map",
        lambda path: ({}, {}),
    )
    monkeypatch.setattr(
        "fineqcomp.information_scaling.apply_adapter_tensors",
        lambda model, tensors: None,
    )
    monkeypatch.setattr(
        "fineqcomp.information_scaling.score",
        lambda *args, **kwargs: SimpleNamespace(
            summary=lambda: {"accuracy": 1.0, "code_bits_per_mapping": 0.5}
        ),
    )

    curve = _rate_curve(
        session=SimpleNamespace(model=object(), tokenizer=object()),
        model_spec=object(),
        labels=LABELS,
        examples=[object()],
        raw_tensors={},
        base_bits_per_mapping=4.0,
        base_accuracy=0.0625,
        raw_saved=3.5,
        codecs=({"key": "binary", "bits": 1, "blend": 0.0},),
        target=0.9,
        gate=1.0,
        out_dir=tmp_path,
        batch_size=1,
        keep_files=True,
    )

    # Without the anchor the crossing can sit below every measured rung and R*
    # reports the ladder floor, which is what censored the first pass.
    assert [point["codec"] for point in curve["points"]] == ["none", "binary"]
    assert curve["points"][0]["bits_saved_per_mapping"] == 0.0
    assert curve["points"][0]["description_bits"] == 0
    assert curve["r_star_effective_bits_per_value"]["bracketed"] is True


def test_the_grid_shards_into_disjoint_cells():
    info = {
        "seeds": [11, 22, 33],
        "conditions": [{"name": name} for name in ("a", "b", "c")],
        "grid": [
            {
                "study": "main",
                "conditions": ["a", "b", "c"],
                "adapters": ["r16"],
                "prefixes": [64, 512],
            },
            {
                "study": "rank",
                "conditions": ["a"],
                "adapters": ["r4", "r64"],
                "prefixes": [512],
            },
        ],
    }
    cells = expand_grid(info)

    assert len(cells) == 3 * 3 + 1 * 2 * 3
    assert cells[0].prefixes == (64, 512)
    rebuilt = [cell for index in range(4) for cell in shard(cells, index, 4)]
    assert sorted(rebuilt, key=lambda cell: cell.slug) == sorted(
        cells, key=lambda cell: cell.slug
    )
    with pytest.raises(ValueError):
        shard(cells, 4, 4)


def test_unknown_conditions_and_duplicate_codecs_are_rejected():
    with pytest.raises(KeyError):
        expand_grid(
            {
                "seeds": [11],
                "conditions": [{"name": "a"}],
                "grid": [
                    {
                        "study": "main",
                        "conditions": ["missing"],
                        "adapters": ["r16"],
                        "prefixes": [64],
                    }
                ],
            }
        )
    with pytest.raises(ValueError):
        rate_codecs({"adapter_codecs": [{"key": "x", "bits": 1}, {"key": "x", "bits": 2}]})
    with pytest.raises(ValueError):
        rate_codecs({"adapter_codecs": [{"key": "x", "bits": 0}]})


def test_a_code_that_beats_the_free_base_code_fails_its_gate():
    preq = [
        {
            "condition": "random",
            "seed": 11,
            "right": 768,
            "encoder": "adapter-trained-on-512",
            "bits_per_mapping": 3.9,
            "cumulative_code_bits": 4000.0,
            "cumulative_base_code_bits": 3000.0,
        }
    ]
    gates = check_gates([], preq)

    assert gates["G2_the_code_never_costs_more_than_the_base"]["passed"] is False
    assert gates["G3_the_code_separates_learnable_from_random"][
        "bits_per_unseen_mapping"
    ]["random"]["mean"] == pytest.approx(3.9)


def test_condition_lookup_reads_every_flag():
    raw = {
        "conditions": [
            {"name": "constant", "constant": True},
            {"name": "p4_cue", "prototype_count": 4, "reveal_prototype": True},
        ]
    }
    assert _condition(raw, "constant") == Condition("constant", None, True, False)
    assert _condition(raw, "p4_cue") == Condition("p4_cue", 4, False, True)
    with pytest.raises(KeyError):
        _condition(raw, "nope")


def test_cells_have_stable_identities():
    cell = Cell(study="main", condition="p4", adapter="all_linear_r16", seed=11,
                prefixes=(64, 512))
    assert cell.slug == "main/p4/all_linear_r16/seed11"
