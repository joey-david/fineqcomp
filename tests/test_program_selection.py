from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from fineqcomp.config import TrainingSpec
from fineqcomp.program_selection import (
    FAMILIES,
    Cell,
    answers,
    arm_names,
    build_arm,
    build_pools,
    check_gates,
    epochs_for_budget,
    expand_cells,
    load_config,
    make_example,
    matched_control,
    split_pools,
)


CONFIG = load_config(Path("configs/program_selection.yaml"))


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_training_region_is_ambiguous_and_probes_are_not(name):
    """The whole claim rests on this: training cannot pick out a program."""
    family = FAMILIES[name]
    pools = build_pools(family)
    assert all(len(set(answers(family, row))) == 1 for row in pools.ambiguous[:2000])
    assert all(
        len(set(answers(family, row))) == len(family.programs)
        for row in pools.separating[:2000]
    )


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_split_slices_never_share_an_input(name):
    family = FAMILIES[name]
    split = split_pools(family, build_pools(family), 11, CONFIG)
    slices = [
        split.base,
        split.extra,
        split.ambiguous_eval,
        split.diagnostic,
        split.candidates,
        split.probe,
    ]
    keys = [{json.dumps(row, sort_keys=True) for row in part} for part in slices]
    total = sum(len(part) for part in keys)
    assert len(set().union(*keys)) == total


def test_prompt_is_the_same_for_every_candidate_answer():
    """A likelihood comparison across programs must vary only the answer."""
    family = FAMILIES["middle"]
    pools = build_pools(family)
    split = split_pools(family, pools, 11, CONFIG)
    row = split.probe[0]
    prompts = {
        make_example(family, row, program, split.base[:8]).prompt
        for program in family.programs
    }
    responses = {
        make_example(family, row, program, split.base[:8]).response
        for program in family.programs
    }
    assert len(prompts) == 1
    assert len(responses) == len(family.programs)


def test_arms_share_the_base_set_and_differ_by_one_addition():
    family = FAMILIES["count"]
    pools = build_pools(family)
    split = split_pools(family, pools, 11, CONFIG)
    lock = {
        "target_program": "count_a",
        "targeted": {"input": split.candidates[0]},
        "controls": {
            "matched_difficulty": {"input": split.candidates[1]},
            "matched_loss": {"input": split.candidates[2]},
            "random_distinguishing": {"input": split.candidates[3]},
        },
    }
    base = build_arm(family, split, lock, "base", None)
    assert len(base) == int(CONFIG["base_rows"])
    for arm in ("targeted", "matched_difficulty", "matched_loss"):
        rows = build_arm(family, split, lock, arm, None)
        assert len(rows) == len(base) + 1
        assert rows[:-1] == base
    influence = build_arm(family, split, lock, "matched_influence", split.candidates[4])
    assert influence[-1] != build_arm(family, split, lock, "targeted", None)[-1]
    for count in CONFIG["ambiguous_extra"]:
        rows = build_arm(family, split, lock, f"ambiguous_{count}", None)
        assert len(rows) == len(base) + int(count)


def test_every_arm_reaches_the_same_update_budget():
    spec = TrainingSpec(
        epochs=1,
        learning_rate=2e-4,
        effective_batch_size=16,
        micro_batch_size=8,
        max_length=512,
    )
    for rows in (128, 129, 228, 1128, 10128):
        epochs = epochs_for_budget(rows, spec, 256)
        per_epoch = -(-(-(-rows // 8)) // 2)
        assert epochs * per_epoch >= 256


def test_matched_control_takes_the_least_committed_of_the_nearest():
    table = [
        {"example_id": "a", "difficulty": 3, "mass_on_dominant": 0.9},
        {"example_id": "b", "difficulty": 3, "mass_on_dominant": 0.1},
        {"example_id": "c", "difficulty": 3, "mass_on_dominant": 0.5},
        {"example_id": "d", "difficulty": 9, "mass_on_dominant": 0.01},
    ]
    chosen = matched_control(table, 0, "difficulty", window=2)
    assert chosen["example_id"] == "b"
    assert chosen["matched_on"] == "difficulty"


def test_grid_covers_every_arm_once_per_cell():
    cells = expand_cells(CONFIG)
    assert len(cells) == (
        len(CONFIG["families"])
        * len(CONFIG["models"])
        * len(arm_names(CONFIG))
        * len(CONFIG["seeds"])
    )
    assert len(set(cell.slug for cell in cells)) == len(cells)
    assert Cell("chain", "mistral_7b_base", "targeted", 11) in cells


def _row(arm: str, **overrides):
    row = {
        "family": "chain",
        "model": "m",
        "arm": arm,
        "seed": 11,
        "dominant_program": "one",
        "target_program": "steps",
        "training_region_accuracy": 0.99,
        "agreement_target": 0.05,
        "agreement_dominant": 0.80,
        "argmax_program": "one",
    }
    row.update(overrides)
    return row


def test_gates_reject_a_switch_a_control_also_produces():
    config = {
        **CONFIG,
        "ambiguous_extra": [10],
        "seeds": [11],
    }
    rows = [
        _row("base"),
        _row("ambiguous_10"),
        _row("targeted", agreement_target=0.70, argmax_program="steps"),
        _row("matched_difficulty", agreement_target=0.65),
        _row("matched_loss"),
        _row("matched_influence"),
        _row("random_distinguishing"),
    ]
    report = check_gates(rows, config)
    cell = report["cells"][0]
    assert cell["g1"] and cell["g2"] and cell["g3"]
    assert not cell["g4"]
    assert not report["passed"]


def test_gates_accept_a_switch_only_the_targeted_example_produces():
    config = {**CONFIG, "ambiguous_extra": [10], "seeds": [11]}
    rows = [
        _row("base"),
        _row("ambiguous_10"),
        _row("targeted", agreement_target=0.75, argmax_program="steps"),
        _row("matched_difficulty"),
        _row("matched_loss"),
        _row("matched_influence"),
        _row("random_distinguishing"),
    ]
    for seed in (11, 22, 33):
        pass
    report = check_gates(
        [{**row, "seed": seed} for seed in (11, 22, 33) for row in rows], config
    )
    assert all(cell["g3"] and cell["g4"] for cell in report["cells"])
    assert report["replication"][0]["g5"]
    assert report["passed"]


def test_config_gates_are_the_registered_ones():
    raw = yaml.safe_load(Path("configs/program_selection.yaml").read_text())
    assert raw["gates"] == {
        "training_region_accuracy": 0.90,
        "ambiguous_stability": 0.10,
        "targeted_switch_gain": 0.30,
        "control_margin": 0.20,
    }
