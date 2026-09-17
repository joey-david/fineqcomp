"""The generated pointer task has to behave like the retrieved ones."""

from __future__ import annotations

import pytest

from fineqcomp.data import permute_rationales
from fineqcomp.pointer_chasing import (
    ANSWER_MARKER,
    build_pointer_splits,
    build_split,
)


def _map_of(prompt: str) -> dict[int, int]:
    body = prompt.split("Map:\n", 1)[1].split("\n\nStart:", 1)[0]
    pairs = (line.split(" -> ") for line in body.splitlines())
    return {int(source): int(target) for source, target in pairs}


def _query(prompt: str) -> tuple[int, int]:
    start = int(prompt.split("Start: ", 1)[1].split("\n", 1)[0])
    hops = int(prompt.split("Steps: ", 1)[1].split("\n", 1)[0])
    return start, hops


def test_the_stated_answer_is_the_walk_the_map_defines():
    for row in build_split("train", 40, nodes=12, hops=4, seed=11):
        mapping = _map_of(row.prompt)
        start, hops = _query(row.prompt)
        current = start
        for _ in range(hops):
            current = mapping[current]
        stated = int(row.response.rsplit(ANSWER_MARKER, 1)[1].strip())
        assert stated == current
        assert row.metadata["answer"] == current


def test_the_written_steps_match_the_map_and_the_hop_count():
    for row in build_split("test", 20, nodes=10, hops=3, seed=0):
        mapping = _map_of(row.prompt)
        body = row.response.rsplit(ANSWER_MARKER, 1)[0]
        steps = [line.strip() for line in body.splitlines() if line.strip().startswith("Step ")]
        assert len(steps) == 3
        start, _ = _query(row.prompt)
        current = start
        for line in steps:
            source, target = line.split(": ", 1)[1].split(" -> ")
            assert int(source) == current
            assert mapping[current] == int(target)
            current = int(target)


def test_no_entry_maps_a_label_to_itself():
    for row in build_split("train", 30, nodes=8, hops=2, seed=3):
        assert all(source != target for source, target in _map_of(row.prompt).items())


def test_the_generator_is_reproducible_and_seed_dependent():
    first = build_split("train", 10, nodes=10, hops=3, seed=11)
    again = build_split("train", 10, nodes=10, hops=3, seed=11)
    other = build_split("train", 10, nodes=10, hops=3, seed=22)
    assert [row.prompt for row in first] == [row.prompt for row in again]
    assert [row.prompt for row in first] != [row.prompt for row in other]


def test_the_splits_do_not_share_items():
    splits = build_pointer_splits(
        {"nodes": 10, "hops": 3, "train_rows": 60, "validation_rows": 20, "test_rows": 20},
        seed=11,
    )
    prompts = {name: {row.prompt for row in rows} for name, rows in splits.items()}
    assert not prompts["train"] & prompts["test"]
    assert not prompts["train"] & prompts["calibration"]
    assert not prompts["calibration"] & prompts["test"]


def test_the_test_split_ignores_the_campaign_seed():
    """Every seed must be scored on the same items, as elsewhere in the project."""
    one = build_pointer_splits(
        {"nodes": 10, "hops": 3, "train_rows": 10, "validation_rows": 5, "test_rows": 15},
        seed=11,
    )
    two = build_pointer_splits(
        {"nodes": 10, "hops": 3, "train_rows": 10, "validation_rows": 5, "test_rows": 15},
        seed=22,
    )
    assert [row.prompt for row in one["test"]] == [row.prompt for row in two["test"]]
    assert [row.prompt for row in one["train"]] != [row.prompt for row in two["train"]]


def test_the_recorded_permutation_corrupts_it_unchanged():
    """The whole point of the numeric labels and the shared answer marker."""
    rows = build_split("train", 24, nodes=10, hops=3, seed=11)
    moved = permute_rationales(rows, ANSWER_MARKER, seed=5)
    assert len(moved) == len(rows)
    for before, after in zip(rows, moved, strict=True):
        assert before.prompt == after.prompt
        # The answer line survives; the working does not belong to it any more.
        assert after.response.rsplit(ANSWER_MARKER, 1)[1] == (
            before.response.rsplit(ANSWER_MARKER, 1)[1]
        )
    bodies_before = sorted(r.response.rsplit(ANSWER_MARKER, 1)[0] for r in rows)
    bodies_after = sorted(r.response.rsplit(ANSWER_MARKER, 1)[0] for r in moved)
    assert bodies_before == bodies_after
    assert any(
        a.response != b.response for a, b in zip(rows, moved, strict=True)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nodes": 1, "hops": 2},
        {"nodes": 5, "hops": 0},
        {"nodes": 200, "hops": 2},
    ],
)
def test_impossible_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        build_split("train", 4, seed=1, **kwargs)
