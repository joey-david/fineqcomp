"""The corruption controls added for the generalisation study.

Each control has one invariant that makes it the control it claims to be, and
these check exactly those: the arithmetic corruption must move the answer along
with the working, the shuffle must preserve every line and the answer, the label
scramble must never leave a label correct, and the response mismatch must be a
derangement that preserves the response multiset exactly.
"""

from __future__ import annotations

import pytest

from fineqcomp.data import (
    Example,
    corrupt_arithmetic,
    mismatch_responses,
    scramble_labels,
    shuffle_rationale_steps,
)

MARKER = "The answer is:"


def chain(index: int) -> Example:
    body = (
        f"Start with {index + 2} apples.\n"
        f"She buys {index + 3} more, so she has {index + 2} + {index + 3} = {2 * index + 5}.\n"
        f"Half of them are green, so {2 * index + 5} / 5 = {(2 * index + 5) // 5}.\n"
    )
    return Example(
        example_id=f"row-{index}",
        prompt=f"Question: problem {index}\nAnswer:",
        response=f"{body}{MARKER} {(2 * index + 5) // 5}",
        metadata={"split": "train", "source_problem": f"p{index}"},
    )


def choice(index: int) -> Example:
    return Example(
        example_id=f"mc-{index}",
        prompt="Choose the best answer.\nA. a\nB. b\nC. c\nD. d\nAnswer:",
        response="ABCD"[index % 4],
        metadata={"split": "train", "choice_count": 4, "label_index": index % 4},
    )


ROWS = [chain(i) for i in range(40)]


def test_arithmetic_moves_the_answer_with_the_working():
    out = corrupt_arithmetic(ROWS, MARKER, seed=3, fraction=1.0)
    assert len(out) == len(ROWS)
    for before, after in zip(ROWS, out, strict=True):
        assert after.response.count(MARKER) == 1, "marker duplicated"
        assert after.prompt == before.prompt
        assert after.metadata["original_answer"] != after.metadata["corrupted_answer"]
        stated = after.response.rsplit(MARKER, 1)[1].strip()
        assert stated == after.metadata["corrupted_answer"]
    assert sum(a.response != b.response for a, b in zip(ROWS, out, strict=True)) == len(ROWS)


def test_arithmetic_is_reproducible_and_seed_dependent():
    first = corrupt_arithmetic(ROWS, MARKER, seed=3, fraction=0.5)
    again = corrupt_arithmetic(ROWS, MARKER, seed=3, fraction=0.5)
    other = corrupt_arithmetic(ROWS, MARKER, seed=4, fraction=0.5)
    assert [r.response for r in first] == [r.response for r in again]
    assert [r.response for r in first] != [r.response for r in other]


def test_shuffle_keeps_every_line_and_the_answer():
    out = shuffle_rationale_steps(ROWS, MARKER, seed=5)
    moved = 0
    for before, after in zip(ROWS, out, strict=True):
        assert after.response.count(MARKER) == 1
        assert after.response.rsplit(MARKER, 1)[1] == before.response.rsplit(MARKER, 1)[1]
        lines = lambda text: sorted(
            l for l in text.rsplit(MARKER, 1)[0].split("\n") if l.strip()
        )
        assert lines(after.response) == lines(before.response)
        moved += after.response != before.response
    assert moved > len(ROWS) // 2


def test_scrambled_labels_are_never_correct_and_stay_in_range():
    rows = [choice(i) for i in range(60)]
    out = scramble_labels(rows, seed=7)
    for before, after in zip(rows, out, strict=True):
        assert after.response != before.response
        index = ord(after.response) - ord("A")
        assert 0 <= index < int(after.metadata["choice_count"])
        assert after.metadata["label_index"] != after.metadata["original_label_index"]


def test_scramble_fraction_leaves_some_rows_alone():
    rows = [choice(i) for i in range(200)]
    out = scramble_labels(rows, seed=7, fraction=0.5)
    moved = sum(a.response != b.response for a, b in zip(rows, out, strict=True))
    assert 0 < moved < len(rows)


def test_mismatch_is_a_derangement_preserving_the_multiset():
    out = mismatch_responses(ROWS, seed=9, block_size=8)
    assert sorted(r.response for r in out) == sorted(r.response for r in ROWS)
    for before, after in zip(ROWS, out, strict=True):
        assert after.response != before.response
        assert after.prompt == before.prompt
        assert after.metadata["donor_id"] != after.example_id


def test_mismatch_needs_two_rows():
    with pytest.raises(ValueError):
        mismatch_responses(ROWS[:1], seed=9)
