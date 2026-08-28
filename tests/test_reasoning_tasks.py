from __future__ import annotations

import pytest

pytest.importorskip("reasoning_gym")

from fineqcomp.reasoning_tasks import (  # noqa: E402
    DIFFICULTIES,
    TASK_SPECS,
    extract_final_answer,
    generate_reasoning_examples,
    score_reasoning_answer,
)


def test_frozen_panel_uses_valid_instance_rich_tasks():
    assert TASK_SPECS["logic"]["task"] == "knights_knaves"
    assert TASK_SPECS["planning"]["task"] == "jugs"
    assert all(
        TASK_SPECS[family]["task"]
        not in {"propositional_logic", "tower_of_hanoi"}
        for family in TASK_SPECS
    )


@pytest.mark.parametrize("family", list(TASK_SPECS))
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_generators_are_distinct_deterministic_and_score_their_oracles(
    family: str, difficulty: str
):
    first = generate_reasoning_examples(family, difficulty, 4, seed=20260828)
    second = generate_reasoning_examples(family, difficulty, 4, seed=20260828)

    assert first == second
    assert len({row.prompt for row in first}) == 4
    for row in first:
        oracle = str(row.metadata["verifier_entry"]["answer"])
        assert score_reasoning_answer(row, oracle) == 1.0
        assert score_reasoning_answer(row, "__definitely_wrong__") == 0.0
        assert score_reasoning_answer(row, "") == 0.0


def test_final_answer_extraction_requires_the_last_exact_marker():
    assert extract_final_answer("work only") is None
    assert extract_final_answer("FINAL ANSWER: 12") == "12"
    assert (
        extract_final_answer(
            "FINAL ANSWER: draft\nmore work\nfinal answer:\n```text\nA B\nC D\n```"
        )
        == "A B\nC D"
    )
