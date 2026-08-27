from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from fineqcomp.grpo_control import (
    GRPORollout,
    aligned_grpo_credit,
    permute_grpo_credit,
    validate_grpo_credit_control,
)


def _rollouts(prompts: int = 8, group_size: int = 4) -> list[GRPORollout]:
    advantages = (-1.2, -0.4, 0.4, 1.2)
    rows = []
    for prompt in range(prompts):
        for completion in range(group_size):
            rows.append(
                GRPORollout(
                    rollout_id=f"p{prompt}-r{completion}",
                    prompt_id=f"p{prompt}",
                    prompt=f"problem {prompt}",
                    completion_index=completion,
                    completion=f"work for prompt {prompt}, sample {completion}",
                    completion_tokens=20 + prompt * 3 + completion,
                    old_logprobs=(-0.5,) * (20 + prompt * 3 + completion),
                    loss_mask=(True,) * (20 + prompt * 3 + completion),
                    reward=float(completion >= 2),
                    advantage=advantages[completion],
                    source_problem=f"source-{prompt}",
                )
            )
    return rows


def test_credit_control_changes_only_a_bijective_advantage_assignment():
    rows = _rollouts()
    aligned = aligned_grpo_credit(rows)
    first = permute_grpo_credit(rows, 11, block_size=4)
    second = permute_grpo_credit(rows, 11, block_size=4)

    assert first == second
    assert [row.completion for row in first] == [row.completion for row in rows]
    assert [row.reward for row in first] == [row.reward for row in rows]
    assert Counter(row.assigned_advantage for row in first) == Counter(
        row.advantage for row in rows
    )
    assert all(row.prompt_id != row.credit_donor_prompt_id for row in first)
    report = validate_grpo_credit_control(aligned, first)
    assert report["prompt_groups"] == 8
    assert report["changed_advantage_fraction"] == 1.0
    assert report["zero_reward_spread_fraction"] == 0.0


def test_credit_control_rejects_changed_rollout_text():
    rows = _rollouts()
    aligned = aligned_grpo_credit(rows)
    permuted = permute_grpo_credit(rows, 22, block_size=4)
    permuted[0] = replace(permuted[0], completion="changed")

    with pytest.raises(ValueError, match="changed rollout content"):
        validate_grpo_credit_control(aligned, permuted)


def test_credit_control_rejects_one_prompt_or_bad_group_shape():
    rows = _rollouts(prompts=1)
    with pytest.raises(ValueError, match="at least two prompt groups"):
        permute_grpo_credit(rows, 11)

    broken = _rollouts(prompts=2)[:-1]
    with pytest.raises(ValueError, match="same number of completions"):
        permute_grpo_credit(broken, 11)
