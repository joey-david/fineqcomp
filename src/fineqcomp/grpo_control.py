"""Matched credit controls for fixed-rollout GRPO experiments."""

from __future__ import annotations

import math
import random
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from statistics import pstdev


@dataclass(frozen=True)
class GRPORollout:
    """One rollout after reward scoring and before a GRPO update."""

    rollout_id: str
    prompt_id: str
    prompt: str
    completion_index: int
    completion: str
    completion_tokens: int
    old_logprobs: tuple[float, ...]
    loss_mask: tuple[bool, ...]
    reward: float
    advantage: float
    source_problem: str = ""
    assigned_advantage: float | None = None
    credit_donor_prompt_id: str | None = None
    credit_donor_rollout_id: str | None = None


def aligned_grpo_credit(rows: list[GRPORollout]) -> list[GRPORollout]:
    """Attach every measured advantage to the rollout that earned it."""
    _group_rows(rows)
    return [
        replace(
            row,
            assigned_advantage=row.advantage,
            credit_donor_prompt_id=row.prompt_id,
            credit_donor_rollout_id=row.rollout_id,
        )
        for row in rows
    ]


def permute_grpo_credit(
    rows: list[GRPORollout], seed: int, *, block_size: int = 16
) -> list[GRPORollout]:
    """Move whole GRPO credit vectors to matched, unrelated prompt groups.

    Prompt text, completions, rewards, old policy scores, and token counts stay
    fixed outside this record. Each target prompt receives the full advantage
    multiset from one other prompt. Small blocks match groups on completion
    length and reward spread. A cyclic slot match then changes as many row-level
    advantages as it can while keeping donor and target lengths close.
    """
    if block_size < 2:
        raise ValueError("GRPO credit blocks need at least two prompt groups")
    groups = _group_rows(rows)
    if len(groups) < 2:
        raise ValueError("GRPO credit permutation needs at least two prompt groups")
    group_size = len(next(iter(groups.values())))
    if any(len(group) != group_size for group in groups.values()):
        raise ValueError("every GRPO prompt must have the same number of completions")

    def signature(prompt_id: str) -> tuple[float, float, str]:
        group = groups[prompt_id]
        return (
            sum(row.completion_tokens for row in group) / len(group),
            pstdev(row.reward for row in group),
            prompt_id,
        )

    ordered = sorted(groups, key=signature)
    blocks = [
        ordered[start : start + block_size]
        for start in range(0, len(ordered), block_size)
    ]
    if len(blocks) > 1 and len(blocks[-1]) == 1:
        blocks[-2].extend(blocks.pop())

    donor_for: dict[str, str] = {}
    for block_index, targets in enumerate(blocks):
        rng = random.Random((int(seed) << 16) + block_index)
        candidates = list(targets)
        for _ in range(10_000):
            rng.shuffle(candidates)
            if all(
                target != donor
                and not _same_named_source(groups[target], groups[donor])
                for target, donor in zip(targets, candidates, strict=True)
            ):
                donor_for.update(zip(targets, candidates, strict=True))
                break
        else:
            raise ValueError(
                "could not permute GRPO credit across unrelated prompt groups; "
                "increase the block size"
            )

    assigned: dict[str, tuple[float, str, str]] = {}
    for target_prompt, donor_prompt in donor_for.items():
        targets = sorted(
            groups[target_prompt], key=lambda row: (row.completion_tokens, row.rollout_id)
        )
        donors = sorted(
            groups[donor_prompt], key=lambda row: (row.completion_tokens, row.rollout_id)
        )
        # Search all cyclic matches. First avoid leaving a rollout with the
        # same numeric credit; among ties, keep donor and target lengths close.
        choices = []
        for shift in range(group_size):
            shifted = donors[shift:] + donors[:shift]
            unchanged = sum(
                _float_key(target.advantage) == _float_key(donor.advantage)
                for target, donor in zip(targets, shifted, strict=True)
            )
            length_delta = sum(
                abs(target.completion_tokens - donor.completion_tokens)
                for target, donor in zip(targets, shifted, strict=True)
            )
            choices.append((unchanged, length_delta, shift, shifted))
        _, _, _, matched = min(choices, key=lambda item: item[:3])
        for target, donor in zip(targets, matched, strict=True):
            assigned[target.rollout_id] = (
                donor.advantage,
                donor.prompt_id,
                donor.rollout_id,
            )

    permuted = [
        replace(
            row,
            assigned_advantage=assigned[row.rollout_id][0],
            credit_donor_prompt_id=assigned[row.rollout_id][1],
            credit_donor_rollout_id=assigned[row.rollout_id][2],
        )
        for row in rows
    ]
    validate_grpo_credit_control(aligned_grpo_credit(rows), permuted)
    return permuted


def validate_grpo_credit_control(
    aligned: list[GRPORollout], permuted: list[GRPORollout]
) -> dict[str, float | int]:
    """Check that a control changes only the rollout-to-credit assignment."""
    if len(aligned) != len(permuted) or not aligned:
        raise ValueError("aligned and permuted GRPO rows must have equal nonzero size")
    fixed_fields = (
        "rollout_id",
        "prompt_id",
        "prompt",
        "completion_index",
        "completion",
        "completion_tokens",
        "old_logprobs",
        "loss_mask",
        "reward",
        "advantage",
        "source_problem",
    )
    for expected, observed in zip(aligned, permuted, strict=True):
        if any(
            getattr(expected, field) != getattr(observed, field)
            for field in fixed_fields
        ):
            raise ValueError("GRPO control changed rollout content or measured reward")
        if observed.assigned_advantage is None:
            raise ValueError("GRPO control left an advantage unassigned")

    aligned_credit = Counter(_float_key(row.assigned_advantage) for row in aligned)
    permuted_credit = Counter(_float_key(row.assigned_advantage) for row in permuted)
    if aligned_credit != permuted_credit:
        raise ValueError("GRPO control changed the global advantage multiset")

    aligned_groups = _group_rows(aligned)
    permuted_groups = _group_rows(permuted)
    donor_prompts = []
    donor_rollouts = []
    for prompt_id, rows in permuted_groups.items():
        prompt_donors = {row.credit_donor_prompt_id for row in rows}
        if len(prompt_donors) != 1 or None in prompt_donors:
            raise ValueError("one target prompt received credit from several prompts")
        donor_prompt = str(next(iter(prompt_donors)))
        donor_prompts.append(donor_prompt)
        if donor_prompt == prompt_id or _same_named_source(
            aligned_groups[prompt_id], aligned_groups[donor_prompt]
        ):
            raise ValueError("GRPO control retained a prompt or source pairing")
        expected = Counter(
            _float_key(row.advantage) for row in aligned_groups[donor_prompt]
        )
        observed = Counter(_float_key(row.assigned_advantage) for row in rows)
        if expected != observed:
            raise ValueError("target prompt did not receive one full donor credit vector")
        donor_rollouts.extend(str(row.credit_donor_rollout_id) for row in rows)
    if Counter(donor_prompts) != Counter(aligned_groups.keys()):
        raise ValueError("GRPO donor prompts do not form a bijection")
    if Counter(donor_rollouts) != Counter(row.rollout_id for row in aligned):
        raise ValueError("GRPO donor rollouts do not form a bijection")

    changed = sum(
        _float_key(expected.assigned_advantage) != _float_key(observed.assigned_advantage)
        for expected, observed in zip(aligned, permuted, strict=True)
    )
    zero_spread = sum(
        math.isclose(pstdev(row.reward for row in group), 0.0, abs_tol=1e-12)
        for group in aligned_groups.values()
    )
    donor_tokens = {row.rollout_id: row.completion_tokens for row in aligned}
    token_delta = sum(
        abs(
            row.completion_tokens
            - donor_tokens[str(row.credit_donor_rollout_id)]
        )
        for row in permuted
    )
    return {
        "prompt_groups": len(aligned_groups),
        "rollouts": len(aligned),
        "changed_advantages": changed,
        "changed_advantage_fraction": changed / len(aligned),
        "zero_reward_spread_groups": zero_spread,
        "zero_reward_spread_fraction": zero_spread / len(aligned_groups),
        "mean_donor_token_delta": token_delta / len(aligned),
    }


def _group_rows(rows: list[GRPORollout]) -> dict[str, list[GRPORollout]]:
    groups: dict[str, list[GRPORollout]] = defaultdict(list)
    ids = set()
    for row in rows:
        if row.rollout_id in ids:
            raise ValueError(f"duplicate GRPO rollout ID: {row.rollout_id}")
        if row.completion_tokens < 1:
            raise ValueError("GRPO completion token counts must be positive")
        if (
            len(row.old_logprobs) != row.completion_tokens
            or len(row.loss_mask) != row.completion_tokens
            or not all(math.isfinite(value) for value in row.old_logprobs)
        ):
            raise ValueError(
                f"{row.rollout_id}: old policy scores and mask must match tokens"
            )
        ids.add(row.rollout_id)
        groups[row.prompt_id].append(row)
    for prompt_id, group in groups.items():
        indices = [row.completion_index for row in group]
        if sorted(indices) != list(range(len(group))):
            raise ValueError(f"{prompt_id}: completion indices are not 0..G-1")
        sources = {row.source_problem for row in group}
        if len(sources) != 1:
            raise ValueError(f"{prompt_id}: one prompt has several source problems")
        prompts = {row.prompt for row in group}
        if len(prompts) != 1:
            raise ValueError(f"{prompt_id}: one prompt ID has several prompt texts")
    return dict(groups)


def _same_named_source(
    left: list[GRPORollout], right: list[GRPORollout]
) -> bool:
    left_source = left[0].source_problem
    right_source = right[0].source_problem
    return bool(left_source) and left_source == right_source


def _float_key(value: float | None) -> bytes:
    if value is None or not math.isfinite(value):
        raise ValueError("GRPO advantages must be finite")
    return struct.pack("!d", float(value))
