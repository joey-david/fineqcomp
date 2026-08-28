"""Frozen procedural reasoning tasks and their programmatic verifiers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict
from time import perf_counter
from typing import Any, Iterable

from fineqcomp.config import ModelSpec
from fineqcomp.data import Example


REASONING_GYM_VERSION = "0.1.25"
FINAL_ANSWER_MARKER = "FINAL ANSWER:"
DIFFICULTIES = ("easy", "medium", "hard")

# Each level fixes every declared difficulty field. Instance seed changes the
# problem, while family and level stay fixed.
TASK_SPECS: dict[str, dict[str, Any]] = {
    "arithmetic": {
        "task": "chain_sum",
        "reward_semantics": "numeric answer",
        "levels": {
            "easy": {
                "min_terms": 3,
                "max_terms": 3,
                "min_digits": 2,
                "max_digits": 2,
                "allow_negation": False,
            },
            "medium": {
                "min_terms": 6,
                "max_terms": 6,
                "min_digits": 3,
                "max_digits": 3,
                "allow_negation": False,
            },
            "hard": {
                "min_terms": 10,
                "max_terms": 10,
                "min_digits": 5,
                "max_digits": 5,
                "allow_negation": False,
            },
        },
    },
    "algebra": {
        "task": "polynomial_equations",
        "reward_semantics": "graded distance between numeric root sets",
        "levels": {
            "easy": {
                "min_degree": 1,
                "max_degree": 1,
                "min_terms": 2,
                "max_terms": 2,
                "min_value": 1,
                "max_value": 10,
            },
            "medium": {
                "min_degree": 2,
                "max_degree": 2,
                "min_terms": 3,
                "max_terms": 3,
                "min_value": 1,
                "max_value": 30,
            },
            "hard": {
                "min_degree": 3,
                "max_degree": 3,
                "min_terms": 4,
                "max_terms": 4,
                "min_value": 1,
                "max_value": 100,
            },
        },
    },
    "logic": {
        "task": "knights_knaves",
        "reward_semantics": "full or partial named truth assignments",
        "levels": {
            "easy": {
                "n_people": 2,
                "depth_constraint": 2,
                "width_constraint": 2,
            },
            "medium": {
                "n_people": 3,
                "depth_constraint": 2,
                "width_constraint": 2,
            },
            "hard": {
                "n_people": 4,
                "depth_constraint": 3,
                "width_constraint": 3,
            },
        },
    },
    "algorithmic": {
        "task": "shortest_path",
        "reward_semantics": "valid shortest path, with partial credit for a longer path",
        "levels": {
            "easy": {
                "min_rows": 6,
                "max_rows": 6,
                "min_cols": 6,
                "max_cols": 6,
                "p_blocked": 0.20,
            },
            "medium": {
                "min_rows": 10,
                "max_rows": 10,
                "min_cols": 10,
                "max_cols": 10,
                "p_blocked": 0.25,
            },
            "hard": {
                "min_rows": 16,
                "max_rows": 16,
                "min_cols": 16,
                "max_cols": 16,
                "p_blocked": 0.30,
            },
        },
    },
    "planning": {
        "task": "jugs",
        "reward_semantics": "simulated valid plan",
        "levels": {
            "easy": {"num_jugs": 3, "difficulty": 5},
            "medium": {"num_jugs": 3, "difficulty": 10},
            "hard": {"num_jugs": 3, "difficulty": 15},
        },
    },
}


def _reasoning_gym() -> Any:
    try:
        import reasoning_gym
    except ImportError as error:
        raise RuntimeError(
            "procedural reasoning tasks need the 'reasoning' project extra"
        ) from error
    installed = importlib.metadata.version("reasoning-gym")
    if installed != REASONING_GYM_VERSION:
        raise RuntimeError(
            f"reasoning-gym {REASONING_GYM_VERSION} is locked; found {installed}"
        )
    return reasoning_gym


def _task_spec(family: str, difficulty: str) -> tuple[str, dict[str, Any]]:
    if family not in TASK_SPECS:
        raise ValueError(f"unknown reasoning family: {family}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unknown reasoning difficulty: {difficulty}")
    spec = TASK_SPECS[family]
    return str(spec["task"]), dict(spec["levels"][difficulty])


def _json_copy(value: Any) -> Any:
    """Remove tuples and package-owned containers from a stored verifier row."""
    return json.loads(json.dumps(value, sort_keys=True))


def _prompt(question: str) -> str:
    return (
        question.rstrip()
        + "\n\nThink through the problem. End with the exact marker "
        + f"'{FINAL_ANSWER_MARKER}' followed only by your answer. A multi-line "
        + "answer may continue on the following lines."
    )


def generate_reasoning_examples(
    family: str,
    difficulty: str,
    rows: int,
    seed: int,
) -> list[Example]:
    """Generate distinct, fixed-difficulty instances with stored verifier state."""
    if rows < 1:
        raise ValueError("reasoning task row count must be positive")
    reasoning_gym = _reasoning_gym()
    task, level_config = _task_spec(family, difficulty)
    pool_size = max(rows * 4, rows + 64)
    dataset = reasoning_gym.create_dataset(
        task, size=pool_size, seed=seed, **level_config
    )
    generator_config = _json_copy(asdict(dataset.config))
    selected: list[Example] = []
    seen_questions: set[str] = set()
    for source_index in range(pool_size):
        entry = dataset[source_index]
        question = str(entry.get("question", "")).strip()
        answer = entry.get("answer")
        if not question or answer is None or question in seen_questions:
            continue
        oracle_reward = float(dataset.score_answer(str(answer), entry))
        if not math.isclose(oracle_reward, 1.0, abs_tol=1e-12):
            raise ValueError(
                f"{task}/{difficulty}/{source_index}: package oracle scores "
                f"{oracle_reward}, not 1"
            )
        seen_questions.add(question)
        problem_hash = hashlib.sha256(question.encode()).hexdigest()
        selected.append(
            Example(
                example_id=(
                    f"rg-{task}-{difficulty}-s{seed}-i{source_index}"
                ),
                prompt=_prompt(question),
                response=f"\n{FINAL_ANSWER_MARKER} {answer}",
                metadata={
                    "split": "procedural",
                    "evaluator": "reasoning_gym",
                    "reasoning_family": family,
                    "reasoning_task": task,
                    "difficulty": difficulty,
                    "source_index": source_index,
                    "source_problem": problem_hash,
                    "generator_package": "reasoning-gym",
                    "generator_version": REASONING_GYM_VERSION,
                    "generator_config": generator_config,
                    "verifier_entry": _json_copy(entry),
                },
            )
        )
        if len(selected) == rows:
            return selected
    raise ValueError(
        f"{task}/{difficulty}: found {len(selected)} distinct valid rows "
        f"in a frozen pool of {pool_size}, need {rows}"
    )


def extract_final_answer(response: str) -> str | None:
    matches = list(re.finditer(re.escape(FINAL_ANSWER_MARKER), response, re.I))
    if not matches:
        return None
    answer = response[matches[-1].end() :].strip()
    if answer.startswith("```"):
        lines = answer.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        answer = "\n".join(lines).strip()
    return answer or None


def score_reasoning_answer(example: Example, answer: str | None) -> float:
    metadata = example.metadata
    if metadata.get("generator_version") != REASONING_GYM_VERSION:
        raise ValueError(f"{example.example_id}: wrong or missing generator version")
    task = str(metadata["reasoning_task"])
    config = dict(metadata["generator_config"])
    entry = dict(metadata["verifier_entry"])
    reasoning_gym = _reasoning_gym()
    dataset = reasoning_gym.create_dataset(task, **config)
    reward = float(dataset.score_answer(answer, entry))
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise ValueError(f"{example.example_id}: verifier returned {reward}")
    return reward


def score_reasoning_completion(
    example: Example, completion: str
) -> tuple[str | None, float]:
    answer = extract_final_answer(completion)
    return answer, score_reasoning_answer(example, answer)


def audit_reasoning_task_panel(rows: int, seed: int) -> dict[str, Any]:
    """Check generator identity, determinism, diversity, and basic scorer cases."""
    cells = []
    for family in TASK_SPECS:
        task = str(TASK_SPECS[family]["task"])
        for difficulty in DIFFICULTIES:
            started = perf_counter()
            examples = generate_reasoning_examples(family, difficulty, rows, seed)
            repeated = generate_reasoning_examples(family, difficulty, rows, seed)
            oracle_rewards = [
                score_reasoning_answer(
                    example, str(example.metadata["verifier_entry"]["answer"])
                )
                for example in examples
            ]
            wrong_rewards = [
                score_reasoning_answer(example, "__definitely_wrong__")
                for example in examples
            ]
            empty_rewards = [
                score_reasoning_answer(example, "") for example in examples
            ]
            unique_prompts = len({example.prompt for example in examples})
            deterministic = [row.to_dict() for row in examples] == [
                row.to_dict() for row in repeated
            ]
            cell_pass = (
                unique_prompts == rows
                and deterministic
                and min(oracle_rewards) == 1.0
                and max(wrong_rewards) == 0.0
                and max(empty_rewards) == 0.0
            )
            cells.append(
                {
                    "family": family,
                    "task": task,
                    "difficulty": difficulty,
                    "rows": rows,
                    "generator_config": examples[0].metadata["generator_config"],
                    "reward_semantics": TASK_SPECS[family]["reward_semantics"],
                    "unique_prompts": unique_prompts,
                    "unique_oracle_answers": len(
                        {
                            str(example.metadata["verifier_entry"]["answer"])
                            for example in examples
                        }
                    ),
                    "deterministic_replay": deterministic,
                    "oracle_reward_min": min(oracle_rewards),
                    "sentinel_wrong_reward_max": max(wrong_rewards),
                    "empty_reward_max": max(empty_rewards),
                    "seconds": perf_counter() - started,
                    "passed": cell_pass,
                }
            )
    return {
        "schema": 1,
        "package": "reasoning-gym",
        "package_version": REASONING_GYM_VERSION,
        "seed": seed,
        "rows_per_cell": rows,
        "cells": cells,
        "passed": all(bool(cell["passed"]) for cell in cells),
    }


def evaluate_reasoning_generation(
    model: Any,
    tokenizer: Any,
    model_spec: ModelSpec,
    families: Iterable[str],
    difficulties: Iterable[str],
    rows_per_cell: int,
    seed: int,
    batch_size: int,
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Greedily generate one answer per fixed procedural instance and score it."""
    from fineqcomp.evaluation import generate_response_records

    examples = [
        example
        for family in families
        for difficulty in difficulties
        for example in generate_reasoning_examples(
            family, difficulty, rows_per_cell, seed
        )
    ]
    generated = generate_response_records(
        model,
        tokenizer,
        examples,
        model_spec,
        batch_size,
        max_new_tokens,
    )
    predictions = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for example, record in zip(examples, generated, strict=True):
        answer, reward = score_reasoning_completion(
            example, str(record["response"])
        )
        prediction = {
            "example_id": example.example_id,
            "family": example.metadata["reasoning_family"],
            "task": example.metadata["reasoning_task"],
            "difficulty": example.metadata["difficulty"],
            "expected": example.metadata["verifier_entry"]["answer"],
            "extracted_answer": answer,
            "reward": reward,
            **record,
        }
        predictions.append(prediction)
        grouped[(str(prediction["family"]), str(prediction["difficulty"]))].append(
            prediction
        )

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "examples": len(rows),
            "mean_reward": sum(float(row["reward"]) for row in rows) / len(rows),
            "full_reward_fraction": sum(row["reward"] == 1.0 for row in rows)
            / len(rows),
            "nonzero_reward_fraction": sum(row["reward"] > 0.0 for row in rows)
            / len(rows),
            "answer_extracted_fraction": sum(
                row["extracted_answer"] is not None for row in rows
            )
            / len(rows),
            "terminated_fraction": sum(row["terminated_with_eos"] for row in rows)
            / len(rows),
            "hit_generation_limit_fraction": sum(
                row["hit_generation_limit"] for row in rows
            )
            / len(rows),
            "mean_completion_tokens": sum(
                int(row["completion_tokens"]) for row in rows
            )
            / len(rows),
        }

    metrics = summarize(predictions)
    metrics["cells"] = [
        {"family": family, "difficulty": difficulty, **summarize(rows)}
        for (family, difficulty), rows in grouped.items()
    ]
    metrics["max_new_tokens"] = max_new_tokens
    return metrics, predictions
