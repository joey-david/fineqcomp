"""Deterministic task evaluation for the campaign."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import torch

from fineqcomp.config import ModelSpec
from fineqcomp.data import Example
from fineqcomp.sandbox import run_humaneval_tests
from fineqcomp.modeling import (
    inference_autocast,
    model_device,
    render_prompt,
    validate_single_token_labels,
)


def _batched(rows: list[Any], size: int) -> list[list[Any]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


@torch.no_grad()
def evaluate_constrained_labels(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Example],
    model_spec: ModelSpec,
    labels: list[str],
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score a fixed 16-label classification task from next-token logits.

    Used by the information-scaling study, where every prompt has exactly one
    correct single-token label, so accuracy and label NLL are exact.
    """
    label_ids = validate_single_token_labels(tokenizer, labels)
    device = model_device(model)
    model.eval()
    predictions = []
    total_nll = 0.0
    confidences = []
    correct_flags = []
    for batch in _batched(examples, batch_size):
        prompts = [render_prompt(tokenizer, row.prompt, model_spec) for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with inference_autocast(model, device):
            logits = model(**encoded).logits
        positions = torch.arange(
            encoded["attention_mask"].shape[1], device=device
        ).expand_as(encoded["attention_mask"])
        final_index = (positions * encoded["attention_mask"]).argmax(dim=1)
        next_logits = logits[torch.arange(len(batch), device=device), final_index]
        class_logits = next_logits[:, label_ids]
        probabilities = torch.softmax(class_logits.float(), dim=-1)
        chosen = probabilities.argmax(dim=-1)
        for row_index, row in enumerate(batch):
            target = int(row.metadata["label_index"])
            predicted = int(chosen[row_index].item())
            confidence = float(probabilities[row_index, predicted].item())
            target_probability = float(probabilities[row_index, target].item())
            correct = predicted == target
            total_nll += -math.log(max(target_probability, 1e-12))
            confidences.append(confidence)
            correct_flags.append(int(correct))
            prediction = {
                "example_id": row.example_id,
                "target": target,
                "prediction": predicted,
                "confidence": confidence,
                "correct": correct,
                # The probability the code actually pays for. Without it a
                # finished run cannot be recalibrated or re-coded, and the
                # only way to ask a new question of it is to train again.
                "target_probability": target_probability,
                "probabilities": [
                    round(float(value), 8)
                    for value in probabilities[row_index].tolist()
                ],
            }
            prediction["family"] = int(row.metadata["family"])
            prediction["item"] = int(row.metadata["item"])
            predictions.append(prediction)
    accuracy = sum(correct_flags) / max(len(correct_flags), 1)
    ece = 0.0
    for lower in [index / 10 for index in range(10)]:
        members = [
            index
            for index, confidence in enumerate(confidences)
            if lower <= confidence < lower + 0.1 or (lower == 0.9 and confidence == 1.0)
        ]
        if members:
            bin_accuracy = sum(correct_flags[index] for index in members) / len(members)
            bin_confidence = sum(confidences[index] for index in members) / len(members)
            ece += len(members) / len(confidences) * abs(bin_accuracy - bin_confidence)
    return (
        {
            "examples": len(examples),
            "accuracy": accuracy,
            "distortion": 1.0 - accuracy,
            "label_nll": total_nll / max(len(examples), 1),
            "ece": ece,
        },
        predictions,
    )



@torch.no_grad()
def evaluate_multiple_choice(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Example],
    model_spec: ModelSpec,
    labels: list[str],
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score standard multiple-choice tasks by constrained answer likelihood."""
    label_ids = validate_single_token_labels(tokenizer, labels)
    device = model_device(model)
    model.eval()
    predictions = []
    total_nll = 0.0
    correct = 0
    for batch in _batched(examples, batch_size):
        prompts = [render_prompt(tokenizer, row.prompt, model_spec) for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with inference_autocast(model, device):
            logits = model(**encoded).logits
        positions = torch.arange(
            encoded["attention_mask"].shape[1], device=device
        ).expand_as(encoded["attention_mask"])
        final_index = (positions * encoded["attention_mask"]).argmax(dim=1)
        next_logits = logits[torch.arange(len(batch), device=device), final_index]
        for row_index, row in enumerate(batch):
            choice_count = int(row.metadata["choice_count"])
            target = int(row.metadata["label_index"])
            probabilities = torch.softmax(
                next_logits[row_index, label_ids[:choice_count]].float(), dim=-1
            )
            predicted = int(probabilities.argmax().item())
            target_probability = float(probabilities[target].item())
            is_correct = predicted == target
            total_nll += -math.log(max(target_probability, 1e-12))
            correct += int(is_correct)
            predictions.append(
                {
                    "example_id": row.example_id,
                    "target": target,
                    "prediction": predicted,
                    "confidence": float(probabilities[predicted].item()),
                    "correct": is_correct,
                }
            )
    return (
        {
            "examples": len(examples),
            "accuracy": correct / max(len(examples), 1),
            "label_nll": total_nll / max(len(examples), 1),
        },
        predictions,
    )


def _normalize_number(text: str) -> str | None:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    value = matches[-1].replace(",", "")
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else str(number)


@torch.no_grad()
def generate_response_records(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Example],
    model_spec: ModelSpec,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    device = model_device(model)
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs: list[dict[str, Any]] = []
    model.eval()
    try:
        for batch in _batched(examples, batch_size):
            prompts = [
                render_prompt(tokenizer, row.prompt, model_spec) for row in batch
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with inference_autocast(model, device):
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            prompt_width = encoded["input_ids"].shape[1]
            for row in generated:
                completion = row[prompt_width:]
                eos = tokenizer.eos_token_id
                eos_positions = (
                    (completion == int(eos)).nonzero(as_tuple=False).flatten()
                    if eos is not None
                    else torch.empty(0, dtype=torch.long, device=completion.device)
                )
                terminated = bool(len(eos_positions))
                token_count = (
                    int(eos_positions[0].item()) + 1
                    if terminated
                    else int(completion.numel())
                )
                outputs.append(
                    {
                        "response": tokenizer.decode(
                            completion, skip_special_tokens=True
                        ),
                        "completion_tokens": token_count,
                        "terminated_with_eos": terminated,
                        "hit_generation_limit": not terminated,
                    }
                )
    finally:
        tokenizer.padding_side = old_padding
    return outputs


def generate_responses(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Example],
    model_spec: ModelSpec,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    """Compatibility wrapper for callers that need response text only."""
    return [
        str(record["response"])
        for record in generate_response_records(
            model,
            tokenizer,
            examples,
            model_spec,
            batch_size,
            max_new_tokens,
        )
    ]


def _generation_metrics(
    records: list[dict[str, Any]], max_new_tokens: int
) -> dict[str, float | int]:
    rows = max(len(records), 1)
    return {
        "max_new_tokens": max_new_tokens,
        "terminated_fraction": sum(
            bool(record["terminated_with_eos"]) for record in records
        )
        / rows,
        "hit_generation_limit_fraction": sum(
            bool(record["hit_generation_limit"]) for record in records
        )
        / rows,
        "mean_completion_tokens": sum(
            int(record["completion_tokens"]) for record in records
        )
        / rows,
    }


def evaluate_natural(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Example],
    model_spec: ModelSpec,
    dataset_key: str,
    batch_size: int,
    multiple_choice_labels: list[str] | None = None,
    max_new_tokens: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluators = []
    for example in examples:
        evaluator = str(example.metadata.get("evaluator", dataset_key))
        if evaluator not in evaluators:
            evaluators.append(evaluator)
    if len(evaluators) > 1:
        all_predictions = []
        results = {}
        for evaluator in evaluators:
            subset = [
                example
                for example in examples
                if str(example.metadata.get("evaluator", dataset_key)) == evaluator
            ]
            metrics, predictions = evaluate_natural(
                model,
                tokenizer,
                subset,
                model_spec,
                evaluator,
                batch_size,
                multiple_choice_labels,
                max_new_tokens,
            )
            results[evaluator] = metrics
            all_predictions.extend(
                {**prediction, "evaluator": evaluator} for prediction in predictions
            )
        primary = results[evaluators[0]]
        return {
            **primary,
            "primary_evaluator": evaluators[0],
            "evaluations": results,
        }, all_predictions
    elif len(evaluators) == 1:
        dataset_key = evaluators[0]
    if examples and "choice_count" in examples[0].metadata:
        if not multiple_choice_labels:
            raise ValueError("multiple-choice evaluation requires answer labels")
        return evaluate_multiple_choice(
            model,
            tokenizer,
            examples,
            model_spec,
            multiple_choice_labels,
            batch_size,
        )
    max_tokens = (
        max_new_tokens
        if max_new_tokens is not None
        else (512 if dataset_key in {"gsm8k", "math", "mbpp", "humaneval"} else 128)
    )
    if max_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    generation_records = generate_response_records(
        model, tokenizer, examples, model_spec, batch_size, max_tokens
    )
    responses = [str(record["response"]) for record in generation_records]
    generation_metrics = _generation_metrics(generation_records, max_tokens)
    predictions = []
    if dataset_key == "gsm8k":
        correct = 0
        extracted = 0
        for example, response, generation in zip(
            examples, responses, generation_records, strict=True
        ):
            expected = _normalize_number(example.response)
            predicted = _normalize_number(response)
            extracted += int(predicted is not None)
            is_correct = predicted is not None and predicted == expected
            correct += int(is_correct)
            predictions.append(
                {
                    "example_id": example.example_id,
                    "expected": expected,
                    "prediction": predicted,
                    "response": response,
                    "correct": is_correct,
                    **generation,
                }
            )
        return {
            "examples": len(examples),
            "exact_match": correct / max(len(examples), 1),
            "answer_extracted_fraction": extracted / max(len(examples), 1),
            **generation_metrics,
        }, predictions
    if dataset_key == "humaneval":
        passed = 0
        statuses: dict[str, int] = {}
        for example, response, generation in zip(
            examples, responses, generation_records, strict=True
        ):
            result = run_humaneval_tests(
                response,
                str(example.metadata["code_prefix"]),
                str(example.metadata["tests"]),
                str(example.metadata["entry_point"]),
            )
            passed += int(result["passed"])
            statuses[result["status"]] = statuses.get(result["status"], 0) + 1
            predictions.append(
                {
                    "example_id": example.example_id,
                    "response": response,
                    **generation,
                    **result,
                }
            )
        return {
            "examples": len(examples),
            "pass_at_1": passed / max(len(examples), 1),
            "failure_counts": statuses,
            **generation_metrics,
        }, predictions
    if dataset_key == "math":
        from math_verify import (
            ExprExtractionConfig,
            LatexExtractionConfig,
            parse,
            verify,
        )

        latex = LatexExtractionConfig(boxed_match_priority=0)
        correct = 0
        extracted = 0
        for example, response, generation in zip(
            examples, responses, generation_records, strict=True
        ):
            try:
                gold = parse(example.response, extraction_config=[latex])
                answer = parse(
                    response,
                    extraction_config=[latex, ExprExtractionConfig()],
                )
                extracted += int(bool(answer))
                is_correct = bool(gold and answer and verify(gold, answer))
            except (ValueError, TypeError):
                is_correct = False
            correct += int(is_correct)
            predictions.append(
                {
                    "example_id": example.example_id,
                    "response": response,
                    "correct": is_correct,
                    "level": example.metadata.get("level"),
                    "subject": example.metadata.get("subject"),
                    **generation,
                }
            )
        return {
            "examples": len(examples),
            "exact_match": correct / max(len(examples), 1),
            "answer_extracted_fraction": extracted / max(len(examples), 1),
            **generation_metrics,
        }, predictions
    if dataset_key == "xsum":
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [
            scorer.score(example.response.strip(), response.strip())["rougeL"].fmeasure
            for example, response in zip(examples, responses, strict=True)
        ]
        for example, response, score, generation in zip(
            examples, responses, scores, generation_records, strict=True
        ):
            predictions.append(
                {
                    "example_id": example.example_id,
                    "response": response,
                    "rouge_l": score,
                    **generation,
                }
            )
        return {
            "examples": len(examples),
            "rouge_l": sum(scores) / max(len(scores), 1),
            **generation_metrics,
        }, predictions
    if dataset_key in _EXACT_STRING_TASKS:
        correct = 0
        for example, response, generation in zip(
            examples, responses, generation_records, strict=True
        ):
            expected = _normalize_answer_text(example.response)
            predicted = _normalize_answer_text(response)
            is_correct = bool(expected) and predicted == expected
            correct += int(is_correct)
            predictions.append(
                {
                    "example_id": example.example_id,
                    "expected": expected,
                    "prediction": predicted,
                    "response": response,
                    "correct": is_correct,
                    **generation,
                }
            )
        return {
            "examples": len(examples),
            "exact_match": correct / max(len(examples), 1),
            **generation_metrics,
        }, predictions
    raise ValueError(f"unsupported natural evaluation: {dataset_key}")


# Tasks whose answer is one short string: a SQL query, an XBRL tag. Scored on a
# normalised exact match rather than a task-specific executor, so the number is
# a lower bound on the real metric -- an equivalent query written differently
# counts as wrong. That is acceptable here because every arm is scored the same
# way and the comparison is between arms.
_EXACT_STRING_TASKS = {"paws", "text_to_sql", "xbrl_tags"}


def _normalize_answer_text(text: str) -> str:
    """Cut the continuation, then collapse whitespace, case and a semicolon.

    Nothing stops generation at the end of the answer, so a model keeps writing
    the next example. The cut is at a blank line or the next section marker
    rather than at the first newline, because a SQL query may legitimately span
    several lines and truncating it would score a right answer wrong.
    """
    body = str(text)
    for stop in ("\n\n", "###", "### "):
        index = body.find(stop)
        if index > 0:
            body = body[:index]
    return " ".join(body.split()).strip().rstrip(";").strip().lower()


def write_predictions(path: str | Path, rows: list[dict[str, Any]]) -> None:
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(target)
