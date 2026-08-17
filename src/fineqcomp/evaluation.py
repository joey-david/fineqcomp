"""Deterministic task evaluation for the campaign."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import torch

from fineqcomp.config import ModelSpec
from fineqcomp.data import Example
from fineqcomp.mbpp import run_humaneval_tests, run_mbpp_tests
from fineqcomp.modeling import model_device, render_prompt, validate_single_token_labels


def _batched(rows: list[Any], size: int) -> list[list[Any]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


@torch.no_grad()
def evaluate_synthetic(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Example],
    model_spec: ModelSpec,
    labels: list[str],
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
            }
            if "binding" in row.metadata:
                prediction["binding"] = int(row.metadata["binding"])
            else:
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
def generate_responses(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Example],
    model_spec: ModelSpec,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    device = model_device(model)
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs: list[str] = []
    model.eval()
    try:
        for batch in _batched(examples, batch_size):
            prompts = [
                render_prompt(tokenizer, row.prompt, model_spec) for row in batch
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_width = encoded["input_ids"].shape[1]
            for row in generated:
                outputs.append(
                    tokenizer.decode(row[prompt_width:], skip_special_tokens=True)
                )
    finally:
        tokenizer.padding_side = old_padding
    return outputs


def evaluate_natural(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Example],
    model_spec: ModelSpec,
    dataset_key: str,
    batch_size: int,
    multiple_choice_labels: list[str] | None = None,
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
    max_tokens = 512 if dataset_key in {"gsm8k", "math", "mbpp", "humaneval"} else 128
    responses = generate_responses(
        model, tokenizer, examples, model_spec, batch_size, max_tokens
    )
    predictions = []
    if dataset_key == "gsm8k":
        correct = 0
        for example, response in zip(examples, responses, strict=True):
            expected = _normalize_number(example.response)
            predicted = _normalize_number(response)
            is_correct = predicted is not None and predicted == expected
            correct += int(is_correct)
            predictions.append(
                {
                    "example_id": example.example_id,
                    "expected": expected,
                    "prediction": predicted,
                    "response": response,
                    "correct": is_correct,
                }
            )
        return {
            "examples": len(examples),
            "exact_match": correct / max(len(examples), 1),
        }, predictions
    if dataset_key == "mbpp":
        passed = 0
        statuses: dict[str, int] = {}
        for example, response in zip(examples, responses, strict=True):
            result = run_mbpp_tests(response, list(example.metadata["tests"]))
            passed += int(result["passed"])
            statuses[result["status"]] = statuses.get(result["status"], 0) + 1
            predictions.append(
                {
                    "example_id": example.example_id,
                    "response": response,
                    **result,
                }
            )
        return {
            "examples": len(examples),
            "pass_at_1": passed / max(len(examples), 1),
            "failure_counts": statuses,
        }, predictions
    if dataset_key == "humaneval":
        passed = 0
        statuses: dict[str, int] = {}
        for example, response in zip(examples, responses, strict=True):
            result = run_humaneval_tests(
                response,
                str(example.metadata["code_prefix"]),
                str(example.metadata["tests"]),
                str(example.metadata["entry_point"]),
            )
            passed += int(result["passed"])
            statuses[result["status"]] = statuses.get(result["status"], 0) + 1
            predictions.append(
                {"example_id": example.example_id, "response": response, **result}
            )
        return {
            "examples": len(examples),
            "pass_at_1": passed / max(len(examples), 1),
            "failure_counts": statuses,
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
        for example, response in zip(examples, responses, strict=True):
            try:
                gold = parse(example.response, extraction_config=[latex])
                answer = parse(
                    response,
                    extraction_config=[latex, ExprExtractionConfig()],
                )
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
                }
            )
        return {
            "examples": len(examples),
            "exact_match": correct / max(len(examples), 1),
        }, predictions
    if dataset_key == "xsum":
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [
            scorer.score(example.response.strip(), response.strip())["rougeL"].fmeasure
            for example, response in zip(examples, responses, strict=True)
        ]
        for example, response, score in zip(examples, responses, scores, strict=True):
            predictions.append(
                {
                    "example_id": example.example_id,
                    "response": response,
                    "rouge_l": score,
                }
            )
        return {
            "examples": len(examples),
            "rouge_l": sum(scores) / max(len(scores), 1),
        }, predictions
    raise ValueError(f"unsupported natural evaluation: {dataset_key}")


def evaluate_ifeval(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Example],
    evaluator_rows: list[dict[str, Any]],
    model_spec: ModelSpec,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate IFEval answers and score them with the pinned official port."""
    from instruction_following_eval.evaluation import evaluate_instruction_following

    responses = generate_responses(
        model,
        tokenizer,
        examples,
        model_spec,
        batch_size=batch_size,
        max_new_tokens=768,
    )
    metrics = evaluate_instruction_following(evaluator_rows, responses)
    predictions = [
        {"example_id": example.example_id, "response": response}
        for example, response in zip(examples, responses, strict=True)
    ]
    return metrics, predictions


def write_predictions(path: str | Path, rows: list[dict[str, Any]]) -> None:
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(target)
