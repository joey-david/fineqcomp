"""How does a compressed corrupted adapter beat the base model on GSM8K?

`denoise_vs_shrinkage` establishes *that* it does: the damaged rank-16 adapter
truncated to rank one and coded at one bit scores 80.2% against the base
model's 74.8%, and a plain rescaling of the same rank-one direction does the
same. It also establishes what the gain is *not*: the surviving update is
nearly orthogonal to an adapter trained on clean rationales (cosine 0.006), so
it is not installing task knowledge.

Reading the generations points somewhere else. The recovered adapter writes
about 44% more words than the base model and almost exactly the same number of
arithmetic operations -- 4.34 against 4.23 on the problems whose reference
solution needs five or more. The accuracy gain is not spread evenly over those
generations: sorted by how much the base model chose to write, it is +16.7
points where the base model wrote under twenty words and -0.6 points where it
wrote seventy or more.

That suggests the update acts on the decision to *keep writing* rather than on
the arithmetic, which this module tests three ways.

  forcing    The causal test, and the one that can refute the account. Ban the
             answer marker and the end-of-text token for the first K generated
             tokens, so the base model cannot stop early, and see whether it
             recovers the adapter's gain with no adapter attached. If length is
             the mechanism, forcing length reproduces the effect; if it does
             not, the update is doing something a length prior cannot.

  bands      Where the effect lives. The rank-one update is applied only inside
             one band of transformer layers and zeroed elsewhere, which asks
             whether the disposition is written early, in the middle, or late.

  pressure   The readout. Teacher-force the base model's own chains and measure
             how the update shifts the log-probability of emitting the answer
             marker at each position. A negative shift that grows along the
             chain is termination being suppressed, measured directly on the
             logits rather than inferred from lengths.

Corpora, prompts and answer scoring are imported from the same places the
recorded studies use, so the accuracies here sit on the recorded axis.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import LogitsProcessor, LogitsProcessorList

from fineqcomp.adapters import apply_adapter_tensors, module_layer_index
from fineqcomp.artifacts import read_json, write_json
from fineqcomp.behavioral_trajectory import sha256, study_data
from fineqcomp.codec import (
    decode_adapter_tensor_map, encode_tensor_map, pad_lora_rank, truncate_lora_rank,
)
from fineqcomp.config import RunSpec
from fineqcomp.evaluation import (
    _normalize_number, render_prompt, repeated_ngram_fraction, write_predictions,
)
from fineqcomp.denoise_vs_shrinkage import (
    NLL_SPLITS, frobenius_inner, lora_pairs, nll_examples, update_geometry,
)
from fineqcomp.modeling import (
    ModelSession, generation_policy, inference_autocast, model_device,
)
from fineqcomp.training import causal_nll


def marker_token_ids(tokenizer: Any) -> list[int]:
    """Every token whose text contains '#', plus end-of-text.

    The benchmark's answer marker is '####', but a tokenizer is free to split
    it, to attach leading whitespace, or to carry it inside a larger piece. So
    the ban is defined over the character rather than over one id: any token
    that could begin or continue the marker is blocked. '#' does not otherwise
    occur in GSM8K working, so this costs nothing the model needed.
    """
    blocked = []
    for token, index in tokenizer.get_vocab().items():
        text = tokenizer.convert_tokens_to_string([token])
        if "#" in text:
            blocked.append(int(index))
    if tokenizer.eos_token_id is not None:
        blocked.append(int(tokenizer.eos_token_id))
    return sorted(set(blocked))


class MeasuredBias(LogitsProcessor):
    """Apply the update's own measured per-token shift to the base model.

    The readout gives, for every token, the expected frequency the update
    assigns it against the frequency the base model assigns it. The log of that
    ratio is the first-order estimate of the logit shift that would produce it,
    so adding it back to the base model's logits reconstructs the update's
    effect on the output distribution -- and only that effect. Restricting the
    reconstruction to the k tokens that move the most mass asks how much of the
    behaviour those tokens carry.

    This is an approximation and worth naming as one: a shift in mean
    probability across positions is not the same object as a constant logit
    offset, because the update's real effect varies by context and this does
    not. It reproduces the average, not the conditioning.
    """

    def __init__(self, ids: list[int], shifts: list[float]):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.shifts = torch.tensor(shifts, dtype=torch.float)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores[:, self.ids.to(scores.device)] += self.shifts.to(
            scores.device, scores.dtype
        )
        return scores


class MarkerBias(LogitsProcessor):
    """Subtract a constant from the answer marker's logits at every position.

    This is the matched form of the forcing control. A hard ban (`HoldOpen`)
    does not lengthen a chain so much as derail it: forbidden to finish, the
    model runs past the point where it had anything left to say and never
    recovers the answer format, which changes far more than the length. A
    constant bias instead shifts the *decision* to stop by a fixed amount in
    log-odds and leaves everything else to the model, so the chain grows where
    the model was nearly ready to stop and is untouched where it was not.

    It is also the intervention the pressure readout measures directly. If the
    update's whole effect is to make finishing less attractive by d nats, then
    biasing the base model by that same d should reproduce it.
    """

    def __init__(self, blocked: list[int], bias: float):
        self.blocked = torch.tensor(blocked, dtype=torch.long)
        self.bias = float(bias)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores[:, self.blocked.to(scores.device)] -= self.bias
        return scores


class HoldOpen(LogitsProcessor):
    """Forbid finishing for the first `hold` generated tokens.

    This is a floor on the chain, not a target length: once the ban lifts the
    model stops wherever it would have stopped anyway, so a chain that was
    already longer than the floor is left untouched. That is what makes the
    comparison against the adapter fair -- the adapter does not lengthen every
    chain either.
    """

    def __init__(self, blocked: list[int], hold: int, prompt_width: int):
        self.blocked = torch.tensor(blocked, dtype=torch.long)
        self.hold = int(hold)
        self.prompt_width = int(prompt_width)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        produced = int(input_ids.shape[1]) - self.prompt_width
        if produced >= self.hold:
            return scores
        scores[:, self.blocked.to(scores.device)] = float("-inf")
        return scores


@torch.no_grad()
def generate(
    session: ModelSession,
    run: RunSpec,
    examples: list,
    config: dict,
    hold: int = 0,
    bias: float = 0.0,
    prefix: str = "",
    measured: LogitsProcessor | None = None,
) -> list[dict[str, Any]]:
    """Greedy generation, optionally biased, floored, or given a prompt prefix."""
    tokenizer, model = session.tokenizer, session.model
    device = model_device(model)
    blocked = marker_token_ids(tokenizer) if (hold > 0 or bias) else []
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    records = []
    model.eval()
    try:
        with torch.random.fork_rng(
            devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        ):
            torch.manual_seed(run.seed)
            size = int(config["generation_batch_size"])
            for start in range(0, len(examples), size):
                batch = examples[start : start + size]
                prompts = [
                    render_prompt(tokenizer, prefix + row.prompt, run.model)
                    for row in batch
                ]
                encoded = tokenizer(prompts, return_tensors="pt", padding=True)
                encoded = {key: value.to(device) for key, value in encoded.items()}
                width = int(encoded["input_ids"].shape[1])
                steps = []
                if measured is not None:
                    steps.append(measured)
                if bias:
                    steps.append(MarkerBias(blocked, bias))
                if hold > 0:
                    steps.append(HoldOpen(blocked, hold, width))
                processors = LogitsProcessorList(steps) if steps else None
                with inference_autocast(model, device):
                    produced = model.generate(
                        **encoded,
                        **generation_policy(run.model),
                        max_new_tokens=int(config["max_new_tokens"]),
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        logits_processor=processors,
                        use_cache=True,
                    )
                for row, example in zip(produced, batch, strict=True):
                    completion = row[width:]
                    eos = tokenizer.eos_token_id
                    hits = (completion == int(eos)).nonzero(as_tuple=False).flatten()
                    text = tokenizer.decode(completion, skip_special_tokens=True)
                    # A few-shot prompt teaches the model to emit question and
                    # answer pairs, so it emits more of them. Everything from
                    # the first invented question on is about a different
                    # problem, and scoring it would read the answer to a
                    # question nobody asked. Cut before any metric is taken, so
                    # the response, its length and its answer all agree.
                    stop = str(config.get("few_shot_stop", "\nQuestion:"))
                    if stop and stop in text:
                        text = text.split(stop)[0]
                    predicted = _normalize_number(text)
                    expected = _normalize_number(example.response)
                    records.append(
                        {
                            "example_id": example.example_id,
                            "response": text,
                            "expected": expected,
                            "prediction": predicted,
                            "correct": predicted is not None and predicted == expected,
                            "completion_tokens": (
                                int(hits[0].item()) + 1 if len(hits)
                                else int(completion.numel())
                            ),
                            "terminated_with_eos": bool(len(hits)),
                            "hit_generation_limit": not bool(len(hits)),
                            "repeated_8gram_fraction": repeated_ngram_fraction(text),
                            "truncated_at_next_question": stop in tokenizer.decode(
                                completion, skip_special_tokens=True),
                            "cluster": example.example_id,
                        }
                    )
    finally:
        tokenizer.padding_side = old_padding
    return records


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    chains = [row["response"].split("####")[0] for row in records]
    return {
        "examples": len(records),
        "exact_match": sum(row["correct"] for row in records) / max(len(records), 1),
        "answer_extracted_fraction": (
            sum(row["prediction"] is not None for row in records) / max(len(records), 1)
        ),
        "mean_completion_tokens": (
            sum(row["completion_tokens"] for row in records) / max(len(records), 1)
        ),
        "mean_chain_words": sum(len(c.split()) for c in chains) / max(len(chains), 1),
        "mean_chain_equals": sum(c.count("=") for c in chains) / max(len(chains), 1),
        "hit_generation_limit_fraction": (
            sum(row["hit_generation_limit"] for row in records) / max(len(records), 1)
        ),
    }


def cached(path: Path, build) -> dict[str, Any]:
    existing = read_json(path)
    if existing is not None:
        return existing
    records = build()
    write_predictions(path.with_suffix(".jsonl"), records)
    metrics = summarise(records)
    write_json(path, metrics)
    print(f"{path.stem}: acc={metrics['exact_match']:.4f} "
          f"words={metrics['mean_chain_words']:.1f} "
          f"eq={metrics['mean_chain_equals']:.2f}", flush=True)
    return metrics


def band_update(
    tensors: dict[str, torch.Tensor], low: int, high: int
) -> dict[str, torch.Tensor]:
    """Keep the update only on layers in [low, high]; zero it everywhere else.

    Zeroing rather than dropping keeps the tensor map the exact shape the
    attached adapter expects, so every band is applied through one unchanged
    model and the bands differ only in which layers carry a nonzero update.
    """
    kept = {}
    for name, value in tensors.items():
        layer = module_layer_index(name)
        inside = layer is not None and low <= layer <= high
        kept[name] = value.clone() if inside else torch.zeros_like(value)
    return kept


def coded_head(tensors: dict[str, torch.Tensor], bits: int, scratch: Path):
    """The recovered update: rank one, through a real file at `bits`."""
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / f"head_b{bits}.fqcb"
    encode_tensor_map(truncate_lora_rank(tensors, 1), path, bits)
    _, decoded = decode_adapter_tensor_map(path)
    path.unlink(missing_ok=True)
    return decoded


@torch.no_grad()
def termination_pressure(
    session: ModelSession,
    run: RunSpec,
    records: list[dict[str, Any]],
    examples_by_id: dict[str, Any],
    update: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
    config: dict,
) -> dict[str, Any]:
    """How the update moves the log-probability of finishing, position by position.

    The chains are the *base model's own*, held fixed and teacher-forced under
    both weight settings, so the two passes see identical text and the only
    thing that varies is the update. Comparing generations instead would
    confound the shift with the different text each setting goes on to write.

    Positions are reported in tenths of the chain, because the interesting
    claim is about where along a chain the pressure to stop is applied, not
    about any single token.
    """
    tokenizer, model = session.tokenizer, session.model
    device = model_device(model)
    marker = torch.tensor(
        [i for i in marker_token_ids(tokenizer) if i != tokenizer.eos_token_id],
        dtype=torch.long, device=device,
    )
    rows = records[: int(config["pressure_rows"])]
    detail: dict[str, dict[int, list[float]]] = {"base": {}, "update": {}}
    totals: dict[str, list[float]] = {"base": [], "update": []}
    model.eval()
    for setting, tensors in (("base", base), ("update", update)):
        apply_adapter_tensors(model, tensors)
        for row in rows:
            example = examples_by_id[row["example_id"]]
            chain = row["response"].split("####")[0]
            if len(chain.split()) < 8:
                continue
            prompt_ids = tokenizer(
                render_prompt(tokenizer, example.prompt, run.model),
                return_tensors="pt",
            )["input_ids"]
            chain_ids = tokenizer(
                chain, return_tensors="pt", add_special_tokens=False
            )["input_ids"]
            ids = torch.cat([prompt_ids, chain_ids], dim=1).to(device)
            if ids.shape[1] > int(config["pressure_max_length"]):
                continue
            with inference_autocast(model, device):
                logits = model(input_ids=ids).logits
            width = int(prompt_ids.shape[1])
            # The logit at position t predicts token t+1, so the slice starts
            # one before the chain and ends at its last token. That scores the
            # marker as the next thing the model could say at every point in
            # the chain -- which is the decision the account is about.
            span = torch.log_softmax(logits[0, width - 1 : -1].float(), dim=-1)
            value = torch.logsumexp(span.index_select(1, marker), dim=1)
            totals[setting].append(float(value.mean()))
            length = int(value.numel())
            for index in range(length):
                bucket = min(9, index * 10 // max(length, 1))
                detail[setting].setdefault(bucket, []).append(float(value[index]))
    shifts = {}
    for bucket in range(10):
        left, right = detail["base"].get(bucket, []), detail["update"].get(bucket, [])
        if not left or not right:
            continue
        shifts[bucket] = {
            "base_logp_marker": sum(left) / len(left),
            "update_logp_marker": sum(right) / len(right),
            "shift": sum(right) / len(right) - sum(left) / len(left),
            "positions": len(left),
        }
    return {
        "chains_scored": len(totals["base"]),
        "mean_logp_marker_base": (
            sum(totals["base"]) / max(len(totals["base"]), 1)
        ),
        "mean_logp_marker_update": (
            sum(totals["update"]) / max(len(totals["update"]), 1)
        ),
        "by_decile": shifts,
    }


def exemplar_block(rows: list, count: int, marker: str) -> str:
    """Worked examples in one register, taken from the training corpus.

    The exemplars are real rows rather than prose written here, because a
    hand-written style would be a second intervention with no way to say which
    part of it mattered. Both registers are drawn the same way and differ only
    in which corpus they come from.

    The rows are training data for the adapter under study, so they cannot leak
    the test split; they are the register the adapter was fitted to.
    """
    parts = []
    for row in rows[:count]:
        # The whole response, answer line included. An exemplar cut before its
        # answer demonstrates a register but not how to finish in it, and the
        # model then has to invent an ending -- which is a second difference
        # between the arms and not the one under test. `marker` is kept in the
        # signature because each register names its answer line differently and
        # the caller has to pass the right one for the check below.
        body = row.response.strip()
        if marker and marker not in body:
            raise ValueError(f"exemplar has no {marker!r} answer line")
        question = row.prompt.split("Question:")[-1].split("Answer:")[0].strip()
        parts.append(f"Question: {question}\nAnswer: {body}")
    return "\n\n".join(parts) + "\n\n" if parts else ""


@torch.no_grad()
def token_readout(
    session: ModelSession,
    run: RunSpec,
    records: list[dict[str, Any]],
    examples_by_id: dict[str, Any],
    update: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
    config: dict,
) -> dict[str, Any]:
    """Which tokens the update promotes, averaged over real chain positions.

    A rank-one update's output direction cannot be read straight off the
    weights, because the sign of an SVD factor pair is arbitrary and only the
    product is fixed -- what the direction writes depends on the sign of its
    input projection, which varies token by token. So the readout is taken
    where the ambiguity does not arise: teacher-force the base model's own
    chains under both weight settings and difference the log-probabilities the
    model actually assigns.

    Averaging over positions gives the update's standing effect on the next
    token, in nats, for every item in the vocabulary.
    """
    tokenizer, model = session.tokenizer, session.model
    device = model_device(model)
    rows = records[: int(config["pressure_rows"])]
    totals: dict[str, torch.Tensor] = {}
    counts = 0
    model.eval()
    for setting, tensors in (("base", base), ("update", update)):
        apply_adapter_tensors(model, tensors)
        running = None
        counts = 0
        for row in rows:
            example = examples_by_id[row["example_id"]]
            chain = row["response"].split("####")[0]
            if len(chain.split()) < 8:
                continue
            prompt_ids = tokenizer(
                render_prompt(tokenizer, example.prompt, run.model), return_tensors="pt"
            )["input_ids"]
            chain_ids = tokenizer(
                chain, return_tensors="pt", add_special_tokens=False
            )["input_ids"]
            ids = torch.cat([prompt_ids, chain_ids], dim=1).to(device)
            if ids.shape[1] > int(config["pressure_max_length"]):
                continue
            with inference_autocast(model, device):
                logits = model(input_ids=ids).logits
            width = int(prompt_ids.shape[1])
            # Probabilities, not log-probabilities. Averaging log p over
            # positions gives a geometric mean, which is negligible for every
            # token in the vocabulary -- at any one position almost nothing is
            # likely -- and differencing two such averages says nothing about
            # what the model would actually emit. The arithmetic mean of p is
            # the expected frequency of the token per position, which is the
            # quantity a reader means by "this update promotes that word".
            span = torch.softmax(logits[0, width - 1 : -1].float(), dim=-1)
            running = span.sum(0) if running is None else running + span.sum(0)
            counts += int(span.shape[0])
        totals[setting] = running / max(counts, 1)
    base_p = totals["base"].cpu()
    update_p = totals["update"].cpu()
    mass = update_p - base_p
    # A ratio on tokens the base model essentially never emits is noise, so the
    # log-ratio ranking is restricted to tokens carrying real expected mass.
    floor = float(config.get("readout_min_prob", 1e-4))
    eligible = base_p > floor
    ratio = torch.log((update_p + 1e-12) / (base_p + 1e-12))
    up, down = ratio.clone(), ratio.clone()
    up[~eligible] = float("-inf")
    down[~eligible] = float("inf")

    def render(indices, source):
        return [
            {"token": tokenizer.convert_ids_to_tokens([int(i)])[0],
             "text": tokenizer.decode([int(i)]),
             "delta": round(float(source[int(i)]), 6),
             "base_prob": round(float(base_p[int(i)]), 6),
             "update_prob": round(float(update_p[int(i)]), 6)}
            for i in indices
        ]

    top = int(config.get("readout_top", 40))
    by_mass = torch.argsort(mass, descending=True)
    return {
        "positions": counts,
        "eligible_tokens": int(eligible.sum()),
        "promoted_by_mass": render(by_mass[:top], mass),
        "suppressed_by_mass": render(by_mass[-top:].flip(0), mass),
        "promoted_by_ratio": render(torch.argsort(up, descending=True)[:top], ratio),
        "suppressed_by_ratio": render(torch.argsort(down)[:top], ratio),
        "total_absolute_mass_moved": float(mass.abs().sum() / 2),
    }


def spectral_slice(
    tensors: dict[str, torch.Tensor], low: int, high: int
) -> dict[str, torch.Tensor]:
    """Keep singular directions [low, high) of the update and drop the rest.

    `truncate_lora_rank` returns balanced factors whose rows are ordered by
    singular value, so a contiguous slice of them is a contiguous slice of the
    spectrum. Taking a middle or tail band rather than a prefix is what lets
    "the top direction carries the signal and the rest is memorised noise" be
    tested rather than assumed: a prefix always contains the top direction, so
    only the complementary bands can falsify it.
    """
    full = max(t.shape[0] for n, t in tensors.items() if ".lora_A." in n)
    balanced = truncate_lora_rank(tensors, full)
    sliced = {}
    for a_name in sorted(n for n in balanced if ".lora_A." in n):
        b_name = a_name.replace(".lora_A.", ".lora_B.")
        sliced[a_name] = balanced[a_name][low:high].contiguous()
        sliced[b_name] = balanced[b_name][:, low:high].contiguous()
    return sliced


def slice_energy(tensors: dict[str, torch.Tensor], low: int, high: int) -> float:
    """Share of the update's squared Frobenius norm in a band of the spectrum."""
    whole = update_geometry(tensors)["update_norm"] ** 2
    part = update_geometry(spectral_slice(tensors, low, high))["update_norm"] ** 2
    return part / max(whole, 1e-30)


def spectral_coordinates(
    trained: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    bands: list[tuple[int, int]],
) -> dict[str, Any]:
    """Project a candidate onto disjoint bands of the trained update.

    LoRA products are compared in weight space with small factor Gram matrices.
    The returned residual is therefore the codec change that no scalar
    attenuation of the trained spectral bands can express.
    """
    native = max(t.shape[0] for n, t in trained.items() if ".lora_A." in n)
    covered = [index for low, high in bands for index in range(low, high)]
    if sorted(covered) != list(range(native)) or len(set(covered)) != native:
        raise ValueError("spectral bands must partition the trained rank")
    parts = [spectral_slice(trained, low, high) for low, high in bands]
    candidate_squared = update_geometry(candidate)["update_norm"] ** 2
    trained_squared = update_geometry(trained)["update_norm"] ** 2
    coefficients = []
    projected_squared = 0.0
    total_cross = 0.0
    for part in parts:
        part_squared = update_geometry(part)["update_norm"] ** 2
        cross = 0.0
        for a_name, b_name in lora_pairs(part):
            cross += frobenius_inner(
                part[a_name], part[b_name], candidate[a_name], candidate[b_name]
            )
        coefficient = cross / max(part_squared, 1e-30)
        coefficients.append(coefficient)
        total_cross += cross
        projected_squared += coefficient * coefficient * part_squared
    residual_squared = max(candidate_squared - projected_squared, 0.0)
    return {
        "coefficients": coefficients,
        "scalar_to_full": total_cross / max(trained_squared, 1e-30),
        "candidate_norm": math.sqrt(candidate_squared),
        "projected_norm": math.sqrt(projected_squared),
        "residual_norm": math.sqrt(residual_squared),
        "residual_fraction": math.sqrt(residual_squared / max(candidate_squared, 1e-30)),
    }


def spectral_projection(
    trained: dict[str, torch.Tensor],
    bands: list[tuple[int, int]],
    coefficients: list[float],
) -> dict[str, torch.Tensor]:
    """Apply one scalar coefficient to each trained spectral band."""
    if len(bands) != len(coefficients):
        raise ValueError("every spectral band needs one coefficient")
    native = max(t.shape[0] for n, t in trained.items() if ".lora_A." in n)
    covered = [index for low, high in bands for index in range(low, high)]
    if sorted(covered) != list(range(native)) or len(set(covered)) != native:
        raise ValueError("spectral bands must partition the trained rank")
    balanced = truncate_lora_rank(trained, native)
    projected = {name: value.clone() for name, value in balanced.items()}
    for a_name, b_name in lora_pairs(projected):
        scales = torch.zeros(native, dtype=projected[b_name].dtype)
        for (low, high), coefficient in zip(bands, coefficients, strict=True):
            scales[low:high] = float(coefficient)
        projected[b_name] *= scales.to(projected[b_name].device)[None, :]
    return projected


def functional_spectrum_predictions(
    rows: list[dict[str, Any]],
    *,
    base_calibration: float,
    full_calibration: float,
    band_marginals: list[float],
) -> list[dict[str, Any]]:
    """Predict test-set NLL gains from calibration-only spectral measurements."""
    full_gain = base_calibration - full_calibration
    predictions = []
    for row in rows:
        coefficients = row["geometry"]["coefficients"]
        signed_gain = full_gain + sum(
            (coefficient - 1.0) * marginal
            for coefficient, marginal in zip(
                coefficients, band_marginals, strict=True
            )
        )
        scalar = float(row["geometry"]["scalar_to_full"])
        energy_gain = scalar * full_gain
        codec_gain = signed_gain + (
            float(row["calibration_gain"])
            - float(row["projection_calibration_gain"])
        )
        predictions.append(
            {
                **row,
                "predicted_gain": {
                    "energy": energy_gain,
                    "signed": signed_gain,
                    "codec_aware": codec_gain,
                    "calibration_only": float(row["calibration_gain"]),
                },
            }
        )
    return predictions


def functional_spectrum_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score surface RMSE and byte-budget selection regret."""
    methods = ("energy", "signed", "codec_aware", "calibration_only")
    scores = {}
    for method in methods:
        errors = [
            float(row["predicted_gain"][method]) - float(row["test_gain"])
            for row in rows
        ]
        scores[method] = {
            "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
            "bias": sum(errors) / len(errors),
        }
    regrets = {method: [] for method in methods}
    for budget in sorted({int(row["file_bits"]) for row in rows}):
        eligible = [row for row in rows if int(row["file_bits"]) <= budget]
        oracle = max(float(row["test_gain"]) for row in eligible)
        for method in methods:
            chosen = max(
                eligible,
                key=lambda row: (float(row["predicted_gain"][method]), -int(row["file_bits"])),
            )
            regrets[method].append(oracle - float(chosen["test_gain"]))
    for method in methods:
        scores[method]["mean_selection_regret"] = sum(regrets[method]) / len(regrets[method])
        scores[method]["max_selection_regret"] = max(regrets[method])
    return {"surface_cells": len(rows), "scores": scores}


def run_experiment(name: str, config: dict, out: Path, runs: Path, prepared: Path) -> None:
    seed = int(config["seed"])
    source_run = str(config.get("functional_source_run", "damaged_run"))
    source_lock = read_json(Path(config["source_lock"]))
    if source_run not in source_lock:
        raise KeyError(f"source lock has no run named {source_run!r}")
    run = RunSpec.from_dict(source_lock[source_run])
    data = study_data(config, seed, prepared)
    examples = data["gsm8k"]
    if config.get("smoke"):
        examples = examples[: int(config["smoke_rows"])]
    by_id = {row.example_id: row for row in examples}
    out.mkdir(parents=True, exist_ok=True)
    session = ModelSession.load(run.model)
    try:
        session.attach(run.adapter, run.seed)
        base = {
            name_: value.detach().cpu().float().clone()
            for name_, value in session.model.named_parameters()
            if "lora_A" in name_ or "lora_B" in name_
        }
        trained = torch.load(runs / run.run_id / "raw_channel.pt",
                             map_location="cpu", weights_only=True)
        native = run.adapter.rank
        recovered = pad_lora_rank(
            coded_head(trained, int(config["recovered_bits"]), out / "scratch"), native
        )

        if name == "forcing":
            # Base weights throughout: the point is to reproduce the adapter's
            # effect without an adapter.
            apply_adapter_tensors(session.model, base)
            # The matched control first: a smooth shift in the stopping
            # decision, which is the shape of intervention the account claims.
            for bias in config["marker_bias"]:
                cached(out / f"forcing_bias{bias:g}.json",
                       lambda b=bias: generate(session, run, examples, config, 0, b))
            # The hard floor, kept because it bounds the other side: it shows
            # what happens when a chain is lengthened by an instrument that
            # does not respect where the model wanted to stop.
            for hold in config["hold_tokens"]:
                cached(out / f"forcing_hold{hold}.json",
                       lambda h=hold: generate(session, run, examples, config, h))
            # The adapter itself, unforced, as the target to reproduce.
            apply_adapter_tensors(session.model, recovered)
            cached(out / "forcing_adapter.json",
                   lambda: generate(session, run, examples, config, 0))
            return

        if name == "bands":
            layers = sorted({
                module_layer_index(n) for n in recovered if module_layer_index(n) is not None
            })
            width = max(1, (max(layers) + 1) // int(config["band_count"]))
            for index in range(int(config["band_count"])):
                low, high = index * width, (index + 1) * width - 1
                if index == int(config["band_count"]) - 1:
                    high = max(layers)
                apply_adapter_tensors(session.model, band_update(recovered, low, high))
                cached(out / f"band_{low:02d}_{high:02d}.json",
                       lambda: generate(session, run, examples, config, 0))
            apply_adapter_tensors(session.model, recovered)
            cached(out / "band_all.json",
                   lambda: generate(session, run, examples, config, 0))
            apply_adapter_tensors(session.model, base)
            cached(out / "band_none.json",
                   lambda: generate(session, run, examples, config, 0))
            return

        if name == "style":
            # No adapter anywhere in this experiment: the question is whether
            # the register the adapter writes in can be induced in context, and
            # an attached update would confound that.
            apply_adapter_tensors(session.model, base)
            # The terse-register exemplars have to come from GSM8K itself, and
            # the only GSM8K rows here are the test rows -- so the rows used as
            # exemplars are held out of scoring for *every* arm, including the
            # ones that do not use them. Otherwise the arms would be compared
            # on different question sets.
            count = int(config["exemplar_count"])
            held, scored = examples[:count], examples[count:]
            cached(out / "style_none.json",
                   lambda: generate(session, run, scored, config, 0, 0.0, ""))
            # MetaMath aligned rows: the register the adapter was fitted to.
            metamath = exemplar_block(
                data["aligned"], count,
                config.get("answer_marker", "The answer is:"),
            )
            cached(out / "style_metamath.json",
                   lambda: generate(session, run, scored, config, 0, 0.0, metamath))
            # GSM8K's own gold solutions: a terser register, same number of
            # exemplars, so whatever in-context learning the exemplars buy is
            # common to both arms and only the register differs.
            gsm = exemplar_block(held, count, "####")
            cached(out / "style_gsm8k.json",
                   lambda: generate(session, run, scored, config, 0, 0.0, gsm))
            return

        if name == "style_matched":
            # The register test with the problem distribution held fixed.
            #
            # The first attempt drew its verbose exemplars from MetaMathQA and
            # its terse ones from GSM8K, so register and problem distribution
            # moved together and the contrast could not separate them. Here both
            # registers describe the *same* three GSM8K problems: the terse
            # solutions are GSM8K's own, and the verbose ones are what the
            # adapter itself wrote for those problems. Neither is written by
            # hand, so no third style enters through the author.
            apply_adapter_tensors(session.model, base)
            source = out / "forcing_adapter.jsonl"
            if not source.exists():
                raise FileNotFoundError(
                    f"{source} is needed for the verbose exemplars; run the"
                    " forcing experiment first"
                )
            written = {
                json.loads(line)["example_id"]: json.loads(line)
                for line in source.read_text().splitlines() if line
            }
            # Exemplars have to be correct in both registers or the contrast is
            # between a good demonstration and a bad one. The gold solution is
            # correct by construction; the adapter's is not, so it is screened.
            pick = [
                row for row in examples
                if written.get(row.example_id, {}).get("correct")
            ][: int(config["exemplar_count"])]
            if len(pick) < int(config["exemplar_count"]):
                raise ValueError("too few correctly answered exemplar problems")
            chosen = {row.example_id for row in pick}
            scored = [row for row in examples if row.example_id not in chosen]

            def block_of(bodies: dict[str, str]) -> str:
                parts = []
                for row in pick:
                    question = row.prompt.split("Question:")[-1]
                    question = question.split("Answer:")[0].strip()
                    parts.append(f"Question: {question}\nAnswer: {bodies[row.example_id].strip()}")
                return "\n\n".join(parts) + "\n\n"

            terse = block_of({row.example_id: row.response for row in pick})
            verbose = block_of(
                {row.example_id: written[row.example_id]["response"] for row in pick}
            )
            cached(out / "matched_none.json",
                   lambda: generate(session, run, scored, config, 0, 0.0, ""))
            cached(out / "matched_terse.json",
                   lambda: generate(session, run, scored, config, 0, 0.0, terse))
            cached(out / "matched_verbose.json",
                   lambda: generate(session, run, scored, config, 0, 0.0, verbose))
            # And the adapter itself on the same scored subset, so the target
            # the prompt is trying to reproduce is measured on the same rows.
            apply_adapter_tensors(session.model, recovered)
            cached(out / "matched_adapter.json",
                   lambda: generate(session, run, scored, config, 0, 0.0, ""))
            return

        if name == "segment":
            # The intervention this experiment applies is derived from the
            # measurement, not chosen: the readout says the update's largest
            # single effect by far is turning sentence-final "." into ".\n",
            # and this puts exactly that shift -- and the next k-1 largest --
            # back onto the base model to see how much of the gain follows.
            apply_adapter_tensors(session.model, base)
            measured = read_json(out / "readout.json")
            if measured is None:
                raise FileNotFoundError(
                    f"{out / 'readout.json'} is needed; run the readout first"
                )
            ranked = sorted(
                measured["promoted_by_mass"] + measured["suppressed_by_mass"],
                key=lambda row: abs(row["delta"]), reverse=True,
            )
            seen, entries = set(), []
            for row in ranked:
                token = row["token"]
                index = session.tokenizer.convert_tokens_to_ids(token)
                if index is None or index in seen:
                    continue
                if row["base_prob"] <= 0 or row["update_prob"] <= 0:
                    continue
                seen.add(index)
                entries.append(
                    (int(index),
                     float(torch.log(torch.tensor(
                         row["update_prob"] / row["base_prob"]))))
                )
            for k in config["segment_top_k"]:
                k = int(k)
                if k > len(entries):
                    continue
                ids = [i for i, _ in entries[:k]]
                shifts = [v for _, v in entries[:k]]
                cached(out / f"segment_top{k}.json",
                       lambda i=ids, v=shifts: generate(
                           session, run, examples, config, 0, 0.0, "",
                           MeasuredBias(i, v)))
            return

        if name == "segment_scaled":
            # The unscaled derived intervention moved nothing, and the reason is
            # in how it was derived: the readout averages over every chain
            # position, but a sentence only ends at a small fraction of them, so
            # the average shift is far smaller than the shift at the positions
            # that matter. Applying it uniformly is therefore much too weak
            # exactly where it would act.
            #
            # Scaling the same two-token shift asks the question the weak
            # version could not: is there *any* strength of pure line-breaking
            # pressure that reproduces the adapter's gain? A multiplier of ten
            # makes the newline all but mandatory at a sentence end, so if the
            # account is right the curve should rise before it does.
            apply_adapter_tensors(session.model, base)
            measured = read_json(out / "readout.json")
            if measured is None:
                raise FileNotFoundError("run the readout first")
            ranked = sorted(
                measured["promoted_by_mass"] + measured["suppressed_by_mass"],
                key=lambda row: abs(row["delta"]), reverse=True,
            )
            entries, seen = [], set()
            for row in ranked:
                index = session.tokenizer.convert_tokens_to_ids(row["token"])
                if index is None or index in seen or row["base_prob"] <= 0:
                    continue
                if row["update_prob"] <= 0:
                    continue
                seen.add(index)
                entries.append((int(index), float(
                    torch.log(torch.tensor(row["update_prob"] / row["base_prob"])))))
            width = int(config["segment_scale_top_k"])
            ids = [i for i, _ in entries[:width]]
            shifts = [v for _, v in entries[:width]]
            print("scaling shifts:", [
                (session.tokenizer.decode([i]), round(v, 3))
                for i, v in zip(ids, shifts)
            ], flush=True)
            for scale in config["segment_scale"]:
                factor = float(scale)
                cached(out / f"segscale_{factor:g}.json",
                       lambda f=factor: generate(
                           session, run, examples, config, 0, 0.0, "",
                           MeasuredBias(ids, [v * f for v in shifts])))
            return

        if name == "spectral":
            # Which part of the damaged update carries the recovery, and which
            # carries the damage? Compression keeps the top of the spectrum, so
            # "compression denoises" predicts the top band recovers and the
            # tail does not. Bands are evaluated on their own, padded back to
            # the adapter's rank so every one is applied through one model.
            native = run.adapter.rank
            bands = [tuple(int(v) for v in pair) for pair in config["spectral_bands"]]
            energies = {f"{lo}_{hi}": slice_energy(trained, lo, hi) for lo, hi in bands}
            write_json(out / "spectral_energy.json", energies)
            print("energy share by band:",
                  {k: round(v, 4) for k, v in energies.items()}, flush=True)
            for lo, hi in bands:
                part = spectral_slice(trained, lo, hi)
                for bits in config["spectral_bits"]:
                    bits = int(bits)
                    key = f"spec_{lo}_{hi}_b{bits}"
                    if bits >= 16:
                        applied = pad_lora_rank(part, native)
                    else:
                        path = out / "scratch" / f"{key}.fqcb"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        encode_tensor_map(part, path, bits)
                        _, decoded = decode_adapter_tensor_map(path)
                        path.unlink(missing_ok=True)
                        applied = pad_lora_rank(decoded, native)
                    apply_adapter_tensors(session.model, applied)
                    cached(out / f"{key}.json",
                           lambda: generate(session, run, examples, config, 0))
            return

        if name == "functional_spectrum":
            bands = [
                tuple(int(value) for value in pair)
                for pair in config["functional_spectral_bands"]
            ]
            calibration = data["calibration"][: int(config["functional_calibration_rows"])]
            test = data["gsm8k"][: int(config["functional_test_rows"])]
            if config.get("smoke"):
                calibration = calibration[: int(config["smoke_rows"])]
                test = test[: int(config["smoke_rows"])]
            if not calibration or not test:
                raise ValueError("functional spectrum needs calibration and test rows")

            def measure(label: str, tensors: dict[str, torch.Tensor], rows: list) -> float:
                path = out / "functional_spectrum" / f"{label}.json"
                found = read_json(path)
                if found is None:
                    apply_adapter_tensors(session.model, pad_lora_rank(tensors, native))
                    found = causal_nll(
                        session.model,
                        session.tokenizer,
                        rows,
                        run.model,
                        run.training.max_length,
                        int(config["nll_batch_size"]),
                        label_span=str(config.get("functional_label_span", "all")),
                        answer_marker=config.get("answer_marker"),
                    )
                    write_json(path, found)
                return float(found["bits_per_token"])

            base_calibration = measure("base_calibration", base, calibration)
            base_test = measure("base_test", base, test)
            full_calibration = measure("full_calibration", trained, calibration)
            full_test = measure("full_test", trained, test)
            band_rows = []
            marginals = []
            for index, (low, high) in enumerate(bands):
                band = spectral_slice(trained, low, high)
                coefficients = [1.0] * len(bands)
                coefficients[index] = 0.0
                complement = spectral_projection(trained, bands, coefficients)
                band_nll = measure(f"band_{low}_{high}_calibration", band, calibration)
                complement_nll = measure(
                    f"without_{low}_{high}_calibration", complement, calibration
                )
                marginal = complement_nll - full_calibration
                marginals.append(marginal)
                band_rows.append(
                    {
                        "band": [low, high],
                        "energy_fraction": slice_energy(trained, low, high),
                        "standalone_gain": base_calibration - band_nll,
                        "ablation_marginal": marginal,
                    }
                )

            surface = []
            scratch = out / "functional_spectrum" / "scratch"
            scratch.mkdir(parents=True, exist_ok=True)
            for rank in config["functional_ranks"]:
                reduced = truncate_lora_rank(trained, int(rank))
                for bits in config["functional_bits"]:
                    key = f"r{int(rank)}_b{int(bits)}"
                    path = scratch / f"{key}.fqcb"
                    storage = encode_tensor_map(reduced, path, int(bits))
                    _, candidate = decode_adapter_tensor_map(path)
                    geometry = spectral_coordinates(trained, candidate, bands)
                    projection = spectral_projection(
                        trained, bands, geometry["coefficients"]
                    )
                    candidate_calibration = measure(
                        f"{key}_calibration", candidate, calibration
                    )
                    projection_calibration = measure(
                        f"{key}_projection_calibration", projection, calibration
                    )
                    candidate_test = measure(f"{key}_test", candidate, test)
                    surface.append(
                        {
                            "key": key,
                            "rank": int(rank),
                            "bits": int(bits),
                            "file_bits": int(storage["file_bits"]),
                            "geometry": geometry,
                            "calibration_gain": base_calibration - candidate_calibration,
                            "projection_calibration_gain": (
                                base_calibration - projection_calibration
                            ),
                            "test_gain": base_test - candidate_test,
                        }
                    )
                    path.unlink(missing_ok=True)
            predictions = functional_spectrum_predictions(
                surface,
                base_calibration=base_calibration,
                full_calibration=full_calibration,
                band_marginals=marginals,
            )
            summary = {
                "status": "complete",
                "source_run": source_run,
                "run_id": run.run_id,
                "utility": "reduction in held-out bits per scored token",
                "calibration_rows": len(calibration),
                "test_rows": len(test),
                "base_calibration_bits_per_token": base_calibration,
                "base_test_bits_per_token": base_test,
                "full_calibration_bits_per_token": full_calibration,
                "full_test_bits_per_token": full_test,
                "bands": band_rows,
                "surface": predictions,
                **functional_spectrum_summary(predictions),
            }
            write_json(out / "functional_spectrum" / "summary.json", summary)
            print(json.dumps({"bands": band_rows, "scores": summary["scores"]}, indent=2))
            return

        if name == "per_row_nll":
            # The companion study reports corruption-preference gaps without
            # intervals, because only the aggregate bits/token was stored. This
            # recomputes them one row at a time so the paired difference can be
            # bootstrapped over rows, which is the unit the claim is about.
            probes = nll_examples(
                study_data(config, int(config["seed"]), prepared), config)
            settings = {
                "base": base,
                "raw": trained,
                "rank1_b1": recovered,
                "norm_matched_b1": None,
                "scale_head_0p5": None,
            }
            head = truncate_lora_rank(trained, 1)
            target = update_geometry(recovered)["update_norm"]
            current = update_geometry(pad_lora_rank(head, native))["update_norm"]
            factor = target / max(current, 1e-30)
            settings["norm_matched_b1"] = pad_lora_rank(
                {n: (v * factor if ".lora_B." in n else v) for n, v in head.items()},
                native)
            settings["scale_head_0p5"] = pad_lora_rank(
                {n: (v * 0.5 if ".lora_B." in n else v) for n, v in head.items()},
                native)
            out.mkdir(parents=True, exist_ok=True)
            for label, tensors in settings.items():
                path = out / f"rownll_{label}.json"
                if path.exists():
                    continue
                apply_adapter_tensors(session.model, tensors)
                record = {}
                for split in NLL_SPLITS:
                    values = []
                    for example in probes[split]:
                        metrics = causal_nll(
                            session.model, session.tokenizer, [example], run.model,
                            run.training.max_length, 1, label_span="reasoning",
                            answer_marker=config["answer_marker"])
                        values.append(float(metrics["bits_per_token"]))
                    record[split] = values
                record["example_ids"] = [e.example_id for e in probes["permuted"]]
                write_json(path, record)
                gap = (sum(record["aligned"]) - sum(record["permuted"])) / len(
                    record["permuted"])
                print(f"{label}: corruption preference {gap:+.4f} bits/token "
                      f"over {len(record['permuted'])} rows", flush=True)
            return

        if name == "coefficient":
            # What does the direction read? A rank-one update writes b·(a·x), so
            # every token has one scalar coefficient per module: how strongly the
            # direction fires there. If the update is a step-boundary detector,
            # that coefficient should be larger where a step is about to end than
            # where it is not -- measured on the base model's own chains, so the
            # text is identical across positions being compared.
            apply_adapter_tensors(session.model, recovered)
            tokenizer = session.tokenizer
            marker = {
                i for i in marker_token_ids(tokenizer) if i != tokenizer.eos_token_id
            }
            # A sentence end is the period token itself; the question is whether
            # the direction fires on the token *before* the model must decide.
            ends = {
                index for token, index in tokenizer.get_vocab().items()
                if tokenizer.convert_tokens_to_string([token]).strip() in {".", ".\n"}
                or tokenizer.convert_tokens_to_string([token]) in {".", ".\n", ". "}
            }
            captured: dict[str, torch.Tensor] = {}
            handles = []

            def hook(name_: str):
                def fn(_module, _inputs, output):
                    # (batch, seq, rank); the padded update keeps only row 0.
                    captured[name_] = output[0, :, 0].detach().float().cpu()
                return fn

            for module_name, module in session.model.named_modules():
                if module_name.endswith("lora_A.default"):
                    handles.append(module.register_forward_hook(hook(module_name)))
            records = [
                json.loads(line) for line in
                (out / "pressure_base_chains.jsonl").read_text().splitlines() if line
            ]
            rows = records[: int(config["pressure_rows"])]
            totals: dict[str, dict[str, list[float]]] = {}
            try:
                for row in rows:
                    example = by_id[row["example_id"]]
                    chain = row["response"].split("####")[0]
                    if len(chain.split()) < 8:
                        continue
                    prompt_ids = tokenizer(
                        render_prompt(tokenizer, example.prompt, run.model),
                        return_tensors="pt")["input_ids"]
                    chain_ids = tokenizer(chain, return_tensors="pt",
                                          add_special_tokens=False)["input_ids"]
                    ids = torch.cat([prompt_ids, chain_ids], dim=1)
                    if ids.shape[1] > int(config["pressure_max_length"]):
                        continue
                    captured.clear()
                    with torch.no_grad(), inference_autocast(
                        session.model, model_device(session.model)
                    ):
                        session.model(input_ids=ids.to(model_device(session.model)))
                    width = int(prompt_ids.shape[1])
                    flat = ids[0].tolist()
                    # Position t is "at a step boundary" when the token it must
                    # predict next is a sentence end.
                    labels = [
                        int(flat[t + 1] in ends) if t + 1 < len(flat) else 0
                        for t in range(len(flat))
                    ]
                    for module_name, values in captured.items():
                        bucket = totals.setdefault(module_name, {"at": [], "away": []})
                        for t in range(width, len(flat) - 1):
                            # Signed, not absolute. The update writes b·(a·x), so
                            # the sign of the coefficient decides which way the
                            # residual stream is pushed; a direction that writes
                            # +b at a boundary and -b elsewhere is perfectly
                            # selective and has an absolute ratio of exactly one.
                            bucket["at" if labels[t] else "away"].append(
                                float(values[t]))
            finally:
                for handle in handles:
                    handle.remove()
            summary = {}
            for module_name, bucket in totals.items():
                if not bucket["at"] or not bucket["away"]:
                    continue
                at = torch.tensor(bucket["at"])
                away = torch.tensor(bucket["away"])
                # A standardised difference, so modules with different coefficient
                # scales can be compared and a shift is read against the spread
                # it has to stand out from.
                pooled = torch.cat([at, away]).std().clamp_min(1e-9)
                summary[module_name] = {
                    "layer": module_layer_index(module_name),
                    "mean_at_boundary": float(at.mean()),
                    "mean_elsewhere": float(away.mean()),
                    "abs_at_boundary": float(at.abs().mean()),
                    "abs_elsewhere": float(away.abs().mean()),
                    "standardised_gap": float((at.mean() - away.mean()) / pooled),
                    "abs_ratio": float(at.abs().mean() / away.abs().mean().clamp_min(1e-9)),
                    "n_at": int(at.numel()), "n_away": int(away.numel()),
                }
            write_json(out / "coefficient.json", summary)
            gaps = sorted((v["standardised_gap"], k) for k, v in summary.items())
            absr = sorted((v["abs_ratio"], k) for k, v in summary.items())
            print(f"{len(summary)} modules, coefficient at a step boundary "
                  f"against elsewhere")
            if gaps:
                print(f"  standardised gap  median {gaps[len(gaps)//2][0]:+.3f}  "
                      f"max {gaps[-1][0]:+.3f} ({gaps[-1][1].split('model.')[-1]})  "
                      f"min {gaps[0][0]:+.3f} ({gaps[0][1].split('model.')[-1]})")
                strong = [g for g in gaps if abs(g[0]) > 0.2]
                print(f"  modules with |gap| > 0.2 std: {len(strong)} of {len(gaps)}")
            if absr:
                print(f"  magnitude ratio   median {absr[len(absr)//2][0]:.3f}")
            return

        if name == "readout":
            apply_adapter_tensors(session.model, base)
            cached(out / "pressure_base_chains.json",
                   lambda: generate(session, run, examples, config, 0))
            chains = [json.loads(l) for l in
                      (out / "pressure_base_chains.jsonl").read_text().splitlines() if l]
            result = token_readout(
                session, run, chains, by_id, recovered, base, config
            )
            write_json(out / "readout.json", result)
            for field in ("promoted_by_mass", "suppressed_by_mass",
                          "promoted_by_ratio", "suppressed_by_ratio"):
                print(f"{field}:",
                      ", ".join(repr(r["text"]) for r in result[field][:20]))
            return

        if name == "pressure":
            apply_adapter_tensors(session.model, base)
            records = cached(out / "pressure_base_chains.json",
                             lambda: generate(session, run, examples, config, 0))
            chains = [json.loads(l) for l in
                      (out / "pressure_base_chains.jsonl").read_text().splitlines() if l]
            result = termination_pressure(
                session, run, chains, by_id, recovered, base, config
            )
            write_json(out / "pressure.json", result)
            print(json.dumps({k: v for k, v in result.items() if k != "by_decile"},
                             indent=2))
            return
        raise ValueError(f"unknown experiment {name!r}")
    finally:
        session.unload()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/upnquick/cot_verbosity.yaml"))
    parser.add_argument("--out", type=Path,
                        default=Path(".cache/reports/cot_verbosity_upnquick"))
    parser.add_argument("--runs", type=Path, default=Path(".cache/runs"))
    parser.add_argument("--prepared", type=Path, default=Path(".cache/prepared"))
    parser.add_argument("--smoke", action="store_true",
                        help="restrict configured measurements to smoke_rows")
    parser.add_argument("--source-run", choices=["damaged_run", "clean_run"],
                        help="select an adapter arm from the source lock")
    parser.add_argument("--experiment", required=True,
                        choices=["forcing", "bands", "pressure", "style", "style_matched",
                                 "readout", "segment", "segment_scaled",
                                 "spectral", "functional_spectrum", "per_row_nll",
                                 "coefficient"])
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.smoke:
        config["smoke"] = True
    if args.source_run:
        config["functional_source_run"] = args.source_run
    started = time.monotonic()
    run_experiment(args.experiment, config, args.out, args.runs, args.prepared)
    print(f"{args.experiment} finished in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
