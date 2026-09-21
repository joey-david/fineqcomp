"""Is the recovery a denoising, or is it shrinkage?

A LoRA fine-tuned on MetaMathQA rows whose worked explanation was taken from a
*different* problem loses most of its GSM8K accuracy. Truncating that finished
rank-16 adapter to rank one and re-coding it at one bit per value puts the
accuracy back above the base model. `matched_budget` established the behavioural
half of that -- the recovery is not reproduced by training small, and at matched
`rank x bits` the coarse cell wins -- and said plainly what it could not rule
out: quantization shrinks an update as well as coarsening it, so a plain
rescaling of the same damaged update could in principle reproduce the pattern.

This module runs that control, and two others that go with it. The three
questions are separate and are answered by separate measurements:

1. **Shrinkage.** Put the one-bit rank-one adapter on the same axis as the
   family `alpha * dW_rank1` at full precision, alpha swept from 0 (the base
   model) to 1 (the fp16 rank-one truncation), together with the alpha that
   matches the *decoded* one-bit update's Frobenius norm. If the coded point
   lies on the alpha curve, the recovery is shrinkage and nothing more. If it
   lies off it, coarse coding is doing something a scalar cannot.

2. **What recovered, and what was forgotten.** Task accuracy alone cannot
   answer question 1, because every arm that turns the adapter down approaches
   the base model and the base model is already good at GSM8K: accuracy
   saturates where the interesting differences are. So every condition is
   scored on two axes that move independently -- GSM8K exact match for the
   clean task, and the model's appetite for the corrupted explanations it was
   trained on, measured as reasoning-span NLL on the permuted training rows
   against NLL on the *aligned* counterparts of those same rows. The gap
   between the two is the quantity that says whether the mismatched rationales
   were forgotten or merely turned down, and a behavioural read of the same
   thing comes free from the stored generations: a damaged adapter opens most
   of its answers with one memorised off-topic rationale, and the share of
   answers in that modal opening is a direct count of the corrupted chain of
   thought being reproduced.

3. **Where it lands.** A rank-16 LoRA trained on the *aligned* rationales --
   identical in every other field -- is the solution the damaged adapter would
   have reached without the corruption. Scoring it alongside the rest puts a
   reference point on both axes, and the report measures how close each
   condition gets to it functionally: agreement of extracted answers item by
   item, and the cosine between the two updates in weight space, computed from
   the small factor Gram matrices rather than any dense weight.

Nothing here trains except the aligned arm, which is an ordinary campaign
(`configs/upnquick/aligned_cot.yaml`). Corpora, splits and scoring are imported
from `behavioral_trajectory` so the numbers land on the axis the recorded
results used, and the damaged adapter is read from the same run the matched
budget grid swept.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import torch
import yaml

from fineqcomp.adapters import adapter_tensors, apply_adapter_tensors
from fineqcomp.artifacts import claim_run, read_json, write_json
from fineqcomp.behavioral_trajectory import paired_interval, score, sha256, study_data
from fineqcomp.campaign import expand_campaign
from fineqcomp.codec import (
    ALLOWED_BITS,
    decode_adapter_tensor_map,
    encode_tensor_map,
    pad_lora_rank,
    truncate_lora_rank,
)
from fineqcomp.config import RunSpec, load_campaign
from fineqcomp.modeling import ModelSession
from fineqcomp.training import causal_nll

# The two likelihood probes. Both run on the same prompts with the same span
# mask; only the rationale text differs, so their difference isolates the
# corruption rather than the prompt distribution or the answer format.
NLL_SPLITS = ("permuted", "aligned")


def scale_label(scale: float) -> str:
    return f"{scale:g}".replace(".", "p").replace("-", "m")


def lora_pairs(tensors: dict[str, torch.Tensor]) -> list[tuple[str, str]]:
    """Every (lora_A, lora_B) name pair in a tensor map, in a fixed order."""
    pairs = []
    for a_name in sorted(name for name in tensors if ".lora_A." in name):
        b_name = a_name.replace(".lora_A.", ".lora_B.")
        if b_name not in tensors:
            raise KeyError(f"{a_name} has no matching lora_B tensor")
        pairs.append((a_name, b_name))
    if not pairs:
        raise ValueError("tensor map holds no LoRA factor pairs")
    return pairs


def frobenius_inner(
    a_left: torch.Tensor,
    b_left: torch.Tensor,
    a_right: torch.Tensor,
    b_right: torch.Tensor,
) -> float:
    """<B_l A_l, B_r A_r>_F without ever forming a dense weight update.

    tr((B_l A_l)^T B_r A_r) = tr((B_l^T B_r)(A_r A_l^T)), and both factors are
    r x r. The dense update for one `gate_proj` is 68M values; this is two
    matrices of at most 16 x 16.
    """
    left = b_left.T @ b_right
    right = a_right @ a_left.T
    return float((left * right.T).sum())


def update_geometry(
    tensors: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    """Frobenius norm of an update, and its cosine against a reference update.

    Both are summed over target modules, which treats the concatenation of every
    module's update as one vector -- the same object a single scalar `alpha`
    rescales, so the norm reported here is exactly the quantity a shrinkage
    account has to move.

    The runtime `alpha / rank` scaling is a constant common to every condition
    here (all of them are padded back to the same rank-16 adapter spec and
    applied through it), so it cancels in the cosine and rescales every norm
    by the same factor. Norms are therefore comparable across conditions and
    are not absolute weight-space distances.
    """
    squared = 0.0
    cross = 0.0
    reference_squared = 0.0
    for a_name, b_name in lora_pairs(tensors):
        a, b = tensors[a_name], tensors[b_name]
        squared += frobenius_inner(a, b, a, b)
        if reference is None:
            continue
        ref_a, ref_b = reference[a_name], reference[b_name]
        cross += frobenius_inner(a, b, ref_a, ref_b)
        reference_squared += frobenius_inner(ref_a, ref_b, ref_a, ref_b)
    geometry = {"update_norm": squared ** 0.5}
    if reference is not None:
        denominator = (squared * reference_squared) ** 0.5
        geometry["cosine_to_clean"] = cross / denominator if denominator > 0 else 0.0
    return geometry


def declare_conditions(config: dict, native_rank: int) -> list[dict[str, Any]]:
    """The full condition list, frozen before any GPU is claimed.

    Declaring the conditions rather than generating them inside the sweep is
    what lets `prepare` write them into the lock, so a condition added later is
    a lock mismatch rather than a quietly different comparison.
    """
    conditions: list[dict[str, Any]] = [
        {"key": "base", "family": "reference", "kind": "base"},
        {"key": "raw", "family": "reference", "kind": "raw", "rank": native_rank},
    ]
    for bits in config["compression_bits"]:
        if int(bits) not in ALLOWED_BITS or int(bits) < 1:
            raise ValueError(f"bits must be one of {sorted(ALLOWED_BITS - {0})}")
        conditions.append(
            {
                "key": f"rank1_b{int(bits)}",
                "family": "compression",
                "kind": "coded",
                "rank": 1,
                "bits": int(bits),
            }
        )
    # The two shrinkage families differ in what is being turned down: the
    # rank-one head alone (the thing the coded arm keeps) or the whole trained
    # update (the thing the coded arm starts from). Only the first is a like-
    # for-like control on the coded point; the second is there so a reader can
    # see that the rank truncation is not itself doing the work.
    # The two sweeps are declared separately because they are not the same
    # experiment: the head sweep is the control the coded arm is read against
    # and needs resolution, while the full sweep only has to show that rank
    # truncation is not what recovers, and three points do that.
    for source, family, key in (
        ("head", "shrinkage_head", "scale_sweep_head"),
        ("full", "shrinkage_full", "scale_sweep_full"),
    ):
        rank = 1 if source == "head" else native_rank
        for scale in config[key]:
            conditions.append(
                {
                    "key": f"scale_{source}_{scale_label(float(scale))}",
                    "family": family,
                    "kind": "scaled",
                    "source": source,
                    "rank": rank,
                    "scale": float(scale),
                }
            )
    # The most informative shrinkage points are not on any round-number grid:
    # they are the alphas that reproduce each coded cell's decoded update norm
    # exactly. One of these per coded cell turns the central comparison from
    # "nearest point on a sweep" into an exact pairing -- a coded update and a
    # scalar rescaling of the same directions with the same norm, differing
    # only in that one has been through a file. If shrinkage is the whole
    # story, each pair agrees on every measure.
    for bits in config["norm_matched_bits"]:
        conditions.append(
            {
                "key": f"norm_matched_b{int(bits)}",
                "family": "shrinkage_matched",
                "kind": "norm_matched",
                "source": "head",
                "rank": 1,
                "target_bits": int(bits),
            }
        )
    if config.get("clean_arm"):
        conditions.append(
            {"key": "clean_raw", "family": "clean", "kind": "clean_raw",
             "rank": native_rank}
        )
        for bits in config["clean_compression_bits"]:
            conditions.append(
                {
                    "key": f"clean_rank1_b{int(bits)}",
                    "family": "clean",
                    "kind": "clean_coded",
                    "rank": 1,
                    "bits": int(bits),
                }
            )
    keys = [condition["key"] for condition in conditions]
    if len(set(keys)) != len(keys):
        raise ValueError("condition keys collide")
    return conditions


def code_through_file(
    tensors: dict[str, torch.Tensor], rank: int, bits: int, path: Path
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Write the real coded file and return what the decoder reconstructs.

    Scoring the decoded tensors rather than the pre-encode ones is the point of
    the comparison: the claim is about what survives a file of this size.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    storage = encode_tensor_map(truncate_lora_rank(tensors, rank), path, bits)
    _, decoded = decode_adapter_tensor_map(path)
    return storage, decoded


def decoded_head_norm(tensors: dict[str, torch.Tensor], bits: int) -> float:
    """Frobenius norm of the rank-one update after a real encode/decode round."""
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "target.fqcb"
        _, decoded = code_through_file(tensors, 1, bits, path)
    return update_geometry(decoded)["update_norm"]


def materialise(
    condition: dict[str, Any],
    trained: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
    clean: dict[str, torch.Tensor] | None,
    native_rank: int,
    scratch: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build one condition's tensors, padded back to the attached adapter rank.

    Every condition is applied through the same rank-16 adapter object, so a
    rank-one update is padded with zero rows rather than attached as a rank-one
    adapter. That keeps the model, its scaling and its generation path fixed
    across the whole study; only the values of the update change.
    """
    kind = condition["kind"]
    extra: dict[str, Any] = {}
    if kind == "base":
        return base, extra
    if kind == "raw":
        return trained, extra
    if kind == "clean_raw":
        if clean is None:
            raise ValueError("clean arm requested without a clean adapter")
        return clean, extra
    source_map = clean if kind.startswith("clean_") else trained
    if source_map is None:
        raise ValueError("clean arm requested without a clean adapter")
    if kind in ("coded", "clean_coded"):
        storage, decoded = code_through_file(
            source_map, int(condition["rank"]), int(condition["bits"]),
            scratch / f"{condition['key']}.fqcb",
        )
        extra = {
            "file_bits": int(storage["file_bits"]),
            "effective_bits_per_value": float(storage["effective_bits_per_value"]),
        }
        # The coded file is a deterministic function of the raw adapter and the
        # two integers in its name, and keeping the whole set runs to gigabytes.
        (scratch / f"{condition['key']}.fqcb").unlink(missing_ok=True)
        return pad_lora_rank(decoded, native_rank), extra

    # Both remaining kinds are a scalar times the rank-r truncation of the
    # trained update. Scaling `lora_B` alone scales the product, and leaving
    # `lora_A` untouched keeps the update's directions exactly as trained --
    # which is what makes this a shrinkage control rather than a second
    # intervention.
    reduced = truncate_lora_rank(source_map, int(condition["rank"]))
    if kind == "norm_matched":
        target = decoded_head_norm(source_map, int(condition["target_bits"]))
        current = update_geometry(reduced)["update_norm"]
        scale = target / current if current > 0 else 0.0
        extra = {"scale": scale, "norm_target": target}
    elif kind == "scaled":
        scale = float(condition["scale"])
    else:
        raise ValueError(f"unknown condition kind {kind!r}")
    scaled = {
        name: (value * scale if ".lora_B." in name else value)
        for name, value in reduced.items()
    }
    return pad_lora_rank(scaled, native_rank), extra


def arm_run(arm: dict[str, Any], seed: int) -> RunSpec:
    """Resolve one arm's run id by re-expanding the campaign that defines it.

    Run ids are content hashes of the run's own identity, so this recovers the
    id exactly without the run existing yet -- and turns an edited training
    config into a missing adapter rather than a mismatched comparison.
    """
    runs = [
        run
        for run in expand_campaign(load_campaign(arm["config"]))
        if run.study == arm["study"]
        and run.model.key == arm["model"]
        and run.adapter.rank == int(arm["rank"])
        and run.seed == int(seed)
    ]
    if len(runs) != 1:
        raise ValueError(
            f"arm {arm['study']} rank {arm['rank']} seed {seed} expands to"
            f" {len(runs)} runs in {arm['config']}"
        )
    return runs[0]


def prepare(config: dict, out: Path, runs: Path, prepared: Path) -> dict:
    """Freeze the condition list and every input digest before any GPU."""
    seed = int(config["seed"])
    damaged = arm_run(config["damaged_arm"], seed)
    clean = arm_run(config["clean_arm"], seed) if config.get("clean_arm") else None
    native_rank = damaged.adapter.rank
    if clean is not None and clean.adapter.rank != native_rank:
        raise ValueError("the clean arm must match the damaged arm's rank")
    inputs = [Path(config["symbolic_source"]), Path(config["damaged_arm"]["config"])]
    if config.get("clean_arm"):
        inputs.append(Path(config["clean_arm"]["config"]))
    study_data(config, seed, prepared)
    inputs.extend(
        (prepared / "natural" / "cot_math_permuted" / f"seed{seed}").glob("*.jsonl")
    )
    record = {
        "config": config,
        "seed": seed,
        "native_rank": native_rank,
        "damaged_run": damaged.to_dict(),
        "clean_run": clean.to_dict() if clean is not None else None,
        "conditions": declare_conditions(config, native_rank),
        "slug": f"{damaged.model.key}/seed{seed}",
        "inputs": {str(path): sha256(path) for path in sorted(set(inputs))},
        "implementation": {
            str(path): sha256(path)
            for path in sorted(Path("src/fineqcomp").glob("*.py"))
        },
    }
    # Compare what would be written, not what is in memory. `RunSpec.to_dict`
    # returns tuples where JSON round-trips to lists, so an unchanged study
    # re-prepared from the same config differs from its own stored lock unless
    # both sides go through the serializer first. Re-preparing an unchanged
    # study has to be a no-op: the driver does it at the start of every night.
    record = json.loads(json.dumps(record, sort_keys=True, default=list))
    previous = read_json(out / "lock.json")
    if previous is not None and previous != record:
        raise ValueError("study changed: use a new output directory")
    write_json(out / "lock.json", record)
    return record


def check(lock: dict, runs: Path) -> dict:
    """Prove every adapter this study reads exists before a GPU hour is spent."""
    def present(label: str) -> bool:
        run = lock.get(label)
        return run is not None and (
            runs / run["run_id"] / "raw_channel.pt"
        ).is_file()

    # Only the damaged adapter is required. The aligned arm adds the reference
    # point; without it the shrinkage control is still a complete experiment,
    # so a missing clean adapter is reported rather than fatal.
    missing = [] if present("damaged_run") else ["damaged_run"]
    optional = (
        [] if lock.get("clean_run") is None or present("clean_run")
        else ["clean_run"]
    )
    return {
        "conditions": len(lock["conditions"]),
        "missing_adapters": missing,
        "missing_optional": optional,
        "passed": not missing,
    }


def nll_probe(
    session: ModelSession,
    run: RunSpec,
    examples: list,
    config: dict,
    path: Path,
) -> dict:
    """Reasoning-span NLL on one rationale variant, cached on its own path.

    The span matters. Scoring the whole response mixes the corrupted working
    with the answer line, and the answer line is identical between the permuted
    and aligned variants of a row -- it would dilute exactly the difference the
    probe is for. `reasoning` scores the working and nothing else.
    """
    cached = read_json(path)
    if cached is not None:
        return cached
    metrics = causal_nll(
        session.model,
        session.tokenizer,
        examples,
        run.model,
        run.training.max_length,
        config["nll_batch_size"],
        label_span="reasoning",
        answer_marker=config["answer_marker"],
    )
    write_json(path, metrics)
    print(f"{path}: bits_per_token={metrics['bits_per_token']:.4f}", flush=True)
    return metrics


def nll_examples(data: dict, config: dict) -> dict[str, list]:
    """The paired rationale probes: same prompts, same answers, both rationales.

    `study_data` returns the permuted training rows and their aligned
    counterparts in the same order, so slicing both to the same count keeps the
    pairing exact and makes the two NLLs differ only in the rationale text.
    """
    rows = int(config["nll_rows"])
    if config.get("smoke"):
        # A smoke run has already sliced the corpus to a handful of rows; the
        # pairing still has to hold exactly, only the count gives way.
        rows = min(rows, len(data["permuted"]), len(data["aligned"]))
    permuted, aligned = data["permuted"][:rows], data["aligned"][:rows]
    if len(permuted) != len(aligned) or len(permuted) < rows:
        raise ValueError("permuted and aligned probes are not paired")
    if [x.example_id for x in permuted] != [x.example_id for x in aligned]:
        raise ValueError("permuted and aligned probes are not row-aligned")
    return {"permuted": permuted, "aligned": aligned}


def sweep(lock: dict, config: dict, out: Path, runs: Path, prepared: Path,
          only: str | None = None) -> None:
    """Score every condition on GSM8K and both likelihood probes, resumably.

    Every measurement is cached on its own path, so a job cut short by walltime
    resumes at the next unmeasured condition rather than repeating the pass.
    """
    root = out / lock["slug"]
    run = RunSpec.from_dict(lock["damaged_run"])
    native_rank = int(lock["native_rank"])
    with claim_run(root) as acquired:
        if not acquired:
            raise RuntimeError(f"study already running: {root}")
        data = study_data(config, int(lock["seed"]), prepared)
        if config.get("smoke"):
            data = {key: value[:4] for key, value in data.items()}
        probes = nll_examples(data, config)
        session = ModelSession.load(run.model)
        try:
            session.attach(run.adapter, run.seed)
            # The attached adapter is zero-initialised, so reading its tensors
            # here is how the base model enters the study -- on the same
            # generation path as every other condition rather than as a
            # separately loaded model.
            base = adapter_tensors(session.model, "full_lora")
            trained = torch.load(
                runs / run.run_id / "raw_channel.pt", map_location="cpu",
                weights_only=True,
            )
            # The clean arm is the only condition that needs a training run
            # this study does not already have, and it is the half of the
            # question that can wait. Loading it lazily lets the shrinkage
            # control -- the part that could overturn the recorded claim --
            # finish a night in which the aligned training did not.
            clean = None
            if lock.get("clean_run") is not None:
                clean_path = runs / lock["clean_run"]["run_id"] / "raw_channel.pt"
                if clean_path.is_file():
                    clean = torch.load(
                        clean_path, map_location="cpu", weights_only=True
                    )
                else:
                    print(f"clean adapter absent: {clean_path}; skipping its"
                          " conditions", flush=True)
            rows = read_json(root / "conditions.json", [])
            done = {row["key"] for row in rows}
            for condition in lock["conditions"]:
                key = condition["key"]
                if key in done or (only is not None and key != only):
                    continue
                if condition["family"] == "clean" and clean is None:
                    continue
                tensors, extra = materialise(
                    condition, trained, base, clean, native_rank, root / "scratch"
                )
                apply_adapter_tensors(session.model, tensors)
                measured = score(
                    session, run, data["gsm8k"], config, root / f"{key}_gsm8k.json"
                )
                likelihood = {
                    split: nll_probe(
                        session, run, probes[split], config,
                        root / f"{key}_nll_{split}.json",
                    )
                    for split in NLL_SPLITS
                }
                geometry = update_geometry(
                    tensors, pad_lora_rank(clean, native_rank) if clean else None
                )
                rows.append(
                    {
                        **{k: v for k, v in condition.items()},
                        **extra,
                        **geometry,
                        "gsm8k_accuracy": measured["exact_match"],
                        **{
                            f"nll_{split}_bits_per_token":
                                likelihood[split]["bits_per_token"]
                            for split in NLL_SPLITS
                        },
                        # Positive means the model finds the aligned rationale
                        # more surprising than the mismatched one it was trained
                        # on -- that is, the corruption is still installed.
                        "corruption_preference_bits":
                            likelihood["aligned"]["bits_per_token"]
                            - likelihood["permuted"]["bits_per_token"],
                    }
                )
                write_json(root / "conditions.json", rows)
            declared = len(lock["conditions"])
            if len(rows) == declared:
                write_json(root / "complete.json",
                           {"complete": True, "conditions": len(rows)})
            else:
                print(f"{len(rows)}/{declared} conditions scored; not complete",
                      flush=True)
        finally:
            session.unload()


def probe(lock: dict, config: dict, out: Path, runs: Path, prepared: Path) -> dict:
    """Time one short generation pass and project the study's walltime.

    Walltime here is set by generation, not by the codec, and the rate depends
    on how long the corrupted adapter's answers run. Measuring it on the raw
    damaged adapter costs minutes and is the difference between a right-sized
    night and one that ends with half the conditions scored.
    """
    run = RunSpec.from_dict(lock["damaged_run"])
    rows = int(config["probe_rows"])
    data = study_data(config, int(lock["seed"]), prepared)
    session = ModelSession.load(run.model)
    try:
        session.attach(run.adapter, run.seed)
        trained = torch.load(
            runs / run.run_id / "raw_channel.pt", map_location="cpu",
            weights_only=True,
        )
        apply_adapter_tensors(session.model, trained)
        started = time.monotonic()
        score(session, run, data["gsm8k"][:rows], config,
              out / "probe" / f"{run.run_id}.json")
        generation = time.monotonic() - started
        probes = nll_examples(data, config)
        started = time.monotonic()
        nll_probe(session, run, probes["permuted"], config,
                  out / "probe" / f"{run.run_id}_nll.json")
        likelihood = time.monotonic() - started
    finally:
        session.unload()
    per_row = generation / max(rows, 1)
    conditions = len(lock["conditions"])
    per_condition = per_row * config["gsm8k_test_rows"] + 2 * likelihood
    return {
        "probe_rows": rows,
        "generation_seconds": generation,
        "generation_seconds_per_row": per_row,
        "nll_seconds": likelihood,
        "conditions": conditions,
        "projected_condition_seconds": per_condition,
        "projected_study_hours": per_condition * conditions / 3600,
        "note": "generation and likelihood only; model load and coding excluded",
    }


NUMBER = re.compile(r"\d+(?:\.\d+)?")


def rationale_of(response: str) -> str:
    """The working, without the answer line the evaluator reads."""
    return response.split("####")[0]


def reproduction_metrics(path: Path, opening_words: int) -> dict[str, float]:
    """How much of one condition's output is the same memorised explanation.

    The damaged adapter answers most questions by reciting a single off-topic
    rationale, so the share of answers that open with the *modal* opening is a
    direct count of the corrupted chain of thought being reproduced -- computed
    from generations that were already paid for. `distinct_openings` is the
    same fact without a privileged opening: it falls as the model collapses
    onto any small set of memorised texts, which is what makes it a check on
    the modal share rather than a restatement of it.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        return {}
    openings = Counter(
        " ".join(rationale_of(row["response"]).split()[:opening_words])
        for row in rows
    )
    return {
        "modal_opening_share": openings.most_common(1)[0][1] / len(rows),
        "distinct_openings": len(openings) / len(rows),
        "mean_completion_tokens":
            sum(row["completion_tokens"] for row in rows) / len(rows),
        "answers": len(rows),
    }


def answer_agreement(left: Path, right: Path) -> dict[str, float]:
    """Item-by-item agreement of two conditions' extracted answers.

    Two adapters can score the same and still be different functions. Agreement
    counts the items where they return the *same* string, and `agreement_wrong`
    restricts that to items both get wrong -- where matching is evidence of a
    shared mechanism rather than of both having found the one right answer.
    """
    rows = {}
    for label, path in (("left", left), ("right", right)):
        rows[label] = {
            json.loads(line)["example_id"]: json.loads(line)
            for line in path.read_text().splitlines() if line
        }
    shared = sorted(set(rows["left"]) & set(rows["right"]))
    if not shared:
        return {}
    same = [
        str(rows["left"][key]["prediction"]) == str(rows["right"][key]["prediction"])
        for key in shared
    ]
    wrong = [
        key for key in shared
        if not rows["left"][key]["correct"] and not rows["right"][key]["correct"]
    ]
    return {
        "items": len(shared),
        "agreement": sum(same) / len(shared),
        "agreement_wrong": (
            sum(
                str(rows["left"][key]["prediction"])
                == str(rows["right"][key]["prediction"])
                for key in wrong
            ) / len(wrong)
        ) if wrong else None,
        "both_wrong_items": len(wrong),
    }


def shrinkage_envelope(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each coded condition against the shrinkage arm nearest it in norm.

    This is the comparison the whole study is for. A coded point and the scalar
    rescaling that reproduces its update norm are matched on the only quantity a
    shrinkage account has to offer; anything they still disagree on is not
    shrinkage. Norm, not accuracy, does the matching -- matching on accuracy
    would assume the answer.
    """
    coded = [row for row in rows if row["family"] == "compression"]
    by_key = {row["key"]: row for row in rows}
    # Prefer the exact norm-matched partner declared for this cell. The nearest
    # point on the alpha sweep is the fallback, and the reported `norm_ratio`
    # is what says which of the two a reader is looking at.
    sweep = [row for row in rows if row["family"] == "shrinkage_head"]
    if not coded:
        return []
    pairs = []
    for row in coded:
        partner = by_key.get(f"norm_matched_b{row['bits']}")
        if partner is None:
            if not sweep:
                continue
            partner = min(
                sweep,
                key=lambda other: abs(other["update_norm"] - row["update_norm"]),
            )
        pairs.append(
            {
                "coded_key": row["key"],
                "shrunk_key": partner["key"],
                "coded_norm": row["update_norm"],
                "shrunk_norm": partner["update_norm"],
                "norm_ratio": (
                    partner["update_norm"] / row["update_norm"]
                    if row["update_norm"] > 0 else None
                ),
                "coded_accuracy": row["gsm8k_accuracy"],
                "shrunk_accuracy": partner["gsm8k_accuracy"],
                "accuracy_gap": row["gsm8k_accuracy"] - partner["gsm8k_accuracy"],
                "coded_corruption_preference": row["corruption_preference_bits"],
                "shrunk_corruption_preference": partner["corruption_preference_bits"],
                "corruption_preference_gap": (
                    row["corruption_preference_bits"]
                    - partner["corruption_preference_bits"]
                ),
                "coded_modal_opening_share": row.get("modal_opening_share"),
                "shrunk_modal_opening_share": partner.get("modal_opening_share"),
            }
        )
    return pairs


def report(lock: dict, out: Path) -> dict:
    config = lock["config"]
    root = out / lock["slug"]
    rows = read_json(root / "conditions.json", [])
    measured = {row["key"] for row in rows}
    declared = [condition["key"] for condition in lock["conditions"]]
    missing = [key for key in declared if key not in measured]
    for row in rows:
        predictions = root / f"{row['key']}_gsm8k.jsonl"
        if predictions.exists():
            row.update(reproduction_metrics(predictions, config["opening_words"]))
        if row["key"] == "base":
            continue
        base_predictions = root / "base_gsm8k.jsonl"
        if predictions.exists() and base_predictions.exists():
            row["gsm8k_change_from_base_ci95"] = paired_interval(
                predictions, base_predictions,
                config["bootstrap_draws"], config["analysis_seed"],
            )
    by_key = {row["key"]: row for row in rows}
    for row in rows:
        if "base" in by_key:
            row["gsm8k_change_from_base"] = (
                row["gsm8k_accuracy"] - by_key["base"]["gsm8k_accuracy"]
            )
        # How close this condition gets to the adapter that never saw the
        # corruption, on outputs rather than on weights.
        clean = root / "clean_raw_gsm8k.jsonl"
        here = root / f"{row['key']}_gsm8k.jsonl"
        if clean.exists() and here.exists() and row["key"] != "clean_raw":
            row["versus_clean"] = answer_agreement(here, clean)
    result = {
        "status": "complete" if not missing else "partial",
        "missing": missing,
        "rows": sorted(rows, key=lambda row: (row["family"], row["key"])),
        "shrinkage_envelope": shrinkage_envelope(rows),
        "claim_boundary": (
            "Qwen2.5-7B, MetaMathQA rationales permuted across problems, one "
            "training seed, GSM8K exact match on the prepared test slice, and "
            "reasoning-span NLL on paired permuted/aligned training rows. "
            "Update norms carry a common alpha/rank factor and are comparable "
            "across conditions rather than absolute. Every declared condition "
            "is measured, so nothing is selected on the test split, but the "
            "intervals are unadjusted for the number of conditions."
        ),
    }
    write_json(out / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/upnquick/denoise_vs_shrinkage.yaml"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path(".cache/reports/denoise_vs_shrinkage_upnquick")
    )
    parser.add_argument("--runs", type=Path, default=Path(".cache/runs"))
    parser.add_argument("--prepared", type=Path, default=Path(".cache/prepared"))
    parser.add_argument(
        "--phase", required=True,
        choices=["prepare", "check", "probe", "sweep", "report"],
    )
    parser.add_argument("--only", type=str, help="score a single condition key")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.phase == "prepare":
        lock = prepare(config, args.out, args.runs, args.prepared)
        print(json.dumps({"conditions": len(lock["conditions"]),
                          "keys": [c["key"] for c in lock["conditions"]],
                          "lock": str(args.out / "lock.json")}, indent=2))
        return
    lock = read_json(args.out / "lock.json")
    if lock is None or lock["config"] != config:
        raise ValueError("prepare this exact config before running")
    for path, digest in {**lock["implementation"], **lock["inputs"]}.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"{path} changed after the study was locked")
    if args.phase == "check":
        result = check(lock, args.runs)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["passed"] else 1)
    if args.phase == "probe":
        print(json.dumps(probe(lock, config, args.out, args.runs, args.prepared),
                         indent=2))
        return
    if args.phase == "report":
        print(json.dumps({k: v for k, v in report(lock, args.out).items()
                          if k != "rows"}, indent=2)[:4000])
        return
    sweep(lock, config, args.out, args.runs, args.prepared, args.only)


if __name__ == "__main__":
    main()
