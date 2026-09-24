"""Select adapter SVD bands on GSM8K, then test the frozen choices elsewhere.

Training, corruption, SVD, coding, generation and uncertainty reuse the existing
owners. Search and finalist selection use separate held-out training subsets;
neither GSM8K test nor transfer answers may choose a filter. Bands index the
singular directions of each LoRA B@A, not the factors separately.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import torch

from fineqcomp.adapters import adapter_tensors, apply_adapter_tensors
from fineqcomp.artifacts import claim_run, read_json, write_json
from fineqcomp.behavioral_trajectory import paired_interval, save_adapter, sha256
from fineqcomp.campaign import expand_campaign
from fineqcomp.codec import ALLOWED_BITS, decode_adapter_tensor_map, encode_tensor_map, pad_lora_rank
from fineqcomp.config import ModelSpec, RunSpec, load_campaign
from fineqcomp.data import (
    _convert_natural, _write_jsonl, prepare_natural_dataset, read_jsonl,
)
from fineqcomp.denoise_vs_shrinkage import update_geometry
from fineqcomp.evaluation import evaluate_natural, write_predictions
from fineqcomp.generalisation import spectral_slice, spectral_subset
from fineqcomp.modeling import ModelSession
from fineqcomp.training import causal_nll, train_adapter


def band_intervals(config: dict, rank: int) -> list[tuple[int, int]]:
    """Which [low, high) windows the grid sweeps.

    `all_contiguous` takes every pair of declared edges, which is the right
    design while the rank is small. It is quadratic, so at rank 64 it asks for
    2,080 windows and stops being runnable. `sweep` instead measures three
    families that each answer a per-direction question directly and grow
    linearly: singletons say what one direction does alone, prefixes [0,k) say
    what the leading block does, and suffixes [k,rank) say what survives once
    the leading block is dropped. Differencing consecutive suffixes gives the
    marginal value of each direction without fitting anything.
    """
    design = str(config.get("band_design", "all_contiguous"))
    if design == "all_contiguous":
        edges = list(config["edges"])
        if not edges or edges != sorted(set(edges)) or edges[0] != 0 or edges[-1] != rank:
            raise ValueError("band edges must increase from zero to the trained rank")
        if any(not isinstance(edge, int) for edge in edges):
            raise ValueError("band edges must be integer direction indices")
        return list(itertools.combinations(edges, 2))
    if design != "sweep":
        raise ValueError(f"unknown band_design {design!r}")
    step = int(config.get("sweep_step", 1))
    if step < 1 or step > rank:
        raise ValueError("sweep_step must be between 1 and the trained rank")
    cuts = sorted({*range(0, rank, step), rank})
    windows = {(lo, min(lo + step, rank)) for lo in cuts if lo < rank}      # singletons
    windows |= {(0, k) for k in cuts if k > 0}                              # prefixes
    windows |= {(k, rank) for k in cuts if k < rank}                        # suffixes
    return sorted(windows)


def grid(config: dict, rank: int) -> list[dict]:
    for code in config["codes"].values():
        if code["bits"] not in ALLOWED_BITS or code["bits"] < 1:
            raise ValueError("unsupported bit width")
        if code.get("quantizer", "midrise") not in {"midrise", "midtread"}:
            raise ValueError("unsupported quantizer")
        if not 0 <= code.get("blend", 0.0) < 1:
            raise ValueError("blend must be in [0, 1)")
    candidates = []
    for low, high in band_intervals(config, rank):
        for name, code in config["codes"].items():
            full = low == 0 and high == rank
            exact = code["bits"] == 16
            if full and exact:
                continue  # The raw reference already covers this condition.
            candidates.append({
                "key": f"band{low:02d}_{high:02d}_{name}", "kind": "coded",
                "arm": "permuted", "low": low, "high": high, "code": code,
                "family": "filter" if exact else "codec" if full else "band_codec",
            })
    if config.get("leave_one_out"):
        # Keep everything except direction i. A contiguous window can never ask
        # this, and it is the only measurement that isolates one direction's
        # contribution *in the presence of all the others*: the drop from the
        # full update is that direction's marginal value, with band width,
        # position and count all held fixed.
        for index in range(rank):
            keep = [i for i in range(rank) if i != index]
            candidates.append({
                "key": f"drop{index:02d}", "kind": "subset", "arm": "permuted",
                "keep": keep, "dropped": index, "family": "leave_one_out",
            })
    for high in (1, rank):
        for scale in config["scales"]:
            if not 0 < scale < 1:
                raise ValueError("shrinkage controls must have scale in (0, 1)")
            candidates.append({
                "key": f"scale_r{high}_{scale:g}", "kind": "scale",
                "arm": "permuted", "low": 0, "high": high,
                "scale": scale, "family": "shrinkage",
            })
    return candidates


def references() -> list[dict]:
    return [
        {"key": "base", "kind": "base", "family": "reference"},
        {"key": "clean_raw", "kind": "raw", "arm": "clean", "family": "reference"},
        {"key": "permuted_raw", "kind": "raw", "arm": "permuted", "family": "reference"},
    ]


def question_key(row) -> str:
    prompt = row.prompt
    for marker in ("Question:", "Problem:"):
        if marker in prompt:
            prompt = prompt.split(marker, 1)[1].rsplit("\nAnswer:", 1)[0]
            break
    return " ".join(prompt.casefold().split())


# The splits that decide anything: what the adapter saw, and what a filter is
# chosen on. These must not share a question with each other or with any probe.
SELECTING_SPLITS = ("train", "training_validation", "search", "selection")


def disjoint(groups: dict[str, list]) -> tuple[dict, list[str]]:
    """Keep the selecting splits clear of each other and of every probe.

    Two *probes* overlapping is a different matter and is not an error: both
    are held out, neither chooses anything, so a shared question cannot leak
    information into a decision. MMLU genuinely ships some questions under two
    subjects -- clinical_knowledge and college_medicine share 78 -- and those
    are still two legitimate test sets. The overlap is reported so an analysis
    that averages across probes knows the samples are not independent.
    """
    keys = {name: {question_key(x) for x in rows} for name, rows in groups.items()}
    protected = [name for name in SELECTING_SPLITS if name in keys]
    for left, right in itertools.combinations(keys, 2):
        if left not in protected and right not in protected:
            continue
        overlap = keys[left] & keys[right]
        if overlap:
            raise ValueError(f"{left} and {right} share {len(overlap)} questions")
    shared = {}
    probes = [name for name in keys if name not in protected]
    for left, right in itertools.combinations(probes, 2):
        count = len(keys[left] & keys[right])
        if count:
            shared[f"{left}|{right}"] = count
    return shared, protected


EVALUATOR_METRIC = {"humaneval": "pass_at_1"}


def probe_spec(cfg: dict, name: str) -> dict:
    """Resolve the evaluator and token budget for any split this study scores.

    Search and selection read held-out rows of the training corpus, so they are
    scored the way the training task is scored, not the way GSM8K is.
    """
    primary = cfg["primary"]
    if name in cfg["transfer"]:
        spec = cfg["transfer"][name]
    elif name in {"search", "selection", "training_validation", primary["key"]}:
        spec = primary
    else:
        raise ValueError(f"no evaluator declared for split {name!r}")
    # The headline metric belongs to the evaluator, so a probe cannot declare
    # HumanEval and forget that it reports pass_at_1 rather than exact_match.
    return {"metric": EVALUATOR_METRIC.get(spec["evaluator"], "exact_match"), **spec}


def study_rows(lock: dict, out: Path) -> dict[str, list]:
    cfg = lock["config"]["spectral_transfer"]
    # Every held-out row the study selects on comes from the clean arm, so a
    # filter can never be chosen for reproducing the corruption.
    clean = lock["runs"]["clean"]
    root = out / "prepared/natural" / clean["dataset_key"] / f"seed{clean['seed']}"
    calibration = read_jsonl(root / "calibration.jsonl")
    a = cfg["training_validation_rows"]
    b = a + cfg["search_rows"]
    c = b + cfg["selection_rows"]
    if len(calibration) != c:
        raise ValueError("reserved GSM8K split does not match the three declared uses")
    return {
        "train": read_jsonl(root / "train.jsonl"),
        "training_validation": calibration[:a], "search": calibration[a:b],
        "selection": calibration[b:c],
        cfg["primary"]["key"]: read_jsonl(root / "test.jsonl"),
        **{name: read_jsonl(out / "data" / f"{name}.jsonl") for name in cfg["transfer"]},
    }


def check_receiver(config: dict, out: Path) -> None:
    """Refuse to train an arm on a receiver the screen did not recommend.

    The study's receiver is written in the config, not read from the screen, so
    the choice stays auditable in one file. This only checks the two agree, so
    a stale model key cannot quietly spend a training budget on a model the
    screen rejected. Overriding the screen is allowed; saying so is not
    optional.
    """
    screened = read_json(out / "screen.json")
    if screened is None:
        return
    chosen = screened["recommendation"]["chosen"]
    receivers = {model for study in config["spectral_transfer"]["arms"].values()
                 for model in config["studies"][study]["models"]}
    if receivers == {chosen} or config["spectral_transfer"].get("override_screen"):
        return
    verdicts = {row["probe"]: row["status"] for row in screened["rows"]
                if row["model"] in receivers}
    raise ValueError(
        f"the screen recommends {chosen!r} but the study trains {sorted(receivers)}; "
        f"its probe verdicts are {verdicts}. Set the study's model to {chosen!r}, "
        "or set spectral_transfer.override_screen with a reason in the config."
    )


def materialize_probes(sources: dict, directory: Path, seed: int) -> dict[str, int]:
    """Freeze each evaluation corpus to one file, shared by prepare and screen.

    A probe may cap its own row count. GSM-Symbolic ships fifty instances of
    each template in order, so the cap samples instead of taking a prefix.
    """
    from datasets import load_dataset

    counts = {}
    for name, source in sources.items():
        path = directory / f"{name}.jsonl"
        if not path.is_file():
            dataset = load_dataset(source["path"], source.get("name"),
                                   revision=source["revision"], split=source["split"])
            examples = _convert_natural(dataset, "test", source["converter"])
            examples = [replace(x, metadata={**x.metadata, "evaluator": source["evaluator"]})
                        for x in examples]
            rows = source.get("rows")
            if rows is not None and rows < len(examples):
                order = random.Random(f"{name}:{seed}").sample(range(len(examples)), int(rows))
                examples = [examples[index] for index in sorted(order)]
            _write_jsonl(path, examples)
        counts[name] = len(read_jsonl(path))
    return counts


def decontaminate_training(lock: dict, out: Path) -> int:
    """Drop training rows that repeat a held-out probe question, in both arms.

    NuminaMath folds in problems from GSM8K-style and SVAMP-style sources, so a
    few of its training rows are test questions of the panel. Dropping the same
    positions from both arms keeps their prompts matched; a corpus with no such
    rows is left untouched.
    """
    data = study_rows(lock, out)
    held = {question_key(x) for name, rows in data.items()
            if name not in SELECTING_SPLITS for x in rows}
    drop = {i for i, x in enumerate(data["train"]) if question_key(x) in held}
    if drop:
        for run in lock["runs"].values():
            path = (out / "prepared/natural" / run["dataset_key"] / f"seed{run['seed']}"
                    / "train.jsonl")
            _write_jsonl(path, [x for i, x in enumerate(read_jsonl(path)) if i not in drop])
    return len(drop)


def prepare(config_path: Path, out: Path) -> dict:
    config = load_campaign(config_path)
    cfg = config["spectral_transfer"]
    if (out / "lock.json").exists():
        return load_lock(config_path, out)
    arms = cfg["arms"]
    check_receiver(config, out)
    expanded = [run for run in expand_campaign(config) if run.study in set(arms.values())]
    by_study = {run.study: run for run in expanded}
    if len(expanded) != 2 or set(by_study) != set(arms.values()):
        raise ValueError("this pilot requires one matched clean/permuted pair")
    runs = {arm: by_study[study].to_dict() for arm, study in arms.items()}
    for run in runs.values():
        prepare_natural_dataset(config, run["dataset_key"], run["seed"], out / "prepared")
    materialize_probes(cfg["transfer"], out / "data", cfg["analysis_seed"])
    lock = {"config": config, "runs": runs,
            "grid": grid(cfg, runs["clean"]["adapter"]["rank"])}
    decontaminated = decontaminate_training(lock, out)
    data = study_rows(lock, out)
    shared_probe_questions, _ = disjoint(data)
    permuted = lock["runs"]["permuted"]
    corrupt = read_jsonl(out / "prepared/natural" / permuted["dataset_key"] /
                         f"seed{permuted['seed']}" / "train.jsonl")
    if [x.prompt for x in corrupt] != [x.prompt for x in data["train"]]:
        raise ValueError("clean/permuted training prompts must match in order")
    lock["rows"] = {key: len(rows) for key, rows in data.items()}
    lock["shared_probe_questions"] = shared_probe_questions
    lock["decontaminated_train_rows"] = decontaminated
    lock["inputs"] = {
        str(path.relative_to(out)): sha256(path)
        for folder in (out / "prepared", out / "data")
        for path in sorted(folder.rglob("*.jsonl"))
    }
    lock["implementation"] = {str(path): sha256(path)
                              for path in sorted(Path("src/fineqcomp").glob("*.py"))}
    lock["claim_boundary"] = (
        "One Mistral-7B seed, GSM8K-only training, fixed rank-16 per-matrix SVD grid. "
        "Only reserved GSM8K validation scores select filters; smoke scores "
        "check execution only. A winning filter "
        "alone does not establish three regimes, intrinsic dimension or generality."
    )
    write_json(out / "lock.json", lock)
    return lock


def load_lock(config_path: Path, out: Path) -> dict:
    lock = read_json(out / "lock.json")
    if lock is None or lock["config"] != load_campaign(config_path):
        raise ValueError("prepare this exact config in a fresh output directory")
    for name, digest in lock["inputs"].items():
        if sha256(out / name) != digest:
            raise ValueError(f"prepared input changed: {name}")
    for name, digest in lock["implementation"].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"implementation changed after prepare: {name}")
    return lock


def train(lock: dict, out: Path, arm: str, smoke: bool = False) -> dict:
    cfg = lock["config"]["spectral_transfer"]
    run = RunSpec.from_dict(lock["runs"][arm])
    root = out / ("smoke_training" if smoke else "adapters") / arm
    with claim_run(root) as acquired:
        if not acquired:
            raise RuntimeError(f"training already running: {root}")
        complete = read_json(root / "complete.json")
        if complete is not None:
            if sha256(root / "raw_channel.pt") != complete["adapter_sha256"]:
                raise ValueError("completed adapter changed")
            return complete
        data_root = out / "prepared/natural" / run.dataset_key / f"seed{run.seed}"
        examples = read_jsonl(data_root / "train.jsonl")
        validation = read_jsonl(data_root / "calibration.jsonl")[:cfg["training_validation_rows"]]
        spec = run.training
        if smoke:
            examples, validation = examples[:32], validation[:4]
            spec = replace(spec, epochs=1, max_updates=2, eval_every_updates=None)
        session = ModelSession.load(run.model)
        try:
            session.attach(run.adapter, run.seed)
            metrics = train_adapter(session.model, session.tokenizer, examples,
                validation, run.model, spec, run.seed, root / "training.jsonl",
                answer_marker="####")
            save_adapter(root / "raw_channel.pt", adapter_tensors(session.model, "full_lora"))
            metrics.update(adapter_sha256=sha256(root / "raw_channel.pt"), run=run.to_dict())
            write_json(root / "complete.json", metrics)
            return metrics
        finally:
            session.unload()


def verdict(metrics: dict, gates: dict, metric: str | None = None) -> dict:
    """Say whether a frozen model is usable as the receiver for one probe.

    The Mistral pilot failed on fluency, not on score: its base arm hit the
    generation cap on three quarters of rows, so "corrupted below base" was
    unreachable and every filter measured termination. Fluency is therefore
    checked before score, and a floor is checked as well as a ceiling.
    """
    score = float(metrics[metric or gates.get("metric", "exact_match")])
    cap = float(metrics["hit_generation_limit_fraction"])
    # A code evaluator like HumanEval has no separate "final answer" step --
    # the whole completion is graded directly -- so it never sets this field.
    # Its absence means nothing to fail, not a fluency gap.
    extracted = float(metrics.get("answer_extracted_fraction", 1.0))
    repetition = float(metrics["mean_repeated_8gram_fraction"])
    reasons = []
    if cap > float(gates["maximum_generation_limit_fraction"]):
        reasons.append("hits the generation cap too often")
    if extracted < float(gates["minimum_answer_extracted_fraction"]):
        reasons.append("does not emit a readable answer")
    if repetition > float(gates["maximum_repeated_8gram_fraction"]):
        reasons.append("repeats itself")
    status = "degenerate" if reasons else (
        "too_easy" if score > float(gates["maximum_baseline_score"]) else
        "too_hard" if score < float(gates["minimum_baseline_score"]) else "usable")
    return {"status": status, "reasons": reasons, "score": score,
            "generation_limit_fraction": cap, "answer_extracted_fraction": extracted,
            "repeated_8gram_fraction": repetition,
            "mean_completion_tokens": metrics["mean_completion_tokens"],
            "headroom_to_ceiling": float(gates["maximum_baseline_score"]) - score}


def screen(config_path: Path, out: Path, *, shard=0, shards=1) -> dict:
    """Score this shard's candidate models on every probe, with no adapter.

    This is the step the Mistral pilot skipped. It costs one generation pass
    per model and rules out receivers that cannot stop or cannot answer before
    any training budget is spent on them. Pure generation, like `search`: a
    later unsharded call reads every shard's files and decides. Folding that
    aggregation into this function too, as an earlier version did, meant the
    unsharded barrier call re-walked every model at its default shard=0,
    shards=1 and silently redid a full multi-model generation pass on one GPU.
    """
    config = load_campaign(config_path)
    cfg = config["screen"]
    materialize_probes(cfg["probes"], out / "screen_data", cfg["seed"])
    # Every declared model is a screen candidate; the study then names one.
    keys = sorted(config["models"])
    for model_key in keys[shard::shards]:
        spec = ModelSpec(key=model_key, **config["models"][model_key])
        session = None
        try:
            for name, source in cfg["probes"].items():
                target = out / "screen" / model_key / f"{name}.json"
                if target.is_file():
                    continue
                with claim_run(target.with_suffix("")) as acquired:
                    if not acquired:
                        raise RuntimeError(f"screen already running: {target}")
                    if session is None:
                        session = ModelSession.load(spec)
                    examples = read_jsonl(out / "screen_data" / f"{name}.jsonl")[:cfg["rows"]]
                    score(session, spec, cfg["seed"], examples, cfg, target,
                          dataset=source["evaluator"],
                          max_tokens=source.get("max_new_tokens"),
                          metric=source.get("metric", "exact_match"),
                          details={"model": model_key, "probe": name,
                                   "family": source["family"]})
        finally:
            if session is not None:
                session.unload()
    return {"shard": shard, "status": "complete"}


def recommend_screen(config_path: Path, out: Path) -> dict:
    """Read every model's scored cells and decide, once every shard is done.

    Pure aggregation, no GPU: the counterpart to `screen` the way `select` is
    the counterpart to `search`.
    """
    config = load_campaign(config_path)
    cfg = config["screen"]
    gates = cfg["gates"]
    counts = materialize_probes(cfg["probes"], out / "screen_data", cfg["seed"])
    keys = sorted(config["models"])
    rows = []
    for model_key in keys:
        for name, source in cfg["probes"].items():
            metrics = read_json(out / "screen" / model_key / f"{name}.json")
            if metrics is None:
                continue
            rows.append({"model": model_key, "probe": name, "family": source["family"],
                         "examples": metrics["examples"],
                         **verdict(metrics, gates, source.get("metric"))})
    usable = {}
    for row in rows:
        if row["status"] == "usable":
            usable.setdefault(row["model"], []).append(row["probe"])
    result = {"rows": rows, "probe_rows": counts, "gates": gates,
              "usable_by_model": usable,
              "complete": len(rows) == len(keys) * len(cfg["probes"]),
              "recommendation": recommend(rows, cfg)}
    write_json(out / "screen.json", result)
    return result


def recommend(rows: list[dict], cfg: dict) -> dict:
    """Pick the receiver with the most usable in-family probes, then headroom.

    The study needs one training task the model can learn and several sibling
    probes it can already answer; a model usable on only the training task
    cannot show dataset agnosticism whatever its score. The primary probe is
    checked first and separately: it is what `search` and `select` actually
    train and choose filters on, so a model that is untrained-usable on every
    sibling but already at ceiling on the primary probe itself is still the
    wrong receiver -- training would leave clean, permuted and coded scores
    clustered near the ceiling with no room to show corruption hurting or
    coding recovering. An earlier version of this ranking had no primary-probe
    check at all and chose a 0.88-untrained-GSM8K model for exactly that
    reason before any training config change caught it.
    """
    family = cfg["training_family"]
    primary_probe = cfg["primary_probe"]
    ranked = []
    for model in sorted({row["model"] for row in rows}):
        mine = [r for r in rows if r["model"] == model]
        if len(mine) != len(cfg["probes"]):
            continue
        infamily = [r for r in mine if r["family"] == family]
        blocked = [r["probe"] for r in mine if r["status"] == "degenerate"]
        primary = next(r for r in mine if r["probe"] == primary_probe)
        ranked.append({"model": model, "degenerate_probes": blocked,
            "primary_probe": primary_probe, "primary_status": primary["status"],
            "usable_in_family": sorted(r["probe"] for r in infamily if r["status"] == "usable"),
            "mean_in_family_headroom": sum(r["headroom_to_ceiling"] for r in infamily) / max(len(infamily), 1),
            "worst_generation_limit_fraction": max(r["generation_limit_fraction"] for r in mine)})
    ranked.sort(key=lambda r: (r["primary_status"] != "usable", len(r["degenerate_probes"]),
                               -len(r["usable_in_family"]), -r["mean_in_family_headroom"]))
    chosen = ranked[0] if ranked else None
    return {"ranking": ranked,
            "chosen": chosen["model"] if chosen and chosen["primary_status"] == "usable" else None,
            "chosen_but_primary_probe_unusable": (
                chosen["model"] if chosen and chosen["primary_status"] != "usable" else None)}


def stage_adapters(lock: dict, out: Path, sources: dict[str, Path]) -> dict:
    """Adopt finished adapters instead of training new ones.

    The recorded recovery result was measured on two specific adapters. Re-running
    the band search against a fresh training run answers a slightly different
    question, so this copies the originals into the layout `train` would have
    written and records where each came from. Shape is checked against the
    config's adapter, because a rank mismatch would otherwise surface much later
    as an empty band.
    """
    rank = RunSpec.from_dict(lock["runs"]["permuted"]).adapter.rank
    adopted = {}
    for arm, source in sources.items():
        if arm not in lock["runs"]:
            raise ValueError(f"unknown arm {arm!r}")
        tensors = torch.load(source, map_location="cpu", weights_only=True)
        widths = {value.shape[0] for name, value in tensors.items() if ".lora_A." in name}
        if widths != {rank}:
            raise ValueError(f"{arm}: adapter ranks {sorted(widths)}, config says {rank}")
        root = out / "adapters" / arm
        root.mkdir(parents=True, exist_ok=True)
        target = root / "raw_channel.pt"
        save_adapter(target, tensors)
        digest = sha256(target)
        write_json(root / "complete.json", {
            "adapter_sha256": digest, "adapter_rank": rank,
            "adopted_from": str(source), "source_sha256": sha256(source),
            "trained_here": False,
            "run": lock["runs"][arm],
        })
        adopted[arm] = {"source": str(source), "sha256": digest,
                        "sites": sum(1 for n in tensors if ".lora_A." in n)}
    return {"adopted": adopted, "rank": rank}


def load_adapters(out: Path) -> dict:
    tensors = {}
    for arm in ("clean", "permuted"):
        path = out / "adapters" / arm / "raw_channel.pt"
        complete = read_json(path.parent / "complete.json")
        if complete is None or sha256(path) != complete["adapter_sha256"]:
            raise ValueError(f"missing or changed completed adapter: {arm}")
        tensors[arm] = torch.load(path, map_location="cpu", weights_only=True)
    return tensors


def materialize(condition: dict, tensors: dict, base: dict, rank: int,
                bands: dict, directory: Path) -> tuple[dict, dict]:
    kind = condition["kind"]
    if kind == "base":
        return base, {"update_norm": 0.0, "file_bits": 0}
    raw = tensors[condition["arm"]]
    if kind == "raw":
        return raw, update_geometry(raw)
    if kind == "subset":
        # An explicit index set, so "everything except direction i" is sayable.
        band_key = (condition["arm"], tuple(condition["keep"]))
        if band_key not in bands:
            bands[band_key] = spectral_subset(raw, condition["keep"])
        band = bands[band_key]
    else:
        if kind == "matched_norm":
            target, _ = materialize(condition["target"], tensors, base, rank, bands, directory)
            source = condition["target"]
            low, high = (0, rank) if condition["scope"] == "full" else (source["low"], source["high"])
        else:
            low, high = condition["low"], condition["high"]
        band_key = (condition["arm"], low, high)
        if band_key not in bands:
            bands[band_key] = spectral_slice(raw, low, high)
        band = bands[band_key]
    scale = condition.get("scale", 1.0)
    if kind == "matched_norm":
        scale = update_geometry(target)["update_norm"] / max(update_geometry(band)["update_norm"], 1e-30)
    # A condition that declares no codec is kept at full precision: the
    # leave-one-out arms select directions and nothing else.
    if kind in {"scale", "matched_norm"} or "code" not in condition:
        decoded = {name: value * scale if ".lora_B." in name else value
                   for name, value in band.items()}
        extra = {"scale": scale}
    else:
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            path = Path(temporary) / "adapter.fqcb"
            storage = encode_tensor_map(band, path, **condition["code"],
                metadata={"singular_start": low, "singular_stop": high})
            _, decoded = decode_adapter_tensor_map(path)
            extra = {"file_bits": storage["file_bits"], "file_sha256": sha256(path)}
    decoded = pad_lora_rank(decoded, rank)
    before = update_geometry(pad_lora_rank(band, rank))["update_norm"]
    extra.update(update_geometry(decoded, tensors["clean"]))
    extra["band_norm_before_coding"] = before
    extra["norm_ratio_to_band"] = extra["update_norm"] / max(before, 1e-30)
    return decoded, extra


def score(session, model_spec, seed, examples, cfg, path: Path, *, dataset="gsm8k",
          max_tokens=None, details=None, metric="exact_match") -> dict:
    cached = read_json(path)
    if cached is not None:
        predictions = [json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines()]
        if [p["example_id"] for p in predictions] != [x.example_id for x in examples]:
            raise ValueError(f"cached prediction IDs differ: {path}")
        if details is not None and any(cached.get(k) != v for k, v in details.items()):
            raise ValueError(f"cached condition differs: {path}")
        return cached
    started = time.monotonic()
    metrics, predictions = evaluate_natural(session.model, session.tokenizer,
        examples, model_spec, dataset, cfg["generation_batch_size"],
        max_new_tokens=max_tokens or cfg["max_new_tokens"], generation_seed=seed)
    # Every evaluator names its own headline metric (HumanEval's is
    # pass_at_1, not exact_match), so this checks whichever one the caller
    # says this scorer actually produces.
    if len(predictions) != len(examples) or not math.isfinite(metrics[metric]):
        raise ValueError(f"generation did not produce a finite {metric} for every row")
    for prediction, example in zip(predictions, examples, strict=True):
        prediction["cluster"] = example.example_id
    metrics["elapsed_seconds"] = time.monotonic() - started
    metrics.update(details or {})
    write_predictions(path.with_suffix(".jsonl"), predictions)
    write_json(path, metrics)
    print(f"{path}: {metric}={metrics[metric]:.4f}, seconds={metrics['elapsed_seconds']:.1f}", flush=True)
    return metrics


def evaluate_conditions(lock, out, conditions, split, root, *, shard=0, shards=1,
                        max_tasks=None):
    cfg = lock["config"]["spectral_transfer"]
    run = RunSpec.from_dict(lock["runs"]["permuted"])
    data = study_rows(lock, out)
    tensors = load_adapters(out)
    tasks = []
    splits = [split] if split != "test" else [cfg["primary"]["key"], *cfg["transfer"]]
    chunk = cfg["evaluation_chunk_rows"] if split == "test" else max(len(data[s]) for s in splits)
    for condition in conditions:
        for name in splits:
            for start in range(0, len(data[name]), chunk):
                tasks.append((condition, name, start, min(start + chunk, len(data[name]))))
    # Group by dataset before distributing chunks: otherwise repeated condition
    # blocks can send most long MATH batches to the same few workers.
    tasks.sort(key=lambda task: task[1])
    assigned = tasks[shard::shards]
    if max_tasks is not None:
        if max_tasks < 1:
            raise ValueError("max_tasks must be positive")
        assigned = assigned[:max_tasks]
    session = None
    bands = {}
    directory = out / "scratch" / f"{split}-{shard}"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for condition, name, start, stop in assigned:
            target = root / condition["key"] / name / f"rows{start:05d}_{stop:05d}.json"
            with claim_run(target.with_suffix("")) as acquired:
                if not acquired:
                    raise RuntimeError(f"evaluation already running: {target}")
                if target.is_file():
                    score(None, run.model, run.seed, data[name][start:stop], cfg, target,
                          details={"condition": condition})
                    continue
                if session is None:
                    session = ModelSession.load(run.model)
                    session.attach(run.adapter, run.seed)
                    base = adapter_tensors(session.model, "full_lora")
                decoded, geometry = materialize(condition, tensors, base, run.adapter.rank, bands, directory)
                apply_adapter_tensors(session.model, decoded)
                probe = probe_spec(cfg, name)
                score(session, run.model, run.seed, data[name][start:stop], cfg, target,
                      dataset=probe["evaluator"], max_tokens=probe.get("max_new_tokens"),
                      metric=probe.get("metric", "exact_match"),
                      details={"condition": condition, **geometry})
    finally:
        if session is not None:
            session.unload()


def likelihood_conditions(lock: dict, out: Path) -> list[dict]:
    """Every candidate the study measured, so each prediction has an outcome.

    The search grid carries search-split accuracy; selected conditions (winners,
    matched-norm and fixed-band controls) also carry test accuracy on probes.
    """
    conditions = {c["key"]: c for c in lock["grid"] + references()}
    selected = read_json(out / "selected.json")
    for condition in (selected or {}).get("conditions", []):
        conditions.setdefault(condition["key"], condition)
    return list(conditions.values())


def likelihood_rows(lock: dict, out: Path) -> dict[str, list]:
    """Rows the predictor may read: none of them decide the outcome it predicts.

    Search accuracy is the outcome on the training task, so the predictor reads
    the selection split instead. On probes it reads the first half of the rows,
    and the outcome is taken from the second half.
    """
    cfg = lock["config"]["spectral_transfer"]
    data = study_rows(lock, out)
    cap = int(cfg.get("likelihood_rows", 64))
    rows = {"selection": data["selection"]}
    for name in [cfg["primary"]["key"], *cfg["transfer"]]:
        rows[name] = data[name][:min(cap, len(data[name]) // 2)]
    return rows


def predict_likelihood(lock, out, *, shard=0, shards=1, max_tasks=None) -> None:
    """Score each candidate update by how well it predicts correct solutions.

    Teacher-forced likelihood of the reference response needs no generation, so
    every token counts and one forward pass replaces a decoding run. It scores
    the materialized update itself -- after band selection and coding -- because
    band effects do not add and coding rotates the update rather than scaling it.
    """
    cfg = lock["config"]["spectral_transfer"]
    run = RunSpec.from_dict(lock["runs"]["permuted"])
    rows = likelihood_rows(lock, out)
    tensors = load_adapters(out)
    tasks = [(c, name) for c in likelihood_conditions(lock, out) for name in rows]
    assigned = tasks[shard::shards]
    if max_tasks is not None:
        assigned = assigned[:max_tasks]
    session, bands = None, {}
    directory = out / "scratch" / f"likelihood-{shard}"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for condition, name in assigned:
            target = out / "likelihood" / condition["key"] / f"{name}.json"
            with claim_run(target.with_suffix("")) as acquired:
                if not acquired:
                    raise RuntimeError(f"likelihood already running: {target}")
                if target.is_file():
                    continue
                if session is None:
                    session = ModelSession.load(run.model)
                    session.attach(run.adapter, run.seed)
                    base = adapter_tensors(session.model, "full_lora")
                decoded, geometry = materialize(condition, tensors, base, run.adapter.rank,
                                                bands, directory)
                apply_adapter_tensors(session.model, decoded)
                started = time.monotonic()
                metrics = causal_nll(session.model, session.tokenizer, rows[name], run.model,
                                     run.training.max_length, cfg.get("likelihood_batch_size", 8))
                if not math.isfinite(metrics["nll"]) or metrics["nll_tokens"] == 0:
                    raise ValueError(f"no finite likelihood for {condition['key']}/{name}")
                write_json(target, {**metrics, "condition": condition, "split": name,
                                    "example_ids": [x.example_id for x in rows[name]],
                                    "seconds": time.monotonic() - started, **geometry})
                print(f"{target}: bits_per_token={metrics['bits_per_token']:.4f}", flush=True)
    finally:
        if session is not None:
            session.unload()


def collect(root: Path, condition: dict, split: str, expected_ids: list[str]) -> tuple[dict, list]:
    files = sorted((root / condition["key"] / split).glob("rows*.json"))
    predictions, records = [], []
    for path in files:
        record = read_json(path)
        if record.get("condition") != condition:
            raise ValueError(f"wrong condition: {path}")
        records.append(record)
        predictions.extend(json.loads(line) for line in path.with_suffix(".jsonl").read_text().splitlines())
    if not expected_ids or [p["example_id"] for p in predictions] != expected_ids:
        raise ValueError(f"incomplete or duplicate rows: {condition['key']}/{split}")
    count = len(predictions)
    row = {"condition": condition, "accuracy": sum(p["correct"] for p in predictions) / count,
           "examples": count, "file_bits": records[0].get("file_bits"),
           "update_norm": records[0].get("update_norm"),
           "cap_fraction": sum(p["hit_generation_limit"] for p in predictions) / count}
    return row, predictions


def best(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot select from an empty family")
    # Accuracy first; exact file size breaks ties, then the stable condition key.
    return min(rows, key=lambda r: (-r["accuracy"],
        r.get("file_bits") if r.get("file_bits") is not None else math.inf,
        r["condition"]["key"]))


SELECTION_FAMILIES = ("filter", "codec", "band_codec", "shrinkage")


def families_present(rows: list[dict]) -> list[str]:
    """Selection families this grid actually has: an fp16-only sweep has no codecs."""
    return [f for f in SELECTION_FAMILIES if any(r["condition"]["family"] == f for r in rows)]


def shortlist(rows: list[dict], count: int) -> list[dict]:
    result = []
    for family in families_present(rows):
        remaining = [r for r in rows if r["condition"]["family"] == family]
        for _ in range(min(count, len(remaining))):
            chosen = best(remaining)
            result.append(chosen["condition"])
            remaining.remove(chosen)
    return result


def selected_conditions(winners: dict, cfg: dict) -> list[dict]:
    conditions = references() + list(winners.values())
    for winner in winners.values():
        if winner["family"] == "shrinkage":
            continue
        scopes = ["full", "band"] if winner["family"] == "band_codec" else ["full"]
        for scope in scopes:
            conditions.append({"key": f"norm_{scope}_{winner['key']}", "kind": "matched_norm",
                "arm": "permuted", "scope": scope, "target": winner, "family": "norm_control"})
    for arm in ("clean", "permuted"):
        for low, high in cfg["fixed_bands"]:
            conditions.append({"key": f"fixed_{arm}_{low}_{high}", "kind": "coded",
                "arm": arm, "low": low, "high": high, "code": {"bits": 16}, "family": "fixed_band"})
    winner = winners["filter"]
    conditions.append({**winner, "key": "clean_selected_band", "arm": "clean", "family": "clean_control"})
    return conditions


def search_finalists(lock: dict, out: Path) -> dict:
    cfg = lock["config"]["spectral_transfer"]
    data = study_rows(lock, out)
    ids = [x.example_id for x in data["search"]]
    search_rows = [collect(out / "search", c, "search", ids)[0]
                   for c in lock["grid"] + references()]
    candidates = shortlist(search_rows, cfg["finalists_per_family"])
    return {"conditions": candidates, "search": search_rows}


def select(lock: dict, out: Path) -> dict:
    cfg = lock["config"]["spectral_transfer"]
    data = study_rows(lock, out)
    finalists = search_finalists(lock, out)
    candidates = finalists["conditions"]
    write_json(out / "shortlist.json", finalists)
    evaluate_conditions(lock, out, candidates + references(), "selection", out / "selection")
    ids = [x.example_id for x in data["selection"]]
    selected_rows = [collect(out / "selection", c, "selection", ids)[0]
                     for c in candidates + references()]
    winners = {family: best([r for r in selected_rows if r["condition"]["family"] == family])["condition"]
               for family in families_present(selected_rows)}
    record = {"winners": winners, "selection": selected_rows,
              "conditions": selected_conditions(winners, cfg),
              "selected_on": "GSM8K reserved training rows; no test scores used",
              "adapter_hashes": {arm: sha256(out / "adapters" / arm / "raw_channel.pt")
                                 for arm in ("clean", "permuted")}}
    previous = read_json(out / "selected.json")
    if previous is not None and previous != record:
        raise ValueError("selection changed; do not overwrite choices after test evaluation")
    write_json(out / "selected.json", record)
    return record


def test_group(conditions: list[dict], group: str) -> list[dict]:
    """Prioritize references and validation winners without dropping controls."""
    core = {"reference", "filter", "codec", "band_codec", "shrinkage", "leave_one_out"}
    if group == "all":
        return conditions
    if group not in {"core", "controls"}:
        raise ValueError(f"unknown test group: {group}")
    return [c for c in conditions if (c["family"] in core) == (group == "core")]


def report(lock: dict, out: Path, group: str = "all") -> dict:
    cfg = lock["config"]["spectral_transfer"]
    selected = read_json(out / "selected.json")
    if selected is None:
        raise ValueError("no frozen selection")
    data = study_rows(lock, out)
    rows, missing = [], []
    for split in (cfg["primary"]["key"], *cfg["transfer"]):
        ids = [x.example_id for x in data[split]]
        available = {}
        for condition in test_group(selected["conditions"], group):
            try:
                row, predictions = collect(out / "test", condition, split, ids)
            except ValueError as error:
                missing.append(str(error))
                continue
            path = out / "joined" / split / f"{condition['key']}.jsonl"
            write_predictions(path, predictions)
            available[condition["key"]] = (row, path)
        for key, (row, path) in available.items():
            row = {**row, "dataset": split, "family": probe_spec(cfg, split)["family"]}
            controls = [c["key"] for c in selected["conditions"]
                        if c.get("target", {}).get("key") == key]
            for reference in ("base", "clean_raw", "permuted_raw", *controls):
                if reference in available:
                    ref, ref_path = available[reference]
                    row[f"difference_vs_{reference}"] = row["accuracy"] - ref["accuracy"]
                    row[f"ci95_vs_{reference}"] = paired_interval(path, ref_path,
                        cfg["bootstrap_draws"], cfg["analysis_seed"])
            rows.append(row)
    result = {"status": "partial" if missing else "complete", "missing": missing,
              "scope": group, "full_study_complete": group == "all" and not missing,
              "rows": rows, "winners": selected["winners"], "claim_boundary": lock["claim_boundary"]}
    if "claim_gates" in cfg:
        result["claims"] = claims(result, cfg, selected["winners"])
    filename = "summary.json" if group == "all" else f"summary_{group}.json"
    write_json(out / filename, result)
    return result


def claims(summary: dict, cfg: dict, winners: dict) -> dict:
    """Read the two frozen claims off a finished report.

    Both are stated before any score exists, and both are refusable. The
    ordering claim is the study's reason to exist: a corrupted adapter must
    hurt the frozen model, and coding it must put the model back in front.
    The transfer claim splits that recovery into a part that follows the task
    family and a part that would follow any task at all.
    """
    gates = cfg["claim_gates"]
    rows = {(row["dataset"], row["condition"]["key"]): row for row in summary["rows"]}
    datasets = sorted({dataset for dataset, _ in rows})
    # The gate names the family frozen in advance; selection decides which
    # member of it is tested, so no key here can be picked after the fact.
    winner = winners[gates["coded_family"]]["key"]
    verdicts = {}
    for dataset in datasets:
        base = rows.get((dataset, "base"))
        permuted = rows.get((dataset, "permuted_raw"))
        coded = rows.get((dataset, winner))
        if base is None or permuted is None or coded is None:
            verdicts[dataset] = {"status": "incomplete"}
            continue
        family = coded["family"]
        low, high = coded.get("ci95_vs_base", (None, None))
        verdicts[dataset] = {
            "family": family,
            "base": base["accuracy"],
            "permuted_raw": permuted["accuracy"],
            "coded": coded["accuracy"],
            "coded_minus_base": coded["accuracy"] - base["accuracy"],
            "permuted_minus_base": permuted["accuracy"] - base["accuracy"],
            "ci95_coded_vs_base": [low, high],
            # "beats" and "matches" are decided by the interval, never the point.
            "coded_beats_base": bool(low is not None and low > 0),
            "coded_matches_base": bool(
                low is not None and low <= 0 <= high
                and high - low <= gates["equivalence_width"]),
            "corruption_hurts": permuted["accuracy"] - base["accuracy"] < -gates["minimum_corruption_harm"],
        }
    training = verdicts.get(cfg["primary"]["key"], {})
    in_family = [v for name, v in verdicts.items()
                 if v.get("family") == gates["transfer_family"] and name != cfg["primary"]["key"]]
    off_family = [v for v in verdicts.values()
                  if v.get("family") not in {gates["transfer_family"], "training_task", None}]
    ordering = bool(training.get("corruption_hurts") and training.get("coded_beats_base"))
    agnostic = bool(in_family) and sum(v["coded_beats_base"] for v in in_family) >= gates[
        "minimum_in_family_datasets"]
    specific = bool(off_family) and all(not v["coded_beats_base"] for v in off_family)
    return {"per_dataset": verdicts,
            "ordering_on_training_task": ordering,
            "dataset_agnostic_within_family": agnostic,
            "task_specific_off_family": specific,
            "all_claims_hold": bool(ordering and agnostic and specific),
            "gates": gates}


def smoke(lock: dict, out: Path, reuse_adapter: Path | None) -> dict:
    cfg = lock["config"]["spectral_transfer"]
    training = train(lock, out, "permuted", smoke=True)
    run = RunSpec.from_dict(lock["runs"]["permuted"])
    path = reuse_adapter or out / "smoke_training/permuted/raw_channel.pt"
    raw = torch.load(path, map_location="cpu", weights_only=True)
    tensors = {"clean": raw, "permuted": raw}
    data = study_rows(lock, out)
    candidate = next(c for c in lock["grid"] if c["low"] == 2 and c["high"] == 8 and c.get("code", {}).get("bits") == 1)
    controls = references()[:1] + [references()[2], candidate,
        {"key": "norm_smoke", "kind": "matched_norm", "arm": "permuted", "scope": "band", "target": candidate}]
    session = ModelSession.load(run.model)
    rows = []
    try:
        session.attach(run.adapter, run.seed)
        base = adapter_tensors(session.model, "full_lora")
        directory = out / "smoke"
        directory.mkdir(parents=True, exist_ok=True)
        for condition in controls:
            decoded, geometry = materialize(condition, tensors, base, run.adapter.rank, {}, directory)
            apply_adapter_tensors(session.model, decoded)
            probe = probe_spec(cfg, "search")
            metrics = score(session, run.model, run.seed,
                            data["search"][:cfg["generation_batch_size"]], cfg,
                            directory / f"{condition['key']}.json",
                            dataset=probe["evaluator"],
                            max_tokens=probe.get("max_new_tokens"),
                            metric=probe.get("metric", "exact_match"))
            rows.append({"key": condition["key"], **metrics, **geometry})
        for name in cfg["transfer"]:
            source = probe_spec(cfg, name)
            metrics = score(session, run.model, run.seed,
                            data[name][:cfg["generation_batch_size"]], cfg,
                            directory / f"{name}.json",
                            dataset=source["evaluator"], max_tokens=source.get("max_new_tokens"),
                            metric=source["metric"])
            rows.append({"key": name, **metrics})
    finally:
        session.unload()
    result = {"status": "passed", "training": training, "rows": rows,
              "reused_adapter": str(path), "reused_sha256": sha256(path),
              "search_candidates": len(lock["grid"]),
              "worst_seconds_per_search_row": max(r["elapsed_seconds"] / r["examples"] for r in rows[:4])}
    write_json(out / "smoke.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/spectral_transfer.yaml"))
    parser.add_argument("--out", type=Path, default=Path(".cache/reports/spectral_transfer_v1"))
    parser.add_argument("--phase", required=True, choices=["screen", "recommend_screen", "prepare", "stage", "smoke", "train", "search", "validate", "select", "test", "report", "likelihood", "map"])
    parser.add_argument("--arm", choices=["clean", "permuted"])
    parser.add_argument("--reuse-adapter", type=Path)
    parser.add_argument("--clean-adapter", type=Path, help="stage: finished aligned adapter")
    parser.add_argument("--permuted-adapter", type=Path, help="stage: finished corrupted adapter")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--group", choices=["all", "core", "controls"], default="all")
    parser.add_argument("--max-tasks", type=int, help="Smoke only: limit tasks per worker in a separate output directory")
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("shard must be in [0, shards)")
    if args.max_tasks is not None and (args.max_tasks < 1 or args.phase not in {"search", "validate", "test", "likelihood", "map"}):
        parser.error("max-tasks must be positive and only applies to evaluation phases")
    # Small factor algebra is much faster without a large BLAS thread pool.
    torch.set_num_threads(4)
    if args.phase == "screen":
        result = screen(args.config, args.out, shard=args.shard, shards=args.shards)
        print(json.dumps(result, indent=2))
        return
    if args.phase == "recommend_screen":
        result = recommend_screen(args.config, args.out)
        print(json.dumps({"recommendation": result["recommendation"],
                          "usable_by_model": result["usable_by_model"],
                          "complete": result["complete"]}, indent=2))
        return
    if args.phase == "prepare":
        lock = prepare(args.config, args.out)
        print(json.dumps({"rows": lock["rows"], "candidates": len(lock["grid"])}, indent=2))
        return
    lock = load_lock(args.config, args.out)
    if args.phase == "stage":
        if args.clean_adapter is None or args.permuted_adapter is None:
            parser.error("stage requires --clean-adapter and --permuted-adapter")
        result = stage_adapters(lock, args.out,
                                {"clean": args.clean_adapter, "permuted": args.permuted_adapter})
    elif args.phase == "smoke":
        result = smoke(lock, args.out, args.reuse_adapter)
    elif args.phase == "train":
        if args.arm is None:
            parser.error("train requires --arm")
        result = train(lock, args.out, args.arm)
    elif args.phase == "search":
        evaluate_conditions(lock, args.out, lock["grid"] + references(), "search",
                            args.out / "search", shard=args.shard, shards=args.shards,
                            max_tasks=args.max_tasks)
        result = {"shard": args.shard, "status": "smoke" if args.max_tasks else "complete"}
    elif args.phase == "validate":
        candidates = search_finalists(lock, args.out)["conditions"] + references()
        evaluate_conditions(lock, args.out, candidates, "selection", args.out / "selection",
                            shard=args.shard, shards=args.shards, max_tasks=args.max_tasks)
        result = {"shard": args.shard, "status": "smoke" if args.max_tasks else "complete"}
    elif args.phase == "map":
        # Every search candidate on the primary test set. Descriptive only: it
        # runs after selection is frozen and nothing reads it to choose.
        if read_json(args.out / "selected.json") is None:
            raise ValueError("select before mapping the grid on test rows")
        primary = lock["config"]["spectral_transfer"]["primary"]["key"]
        evaluate_conditions(lock, args.out, lock["grid"] + references(), primary,
                            args.out / "map", shard=args.shard, shards=args.shards,
                            max_tasks=args.max_tasks)
        result = {"shard": args.shard, "status": "smoke" if args.max_tasks else "complete"}
    elif args.phase == "likelihood":
        predict_likelihood(lock, args.out, shard=args.shard, shards=args.shards,
                           max_tasks=args.max_tasks)
        result = {"shard": args.shard, "status": "smoke" if args.max_tasks else "complete"}
    elif args.phase == "select":
        result = select(lock, args.out)
    elif args.phase == "test":
        selected = read_json(args.out / "selected.json")
        if selected is None:
            raise ValueError("select before scoring test datasets")
        for arm, digest in selected["adapter_hashes"].items():
            if sha256(args.out / "adapters" / arm / "raw_channel.pt") != digest:
                raise ValueError("adapter changed after selection")
        conditions = test_group(selected["conditions"], args.group)
        evaluate_conditions(lock, args.out, conditions, "test", args.out / "test",
                            shard=args.shard, shards=args.shards, max_tasks=args.max_tasks)
        result = {"shard": args.shard, "status": "smoke" if args.max_tasks else "complete"}
    else:
        result = report(lock, args.out, args.group)
    print(json.dumps(result, indent=2)[:12000])


if __name__ == "__main__":
    main()
