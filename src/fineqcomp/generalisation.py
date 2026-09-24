"""Does the compression recovery need a chain of thought, or any corrupted task?

`denoise_vs_shrinkage` and `cot_verbosity_mechanism` establish, on one task and
one corruption, that an adapter trained on mismatched rationales recovers when
compressed and overshoots the base model it was damaged from, and that what
survives the compression is a chain-of-thought format prior rather than task
knowledge. Both findings are about GSM8K reasoning with one kind of damage.

This module asks how far either travels, by holding the whole measurement
apparatus fixed and varying two things:

  the corruption   Permuting a rationale leaves correct arithmetic attached to
                   the wrong problem. Corrupting the arithmetic instead leaves
                   the right problem with wrong working and an answer that
                   agrees with the wrong working -- a different lesson taught
                   to the adapter, in the same format. Shuffling the steps
                   keeps content, format and answer and destroys only order.

  the task         Multiple choice has no chain to segment and a one-token
                   answer, so a format prior has nowhere to live: if the
                   overshoot is a format effect it should vanish there while
                   the recovery toward base survives. Summarisation is
                   generative and has a strong format but no reasoning, which
                   separates "generative structure" from "reasoning" as the
                   thing being restored.

Every arm is scored on the same six conditions, so the comparison across tasks
is a comparison of one number computed the same way:

  base            the model with a zero update, on this task's own test split.
  raw             the damaged adapter as trained.
  rank1_b1        rank one through a one-bit file: the recovery.
  rank1_b16       rank one at full precision: truncation without coding.
  scale_head_0p5  the rank-one direction scaled by a half: shrinkage alone.
  tail_b16        singular directions 4 and up at full precision, which on the
                  recorded task beat every compressed condition.

Nothing here trains. Each arm's adapter is an ordinary campaign named by study
and rank rather than by run id, so an edited training config comes back as a
missing adapter rather than as a silently mismatched comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from fineqcomp.adapters import adapter_tensors, apply_adapter_tensors
from fineqcomp.artifacts import claim_run, read_json, write_json
from fineqcomp.behavioral_trajectory import sha256
from fineqcomp.campaign import expand_campaign
from fineqcomp.codec import (
    decode_adapter_tensor_map, encode_tensor_map, pad_lora_rank, truncate_lora_rank,
)
from fineqcomp.config import RunSpec, load_campaign
from fineqcomp.data import read_jsonl
from fineqcomp.denoise_vs_shrinkage import update_geometry
from fineqcomp.evaluation import evaluate_natural, write_predictions
from fineqcomp.modeling import ModelSession

# The measure each task is judged on. Keeping the mapping here rather than in
# the config stops an arm being quietly scored on a metric that flatters it.
PRIMARY_METRIC = {
    "gsm8k": "exact_match",
    "multiple_choice": "accuracy",
    "paws": "exact_match",
    "xsum": "rouge_l",
    # Format-shift tasks: the model has to emit SQL or XBRL tags rather than
    # prose, and both are scored by exact match on the emitted string.
    "text_to_sql": "exact_match",
    "xbrl_tags": "exact_match",
}


def arm_run(arm: dict[str, Any], seed: int) -> RunSpec:
    """Resolve an arm's run id by re-expanding the campaign that defines it."""
    runs = [
        run
        for run in expand_campaign(load_campaign(arm["config"]))
        if run.study == arm["study"]
        and run.model.key == arm["model"]
        and run.adapter.rank == int(arm["rank"])
        and run.seed == int(seed)
        # A study that sweeps several corpora -- three pointer difficulties, say
        # -- expands to several runs per model, and an arm names exactly one of
        # them. Every arm already carries the dataset it is scored on, so the
        # same field is what identifies the adapter.
        and str(run.dataset_key) == str(arm["dataset"])
    ]
    if len(runs) != 1:
        raise ValueError(
            f"arm {arm['study']}/{arm['dataset']} on {arm['model']} rank"
            f" {arm['rank']} seed {seed} expands to {len(runs)} runs in"
            f" {arm['config']}"
        )
    return runs[0]


def spectral_slice(
    tensors: dict[str, torch.Tensor], low: int, high: int
) -> dict[str, torch.Tensor]:
    """Keep singular directions [low, high) of the update and drop the rest."""
    full = max(t.shape[0] for n, t in tensors.items() if ".lora_A." in n)
    balanced = truncate_lora_rank(tensors, full)
    sliced = {}
    for a_name in sorted(n for n in balanced if ".lora_A." in n):
        b_name = a_name.replace(".lora_A.", ".lora_B.")
        sliced[a_name] = balanced[a_name][low:high].contiguous()
        sliced[b_name] = balanced[b_name][:, low:high].contiguous()
    return sliced


def spectral_profile(tensors: dict[str, torch.Tensor]) -> dict[str, float]:
    """Summarise how a LoRA update spreads its energy, from the weights alone.

    Each site's update B @ A has at most r singular values, taken exactly from
    the r x r core of the two thin QR factors, so no d x d matrix is formed.
    Site statistics are pooled with each site's share of the total energy as
    its weight. `pair_overlap` compares the energy the r stored LoRA pairs
    carry one by one with the update's true energy: 1 means orthogonal pairs,
    above 1 means pairs that cancel, which is what dropping random pairs below
    one bit per value acts on.
    """
    sites = []
    for a_name in sorted(n for n in tensors if ".lora_A." in n):
        a = tensors[a_name].double()
        b = tensors[a_name.replace(".lora_A.", ".lora_B.")].double()
        _, rb = torch.linalg.qr(b)
        _, ra = torch.linalg.qr(a.T)
        sigma = torch.linalg.svdvals(rb @ ra.T)
        energy = float((sigma ** 2).sum())
        if energy == 0.0:
            continue
        share = sigma ** 2 / energy
        entropy = float(-(share[share > 0] * share[share > 0].log()).sum())
        pairs = float(((b ** 2).sum(0) * (a ** 2).sum(1)).sum())
        rows = torch.cat([a, b.T], dim=1)
        centred = rows - rows.mean(1, keepdim=True)
        kurtosis = float(((centred ** 4).mean(1) / (centred ** 2).mean(1).clamp_min(1e-30) ** 2).mean())
        sites.append({"energy": energy, "top1": float(share[0]), "top4": float(share[:4].sum()),
                      "effective_rank": float(torch.exp(torch.tensor(entropy))),
                      "stable_rank": energy / float(sigma[0] ** 2),
                      "pair_overlap": pairs / energy, "kurtosis": kurtosis})
    if not sites:
        raise ValueError("adapter has no non-zero LoRA site")
    total = sum(site["energy"] for site in sites)
    weights = [site["energy"] / total for site in sites]
    profile = {f"spectrum_{key}": sum(w * site[key] for w, site in zip(weights, sites))
               for key in ("top1", "top4", "effective_rank", "stable_rank", "pair_overlap", "kurtosis")}
    ordered = sorted(weights, reverse=True)
    profile["spectrum_log_energy"] = float(torch.log(torch.tensor(total)))
    profile["spectrum_site_top10pct_share"] = sum(ordered[:max(1, len(ordered) // 10)])
    profile["spectrum_sites"] = float(len(sites))
    return profile


@torch.no_grad()
def functional_profile(session: Any, rows: list, model_spec: Any, max_length: int) -> dict[str, float]:
    """Spectrum of an attached LoRA update as the base model exercises it on data.

    The weight spectrum says how the update is shaped; this says how the base
    model's own activations on the training rows excite it. Per site it
    accumulates the r x r covariance C of the bottleneck A h over every token,
    so the update's output covariance B C B^T has the eigenvalues of
    C^1/2 B^T B C^1/2, and it records the frozen layer's own output energy
    E||W0 h||^2 on the same tokens. Adapter, base model and data all enter.
    """
    from torch.utils.data import DataLoader

    from fineqcomp.modeling import CausalExampleDataset, causal_collate, model_device

    sites, hooks = {}, []
    for name, module in session.model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "base_layer")
                and "default" in getattr(module, "lora_A", {})):
            continue
        site = sites[name] = {"module": module, "cov": None, "base": 0.0, "tokens": 0}

        def bottleneck(_, __, output, site=site):
            z = output.detach().reshape(-1, output.shape[-1]).double()
            site["cov"] = z.T @ z if site["cov"] is None else site["cov"] + z.T @ z
            site["tokens"] += z.shape[0]

        def frozen(_, __, output, site=site):
            site["base"] += float(output.detach().double().pow(2).sum())

        hooks.append(module.lora_A["default"].register_forward_hook(bottleneck))
        hooks.append(module.base_layer.register_forward_hook(frozen))
    if not sites:
        raise ValueError("no LoRA site is attached")
    dataset = CausalExampleDataset(session.tokenizer, rows, model_spec, max_length)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=lambda batch: causal_collate(batch, session.tokenizer.pad_token_id))
    device = model_device(session.model)
    session.model.eval()
    try:
        for batch in loader:
            session.model(input_ids=batch["input_ids"].to(device),
                          attention_mask=batch["attention_mask"].to(device))
    finally:
        for hook in hooks:
            hook.remove()
    stats = []
    for site in sites.values():
        module = site["module"]
        b = module.lora_B["default"].weight.detach().double().cpu()
        scale = float(module.scaling["default"]) ** 2 / site["tokens"]
        cov = site["cov"].cpu() * scale
        values, vectors = torch.linalg.eigh(cov)
        root = vectors @ torch.diag(values.clamp_min(0).sqrt()) @ vectors.T
        spectrum = torch.linalg.eigvalsh(root @ b.T @ b @ root).flip(0).clamp_min(0)
        energy = float(spectrum.sum())
        if energy == 0.0:
            continue
        share = spectrum / energy
        nonzero = share[share > 0]
        pairs = float(((b ** 2).sum(0) * torch.diagonal(cov)).sum())
        stats.append({"energy": energy, "base": site["base"] / site["tokens"],
                      "top1": float(share[0]), "top4": float(share[:4].sum()),
                      "effective_rank": float(torch.exp(-(nonzero * nonzero.log()).sum())),
                      "pair_overlap": pairs / energy})
    total = sum(s["energy"] for s in stats)
    weights = [s["energy"] / total for s in stats]
    profile = {f"functional_{key}": sum(w * s[key] for w, s in zip(weights, stats))
               for key in ("top1", "top4", "effective_rank", "pair_overlap")}
    profile["functional_log_relative_energy"] = float(
        torch.log(torch.tensor(total / sum(s["base"] for s in stats))))
    profile["functional_site_log_relative"] = sum(
        w * float(torch.log(torch.tensor(s["energy"] / s["base"]))) for w, s in zip(weights, stats))
    ordered = sorted(weights, reverse=True)
    profile["functional_site_top10pct_share"] = sum(ordered[:max(1, len(ordered) // 10)])
    return profile


@torch.no_grad()
def gain_profile(session: Any, tensors: dict[str, torch.Tensor], rows: list, model_spec: Any,
                 max_length: int, batch_size: int = 4) -> dict[str, float]:
    """Which singular directions carry the adapter's likelihood gain on its data.

    With an adapter attached (at the same rank as `tensors`), scores the base
    model, the full update, and the full update with each singular direction
    removed in turn, all on the same rows. The loss from dropping direction i
    is its marginal share of the gain, with every other direction present --
    the leave-one-out question the pass-band studies showed matters. Adapter,
    base model and data all enter; nothing is compressed.
    """
    from fineqcomp.codec import pad_lora_rank
    from fineqcomp.training import causal_nll

    rank = max(t.shape[0] for n, t in tensors.items() if ".lora_A." in n)

    def nll(update: dict[str, torch.Tensor]) -> float:
        apply_adapter_tensors(session.model, pad_lora_rank(update, rank))
        return float(causal_nll(session.model, session.tokenizer, rows, model_spec,
                                max_length, batch_size)["nll"])

    base = nll({n: (t * 0 if ".lora_B." in n else t) for n, t in tensors.items()})
    full = nll(tensors)
    losses = [nll(spectral_subset(tensors, [j for j in range(rank) if j != i])) - full
              for i in range(rank)]
    apply_adapter_tensors(session.model, tensors)
    helpful = sorted((x for x in losses if x > 0), reverse=True)
    total_helpful = sum(helpful)
    cumulative, needed = 0.0, len(helpful)
    for count, value in enumerate(helpful, start=1):
        cumulative += value
        if cumulative >= 0.9 * total_helpful:
            needed = count
            break
    gain = base - full
    return {
        "gain_bits_per_token": gain / 0.6931471805599453,
        "gain_top1_share": helpful[0] / total_helpful if helpful else 0.0,
        "gain_directions_90": float(needed),
        "gain_negative_share": sum(-x for x in losses if x < 0) / max(sum(abs(x) for x in losses), 1e-30),
        "gain_additivity": sum(losses) / gain if gain else float("nan"),
        "gain_leading_direction_share": max(losses[0], 0.0) / total_helpful if total_helpful else 0.0,
    }


def spectral_subset(
    tensors: dict[str, torch.Tensor], indices: "list[int]"
) -> dict[str, torch.Tensor]:
    """Keep an arbitrary set of singular directions and drop the rest.

    `spectral_slice` can only express a contiguous window, which cannot ask
    what one direction contributes *in the presence of all the others* -- the
    leave-one-out question. This takes an explicit index list instead.
    """
    full = max(t.shape[0] for n, t in tensors.items() if ".lora_A." in n)
    balanced = truncate_lora_rank(tensors, full)
    keep = sorted(set(int(i) for i in indices))
    if not keep or keep[0] < 0 or keep[-1] >= full:
        raise ValueError(f"indices must lie inside [0, {full})")
    picker = torch.tensor(keep, dtype=torch.long)
    sliced = {}
    for a_name in sorted(n for n in balanced if ".lora_A." in n):
        b_name = a_name.replace(".lora_A.", ".lora_B.")
        sliced[a_name] = balanced[a_name].index_select(0, picker).contiguous()
        sliced[b_name] = balanced[b_name].index_select(1, picker).contiguous()
    return sliced


def build_condition(
    key: str,
    trained: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
    native: int,
    scratch: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Materialise one named condition, padded back to the attached rank."""
    if key == "base":
        return base, {}
    if key == "raw":
        return trained, {}
    if key == "tail_b16":
        return pad_lora_rank(spectral_slice(trained, 4, native), native), {}
    if key.startswith("rank1_b"):
        bits = int(key.split("_b")[1])
        scratch.mkdir(parents=True, exist_ok=True)
        path = scratch / f"{key}.fqcb"
        storage = encode_tensor_map(truncate_lora_rank(trained, 1), path, bits)
        _, decoded = decode_adapter_tensor_map(path)
        path.unlink(missing_ok=True)
        return pad_lora_rank(decoded, native), {
            "file_bits": int(storage["file_bits"])
        }
    if key.startswith("scale_head_"):
        scale = float(key.replace("scale_head_", "").replace("p", "."))
        head = truncate_lora_rank(trained, 1)
        scaled = {
            name: (value * scale if ".lora_B." in name else value)
            for name, value in head.items()
        }
        return pad_lora_rank(scaled, native), {"scale": scale}
    if key == "norm_matched_b1":
        # The shrinkage control: the same rank-one directions rescaled to the
        # norm the one-bit code produces, so the pair differ only in coding.
        scratch.mkdir(parents=True, exist_ok=True)
        path = scratch / "target.fqcb"
        encode_tensor_map(truncate_lora_rank(trained, 1), path, 1)
        _, decoded = decode_adapter_tensor_map(path)
        path.unlink(missing_ok=True)
        target = update_geometry(decoded)["update_norm"]
        head = truncate_lora_rank(trained, 1)
        current = update_geometry(head)["update_norm"]
        factor = target / current if current > 0 else 0.0
        scaled = {
            name: (value * factor if ".lora_B." in name else value)
            for name, value in head.items()
        }
        return pad_lora_rank(scaled, native), {"scale": factor}
    raise ValueError(f"unknown condition {key!r}")


def test_rows(arm: dict[str, Any], seed: int, prepared: Path, limit: int) -> list:
    """The arm's own prepared test split, capped."""
    path = prepared / "natural" / arm["dataset"] / f"seed{seed}" / "test.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path}; prepare the arm's campaign first")
    return read_jsonl(path)[:limit]


def score(
    session: ModelSession,
    run: RunSpec,
    rows: list,
    arm: dict[str, Any],
    config: dict,
    path: Path,
) -> dict:
    """Evaluate on this arm's own task, cached on its own path."""
    cached = read_json(path)
    if cached is not None:
        return cached
    metrics, predictions = evaluate_natural(
        session.model,
        session.tokenizer,
        rows,
        run.model,
        arm["evaluator"],
        int(config["evaluation_batch_size"]),
        multiple_choice_labels=list(config.get("multiple_choice_labels", [])) or None,
        max_new_tokens=int(arm.get("max_new_tokens", config["max_new_tokens"])),
        generation_seed=run.seed,
    )
    for prediction, row in zip(predictions, rows, strict=True):
        prediction.setdefault("cluster", row.example_id)
    write_predictions(path.with_suffix(".jsonl"), predictions)
    write_json(path, metrics)
    metric = PRIMARY_METRIC[arm["evaluator"]]
    print(f"{path.stem}: {metric}={metrics[metric]:.4f}", flush=True)
    return metrics


def prepare(config: dict, out: Path, runs: Path, prepared: Path) -> dict:
    seed = int(config["seed"])
    cells = []
    for name, arm in sorted(config["arms"].items()):
        if arm["evaluator"] not in PRIMARY_METRIC:
            raise ValueError(f"arm {name}: unscored evaluator {arm['evaluator']!r}")
        run = arm_run(arm, seed)
        cells.append(
            {
                "id": len(cells),
                "arm": name,
                "run": run.to_dict(),
                "dataset": arm["dataset"],
                "evaluator": arm["evaluator"],
                "slug": f"{name}/seed{seed}",
            }
        )
    record = {
        "config": config,
        "seed": seed,
        "cells": cells,
        "conditions": list(config["conditions"]),
        "inputs": {
            str(Path(arm["config"])): sha256(Path(arm["config"]))
            for arm in config["arms"].values()
        },
        "implementation": {
            str(p): sha256(p) for p in sorted(Path("src/fineqcomp").glob("*.py"))
        },
    }
    record = json.loads(json.dumps(record, sort_keys=True, default=list))
    previous = read_json(out / "lock.json")
    if previous is not None and previous != record:
        raise ValueError("study changed: use a new output directory")
    write_json(out / "lock.json", record)
    return record


def check(lock: dict, runs: Path, prepared: Path) -> dict:
    missing_adapter, missing_data = [], []
    for cell in lock["cells"]:
        if not (runs / cell["run"]["run_id"] / "raw_channel.pt").is_file():
            missing_adapter.append(cell["arm"])
        path = (prepared / "natural" / cell["dataset"]
                / f"seed{lock['seed']}" / "test.jsonl")
        if not path.is_file():
            missing_data.append(cell["arm"])
    return {
        "arms": len(lock["cells"]),
        "conditions": len(lock["conditions"]),
        "missing_adapters": missing_adapter,
        "missing_data": missing_data,
        "passed": not missing_adapter and not missing_data,
    }


def sweep(lock: dict, config: dict, out: Path, runs: Path, prepared: Path,
          only: str | None = None) -> None:
    for cell in lock["cells"]:
        if only is not None and cell["arm"] != only:
            continue
        root = out / cell["slug"]
        run = RunSpec.from_dict(cell["run"])
        arm = config["arms"][cell["arm"]]
        adapter = runs / run.run_id / "raw_channel.pt"
        if not adapter.is_file():
            print(f"{cell['arm']}: adapter absent, skipping", flush=True)
            continue
        with claim_run(root) as acquired:
            if not acquired:
                raise RuntimeError(f"arm already running: {root}")
            if (root / "complete.json").exists():
                continue
            rows = test_rows(arm, int(lock["seed"]), prepared,
                             int(config["test_rows"]))
            if config.get("smoke"):
                rows = rows[: int(config["smoke_rows"])]
            session = ModelSession.load(run.model)
            try:
                session.attach(run.adapter, run.seed)
                base = adapter_tensors(session.model, "full_lora")
                trained = torch.load(adapter, map_location="cpu", weights_only=True)
                native = run.adapter.rank
                results = read_json(root / "conditions.json", [])
                done = {r["condition"] for r in results}
                for key in lock["conditions"]:
                    if key in done:
                        continue
                    tensors, extra = build_condition(
                        key, trained, base, native, root / "scratch"
                    )
                    apply_adapter_tensors(session.model, tensors)
                    metrics = score(session, run, rows, arm, config,
                                    root / f"{key}.json")
                    results.append({
                        "condition": key,
                        "arm": cell["arm"],
                        "evaluator": arm["evaluator"],
                        "score": metrics[PRIMARY_METRIC[arm["evaluator"]]],
                        **extra,
                        **update_geometry(tensors),
                    })
                    write_json(root / "conditions.json", results)
                if len(results) == len(lock["conditions"]):
                    write_json(root / "complete.json", {"complete": True})
            finally:
                session.unload()


def report(lock: dict, out: Path) -> dict:
    arms = {}
    for cell in lock["cells"]:
        rows = read_json(out / cell["slug"] / "conditions.json", [])
        if not rows:
            continue
        by = {r["condition"]: r for r in rows}
        base = by.get("base", {}).get("score")
        raw = by.get("raw", {}).get("score")
        entry = {
            "evaluator": cell["evaluator"],
            "dataset": cell["dataset"],
            "scores": {k: v["score"] for k, v in by.items()},
        }
        if base is not None and raw is not None:
            entry["damage"] = raw - base
            # The two quantities the whole line of work is about: how much of
            # the damage a condition undoes, and whether it ends up above the
            # model it was damaged from.
            entry["recovery"] = {
                k: (v["score"] - raw) / (base - raw) if base != raw else None
                for k, v in by.items()
            }
            entry["overshoot"] = {k: v["score"] - base for k, v in by.items()}
        arms[cell["arm"]] = entry
    result = {
        "status": "complete" if len(arms) == len(lock["cells"]) else "partial",
        "arms": arms,
        "claim_boundary": (
            "One model family, one seed, one adapter per arm. Every arm is "
            "scored on its own task's primary metric, so 'recovery' is "
            "comparable across arms as a fraction of that arm's own damage "
            "while the raw scores are not comparable across tasks."
        ),
    }
    write_json(out / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path(".cache/runs"))
    parser.add_argument("--prepared", type=Path, default=Path(".cache/prepared"))
    parser.add_argument("--phase", required=True,
                        choices=["prepare", "check", "sweep", "report"])
    parser.add_argument("--arm", type=str)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.phase == "prepare":
        lock = prepare(config, args.out, args.runs, args.prepared)
        print(json.dumps({"arms": [c["arm"] for c in lock["cells"]],
                          "conditions": lock["conditions"]}, indent=2))
        return
    lock = read_json(args.out / "lock.json")
    if lock is None or lock["config"] != config:
        raise ValueError("prepare this exact config before running")
    if args.phase == "check":
        result = check(lock, args.runs, args.prepared)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["passed"] else 1)
    if args.phase == "report":
        print(json.dumps(report(lock, args.out), indent=2)[:4000])
        return
    started = time.monotonic()
    sweep(lock, config, args.out, args.runs, args.prepared, args.arm)
    print(f"sweep finished in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
