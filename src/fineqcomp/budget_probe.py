"""How many bits a fine-tune needs, and whether one gradient step can say so.

Two things about the campaign's rate axis have to change before the question
is answerable, and both are repairs rather than new apparatus.

*The target was not a bit budget.*  Below one bit the shipped ladder is not a
precision code: `codec.blend_widths` at zero bits keeps a random fraction of
LoRA rank directions and drops the rest, keyed on the pair name so it knows
nothing about the weights.  Almost every R\\* this project has recorded falls
between 0.2 and 1.2 bits per value, so almost every one of them measures how
much *random rank pruning* an adapter survives.  Informed truncation beats that
mask by a median of 28 retained-gain points on the same files.  So the target
here is the lower envelope of the rank-by-precision surface -- `rank_frontier`
already computes it -- and the summary is `B*`, the smallest file in bits that
keeps 90% of the trained gain.  Bits per value is kept alongside it, against
the full-rank container, so the new numbers can be read next to the old ones.

*The predictors were the wrong kind of object.*  Forty-six candidates have been
screened, all of them scalar summaries of a gradient sketch, all of them fitted
with a per-receiver intercept.  They reverse sign between the within-receiver
and the across-receiver axis, which no single slope can absorb.  The predictor
here is not a scalar at all: it is the same measurement on an adapter that was
never trained.  A run with the optimizer capped at `k` updates over a 256-row
batch produces a real adapter; rescaling it along its own update direction to
the best point on the validation split produces the strongest adapter available
in the direction the corpus first pushes this model; and sweeping *that* over
the identical rank-by-precision surface gives `B*_probe`.

The prediction is `B*_probe = B*_trained` on the identity line.  No fitted
constant, no receiver term, no ranking -- the same units on both axes.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

from fineqcomp.adapters import adapter_tensors, apply_adapter_tensors
from fineqcomp.config import RunSpec, load_campaign
from fineqcomp.modeling import ModelSession
from fineqcomp.rank_frontier import (
    DEFAULT_RANKS,
    DEFAULT_RATES,
    baseline_heldout_bits,
    frontier,
    random_mask_ladder,
    sweep_tensors,
)

DEFAULT_RETENTION = 0.90
# One Adam step is `lr * sign(g)` whatever the corpus, so a capped run arrives
# with a meaningful direction and an arbitrary magnitude: a one-update adapter
# is orders of magnitude shorter than a finished one, and how many orders
# depends on the learning rate rather than on the corpus. The search below
# walks outward in powers of two between these bounds until the held-out gain
# stops improving, so it costs about six forward passes on a well-placed cell
# and cannot be missed by a grid that was drawn too narrow.
SCALE_FLOOR = 1.0 / 64
SCALE_CAP = 4096.0


def budget_at(
    cells: list[dict[str, Any]], retention: float = DEFAULT_RETENTION
) -> dict[str, Any]:
    """The smallest file on the rank-rate envelope that keeps `retention`.

    Interpolation is linear in file bits between the two envelope points that
    bracket the target.  `bracketed` is false when the whole envelope sits on
    one side of it, and an unbracketed cell is read by nothing downstream: an
    extrapolated budget is a number the sweep did not measure.
    """
    if not cells:
        raise ValueError("no swept cells to read a budget from")
    ceiling = float(cells[0]["ceiling_heldout_bits_saved"])
    container = max(int(cell["tensor_values"]) for cell in cells)
    target = retention * ceiling
    envelope = frontier(cells)
    below: dict[str, Any] | None = None
    for cell in envelope:
        if float(cell["heldout_bits_saved"]) >= target:
            if below is None:
                return {
                    "retention": retention,
                    "bracketed": False,
                    "file_bits": float(cell["file_bits"]),
                    "bits_per_value": float(cell["file_bits"]) / container,
                    "rank": int(cell["rank"]),
                    "container_values": container,
                    "ceiling_heldout_bits_saved": ceiling,
                }
            low, high = float(below["file_bits"]), float(cell["file_bits"])
            gap = float(cell["heldout_bits_saved"]) - float(
                below["heldout_bits_saved"]
            )
            share = (
                (target - float(below["heldout_bits_saved"])) / gap if gap > 0 else 0.0
            )
            bits = low + share * (high - low)
            return {
                "retention": retention,
                "bracketed": True,
                "file_bits": bits,
                "bits_per_value": bits / container,
                "rank": int(cell["rank"]),
                "container_values": container,
                "ceiling_heldout_bits_saved": ceiling,
            }
        below = cell
    return {
        "retention": retention,
        "bracketed": False,
        "file_bits": float("nan"),
        "bits_per_value": float("nan"),
        "rank": None,
        "container_values": container,
        "ceiling_heldout_bits_saved": ceiling,
    }


def update_direction(
    initial: dict[str, torch.Tensor], trained: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """What the optimizer moved, as a tensor map over the same container."""
    missing = sorted(set(initial) ^ set(trained))
    if missing:
        raise KeyError(f"adapter tensor maps do not match: {missing[:3]}")
    return {name: trained[name] - initial[name] for name in initial}


def scaled_adapter(
    initial: dict[str, torch.Tensor],
    update: dict[str, torch.Tensor],
    scale: float,
) -> dict[str, torch.Tensor]:
    return {name: initial[name] + scale * update[name] for name in initial}


def scale_search(
    engine: Any,
    session: ModelSession,
    run: RunSpec,
    data: dict[str, list[Any]],
    initial: dict[str, torch.Tensor],
    update: dict[str, torch.Tensor],
    floor: float = SCALE_FLOOR,
    cap: float = SCALE_CAP,
) -> dict[str, Any]:
    """Take the probe as far along its own direction as the held-out split likes.

    The direction is what the corpus supplies; the length is measured.  The
    walk starts at the length the optimizer actually took, doubles while the
    held-out gain improves and halves if it does not, so it brackets the peak
    of a single-humped curve without a grid to draw.  `at_search_edge` says the
    walk stopped at a bound rather than at a peak, which means the probe is not
    the best adapter available in its own direction and its budget is not the
    number this study is about.

    Scale zero is the frozen model exactly, because a PEFT LoRA starts with a
    zero B, so the same loop supplies the baseline the gains are measured
    against.  Taking it here rather than from the run record is what lets a
    probe be swept at all: one update rarely clears the campaign's learning
    gate, and a run that misses that gate never writes a record.
    """
    trials: dict[float, float] = {}

    def total_bits(scale: float) -> float:
        if scale not in trials:
            apply_adapter_tensors(
                session.model, scaled_adapter(initial, update, scale)
            )
            measured = engine._information_measure(
                session, run, data, parts=("heldout",)
            )["heldout"]
            trials[scale] = float(measured["total_bits"])
            if scale:
                print(
                    f"  scale {scale:8.3f} ->"
                    f" {baseline - trials[scale]:10.1f} held-out bits",
                    flush=True,
                )
        return trials[scale]

    for name, tensor in initial.items():
        if ".lora_B." in name and bool(tensor.any()):
            raise ValueError(
                f"{run.run_id}: {name} is not zero at attach, so scale zero is"
                " not the frozen model and the probe has no baseline"
            )
    baseline = total_bits(0.0)

    def saved(scale: float) -> float:
        return baseline - total_bits(scale)

    here = 1.0
    step = 2.0 if saved(2.0) > saved(1.0) else 0.5
    while floor <= here * step <= cap and saved(here * step) > saved(here):
        here *= step
    return {
        "scales": [
            {"scale": scale, "heldout_bits_saved": baseline - trials[scale]}
            for scale in sorted(trials)
        ],
        "scale": here,
        "heldout_bits_saved": baseline - trials[here],
        "baseline_heldout_bits": baseline,
        "at_search_edge": not floor <= here * step <= cap,
    }


def _sweep_one(
    engine: Any,
    session: ModelSession,
    run: RunSpec,
    run_dir: Path,
    data: dict[str, list[Any]],
    out_dir: Path,
    *,
    ranks: Iterable[int],
    rates: Iterable[tuple[int, float]],
    force: bool,
) -> dict[str, Any]:
    """Sweep one adapter, building the probe first when the run was capped.

    A capped run and a finished run reach the same sweep by different routes:
    the finished adapter is swept as it is against its own trained gain, the
    capped one is first rescaled along its update direction and then swept
    against the gain that rescaling reached.  The starting adapter is not
    stored anywhere because it does not have to be -- attaching with the run's
    own seed reproduces it exactly, which is also what makes the update
    direction well defined.
    """
    adapter_path = run_dir / "raw_channel.pt"
    if not adapter_path.is_file():
        raise FileNotFoundError(f"missing adapter: {adapter_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    session.attach(run.adapter, run.seed)
    try:
        trained = torch.load(adapter_path, map_location="cpu", weights_only=True)
        if run.training.max_updates is None:
            record = json.loads((run_dir / "metrics.json").read_text())
            baseline_bits = baseline_heldout_bits(record)
            tensors = trained
            ceiling = float(
                (record.get("raw_behavioral_write") or {}).get("heldout_bits_saved")
                or 0.0
            )
            search: dict[str, Any] = {}
        else:
            initial = adapter_tensors(session.model, run.adapter.method)
            update = update_direction(initial, trained)
            search = scale_search(engine, session, run, data, initial, update)
            baseline_bits = search["baseline_heldout_bits"]
            tensors = scaled_adapter(initial, update, search["scale"])
            ceiling = search["heldout_bits_saved"]
            # Cells are cached by rank and rate so an interrupted sweep can
            # resume, but they are cells of one particular adapter. A search
            # that lands somewhere new makes every stored cell stale.
            previous = out_dir / "probe.json"
            if previous.is_file():
                stored = json.loads(previous.read_text()).get("scale_search", {})
                force = force or stored.get("scale") != search["scale"]
        cells = sweep_tensors(
            engine,
            session,
            run,
            data,
            tensors,
            out_dir,
            baseline_bits=baseline_bits,
            ceiling=ceiling,
            ranks=ranks,
            rates=rates,
            force=force,
        )
    finally:
        session.unload()
    summary = {
        "run_id": run.run_id,
        "model_key": run.model.key,
        "dataset_key": run.dataset_key,
        "adapter_key": run.adapter.key,
        "seed": run.seed,
        "probe_updates": run.training.max_updates,
        "probe_rows": (
            int(run.training.max_updates) * int(run.training.effective_batch_size)
            if run.training.max_updates is not None
            else None
        ),
        "scale_search": search,
        "budget": budget_at(cells),
        "cells": len(cells),
    }
    (out_dir / "probe.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def find_targets(
    specs: list[RunSpec], runs_root: Path
) -> list[tuple[Path, RunSpec]]:
    """The finished run each capped probe is trying to predict.

    A target is not named in the probe's config, because this panel does not
    declare the runs it predicts -- they were trained by the campaigns that
    already paid for them, under several different configs. What identifies a
    pair is the receiver, the corpus, the seed and the container, so that is
    what is matched, and only against runs that were never capped.
    """
    wanted = {
        (spec.model.key, str(spec.dataset_key), spec.seed, spec.adapter.key)
        for spec in specs
        if spec.training.max_updates is not None
    }
    found: dict[tuple[str, str, int, str], tuple[Path, RunSpec]] = {}
    for path in sorted(Path(runs_root).glob("*/config.json")):
        try:
            other = RunSpec.from_dict(json.loads(path.read_text()))
        except Exception:
            continue
        if other.training.max_updates is not None:
            continue
        key = (other.model.key, str(other.dataset_key), other.seed, other.adapter.key)
        if key in wanted and key not in found:
            if (path.parent / "raw_channel.pt").is_file():
                found[key] = (path.parent, other)
    for key in sorted(key for key in wanted if key not in found):
        # stderr, so a caller can read this command's stdout as JSON.
        print(f"no finished adapter to predict for {key}", file=sys.stderr,
              flush=True)
    return list(found.values())


def training_contract(spec: RunSpec) -> tuple[Any, ...]:
    """What a run spent, as the tuple two targets have to agree on.

    Targets are matched by receiver, corpus, seed and container, because that
    is what a probe cell names. It is not enough on its own: several campaigns
    trained the same cell under their own study names, and if two of them had
    used different budgets the receiver axis would be carrying a difference in
    training recipe. They do not -- every target in this panel is four epochs
    at a batch of sixteen -- and the preflight now checks that rather than
    trusting it.
    """
    training = spec.training
    return (
        training.epochs,
        training.effective_batch_size,
        training.micro_batch_size,
        training.max_length,
        training.learning_rate,
        training.label_span,
    )


def check_panel(
    config_path: Path, runs_root: Path, prepared_root: Path
) -> dict[str, Any]:
    """Prove the panel can run before a single GPU is asked for.

    This study predicts adapters other campaigns trained and reads corpora
    other campaigns prepared, so both are things it can only find, never make.
    A missing one is silent at submit time and fatal an hour later, so the
    chain runs this first and stops on any gap.
    """
    from fineqcomp.campaign import expand_campaign
    from fineqcomp.data import load_natural_dataset

    campaign = load_campaign(config_path)
    panel = expand_campaign(campaign)
    targets = find_targets(panel, Path(runs_root))
    paired = {
        (spec.model.key, str(spec.dataset_key), spec.seed) for _, spec in targets
    }
    contracts = sorted({training_contract(spec) for _, spec in targets})
    orphans = sorted(
        {
            (run.model.key, str(run.dataset_key), run.seed)
            for run in panel
            if (run.model.key, str(run.dataset_key), run.seed) not in paired
        }
    )
    unprepared = []
    for dataset_key, seed in sorted(
        {(str(run.dataset_key), run.seed) for run in panel}
    ):
        try:
            load_natural_dataset(campaign, dataset_key, seed, Path(prepared_root))
        except Exception as error:
            unprepared.append({"dataset": dataset_key, "seed": seed,
                               "error": repr(error)})
    return {
        "probe_cells": len(panel),
        "cells_with_a_target": len(paired),
        "orphan_cells": orphans,
        "unprepared_cells": unprepared,
        "target_contracts": [list(contract) for contract in contracts],
        "ready": not orphans and not unprepared and len(contracts) == 1,
    }


def sweep_many(
    run_dirs: Iterable[Path],
    config_path: Path,
    prepared_root: Path,
    out_root: Path,
    *,
    ranks: Iterable[int] = DEFAULT_RANKS,
    rates: Iterable[tuple[int, float]] = DEFAULT_RATES,
    models: Iterable[str] = (),
    with_targets: bool = False,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Sweep a list of finished runs, loading each receiver's weights once.

    Loading a 9B model in NF4 costs more than a sweep does, and a shard holds
    many runs on the same receiver, so the session is owned here rather than
    per run.
    """
    from fineqcomp.runner import RunEngine

    directories = [Path(path).resolve() for path in run_dirs]
    specs = [
        RunSpec.from_dict(json.loads((path / "config.json").read_text()))
        for path in directories
    ]
    if with_targets:
        for path, spec in find_targets(specs, directories[0].parent):
            if path not in directories:
                directories.append(path)
                specs.append(spec)
    if models:
        keep = set(models)
        pairs = [
            (path, spec)
            for path, spec in zip(directories, specs, strict=True)
            if spec.model.key in keep
        ]
        directories = [path for path, _ in pairs]
        specs = [spec for _, spec in pairs]
    if not directories:
        return []
    engine = RunEngine(load_campaign(config_path), prepared_root, directories[0].parent)
    order = sorted(
        range(len(specs)),
        key=lambda index: (specs[index].model.key, str(specs[index].dataset_key)),
    )
    out_root = Path(out_root)
    summaries: list[dict[str, Any]] = []
    session: ModelSession | None = None
    loaded: str | None = None
    try:
        for index in order:
            run, run_dir = specs[index], directories[index]
            if loaded != run.model.key:
                del session
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                session = ModelSession.load(run.model)
                loaded = run.model.key
            print(f"sweeping {run.run_id}", flush=True)
            try:
                summaries.append(
                    _sweep_one(
                        engine,
                        session,
                        run,
                        run_dir,
                        engine._load_data(run)[0],
                        out_root / run.run_id,
                        ranks=ranks,
                        rates=rates,
                        force=force,
                    )
                )
            except Exception as error:
                # One unreadable run must not take the other forty in the
                # shard with it. The failure is recorded where the cells would
                # have gone, so the report can see that the cell is missing
                # rather than silently reading a shorter panel.
                failure = {"run_id": run.run_id, "error": repr(error)}
                (out_root / run.run_id).mkdir(parents=True, exist_ok=True)
                (out_root / run.run_id / "failed.json").write_text(
                    json.dumps(failure, indent=2, sort_keys=True)
                )
                print(f"  failed: {error!r}", flush=True)
                summaries.append(failure)
    finally:
        del session
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summaries


def shipped_ladder_budget(
    record: dict[str, Any], retention: float = DEFAULT_RETENTION
) -> dict[str, Any]:
    """The same crossing read off the codec ladder the run already measured.

    This is the campaign's R*, recomputed here so that every swept adapter
    reports the old number and the new one side by side.  The two are not the
    same kind of quantity: below one bit the shipped ladder drops a random
    subset of rank directions at a fixed container rank, while the budget above
    is free to spend the file on fewer directions at higher precision.
    """
    rungs = random_mask_ladder(record)
    if not rungs:
        return {"bits_per_value": float("nan"), "file_bits": float("nan")}
    ceiling = max(float(rung["heldout_bits_saved"]) for rung in rungs)
    target = retention * ceiling
    below: dict[str, Any] | None = None
    for rung in rungs:
        if float(rung["heldout_bits_saved"]) >= target:
            if below is None:
                break
            gap = float(rung["heldout_bits_saved"]) - float(
                below["heldout_bits_saved"]
            )
            share = (
                (target - float(below["heldout_bits_saved"])) / gap if gap > 0 else 0.0
            )
            return {
                "bits_per_value": float(below["effective_bits_per_value"])
                + share
                * (
                    float(rung["effective_bits_per_value"])
                    - float(below["effective_bits_per_value"])
                ),
                "file_bits": float(below["file_bits"])
                + share * (float(rung["file_bits"]) - float(below["file_bits"])),
            }
        below = rung
    return {"bits_per_value": float("nan"), "file_bits": float("nan")}


def _run_kind(run_dir: Path) -> tuple[str, int | None]:
    """Whether a swept run is a target or a probe, from its own config."""
    config = json.loads((run_dir / "config.json").read_text())
    updates = (config.get("training") or {}).get("max_updates")
    return ("probe", int(updates)) if updates is not None else ("trained", None)


def collect_budgets(
    sweep_root: Path, runs_root: Path, retention: float = DEFAULT_RETENTION
) -> list[dict[str, Any]]:
    """One row per swept adapter: its budget, and whether it was a probe."""
    rows: list[dict[str, Any]] = []
    for cell_dir in sorted(path for path in Path(sweep_root).iterdir() if path.is_dir()):
        cells = [
            json.loads(path.read_text())
            for path in sorted(cell_dir.glob("r*_b*.json"))
        ]
        if not cells:
            continue
        run_dir = Path(runs_root) / cell_dir.name
        if not (run_dir / "config.json").is_file():
            continue
        kind, updates = _run_kind(run_dir)
        budget = budget_at(cells, retention)
        # Only a finished run has a shipped ladder to compare against: a capped
        # probe rarely clears the learning gate, so it never writes a record.
        shipped = (
            shipped_ladder_budget(
                json.loads((run_dir / "metrics.json").read_text()), retention
            )
            if kind == "trained" and (run_dir / "metrics.json").is_file()
            else {"bits_per_value": float("nan"), "file_bits": float("nan")}
        )
        probe = cell_dir / "probe.json"
        search = json.loads(probe.read_text())["scale_search"] if probe.is_file() else {}
        rows.append(
            {
                "run_id": cell_dir.name,
                "model_key": cells[0]["model_key"],
                "dataset_key": cells[0]["dataset_key"],
                "seed": int(cells[0]["seed"]),
                "kind": kind,
                "probe_updates": updates,
                "probe_scale": search.get("scale"),
                "probe_scale_at_edge": search.get("at_search_edge"),
                "cells": len(cells),
                **{f"budget_{key}": value for key, value in budget.items()},
                "shipped_ladder_bits_per_value": shipped["bits_per_value"],
                "shipped_ladder_file_bits": shipped["file_bits"],
                "shipped_over_budget": (
                    shipped["bits_per_value"] / budget["bits_per_value"]
                    if budget["bits_per_value"]
                    else float("nan")
                ),
            }
        )
    return rows


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def pair_budgets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join every probe to the finished run it is trying to predict.

    Seeds are averaged inside a cell before pairing.  A probe and a target are
    the same cell when they share a receiver, a corpus and a seed; the probe
    ladder means one target can have several probes, one per update budget.
    """
    targets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    probes: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if not row["budget_bracketed"]:
            continue
        key = (row["model_key"], row["dataset_key"])
        if row["kind"] == "trained":
            targets.setdefault(key, []).append(row)
        else:
            probes.setdefault((*key, int(row["probe_updates"])), []).append(row)
    paired = []
    for (model_key, dataset_key, updates), group in sorted(probes.items()):
        target = targets.get((model_key, dataset_key))
        if not target:
            continue
        paired.append(
            {
                "model_key": model_key,
                "dataset_key": dataset_key,
                "probe_updates": updates,
                "probe_seeds": len(group),
                "target_seeds": len(target),
                "probe_file_bits": _mean([row["budget_file_bits"] for row in group]),
                "target_file_bits": _mean(
                    [row["budget_file_bits"] for row in target]
                ),
                "probe_bits_per_value": _mean(
                    [row["budget_bits_per_value"] for row in group]
                ),
                "target_bits_per_value": _mean(
                    [row["budget_bits_per_value"] for row in target]
                ),
                "probe_rank": _mean(
                    [float(row["budget_rank"] or float("nan")) for row in group]
                ),
                "target_rank": _mean(
                    [float(row["budget_rank"] or float("nan")) for row in target]
                ),
                "probe_ceiling": _mean(
                    [row["budget_ceiling_heldout_bits_saved"] for row in group]
                ),
                "target_ceiling": _mean(
                    [row["budget_ceiling_heldout_bits_saved"] for row in target]
                ),
            }
        )
    return paired


def _spearman(left: list[float], right: list[float]) -> float:
    """Rank correlation over the pairs where both sides are finite."""
    import numpy as np

    from fineqcomp.relative_info import _spearman as rank_correlation

    pairs = [
        (one, two)
        for one, two in zip(left, right, strict=True)
        if math.isfinite(one) and math.isfinite(two)
    ]
    if len(pairs) < 3:
        return float("nan")
    return rank_correlation(
        np.asarray([pair[0] for pair in pairs], dtype=float),
        np.asarray([pair[1] for pair in pairs], dtype=float),
    )


def _rmse(predicted: list[float], observed: list[float]) -> float:
    pairs = [
        (one, two)
        for one, two in zip(predicted, observed, strict=True)
        if math.isfinite(one) and math.isfinite(two)
    ]
    if not pairs:
        return float("nan")
    return math.sqrt(sum((a - b) ** 2 for a, b in pairs) / len(pairs))


def score_probe(paired: list[dict[str, Any]], updates: int) -> dict[str, Any]:
    """Identity-line accuracy of one probe budget against the receiver mean.

    The comparison is not a correlation.  `predicted` is the probe's own
    budget in bits per value, `observed` is the finished run's, and the error
    that matters is the distance to `predicted == observed`.  The baseline is
    the leave-one-out mean of the other receivers on the same corpus, which is
    what someone who had measured nothing would say.
    """
    group = [row for row in paired if row["probe_updates"] == updates]
    if not group:
        return {"probe_updates": updates, "arms": 0}
    predicted = [row["probe_bits_per_value"] for row in group]
    observed = [row["target_bits_per_value"] for row in group]
    baseline = []
    for index, row in enumerate(group):
        others = [
            other["target_bits_per_value"]
            for position, other in enumerate(group)
            if position != index and other["dataset_key"] == row["dataset_key"]
        ]
        baseline.append(_mean(others) if others else float("nan"))
    signs = []
    for corpus in sorted({row["dataset_key"] for row in group}):
        block = [row for row in group if row["dataset_key"] == corpus]
        for first in range(len(block)):
            for second in range(first + 1, len(block)):
                one, two = block[first], block[second]
                predicted_gap = one["probe_bits_per_value"] - two["probe_bits_per_value"]
                observed_gap = (
                    one["target_bits_per_value"] - two["target_bits_per_value"]
                )
                if predicted_gap == 0 or observed_gap == 0:
                    continue
                signs.append((predicted_gap > 0) == (observed_gap > 0))
    return {
        "probe_updates": updates,
        "arms": len(group),
        "corpora": len({row["dataset_key"] for row in group}),
        "receivers": len({row["model_key"] for row in group}),
        "identity_rmse": _rmse(predicted, observed),
        "corpus_mean_rmse": _rmse(baseline, observed),
        "spearman": _spearman(predicted, observed),
        "receiver_pair_sign_accuracy": (
            sum(signs) / len(signs) if signs else float("nan")
        ),
        "receiver_pairs": len(signs),
    }


def check_gates(
    rows: list[dict[str, Any]], paired: list[dict[str, Any]], gates: dict[str, Any]
) -> dict[str, Any]:
    """The pre-registered decision rule, read off whatever has finished."""
    swept = [row for row in rows if row["cells"]]
    bracketed = [row for row in swept if row["budget_bracketed"]]
    scores = [
        score_probe(paired, updates)
        for updates in sorted({row["probe_updates"] for row in paired})
    ]
    best = max(
        (score for score in scores if score.get("arms")),
        key=lambda score: (
            -score["identity_rmse"]
            if math.isfinite(score["identity_rmse"])
            else -math.inf
        ),
        default={},
    )
    p1 = bool(swept) and len(bracketed) / len(swept) >= float(gates["bracketed_share"])
    p2 = bool(best) and best.get("spearman", float("nan")) >= float(
        gates["probe_spearman"]
    )
    p3 = bool(best) and best.get("identity_rmse", math.inf) <= float(
        gates["identity_over_corpus_mean"]
    ) * best.get("corpus_mean_rmse", 0.0)
    p4 = bool(best) and best.get(
        "receiver_pair_sign_accuracy", float("nan")
    ) >= float(gates["receiver_sign_accuracy"])
    return {
        "swept_runs": len(swept),
        "bracketed_runs": len(bracketed),
        "scores": scores,
        "best_probe": best,
        "p1_budgets_are_bracketed": p1,
        "p2_probe_orders_the_arms": p2,
        "p3_identity_beats_the_corpus_mean": p3,
        "p4_probe_orders_receivers_on_one_corpus": p4,
        "passed": bool(p1 and p2 and p3 and p4),
    }


def write_budget_report(
    sweep_root: Path,
    runs_root: Path,
    out_dir: Path,
    gates: dict[str, Any],
    retention: float = DEFAULT_RETENTION,
) -> dict[str, Any]:
    from fineqcomp.artifacts import write_json
    from fineqcomp.relative_validation import _write_csv

    rows = collect_budgets(Path(sweep_root), Path(runs_root), retention)
    paired = pair_budgets(rows)
    report = check_gates(rows, paired, gates)
    out_dir = Path(out_dir)
    _write_csv(out_dir / "budgets.csv", rows)
    _write_csv(out_dir / "probe_vs_trained.csv", paired)
    _write_csv(out_dir / "probe_scores.csv", report["scores"])
    write_json(out_dir / "summary.json", report)
    return report
