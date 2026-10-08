#!/usr/bin/env python
"""Plan, check and collect the reparameterization ablation on Jean-Zay.

`reparameterization.sh` calls `plan` before it submits anything. The jobs call
`cells`, `data-check`, `smoke-check` and `collect`, and `status` reports
progress at any time. Nothing here needs a GPU, and `collect` never raises:
whatever it cannot read goes into the summary instead.

Usage: python scripts/jean_zay/reparameterization.py COMMAND --state DIR [...]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

CONFIG = Path(os.environ.get("FQ_CONFIG", REPO / "configs/rate/reparameterization.yaml"))

# The post-hoc sweep. identity reproduces the run's own ladder; diagonal is
# a symmetry of the codec and must match it (kept to a 100x range: wider
# rescaling pushes small B columns' fp16 scales into underflow); permutations
# only redraw the codec's random masks and give its noise floor; the rest are
# the test. FQ_GAUGES replaces the list, for a cheaper rerun or a rehearsal.
GAUGES = tuple(os.environ.get("FQ_GAUGES", "").split()) or (
    "identity", "diagonal-k100",
    "permutation-d0", "permutation-d1", "permutation-d2",
    "orthogonal-d0", "orthogonal-d1", "orthogonal-d2",
    "conditioned-k10-d0", "conditioned-k10-d1",
    "conditioned-k100-d0", "conditioned-k100-d1",
    "conditioned-k1000-d0", "conditioned-k1000-d1",
    "svd",
)
SMOKE_GAUGES = ("identity", "conditioned-k100-d0", "svd")
SWEEP_SEED = 11
# Walltimes in minutes on an H100 for a 7-9B receiver: the panel's measured
# durations plus a third (Mistral's code adapter trains in 104 min and its
# baseline and ladder add about 30). A full gpu_p6 starts jobs by backfill, so
# each array asks for what its slowest task needs, scaled by receiver size;
# the launcher doubles them on A100s.
MINUTES = {
    "train": {"kind_code": 210, "panel_math": 180, "xbrl_tags": 90},
    "sweep": {"kind_code": 240, "panel_math": 180, "xbrl_tags": 240},
    "frontier": {"kind_code": 90, "panel_math": 60, "xbrl_tags": 90},
}
SIZE = {"qwen25_0p5b_base": 0.4, "qwen25_1p5b_base": 0.5, "qwen25_3b_base": 0.7,
        "qwen25_14b_base": 1.3}
PROFILE_MINUTES = 90
# One pilot run per receiver, covering every corpus's scorer and one
# non-default initialisation. The last one also takes the gauge smoke.
SMOKE_CELLS = (
    ("llama31_8b_base", "xbrl_tags", "default"),
    ("qwen25_7b_base", "panel_math", "default"),
    ("mistral_7b_base", "kind_code", "conditioned"),
)
# Example-id fingerprints of the prepared splits, identical for every seed.
# Built locally with the Jean-Zay package versions; the seed-11 calibration
# order also matches the ids recorded by the paper's Jean-Zay runs.
DATA_IDS = {
    "kind_code": {"train": "2ac32f13ea8015e6", "calibration": "7767e301edaaa2c5", "test": "bbae451a24d2543e"},
    "panel_math": {"train": "6126e971f6272326", "calibration": "8a1ac08076e9272f", "test": "95979f484a11d16b"},
    "xbrl_tags": {"train": "9cb46b35602184b2", "calibration": "1aed8d0a45cd87b6", "test": "50d9779f2ffcc45d"},
}
PAPER_TARGETS = REPO / "results/rate/adapter_spectrum/attenuation_predictor/replication/targets.csv"


class State:
    """Every path the pipeline writes, under one root."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.manifest = self.root / "manifest.jsonl"
        self.prepared = self.root / "prepared"
        self.runs = self.root / "runs"
        self.lists = self.root / "lists"
        self.sweep = self.root / "sweep"
        self.frontier = self.root / "frontier"
        self.profile = self.root / "profile"
        self.smoke = self.root / "smoke"
        self.report = self.root / "report"
        self.logs = self.root / "logs"


def _campaign():
    from fineqcomp.campaign import expand_campaign
    from fineqcomp.config import load_campaign

    campaign = load_campaign(CONFIG)
    return campaign, expand_campaign(campaign)


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open() as stream:
            return list(csv.DictReader(stream))
    except FileNotFoundError:
        return []


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def model_problem(path: str) -> str | None:
    """Why a model snapshot cannot be loaded by this account, or None."""
    root = Path(path)
    try:
        if not root.is_dir():
            return f"{root} is missing"
        if not (root / "config.json").is_file():
            return f"{root}/config.json is missing"
        (root / "config.json").read_bytes()
        if not any((root / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
            return f"{root} has no tokenizer"
        shards = sorted(root.glob("*.safetensors"))
        if not shards:
            return f"{root} has no safetensors weights"
        for shard in shards:
            with shard.open("rb") as stream:
                stream.read(1)
    except PermissionError as error:
        return f"permission denied: {error.filename}"
    except OSError as error:
        return f"unreadable: {error}"
    return None


def _status(state: State, run_id: str, runs_root: Path | None = None) -> str:
    from fineqcomp.artifacts import read_json

    status = read_json((runs_root or state.runs) / run_id / "status.json", {}) or {}
    return str(status.get("state", "pending"))


def _done(state: State, run) -> bool:
    from fineqcomp.artifacts import run_complete

    return run_complete(state.runs / run.run_id, tuple(codec.key for codec in run.codecs))


def gauge_rows(state: State) -> list[dict[str, str]]:
    return [row for path in sorted(state.sweep.glob("*/gauge_0.csv")) for row in _read_csv(path)]


def attenuation_rows(state: State) -> dict[str, dict[str, str]]:
    """Latest attenuation profile per run, from the sweep and the profile jobs."""
    paths = sorted(state.sweep.glob("*/attenuation_0.csv")) + sorted(
        state.profile.glob("attenuation_*_0.csv"), key=lambda p: p.stat().st_mtime)
    return {row["run_id"]: row for path in paths for row in _read_csv(path)}


def _sweep_runs(runs, excluded: set[str]):
    """The seed-11 default adapters of the panel receivers: the factorization sweep."""
    return [run for run in runs if run.seed == SWEEP_SEED and run.adapter.init == "default"
            and not run.study.startswith("scale_") and run.model.key not in excluded]


def _frontier_runs(runs, excluded: set[str]):
    """Adapters whose R* is also read under rank truncation: the sweep's and the scale ladder's."""
    return _sweep_runs(runs, excluded) + [run for run in runs if run.study.startswith("scale_")
                                          and run.model.key not in excluded]


def _profile_runs(runs, excluded: set[str]):
    """Adapters that get an attenuation probe outside the sweep, for the rate rule."""
    return [run for run in runs if run.study.startswith(("init_", "scale_"))
            and run.model.key not in excluded]


def frontier_cells(state: State, run_id: str) -> list[dict[str, Any]]:
    return [json.loads(path.read_text())
            for path in sorted((state.frontier / run_id).glob("r*_b*.json"))]


def _minutes(stage: str, run) -> int:
    return round(MINUTES[stage][run.dataset_key] * SIZE.get(run.model.key, 1.0))


def plan(state: State) -> dict[str, Any]:
    """Write the manifest and the work that remains, as arrays the launcher submits.

    `arrays.tsv` has one line per array: stage, list file, tasks, minutes on an
    H100, and the training lists it waits for (or `-`). An array never mixes
    corpora or walltimes, so each asks for what its own tasks need.
    """
    from fineqcomp.campaign import write_manifest
    from fineqcomp.studies.rank_frontier import DEFAULT_RANKS, DEFAULT_RATES

    campaign, runs = _campaign()
    write_manifest(runs, state.manifest)
    used = sorted({run.model.key for run in runs})
    problems = {key: problem for key in used
                if (problem := model_problem(campaign["models"][key]["name"]))}
    excluded = set(problems)
    usable = [run for run in runs if run.model.key not in excluded]
    sweep = _sweep_runs(runs, excluded)
    sweep_ids = {run.run_id for run in sweep}

    def group(run) -> str:
        if run.run_id in sweep_ids:
            return "e1"
        return "scale" if run.study.startswith("scale_") else "e2"

    training = [run for run in usable if not _done(state, run)]
    measured: dict[str, set[str]] = {}
    for row in gauge_rows(state):
        measured.setdefault(row["run_id"], set()).add(row["gauge"])
    sweep_todo = [run for run in sweep if _status(state, run.run_id) != "screened_out"
                  and not set(GAUGES) <= measured.get(run.run_id, set())]
    full = len(DEFAULT_RATES) * len(set(DEFAULT_RANKS) | {16})
    frontier_todo = [run for run in _frontier_runs(runs, excluded)
                     if _status(state, run.run_id) != "screened_out"
                     and len(frontier_cells(state, run.run_id)) < full]
    profiled = attenuation_rows(state)
    profile_todo = [run for run in _profile_runs(runs, excluded) if run.run_id not in profiled]
    smoke = []
    for model, dataset, init in SMOKE_CELLS:
        smoke += [run for run in usable if (run.model.key, run.dataset_key, run.adapter.init,
                                            run.seed) == (model, dataset, init, SWEEP_SEED)][:1]
    covered = {run.model.key for run in smoke}
    for key in sorted({run.model.key for run in usable if not run.study.startswith("scale_")} - covered):
        smoke.append(next(run for run in usable if run.model.key == key))
    frontier_probe = next((run for run in smoke if run.dataset_key == "panel_math"), smoke[0] if smoke else None)
    marker = state.smoke / "PASSED"
    smoke_needed = not (marker.is_file() and marker.read_text().strip() == _git_head())

    state.lists.mkdir(parents=True, exist_ok=True)
    for stale in state.lists.glob("*.txt"):
        stale.unlink()
    lists: dict[str, list] = {"smoke": smoke, "smoke_frontier": [frontier_probe] if frontier_probe else []}
    arrays: list[str] = []
    trainer: dict[str, str] = {}

    def add(stage: str, name: str, chosen: list, minutes: int, needs: set[str]) -> None:
        lists[name] = chosen
        arrays.append(f"{stage} {name}.txt {len(chosen)} {minutes} {','.join(sorted(needs)) or '-'}\n")

    for dataset in sorted({run.dataset_key for run in runs}):
        batches: dict[tuple[str, int], list] = {}
        for run in training:
            if run.dataset_key == dataset:
                batches.setdefault((group(run), _minutes("train", run)), []).append(run)
        for (name, minutes), chosen in sorted(batches.items()):
            list_name = f"train_{name}_{dataset}_{minutes}m"
            add("train", list_name, chosen, minutes, set())
            trainer.update({run.run_id: f"{list_name}.txt" for run in chosen})
        for stage, todo in (("sweep", sweep_todo), ("frontier", frontier_todo)):
            batches = {}
            for run in todo:
                if run.dataset_key == dataset:
                    batches.setdefault(_minutes(stage, run), []).append(run)
            for minutes, chosen in sorted(batches.items()):
                add(stage, f"{stage}_{dataset}_{minutes}m", chosen, minutes,
                    {trainer[run.run_id] for run in chosen if run.run_id in trainer})
    if profile_todo:
        # One task walks the whole list, loading each receiver once.
        lists["profile"] = profile_todo
        needs = {trainer[run.run_id] for run in profile_todo if run.run_id in trainer}
        arrays.append(f"profile profile.txt 1 {PROFILE_MINUTES} {','.join(sorted(needs)) or '-'}\n")
    for name, chosen in lists.items():
        (state.lists / f"{name}.txt").write_text("".join(f"{run.run_id}\n" for run in chosen))
    (state.lists / "arrays.tsv").write_text("".join(arrays))
    (state.lists / "gauges.txt").write_text(" ".join(GAUGES) + "\n")
    (state.lists / "smoke_gauges.txt").write_text(" ".join(SMOKE_GAUGES) + "\n")
    counts = {
        "N_TRAIN": len(training), "N_SWEEP": len(sweep_todo), "N_FRONTIER": len(frontier_todo),
        "N_PROFILE": len(profile_todo), "N_SMOKE": len(smoke),
        "SMOKE_NEEDED": int(smoke_needed and bool(smoke)), "N_RUNS": len(runs),
        "N_USABLE": len(usable), "EXCLUDED_MODELS": ",".join(sorted(excluded)),
    }
    (state.lists / "counts.env").write_text("".join(f"{key}={value}\n" for key, value in counts.items()))
    summary = {**counts, "model_problems": problems, "config": str(CONFIG), "git": _git_head()}
    (state.lists / "plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def cells(state: State, run_ids: list[str]) -> str:
    """CSV rows that adapter_spectrum_profile.py reads, for the given runs."""
    _, runs = _campaign()
    by_id = {run.run_id: run for run in runs}
    lines = ["run_id,model_key,dataset_key,seed"]
    for run_id in run_ids:
        run = by_id[run_id]
        lines.append(f"{run.run_id},{run.model.key},{run.dataset_key},{run.seed}")
    return "\n".join(lines) + "\n"


def present_cells(state: State, list_name: str) -> list[str]:
    """Runs of a list that have a trained adapter and still lack a profile."""
    wanted = [line.strip() for line in (state.lists / f"{list_name}.txt").read_text().splitlines()
              if line.strip()]
    profiled = attenuation_rows(state)
    return [run_id for run_id in wanted if (state.runs / run_id / "raw_channel.pt").is_file()
            and run_id not in profiled]


def data_check(state: State) -> dict[str, Any]:
    """Compare the prepared splits with the paper's rows; warn, never fail."""
    report: dict[str, Any] = {"mismatches": [], "checked": 0}
    for directory in sorted(state.prepared.glob("natural/*/seed*")):
        expected = DATA_IDS.get(directory.parent.name)
        if expected is None:
            continue
        for split, digest in expected.items():
            path = directory / f"{split}.jsonl"
            ids = [json.loads(line)["example_id"] for line in path.open()] if path.is_file() else []
            found = hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]
            report["checked"] += 1
            if found != digest:
                report["mismatches"].append(f"{directory.parent.name}/{directory.name}/{split}")
    (state.root / "data_check.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def smoke_check(state: State) -> list[str]:
    """Everything the pilot jobs had to produce; an empty list means it passed."""
    from fineqcomp.artifacts import read_json

    problems = []
    smoke_ids = [line.strip() for line in (state.lists / "smoke.txt").read_text().splitlines() if line.strip()]
    _, runs = _campaign()
    by_id = {run.run_id: run for run in runs}
    for run_id in smoke_ids:
        root = state.smoke / "runs" / run_id
        status = read_json(root / "status.json", {}) or {}
        if status.get("state") != "complete":
            problems.append(f"{run_id}: state {status.get('state')!r} {status.get('error', '')}")
            continue
        record = read_json(root / "metrics.json", {}) or {}
        saved = [(codec.get("behavioral_write") or {}).get("heldout_bits_saved_per_token")
                 for codec in record.get("codecs", [])]
        if len(saved) != len(by_id[run_id].codecs) or any(_float(v) is None for v in saved):
            problems.append(f"{run_id}: codec ladder incomplete or not finite")
        if not (root / "raw_channel.pt").is_file():
            problems.append(f"{run_id}: no saved adapter")
    rows = _read_csv(state.smoke / "gauge_0.csv")
    if sorted(row["gauge"] for row in rows) != sorted(SMOKE_GAUGES):
        problems.append(f"gauge smoke wrote {[row['gauge'] for row in rows]}")
    identity = next((row for row in rows if row["gauge"] == "identity"), None)
    for row in rows:
        drift, gain = _float(row.get("update_drift")), _float(row.get("raw_gain"))
        if drift is None or drift > 1e-3:
            problems.append(f"gauge {row['gauge']}: update drift {row.get('update_drift')}")
        if gain is None or _float(row.get("binary_projection")) is None:
            problems.append(f"gauge {row['gauge']}: non-finite gain or projection")
        elif identity and _float(identity.get("raw_gain")) is not None and abs(
                gain - float(identity["raw_gain"])) > 0.01:
            problems.append(f"gauge {row['gauge']}: uncoded gain moved by "
                            f"{gain - float(identity['raw_gain']):+.4f} bits/token")
    attenuation = _read_csv(state.smoke / "attenuation_0.csv")
    if len(attenuation) != 1 or _float(attenuation[0].get("attenuation_binary_projection")) is None:
        problems.append("attenuation smoke missing or not finite")
    probes = [json.loads(path.read_text()) for path in sorted((state.smoke / "frontier").glob("r*_b*.json"))]
    if len(probes) != 12 or any(_float(cell.get("heldout_bits_saved")) is None for cell in probes):
        problems.append(f"rank-truncation smoke wrote {len(probes)} of 12 finite cells")
    if not problems:
        (state.smoke / "PASSED").write_text(_git_head() + "\n")
    return problems


def _alpha90(row: dict[str, str]) -> float | None:
    """Smallest scale of the uncoded update that keeps 90% of its gain."""
    from fineqcomp.rstar import r_star

    try:
        base = float(row["attenuation_loss_0"])
        gain = base - float(row["attenuation_loss_1"])
        if gain <= 0:
            return None
        points = [{"scale": scale, "gain": base - float(row[f"attenuation_loss_{scale:g}"])}
                  for scale in (0, 0.125, 0.25, 0.5, 1)]
        return r_star(points, rate_key="scale", value_key="gain", reference=gain)["r_star"]
    except (KeyError, TypeError, ValueError):
        return None


def _truncation_r_star(state: State, run_id: str, record: dict[str, Any]) -> float | None:
    """R*(0.90) when the codec keeps the strongest singular directions (rank-frontier cells).

    Rate is file bits per value of the full rank-16 adapter, so it is on the same
    axis as the uniform ladder's; the reference is the best of the raw and coded
    gains, as for the paper's R*.
    """
    from fineqcomp.rstar import r_star

    cells = frontier_cells(state, run_id)
    values = record.get("nominal_channel_values")
    raw = (record.get("raw_behavioral_write") or {}).get("heldout_bits_saved")
    if not cells or not values or raw is None:
        return None
    points = [{"rate": cell["file_bits"] / values, "saved": cell["heldout_bits_saved"]} for cell in cells]
    reference = max([float(raw)] + [point["saved"] for point in points])
    return r_star(points, rate_key="rate", value_key="saved", reference=reference)["r_star"]


def _mean(values: list[float | None]) -> float | None:
    numbers = [v for v in values if v is not None]
    return statistics.fmean(numbers) if numbers else None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _run_table(state: State, runs) -> list[dict[str, Any]]:
    from fineqcomp.artifacts import read_json
    from fineqcomp.rstar import from_run, r_star

    paper = {(row["model_key"], row["dataset_key"], int(row["seed"])): _float(row["r_star"])
             for row in _read_csv(PAPER_TARGETS)}
    profiles = attenuation_rows(state)
    table = []
    for run in runs:
        root = state.runs / run.run_id
        status = read_json(root / "status.json", {}) or {}
        row: dict[str, Any] = {
            "run_id": run.run_id, "study": run.study, "model_key": run.model.key,
            "dataset_key": run.dataset_key, "seed": run.seed, "init": run.adapter.init,
            "state": status.get("state", "pending"), "error": status.get("error", ""),
        }
        if (root / "metrics.json").is_file():
            record = read_json(root / "metrics.json", {}) or {}
            row["truncation_r_star"] = _truncation_r_star(state, run.run_id, record)
            result = from_run(root)
            raw = result.get("raw_reference")
            row.update(r_star=result.get("r_star"), bracketed=result.get("bracketed"),
                       raw_gain_bits_per_token=raw,
                       r_star_raw=r_star([{"effective_bits_per_value": rate,
                                           "heldout_bits_saved_per_token": value * result["reference"]}
                                          for rate, value in result.get("points", [])],
                                         reference=raw)["r_star"] if raw else None)
            training = read_json(root / "training_metrics.json", {}) or {}
            row["train_minutes"] = round(float(training.get("elapsed_seconds", 0)) / 60, 1)
        if run.adapter.init == "default":
            row["paper_r_star"] = paper.get((run.model.key, run.dataset_key, run.seed))
        profile = profiles.get(run.run_id)
        if profile:
            row["alpha90"] = _alpha90(profile)
            row["gamma"] = _float(profile.get("attenuation_binary_projection"))
            if row["alpha90"] is not None and row["gamma"]:
                row["rule_r_star"] = row["alpha90"] / row["gamma"]
        table.append(row)
    return table


def _gauge_table(state: State, run_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alpha = {row["run_id"]: row.get("alpha90") for row in run_table}
    meta = {row["run_id"]: row for row in run_table}
    rows = gauge_rows(state)
    identity = {row["run_id"]: row for row in rows if row["gauge"] == "identity"}
    table = []
    for row in rows:
        base = identity.get(row["run_id"], {})
        r, r0 = _float(row.get("r_star")), _float(base.get("r_star"))
        gamma, gamma0 = _float(row.get("binary_projection")), _float(base.get("binary_projection"))
        a = alpha.get(row["run_id"])
        info = meta.get(row["run_id"], {})
        table.append({
            "run_id": row["run_id"], "model_key": info.get("model_key"),
            "dataset_key": info.get("dataset_key"), "gauge": row["gauge"],
            "gauge_kind": row.get("gauge_kind"), "kappa": _float(row.get("gauge_kappa")),
            "r_star": r, "r_star_raw": _float(row.get("r_star_raw")),
            "bracketed": row.get("bracketed") == "True",
            "delta_r_star": None if r is None or r0 is None else r - r0,
            "gamma": gamma, "delta_gamma": None if gamma is None or gamma0 is None else gamma - gamma0,
            "rule_r_star": a / gamma if a is not None and gamma else None,
            "rule_delta_r_star": (a / gamma - a / gamma0) if a is not None and gamma and gamma0 else None,
            "raw_gain": _float(row.get("raw_gain")),
            "raw_gain_shift": (None if _float(row.get("raw_gain")) is None
                               or _float(base.get("raw_gain")) is None
                               else float(row["raw_gain"]) - float(base["raw_gain"])),
            "update_drift": _float(row.get("update_drift")),
            "runner_r_star": info.get("r_star"),
        })
    return table


def _family(gauge: str) -> str:
    kind, *rest = gauge.split("-")
    kappa = next((part for part in rest if part.startswith("k")), "")
    return f"{kind}-{kappa}" if kind == "conditioned" else kind


WELL_CONDITIONED = ("diagonal", "permutation", "orthogonal", "conditioned-k10", "svd")
GATE_RMSE = 0.10
GATE_SIGN = 0.80


def gates(run_table: list[dict[str, Any]], gauge_table: list[dict[str, Any]]) -> dict[str, Any]:
    """The tests written down in results/rate/reparameterization/README.md before any result.

    The rule is R*(0.90) ~ alpha90 / gamma with no fitted coefficient. G1: new
    adapters (non-default initialisations and the scale ladder). G2: the sweep's
    well-conditioned factorizations, whose alpha90 is the identity's. G3: the sign
    of each factorization's R* shift, where the shift exceeds that adapter's
    codec noise (its largest shift over the three permutation draws).
    """
    def rmse(pairs):
        return (sum((p - m) ** 2 for p, m in pairs) / len(pairs)) ** 0.5 if pairs else None

    new = [row for row in run_table
           if (row["study"].startswith("init_") and row["init"] != "default") or row["study"].startswith("scale_")]
    g1 = [(row["rule_r_star"], row["r_star"]) for row in new
          if row.get("rule_r_star") is not None and row.get("r_star") is not None and row.get("bracketed")]
    moved_rows = [row for row in gauge_table if row["gauge"] != "identity"]
    g2 = [(row["rule_r_star"], row["r_star"]) for row in moved_rows
          if _family(row["gauge"]) in WELL_CONDITIONED and row["bracketed"]
          and row["rule_r_star"] is not None and row["r_star"] is not None]
    floor: dict[str, float] = {}
    for row in moved_rows:
        if row["gauge_kind"] == "permutation" and row["delta_r_star"] is not None:
            floor[row["run_id"]] = max(floor.get(row["run_id"], 0.0), abs(row["delta_r_star"]))
    beyond = [row for row in moved_rows if row["delta_r_star"] is not None and row["rule_delta_r_star"] is not None
              and abs(row["delta_r_star"]) > floor.get(row["run_id"], 0.0)]
    agree = [(row["rule_delta_r_star"] > 0) == (row["delta_r_star"] > 0) for row in beyond]
    ill = [(row["rule_r_star"], row["r_star"]) for row in moved_rows
           if _family(row["gauge"]) in ("conditioned-k100", "conditioned-k1000") and row["bracketed"]
           and row["rule_r_star"] is not None and row["r_star"] is not None]

    def verdict(value, threshold, below):
        if value is None:
            return "no data"
        return "PASS" if (value <= threshold if below else value >= threshold) else "FAIL"

    share = sum(agree) / len(agree) if agree else None
    return {
        "G1 rule on new adapters (RMSE, bits/value)": {
            "n": len(g1), "value": rmse(g1), "threshold": GATE_RMSE, "verdict": verdict(rmse(g1), GATE_RMSE, True)},
        "G2 rule on well-conditioned factorizations (RMSE)": {
            "n": len(g2), "value": rmse(g2), "threshold": GATE_RMSE, "verdict": verdict(rmse(g2), GATE_RMSE, True)},
        "G3 sign of R* shifts beyond codec noise (share)": {
            "n": len(agree), "value": share, "threshold": GATE_SIGN, "verdict": verdict(share, GATE_SIGN, False)},
        "ill-conditioned factorizations, not gated (RMSE; mean predicted minus measured)": {
            "n": len(ill), "value": rmse(ill),
            "bias": _mean([p - m for p, m in ill]) if ill else None},
    }


def _summary_markdown(state: State, run_table, gauge_table, notes: list[str]) -> str:
    lines = ["# Reparameterization ablation: collected results", "",
             f"Collected {time.strftime('%Y-%m-%d %H:%M')} from `{state.root}`, code {_git_head()[:10]}.", ""]
    states: dict[str, int] = {}
    for row in run_table:
        states[row["state"]] = states.get(row["state"], 0) + 1
    lines += ["## Runs", "", ", ".join(f"{k}: {v}" for k, v in sorted(states.items())), ""]
    failed = [row for row in run_table if row["state"] not in {"complete"}]
    for row in failed:
        lines.append(f"- `{row['run_id']}`: {row['state']} {str(row.get('error', ''))[:160]}")
    lines += ["", "## Seed-11 default adapters against the paper", "",
              "| model | corpus | R* runner | R* sweep identity | R* paper | gamma |",
              "|---|---|---|---|---|---|"]
    identity = {row["run_id"]: row for row in gauge_table if row["gauge"] == "identity"}
    for row in run_table:
        if row["seed"] == SWEEP_SEED and row["init"] == "default":
            lines.append(f"| {row['model_key']} | {row['dataset_key']} | {_fmt(row.get('r_star'))} | "
                         f"{_fmt(identity.get(row['run_id'], {}).get('r_star'))} | "
                         f"{_fmt(row.get('paper_r_star'))} | {_fmt(row.get('gamma'))} |")
    lines += ["", "## R* under reparameterization (change from the identity factorization)", "",
              "| gauge family | rows | mean ΔR* | mean abs ΔR* | max abs ΔR* | mean Δgamma | rule mean ΔR* | max uncoded gain shift |",
              "|---|---|---|---|---|---|---|---|"]
    families: dict[str, list[dict[str, Any]]] = {}
    for row in gauge_table:
        families.setdefault(_family(row["gauge"]), []).append(row)
    for family, rows in sorted(families.items()):
        deltas = [row["delta_r_star"] for row in rows if row["delta_r_star"] is not None]
        shifts = [abs(row["raw_gain_shift"]) for row in rows if row["raw_gain_shift"] is not None]
        lines.append(
            f"| {family} | {len(rows)} | {_fmt(_mean(deltas))} | "
            f"{_fmt(_mean([abs(d) for d in deltas]))} | {_fmt(max(map(abs, deltas)) if deltas else None)} | "
            f"{_fmt(_mean([row['delta_gamma'] for row in rows]))} | "
            f"{_fmt(_mean([row['rule_delta_r_star'] for row in rows]))} | {_fmt(max(shifts) if shifts else None, 5)} |")
    lines += ["", "## Pre-registered tests of the rate rule", "",
              "| test | n | value | threshold | verdict |", "|---|---|---|---|---|"]
    for name, gate in gates(run_table, gauge_table).items():
        lines.append(f"| {name} | {gate['n']} | {_fmt(gate['value'])} | "
                     f"{_fmt(gate.get('threshold'))} | {gate.get('verdict', _fmt(gate.get('bias')))} |")
    lines += ["", "## Second codec: keep the strongest singular directions", "",
              "| model | corpus | R* uniform | R* truncation |", "|---|---|---|---|"]
    both = [row for row in run_table if row.get("truncation_r_star") is not None]
    for row in both:
        lines.append(f"| {row['model_key']} | {row['dataset_key']} | {_fmt(row.get('r_star'))} | "
                     f"{_fmt(row['truncation_r_star'])} |")
    paired = [row for row in both if row.get("r_star") is not None]
    if len(paired) > 2:
        from fineqcomp.rstar import kendall_tau_b

        lines.append("")
        lines.append(f"Kendall tau-b between the two codecs' R*, all {len(paired)} adapters: "
                     f"{kendall_tau_b([r['r_star'] for r in paired], [r['truncation_r_star'] for r in paired]):.2f}.")
        for dataset in sorted({row["dataset_key"] for row in paired}):
            group = [row for row in paired if row["dataset_key"] == dataset]
            if len(group) > 2:
                tau = kendall_tau_b([r["r_star"] for r in group], [r["truncation_r_star"] for r in group])
                lines.append(f"Receiver order within {dataset} ({len(group)} receivers): tau-b {tau:.2f}.")
    lines += ["", "## Receiver scale (Qwen2.5, seed 11)", "",
              "| corpus | model | state | R* uniform | R* truncation | rule |", "|---|---|---|---|---|---|"]
    sizes = ("qwen25_0p5b_base", "qwen25_1p5b_base", "qwen25_3b_base", "qwen25_7b_base", "qwen25_14b_base")
    for row in sorted((r for r in run_table if r["model_key"] in sizes and r["seed"] == SWEEP_SEED
                       and r["init"] == "default"),
                      key=lambda r: (r["dataset_key"], sizes.index(r["model_key"]))):
        lines.append(f"| {row['dataset_key']} | {row['model_key']} | {row['state']} | {_fmt(row.get('r_star'))} | "
                     f"{_fmt(row.get('truncation_r_star'))} | {_fmt(row.get('rule_r_star'))} |")
    lines += ["", "## Initialisation (Mistral-7B, three seeds)", "",
              "| corpus | init | n | R* mean | R* sd | gamma mean |", "|---|---|---|---|---|---|"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in run_table:
        if row["study"].startswith("init_"):
            groups.setdefault((row["dataset_key"], row["init"]), []).append(row)
    for (dataset, init), rows in sorted(groups.items()):
        values = [row["r_star"] for row in rows if row.get("r_star") is not None]
        lines.append(f"| {dataset} | {init} | {len(values)} | {_fmt(_mean(values))} | "
                     f"{_fmt(statistics.stdev(values) if len(values) > 1 else None)} | "
                     f"{_fmt(_mean([row.get('gamma') for row in rows]))} |")
    if notes:
        lines += ["", "## Collection notes", ""] + [f"- {note}" for note in notes]
    lines += ["", "Files: `runs.csv` (one row per trained adapter), `gauges.csv` (one row per",
              "adapter and factorization), `plan.json`, `data_check.json`, Slurm logs under `logs/`.", ""]
    return "\n".join(lines)


def collect(state: State, archive: Path | None = None) -> Path:
    """Tables, a summary and one archive to send back. Never raises."""
    notes: list[str] = []
    run_table: list[dict[str, Any]] = []
    gauge_table: list[dict[str, Any]] = []
    state.report.mkdir(parents=True, exist_ok=True)
    try:
        _, runs = _campaign()
        run_table = _run_table(state, runs)
        _write_csv(state.report / "runs.csv", run_table)
    except Exception:
        notes.append("run table failed: " + traceback.format_exc(limit=3).replace("\n", " "))
    try:
        gauge_table = _gauge_table(state, run_table)
        _write_csv(state.report / "gauges.csv", gauge_table)
    except Exception:
        notes.append("gauge table failed: " + traceback.format_exc(limit=3).replace("\n", " "))
    try:
        (state.report / "gates.json").write_text(json.dumps(gates(run_table, gauge_table), indent=2) + "\n")
    except Exception:
        notes.append("gates failed: " + traceback.format_exc(limit=3).replace("\n", " "))
    try:
        (state.report / "summary.md").write_text(_summary_markdown(state, run_table, gauge_table, notes))
    except Exception:
        (state.report / "summary.md").write_text("summary failed:\n" + traceback.format_exc())
    archive = archive or state.root / "fineqcomp_reparameterization_results.tar.gz"
    keep = ("config.json", "status.json", "metrics.json", "training_metrics.json",
            "learning_gate.json", "screening.json")
    temporary = archive.with_name(archive.name + ".tmp")
    with tarfile.open(temporary, "w:gz") as tar:
        def add(path: Path) -> None:
            if path.is_file() and path.stat().st_size < 50_000_000:
                tar.add(path, arcname=str(Path("reparameterization") / path.relative_to(state.root)))
        for path in sorted(state.report.rglob("*")):
            add(path)
        for path in sorted(state.lists.glob("*")):
            add(path)
        for path in (state.root / "jobs.txt", state.root / "data_check.json",
                     state.root / "preflight.json", state.smoke / "gauge_0.csv",
                     state.smoke / "attenuation_0.csv"):
            add(path)
        for run_dir in sorted(p for p in state.runs.glob("*") if p.is_dir()):
            for name in keep:
                add(run_dir / name)
            for path in sorted((run_dir / "codec_metrics").glob("*.json")):
                add(path)
            add(run_dir / "logs" / "training.jsonl")
        for path in sorted(state.runs.glob("baselines/*/metrics.json")):
            add(path)
        for path in (sorted(state.sweep.glob("*/*.csv")) + sorted(state.profile.glob("*.csv"))
                     + sorted(state.frontier.glob("*/r*_b*.json")) + sorted(state.smoke.glob("frontier/*.json"))):
            add(path)
        for path in sorted(state.logs.glob("*")):
            add(path)
    temporary.replace(archive)
    return archive


def status(state: State) -> str:
    """Progress in French, for whoever launched the jobs."""
    from fineqcomp.artifacts import read_json

    lines = []
    plan_info = read_json(state.lists / "plan.json", {}) or {}
    if not plan_info:
        return "Rien n'a encore été planifié ici : lance d'abord `bash scripts/jean_zay/reparameterization.sh`."
    _, runs = _campaign()
    excluded = set(filter(None, str(plan_info.get("EXCLUDED_MODELS", "")).split(",")))
    usable = [run for run in runs if run.model.key not in excluded]
    states: dict[str, int] = {}
    for run in usable:
        key = _status(state, run.run_id)
        states[key] = states.get(key, 0) + 1
    labels = {"complete": "terminés", "running": "en cours", "failed": "échoués",
              "pending": "pas encore commencés", "no_learning": "sans apprentissage",
              "screened_out": "écartés (modèle de base trop fort)"}
    lines.append("Entraînements : " + ", ".join(f"{n} {labels.get(k, k)}" for k, n in sorted(states.items()))
                 + f" (sur {len(usable)})")
    sweep = _sweep_runs(runs, excluded)
    measured: dict[str, set[str]] = {}
    for row in gauge_rows(state):
        measured.setdefault(row["run_id"], set()).add(row["gauge"])
    complete = sum(set(GAUGES) <= measured.get(run.run_id, set()) for run in sweep)
    lines.append(f"Balayage des factorisations : {complete}/{len(sweep)} adaptateurs complets, "
                 f"{sum(len(v) for v in measured.values())}/{len(sweep) * len(GAUGES)} mesures")
    from fineqcomp.studies.rank_frontier import DEFAULT_RANKS, DEFAULT_RATES

    full = len(DEFAULT_RATES) * len(set(DEFAULT_RANKS) | {16})
    truncated = _frontier_runs(runs, excluded)
    lines.append("Second codec (troncature de rang) : "
                 f"{sum(len(frontier_cells(state, run.run_id)) >= full for run in truncated)}/{len(truncated)} adaptateurs")
    profile_runs = _profile_runs(runs, excluded)
    profiled = attenuation_rows(state)
    lines.append(f"Profils d'atténuation : {sum(run.run_id in profiled for run in profile_runs)}/{len(profile_runs)}")
    smoke = "réussi" if (state.smoke / "PASSED").is_file() else "pas encore réussi"
    lines.append(f"Test rapide (smoke) : {smoke}")
    archive = state.root / "fineqcomp_reparameterization_results.tar.gz"
    if archive.is_file():
        lines.append(f"Archive prête : {archive} ({archive.stat().st_size / 1e6:.1f} Mo, "
                     f"{time.strftime('%d/%m %H:%M', time.localtime(archive.stat().st_mtime))})")
    else:
        lines.append("Archive finale : pas encore produite")
    if excluded:
        lines.append(f"Modèles exclus (illisibles pour ce compte) : {', '.join(sorted(excluded))}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "cells", "present-cells", "data-check",
                                            "smoke-check", "collect", "status"))
    parser.add_argument("--state", required=True)
    parser.add_argument("--run-id", nargs="*", default=[])
    parser.add_argument("--list", default="profile")
    parser.add_argument("--archive")
    args = parser.parse_args()
    state = State(args.state)
    if args.command == "plan":
        print(json.dumps(plan(state), indent=2))
    elif args.command == "cells":
        sys.stdout.write(cells(state, args.run_id))
    elif args.command == "present-cells":
        chosen = present_cells(state, args.list)
        sys.stdout.write(cells(state, chosen))
    elif args.command == "data-check":
        report = data_check(state)
        if report["mismatches"]:
            print(f"WARNING: prepared rows differ from the paper's for {report['mismatches']}")
        print(f"data check: {report['checked']} splits, {len(report['mismatches'])} mismatches")
    elif args.command == "smoke-check":
        problems = smoke_check(state)
        for problem in problems:
            print(f"SMOKE FAILED: {problem}")
        if problems:
            return 1
        print("smoke passed")
    elif args.command == "collect":
        archive = collect(state, Path(args.archive) if args.archive else None)
        print(f"archive: {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
        print((state.report / "summary.md").read_text())
    elif args.command == "status":
        print(status(state))
    return 0


if __name__ == "__main__":
    sys.exit(main())
