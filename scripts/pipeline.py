"""Advance the overnight campaign one stage at a time, without a human.

Every stage names the stages it waits for, a gate that reads results already on
disk, and the exact `sbatch` line it submits. One tick reads the state file,
marks finished stages, evaluates the gates of anything now unblocked, submits
what passes, records what did not and why, and re-arms itself for the next tick
unless everything has settled.

Three properties keep this safe to leave running.

Stages are declared here, not discovered. The driver can only ever submit the
lines written in `STAGES`, so a bug can waste GPU hours but cannot invent an
experiment or overwrite a result.

Gates read results, not job states. A stage whose jobs finished but whose runs
failed the learning gate is `skipped`, with the numbers that skipped it written
to the log, rather than silently proceeding on nothing.

Everything is bounded: a maximum number of ticks, a wall-clock deadline, and a
cap on how many jobs one run of the driver may submit. Reaching any of them
stops the chain rather than pausing it, so the failure mode is an idle cluster.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parent.parent
STATE_PATH = REPO / "pipeline" / "state.json"
LOG_PATH = REPO / "pipeline" / "log.jsonl"


# ------------------------------------------------------------------ utilities


def _log(event: str, **fields: Any) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {"at": time.time(), "when": time.strftime("%Y-%m-%d %H:%M:%S"),
           "event": event, **fields}
    with LOG_PATH.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def _read_state() -> dict[str, Any]:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    return {"stages": {}, "ticks": 0, "submitted": 0}


def _write_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=1, sort_keys=True))
    temporary.replace(STATE_PATH)


def _active_jobs(attempts: int = 3) -> set[str]:
    """Job ids still queued or running, as bare ids without array suffixes.

    A failed query must never be read as "everything finished", which would
    mark every stage done and submit the whole rest of the night at once. The
    controller is occasionally busy, so this retries and then gives up; giving
    up ends the tick and the caller re-arms without changing any state.
    """
    last = ""
    for attempt in range(attempts):
        try:
            return {
                line.split("_")[0]
                for line in subprocess.run(
                    ["squeue", "-h", "-u", __import__("getpass").getuser(),
                     "-o", "%i"],
                    capture_output=True, text=True, timeout=120, check=True,
                ).stdout.split()
            }
        except (subprocess.SubprocessError, FileNotFoundError) as error:
            last = f"{type(error).__name__}: {error}"
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"squeue failed {attempts} times ({last})")


def _run_ids(config: str, studies: list[str] | None = None) -> list[str]:
    from fineqcomp.campaign import expand_campaign
    from fineqcomp.config import load_campaign

    runs = expand_campaign(load_campaign(str(REPO / config)))
    return [
        run.run_id for run in runs
        if studies is None or run.study in set(studies)
    ]


# ---------------------------------------------------------------------- gates


def learned(config: str, studies: list[str], minimum_gain: float = 0.05,
            fraction: float = 0.6) -> tuple[bool, dict[str, Any]]:
    """Did enough runs of an earlier stage actually learn anything?

    The gain is the base model's held-out bits per token minus the adapter's,
    which is the same quantity the campaign's own learning gate uses. A stage
    that trains a diversity sweep on a corpus nothing learned would burn a
    night for nothing.
    """
    ids = _run_ids(config, studies)
    gains = []
    for run_id in ids:
        path = REPO / "runs" / run_id / "learning_gate.json"
        if path.is_file():
            gains.append(float(json.loads(path.read_text()).get("gain", 0.0)))
    passing = [g for g in gains if g >= minimum_gain]
    detail = {"runs": len(ids), "measured": len(gains), "passing": len(passing),
              "minimum_gain": minimum_gain,
              "gains": [round(g, 4) for g in sorted(gains)]}
    return (bool(gains) and len(passing) >= fraction * len(ids)), detail


def r_star_available(out_root: str, minimum: int = 4) -> tuple[bool, dict[str, Any]]:
    """Did the synthetic grid produce enough usable crossings to fit anything?

    The crossing lives per checkpoint, not at the top level of the record: a
    cell measures every prefix it was given. Reading `record["r_star"]` finds a
    null on every completed cell and skips a stage whose data is fine.
    """
    found = 0
    total = 0
    for path in (REPO / out_root).rglob("result.json"):
        total += 1
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for checkpoint in record.get("checkpoints") or []:
            summary = checkpoint.get("rate_summary") or {}
            crossing = summary.get("r_star_effective_bits_per_value") or {}
            found += int(crossing.get("r_star") is not None)
    return found >= minimum, {"cells": total, "crossings": found,
                              "minimum": minimum}


def always() -> tuple[bool, dict[str, Any]]:
    return True, {}


# --------------------------------------------------------------------- stages


@dataclass
class Stage:
    name: str
    after: list[str]
    gate: Callable[[], tuple[bool, dict[str, Any]]]
    submit: list[list[str]] = field(default_factory=list)
    note: str = ""


def _compress(config: str, manifest: str, count: int, name: str,
              hours: str = "06:00:00") -> list[str]:
    return [
        "sbatch", f"--array=0-{count - 1}%{min(count, 24)}", f"--time={hours}",
        f"--job-name={name}",
        "--export=ALL,BY_INDEX=1,"
        f"CONFIG={config},MANIFEST={manifest}",
        "scripts/jean_zay_compress.sbatch",
    ]


def build_stages() -> list[Stage]:
    return [
        # Already submitted by hand before the driver existed; the driver only
        # waits on them.
        Stage("high_gain_panel", [], always,
              note="job 1276327, submitted by hand"),
        Stage("payload_grid", [], always,
              note="job 1276363, submitted by hand"),

        # The one intervention that ever worked, on the two new corpora. Only
        # worth running where the baseline panel shows the corpus was learned.
        Stage(
            "sql_diversity", ["high_gain_panel"],
            lambda: learned("configs/high_gain_panel.yaml",
                            ["mistral_text_to_sql", "qwen25_text_to_sql"]),
            [_compress("configs/diversity_sql.yaml",
                       "prepared/diversity-sql-manifest.jsonl", 18, "fqdivsql")],
            "18 runs: 10 / 25 / 100 domains x 2 models x 3 seeds",
        ),
        Stage(
            "xbrl_diversity", ["high_gain_panel"],
            lambda: learned("configs/high_gain_panel.yaml",
                            ["mistral_xbrl_tags", "qwen25_xbrl_tags"]),
            [_compress("configs/diversity_xbrl.yaml",
                       "prepared/diversity-xbrl-manifest.jsonl", 18, "fqdivxbrl",
                       hours="04:00:00")],
            "18 runs: 10 / 18 / 30 companies x 2 models x 3 seeds",
        ),

        # The chain-of-thought split on a second substrate. Independent of the
        # new corpora, so it has no gate beyond being declared.
        Stage(
            "cot_second_model", [], always,
            [_compress("configs/cot_panel.yaml", "prepared/cot-panel-manifest.jsonl",
                       18, "fqcotpanel")],
            "18 runs: 3 supervision spans x 2 models x 3 seeds",
        ),

        # Denser payload sweep and a second rule, once the first grid shows the
        # measurement produces crossings at all.
        Stage(
            "payload_wide", ["payload_grid"],
            lambda: r_star_available("runs_information_scaling/payload", 4),
            [[
                "sbatch", "--array=0-10%11", "--time=04:00:00",
                "--job-name=fqpayloadwide",
                "--export=ALL,MODE=validate,"
                "INFO_CONFIG=configs/program_and_payload_wide.yaml,"
                "INFO_OUT=runs_information_scaling/payload_wide,INFO_SHARDS=11",
                "scripts/jean_zay_information.sbatch",
            ]],
            "33 cells over 11 shards: seven payload levels and a second rule",
        ),
    ]


# ----------------------------------------------------------------------- tick


def tick(arguments: argparse.Namespace) -> int:
    state = _read_state()
    state["ticks"] = int(state.get("ticks", 0)) + 1
    stages = {stage.name: stage for stage in build_stages()}
    records = state.setdefault("stages", {})
    for name in stages:
        records.setdefault(name, {"status": "pending", "jobs": []})

    # Seed the hand-submitted stages so the driver waits on the right ids.
    for name, jobs in (arguments.adopt or {}).items():
        if records.get(name, {}).get("status") == "pending":
            records[name] = {"status": "running", "jobs": jobs,
                             "note": "adopted"}

    try:
        active = _active_jobs()
    except RuntimeError as error:
        # Nothing is marked, nothing is submitted; try again next tick.
        _log("squeue_unavailable", error=str(error))
        _write_state(state)
        return _rearm(arguments, ["squeue unavailable"])
    for name, record in records.items():
        if record["status"] == "running" and not (set(record["jobs"]) & active):
            record["status"] = "done"
            record["finished_at"] = time.time()
            _log("stage_finished", stage=name, jobs=record["jobs"])

    submitted_now = 0
    for name, stage in stages.items():
        record = records[name]
        if record["status"] != "pending":
            continue
        if any(records[other]["status"] != "done" for other in stage.after):
            continue
        if not stage.submit:
            record["status"] = "done"
            _log("stage_noop", stage=name, note=stage.note)
            continue
        try:
            passed, detail = stage.gate()
        except Exception as error:  # a broken gate must not submit anything
            _log("gate_error", stage=name, error=f"{type(error).__name__}: {error}")
            continue
        if not passed:
            # A gate can fail because the upstream results are still being
            # written, so a skip is provisional. Marking it final on the first
            # miss would kill a stage over a few seconds of timing.
            attempts = int(record.get("skips", 0)) + 1
            record["skips"] = attempts
            record["reason"] = detail
            if attempts >= arguments.max_skips:
                record["status"] = "skipped"
                _log("stage_skipped_final", stage=name, attempts=attempts,
                     detail=detail)
            else:
                _log("stage_waiting", stage=name, attempts=attempts,
                     detail=detail)
            continue
        if submitted_now + len(stage.submit) > arguments.max_submissions:
            _log("submission_cap", stage=name, cap=arguments.max_submissions)
            continue
        jobs = []
        for command in stage.submit:
            if arguments.dry_run:
                _log("would_submit", stage=name, command=command, detail=detail)
                jobs.append("dry-run")
                continue
            result = subprocess.run(command, cwd=REPO, capture_output=True,
                                    text=True, timeout=300)
            if result.returncode:
                _log("submit_failed", stage=name, command=command,
                     stderr=result.stderr.strip()[:400])
                break
            job = result.stdout.strip().split()[-1]
            jobs.append(job)
            submitted_now += 1
            _log("submitted", stage=name, job=job, command=command, detail=detail)
        if jobs:
            record["status"] = "running" if not arguments.dry_run else "pending"
            record["jobs"] = jobs
            record["gate"] = detail

    state["submitted"] = int(state.get("submitted", 0)) + submitted_now
    outstanding = [n for n, r in records.items()
                   if r["status"] in {"pending", "running"}]
    _write_state(state)

    if not outstanding:
        _log("pipeline_finished", ticks=state["ticks"], submitted=state["submitted"])
        return 0
    if state["ticks"] >= arguments.max_ticks:
        _log("tick_cap", ticks=state["ticks"], outstanding=outstanding)
        return 0
    if time.time() > arguments.deadline:
        _log("deadline", outstanding=outstanding)
        return 0
    if arguments.dry_run:
        _log("dry_run_end", outstanding=outstanding)
        return 0
    return _rearm(arguments, outstanding)


def _rearm(arguments: argparse.Namespace, outstanding: list[str]) -> int:
    if arguments.dry_run:
        _log("would_rearm", outstanding=outstanding)
        return 0
    rearm = [
        "sbatch", f"--begin=now+{arguments.interval_minutes}minutes",
        "--job-name=fqdriver",
        f"--export=ALL,PIPELINE_DEADLINE={int(arguments.deadline)},"
        f"PIPELINE_MAX_TICKS={arguments.max_ticks},"
        f"PIPELINE_INTERVAL={arguments.interval_minutes}",
        "scripts/jean_zay_pipeline.sbatch",
    ]
    result = subprocess.run(rearm, cwd=REPO, capture_output=True, text=True,
                            timeout=300)
    if result.returncode:
        _log("rearm_failed", stderr=result.stderr.strip()[:400])
        return 1
    _log("rearmed", job=result.stdout.strip().split()[-1],
         minutes=arguments.interval_minutes, outstanding=outstanding)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--interval-minutes", type=int,
                        default=int(__import__("os").environ.get("PIPELINE_INTERVAL", 20)))
    parser.add_argument("--max-ticks", type=int,
                        default=int(__import__("os").environ.get("PIPELINE_MAX_TICKS", 60)))
    parser.add_argument("--max-submissions", type=int, default=4)
    parser.add_argument("--max-skips", type=int, default=6,
                        help="failed gate evaluations before a stage is dropped")
    parser.add_argument("--deadline", type=float,
                        default=float(__import__("os").environ.get(
                            "PIPELINE_DEADLINE", time.time() + 20 * 3600)))
    parser.add_argument("--adopt", type=json.loads, default=None,
                        help='JSON: {"stage": ["jobid", ...]} for hand-submitted work')
    parser.add_argument("--status", action="store_true",
                        help="print the state file and exit")
    arguments = parser.parse_args(argv)
    if arguments.status:
        print(json.dumps(_read_state(), indent=1, sort_keys=True))
        return 0
    return tick(arguments)


if __name__ == "__main__":
    sys.exit(main())
