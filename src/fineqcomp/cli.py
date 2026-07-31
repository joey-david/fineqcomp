"""Command-line interface for preparation, execution, and analysis."""

from __future__ import annotations

import argparse
import json
import sys

from fineqcomp.analysis import analyze
from fineqcomp.artifacts import write_json
from fineqcomp.campaign import expand_campaign, read_manifest, write_manifest
from fineqcomp.config import load_campaign
from fineqcomp.data import prepare_all_controlled, prepare_all_synthetic
from fineqcomp.preflight import (
    environment_report,
    model_smoke,
    validate_prepared,
    validate_tokenizers,
    write_report,
)
from fineqcomp.runner import RunEngine, describe_partition, partition_runs


def _prepare(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.config)
    runs = expand_campaign(campaign)
    write_manifest(runs, args.manifest)
    datasets = []
    controlled = []
    if not args.no_data:
        datasets = prepare_all_synthetic(campaign, runs, args.prepared_root)
        controlled = prepare_all_controlled(campaign, runs, args.prepared_root)
    print(
        json.dumps(
            {
                "runs": len(runs),
                "synthetic_datasets": len(datasets),
                "controlled_datasets": len(controlled),
                "manifest": str(args.manifest),
            },
            indent=2,
        )
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.config)
    runs = read_manifest(args.manifest)
    partitions = partition_runs(runs, args.shards)
    if not 0 <= args.shard < args.shards:
        raise ValueError("shard index must be less than shard count")
    if args.run_id is not None:
        selected = [run for run in runs if run.run_id == args.run_id]
        if len(selected) != 1:
            raise ValueError(f"unknown run ID: {args.run_id}")
    else:
        selected = partitions[args.shard]
    if args.dry_run:
        print(json.dumps(describe_partition(runs, args.shards), indent=2))
        shown = selected if args.limit is None else selected[: args.limit]
        for run in shown:
            print(run.run_id)
        return 0
    engine = RunEngine(campaign, args.prepared_root, args.runs_root)
    counts = engine.run_many(selected, force=args.force, limit=args.limit)
    print(json.dumps(counts, indent=2))
    return 1 if counts["failed"] else 0


def _analyze(args: argparse.Namespace) -> int:
    print(json.dumps(analyze(args.root, args.out), indent=2))
    return 0


def _screen(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.config)
    runs = read_manifest(args.manifest)
    engine = RunEngine(campaign, args.prepared_root, args.runs_root)
    records = engine.screen_natural(runs, args.shard, args.shards)
    write_json(args.out, records)
    print(json.dumps(records, indent=2))
    return 0


def _preflight(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.config)
    runs = read_manifest(args.manifest)
    report = {
        "environment": environment_report(args.require_gpus),
        "manifest_runs": len(runs),
        "prepared_dataset_cells": validate_prepared(runs, args.prepared_root),
        "partition": describe_partition(runs, 2),
    }
    if args.tokenizers:
        report["label_token_ids"] = validate_tokenizers(campaign)
    if args.model_smoke:
        report["model_smoke"] = model_smoke(campaign, runs, args.prepared_root)
    write_report(args.report, report)
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fineqcomp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="materialize the fixed manifest and synthetic data"
    )
    prepare.add_argument("--config", default="configs/campaign.yaml")
    prepare.add_argument("--manifest", default="prepared/manifest.jsonl")
    prepare.add_argument("--prepared-root", default="prepared")
    prepare.add_argument("--no-data", action="store_true")
    prepare.set_defaults(func=_prepare)

    run = subparsers.add_parser("run", help="run one cost-balanced manifest shard")
    run.add_argument("--config", default="configs/campaign.yaml")
    run.add_argument("--manifest", default="prepared/manifest.jsonl")
    run.add_argument("--prepared-root", default="prepared")
    run.add_argument("--runs-root", default="runs")
    run.add_argument("--shard", type=int, required=True)
    run.add_argument("--shards", type=int, default=2)
    run.add_argument("--limit", type=int)
    run.add_argument("--run-id")
    run.add_argument("--force", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=_run)

    analysis = subparsers.add_parser(
        "analyze", help="make fixed campaign tables and plots"
    )
    analysis.add_argument("--root", default="runs")
    analysis.add_argument("--out", default="reports")
    analysis.set_defaults(func=_analyze)

    screen = subparsers.add_parser(
        "screen", help="measure base-task headroom before natural-task training"
    )
    screen.add_argument("--config", default="configs/campaign.yaml")
    screen.add_argument("--manifest", default="prepared/manifest.jsonl")
    screen.add_argument("--prepared-root", default="prepared")
    screen.add_argument("--runs-root", default="runs")
    screen.add_argument("--out", default="prepared/baseline_screening.json")
    screen.add_argument("--shard", type=int, default=0)
    screen.add_argument("--shards", type=int, default=1)
    screen.set_defaults(func=_screen)

    preflight = subparsers.add_parser(
        "preflight", help="check the remote runtime without starting the grid"
    )
    preflight.add_argument("--config", default="configs/campaign.yaml")
    preflight.add_argument("--manifest", default="prepared/manifest.jsonl")
    preflight.add_argument("--prepared-root", default="prepared")
    preflight.add_argument("--report", default="prepared/preflight.json")
    preflight.add_argument("--require-gpus", action="store_true")
    preflight.add_argument("--tokenizers", action="store_true")
    preflight.add_argument("--model-smoke", action="store_true")
    preflight.set_defaults(func=_preflight)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        status = args.func(args)
    except Exception as error:
        print(f"fineqcomp: {error}", file=sys.stderr)
        raise
    raise SystemExit(status)
