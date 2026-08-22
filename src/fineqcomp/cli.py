"""Command-line interface for preparation, execution, and analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from dataclasses import replace
from typing import Any

from fineqcomp.analysis import analyze
from fineqcomp.artifacts import write_json
from fineqcomp.campaign import (
    expand_campaign,
    read_manifest,
    validate_manifest,
    write_manifest,
)
from fineqcomp.config import load_campaign
from fineqcomp.data import prepare_all_natural
from fineqcomp.dataset_info import measure_arms
from fineqcomp.rstar import report as rstar_report
from fineqcomp.preflight import (
    cache_models,
    environment_report,
    model_smoke,
    validate_prepared,
    validate_tokenizers,
    write_report,
)
from fineqcomp.runner import (
    RunEngine,
    describe_partition,
    estimate_run_cost,
    partition_runs,
    partition_runs_weighted,
)


def _unraisable_hook(unraisable: Any) -> None:
    if unraisable.exc_type is BrokenPipeError:
        return
    sys.__unraisablehook__(unraisable)


def _prepare(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.config)
    runs = expand_campaign(campaign)
    write_manifest(runs, args.manifest)
    natural = []
    if not args.no_data:
        natural = prepare_all_natural(campaign, runs, args.prepared_root)
    print(
        json.dumps(
            {
                "runs": len(runs),
                "natural_datasets": len(natural),
                "manifest": str(args.manifest),
            },
            indent=2,
        )
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.config)
    runs = read_manifest(args.manifest)
    validate_manifest(runs, campaign)
    if args.models:
        model_keys = set(args.models)
        runs = [run for run in runs if run.model.key in model_keys]
    if args.adapters:
        adapter_keys = set(args.adapters)
        runs = [run for run in runs if run.adapter.key in adapter_keys]
    if args.datasets:
        dataset_keys = set(args.datasets)
        runs = [run for run in runs if run.dataset_key in dataset_keys]
    if args.backbones:
        backbones = set(args.backbones)
        runs = [run for run in runs if run.model.backbone in backbones]
    if args.max_length is not None:
        runs = [run for run in runs if run.training.max_length <= args.max_length]
    if args.micro_batch_size is not None:
        if any(
            run.training.effective_batch_size % args.micro_batch_size for run in runs
        ):
            raise ValueError("micro batch size must divide every effective batch size")
        runs = [
            replace(
                run,
                training=replace(
                    run.training, micro_batch_size=args.micro_batch_size
                ),
            )
            for run in runs
        ]
    if args.codecs:
        codec_keys = set(args.codecs)
        available = {codec.key for run in runs for codec in run.codecs}
        unknown = sorted(codec_keys - available)
        if unknown:
            raise ValueError(f"unknown codec keys: {unknown}")
        runs = [
            replace(
                run,
                codecs=tuple(
                    codec for codec in run.codecs if codec.key in codec_keys
                ),
            )
            for run in runs
        ]
    if args.pilot_rows is not None:
        runs = [
            replace(
                run,
                training=replace(
                    run.training,
                    epochs=1,
                    effective_batch_size=1,
                    micro_batch_size=1,
                ),
            )
            for run in runs
        ]
    if args.worker_weights:
        if args.worker is None:
            raise ValueError("--worker is required with --worker-weights")
        partitions = partition_runs_weighted(runs, args.worker_weights)
        if not 0 <= args.worker < len(partitions):
            raise ValueError("worker index must be less than the worker count")
        selected = partitions[args.worker]
    else:
        partitions = partition_runs(runs, args.shards)
        if not 0 <= args.shard < args.shards:
            raise ValueError("shard index must be less than shard count")
        selected = partitions[args.shard]
    if args.run_id is not None:
        selected = [run for run in runs if run.run_id == args.run_id]
        if len(selected) != 1:
            raise ValueError(f"unknown run ID: {args.run_id}")
    if args.dry_run:
        if args.worker_weights:
            descriptions = [
                {
                    "worker": index,
                    "weight": args.worker_weights[index],
                    "runs": len(partition),
                    "relative_cost": sum(
                        estimate_run_cost(run) for run in partition
                    ),
                    "models": sorted({run.model.key for run in partition}),
                }
                for index, partition in enumerate(partitions)
            ]
        else:
            descriptions = describe_partition(runs, args.shards)
        print(json.dumps(descriptions, indent=2))
        shown = selected if args.limit is None else selected[: args.limit]
        for run in shown:
            print(run.run_id)
        return 0
    engine = RunEngine(
        campaign,
        args.prepared_root,
        args.runs_root,
        pilot_rows=args.pilot_rows,
    )
    counts = engine.run_many(selected, force=args.force, limit=args.limit)
    print(json.dumps(counts, indent=2))
    return 1 if counts["failed"] else 0


def _analyze(args: argparse.Namespace) -> int:
    print(json.dumps(analyze(args.root, args.out), indent=2))
    return 0


def _dataset_information(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.config)
    runs = read_manifest(args.manifest)
    validate_manifest(runs, campaign)
    session = None
    if args.with_base_model:
        from fineqcomp.campaign import _model
        from fineqcomp.modeling import ModelSession

        session = ModelSession.load(_model(args.model, campaign))
    try:
        records = measure_arms(
            campaign,
            runs,
            args.prepared_root,
            session,
            args.sample_rows,
            args.max_length,
            args.micro_batch_size,
            args.split,
        )
    finally:
        if session is not None:
            try:
                session.unload()
            except Exception:
                pass
    write_json(args.out, records)
    print(json.dumps(records, indent=2))
    return 0


def _layer_profile(args: argparse.Namespace) -> int:
    from fineqcomp.campaign import _model
    from fineqcomp.data import load_natural_dataset
    from fineqcomp.layer_profile import profile_run
    from fineqcomp.modeling import ModelSession

    campaign = load_campaign(args.config)
    runs = read_manifest(args.manifest)
    validate_manifest(runs, campaign)
    selected = [
        run
        for run in runs
        if (not args.studies or run.study in set(args.studies))
        and (not args.seeds or run.seed in set(args.seeds))
    ]
    ready = [
        run
        for run in selected
        if (Path(args.runs_root) / run.run_id / "raw_channel.pt").is_file()
    ]
    if not ready:
        raise SystemExit(
            f"no finished runs with a saved adapter under {args.runs_root}"
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = ModelSession.load(_model(ready[0].model.key, campaign))
    written = []
    try:
        for run in ready:
            target = out / f"{run.run_id}.json"
            if target.is_file() and not args.force:
                continue
            data = load_natural_dataset(
                campaign, run.dataset_key, run.seed, args.prepared_root
            )
            session.attach(run.adapter, run.seed)
            try:
                marker = campaign["datasets"][str(run.dataset_key)].get(
                    "answer_marker"
                )
                record = profile_run(
                    session,
                    Path(args.runs_root) / run.run_id,
                    data["calibration"][: args.rows],
                    run.model,
                    Path(args.work_dir) / run.run_id,
                    max_length=run.training.max_length,
                    batch_size=run.training.micro_batch_size,
                    probe_rows=args.probe_rows,
                    measures=args.measures,
                    answer_marker=str(marker) if marker else None,
                    taylor_codecs=args.taylor_codecs,
                )
            finally:
                session.unload()
            record["run_id"] = run.run_id
            record["study"] = run.study
            record["seed"] = run.seed
            write_json(target, record)
            written.append(str(target))
            print(json.dumps({"profiled": run.run_id}), flush=True)
    finally:
        try:
            session.unload()
        except Exception:
            pass
    print(json.dumps({"written": len(written), "out": str(out)}, indent=2))
    return 0


def _rstar(args: argparse.Namespace) -> int:
    information = Path(args.information) if args.information else None
    if information is not None and not information.is_file():
        information = None
    keep = None
    if not args.all_runs:
        keep = {run.run_id for run in expand_campaign(load_campaign(args.config))}
    print(
        json.dumps(
            rstar_report(
                Path(args.root), information, Path(args.out), args.target, keep
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _screen(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.config)
    runs = read_manifest(args.manifest)
    validate_manifest(runs, campaign)
    engine = RunEngine(campaign, args.prepared_root, args.runs_root)
    records = engine.screen_natural(runs, args.shard, args.shards)
    write_json(args.out, records)
    print(json.dumps(records, indent=2))
    return 0


def _preflight(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.config)
    runs = read_manifest(args.manifest)
    validate_manifest(runs, campaign)
    report = {
        "environment": environment_report(
            args.require_gpus,
            args.require_dependencies,
            args.gpu_count,
            args.min_gpu_memory_gib,
        ),
        "manifest_runs": len(runs),
        "prepared_dataset_cells": validate_prepared(runs, args.prepared_root),
        "partition": describe_partition(runs, args.shards),
    }
    if args.tokenizers:
        report["label_token_ids"] = validate_tokenizers(campaign)
    if args.model_smoke:
        report["model_smoke"] = model_smoke(campaign, runs, args.prepared_root)
    write_report(args.report, report)
    print(json.dumps(report, indent=2))
    return 0


def _cache_models(args: argparse.Namespace) -> int:
    report = cache_models(load_campaign(args.config))
    write_report(args.report, report)
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fineqcomp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="materialize the fixed manifest and pinned datasets"
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
    run.add_argument("--shard", type=int, default=0)
    run.add_argument("--shards", type=int, default=2)
    run.add_argument("--worker", type=int)
    run.add_argument("--worker-weights", type=float, nargs="+")
    run.add_argument("--models", nargs="+")
    run.add_argument("--adapters", nargs="+")
    run.add_argument("--datasets", nargs="+")
    run.add_argument("--backbones", nargs="+", choices=("nf4", "bf16"))
    run.add_argument("--max-length", type=int)
    run.add_argument("--micro-batch-size", type=int)
    run.add_argument("--codecs", nargs="+")
    run.add_argument("--pilot-rows", type=int)
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

    information = subparsers.add_parser(
        "dataset-information",
        help="measure how much information each arm's training data carries",
    )
    information.add_argument("--config", default="configs/compressibility.yaml")
    information.add_argument("--manifest", default="prepared/manifest.jsonl")
    information.add_argument("--prepared-root", default="prepared")
    information.add_argument("--out", default="reports/dataset_information.json")
    information.add_argument("--sample-rows", type=int, default=1024)
    information.add_argument(
        "--split", default="train", choices=["train", "calibration"],
        help="calibration is the split R* is scored on",
    )
    information.add_argument("--max-length", type=int, default=1024)
    information.add_argument("--micro-batch-size", type=int, default=4)
    information.add_argument("--model", default="mistral_7b_base")
    information.add_argument(
        "--with-base-model",
        action="store_true",
        help="also take the duplication-blind measure, which needs a GPU",
    )
    information.set_defaults(func=_dataset_information)

    profile = subparsers.add_parser(
        "layer-profile",
        help="where the adapter bits are needed: layer allocation and"
        " representation shift, on adapters that already exist",
    )
    profile.add_argument("--config", default="configs/compressibility.yaml")
    profile.add_argument("--manifest", default="prepared/compressibility-manifest.jsonl")
    profile.add_argument("--prepared-root", default="prepared")
    profile.add_argument("--runs-root", default="runs")
    profile.add_argument("--out", default="reports/layer_profile")
    profile.add_argument("--work-dir", default="reports/layer_profile/work")
    profile.add_argument("--studies", nargs="+")
    profile.add_argument("--seeds", nargs="+", type=int)
    profile.add_argument("--rows", type=int, default=256)
    profile.add_argument("--probe-rows", type=int, default=32)
    profile.add_argument(
        "--measures",
        nargs="+",
        default=["weights", "representation", "allocation"],
        choices=["weights", "representation", "allocation", "distribution",
                 "taylor"],
        help="which measurements to take; the allocation sweep is by far the"
        " most expensive and a distribution pass does not need it",
    )
    profile.add_argument(
        "--taylor-codecs", nargs="+", default=["uniform2"],
        help="codec keys whose coded containers give the perturbations to expand"
        " around the trained adapter",
    )
    profile.add_argument("--force", action="store_true")
    profile.set_defaults(func=_layer_profile)

    star = subparsers.add_parser(
        "rstar", help="R* against both information measures, with the overfit gap"
    )
    star.add_argument("--root", default="runs")
    star.add_argument("--information", default="reports/dataset_information.json")
    star.add_argument("--out", default="reports")
    star.add_argument("--target", type=float, default=0.90)
    star.add_argument("--config", default="configs/compressibility.yaml")
    star.add_argument(
        "--all-runs",
        action="store_true",
        help="include runs from earlier configs, which the campaign filter drops",
    )
    star.set_defaults(func=_rstar)

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
    preflight.add_argument("--shards", type=int, default=5)
    preflight.add_argument("--require-gpus", action="store_true")
    preflight.add_argument("--require-dependencies", action="store_true")
    preflight.add_argument("--gpu-count", type=int, default=2)
    preflight.add_argument("--min-gpu-memory-gib", type=float, default=75.0)
    preflight.add_argument("--tokenizers", action="store_true")
    preflight.add_argument("--model-smoke", action="store_true")
    preflight.set_defaults(func=_preflight)

    cache = subparsers.add_parser(
        "cache-models", help="download and verify every pinned model snapshot"
    )
    cache.add_argument("--config", default="configs/campaign.yaml")
    cache.add_argument("--report", default="prepared/model-cache.json")
    cache.set_defaults(func=_cache_models)
    return parser


def main(argv: list[str] | None = None) -> None:
    sys.unraisablehook = _unraisable_hook
    args = build_parser().parse_args(argv)
    try:
        status = args.func(args)
    except Exception as error:
        print(f"fineqcomp: {error}", file=sys.stderr)
        raise
    raise SystemExit(status)
