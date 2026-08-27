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
    validate_answer_retention,
    validate_rationale_tokenization,
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
    validate_prepared(
        campaign,
        selected if args.limit is None else selected[: args.limit],
        args.prepared_root,
    )
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


def _frozen_measurement_rows(args: argparse.Namespace):
    """Select one manifest run and the prepared rows a frozen pass will read."""
    from fineqcomp.data import load_natural_dataset
    from fineqcomp.relative_info import sample_examples

    campaign = load_campaign(args.config)
    runs = read_manifest(args.manifest)
    validate_manifest(runs, campaign)
    selected = [run for run in runs if run.run_id == args.run_id]
    if len(selected) != 1:
        raise ValueError(f"unknown run ID: {args.run_id}")
    run = selected[0]
    data = load_natural_dataset(
        campaign, run.dataset_key, run.seed, args.prepared_root
    )
    available_rows = data[args.split]
    rows = (
        sample_examples(available_rows, args.rows, seed=args.row_sample_seed)
        if args.row_sampling == "uniform"
        else available_rows[: args.rows]
    )
    marker = campaign["datasets"][str(run.dataset_key)].get("answer_marker")
    return run, available_rows, rows, marker


def _frozen_measurement_record(args, run, available_rows, rows) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "study": run.study,
        "model_key": run.model.key,
        "dataset_key": run.dataset_key,
        "seed": run.seed,
        "split": args.split,
        "requested_rows": args.rows,
        "measured_rows": len(rows),
        "available_rows": len(available_rows),
        "row_sampling": args.row_sampling,
        "row_sample_seed": args.row_sample_seed,
    }


def _trace_retrieval(args: argparse.Namespace) -> int:
    from fineqcomp.modeling import ModelSession
    from fineqcomp.relative_info import trace_retrieval_load

    run, available_rows, rows, marker = _frozen_measurement_rows(args)
    if not marker:
        raise ValueError(f"dataset {run.dataset_key} declares no answer_marker")
    session = ModelSession.load(run.model)
    try:
        measured = trace_retrieval_load(
            session,
            rows,
            run.model,
            run.training.max_length,
            args.micro_batch_size,
            marker=str(marker),
            candidates=args.candidates,
        )
    finally:
        try:
            session.unload()
        except Exception:
            pass
    record = {
        **_frozen_measurement_record(args, run, available_rows, rows),
        "max_length": run.training.max_length,
        **measured,
    }
    write_json(args.out, record)
    print(json.dumps(record, indent=2))
    return 0


def _generation_smoke(args: argparse.Namespace) -> int:
    from fineqcomp.evaluation import evaluate_natural
    from fineqcomp.modeling import ModelSession

    if args.max_new_tokens < 1:
        raise ValueError("generation smoke max_new_tokens must be positive")
    run, available_rows, rows, _ = _frozen_measurement_rows(args)
    session = ModelSession.load(run.model)
    try:
        metrics, predictions = evaluate_natural(
            session.model,
            session.tokenizer,
            rows,
            run.model,
            str(run.dataset_key),
            args.micro_batch_size,
            multiple_choice_labels=[],
            max_new_tokens=args.max_new_tokens,
        )
    finally:
        try:
            session.unload()
        except Exception:
            pass
    record = {
        **_frozen_measurement_record(args, run, available_rows, rows),
        "metrics": metrics,
        "predictions": predictions,
    }
    write_json(args.out, record)
    print(json.dumps(record, indent=2))
    return 0


def _relative_information(args: argparse.Namespace) -> int:
    from fineqcomp.modeling import ModelSession
    from fineqcomp.relative_info import (
        SketchSpec,
        measure_layer_energy,
        measure_relative_information,
        sample_examples,
    )

    run, available_rows, rows, marker = _frozen_measurement_rows(args)
    distinct_rows = len({(row.prompt, row.response) for row in available_rows})
    session = ModelSession.load(run.model)
    try:
        measured = measure_relative_information(
            session,
            rows,
            run.model,
            min(run.training.max_length, args.max_length),
            args.micro_batch_size,
            label_span=run.training.label_span,
            answer_marker=str(marker) if marker else None,
            sketch=SketchSpec(
                hidden_dim=args.hidden_dim,
                residual_dim=args.residual_dim,
                response_tokens=args.response_tokens,
                seed=args.sketch_seed,
                ntk_ridge=args.ntk_ridge,
            ),
            population_rows=distinct_rows,
        )
        if args.layer_energy_rows:
            measured.update(
                measure_layer_energy(
                    session,
                    sample_examples(
                        rows, args.layer_energy_rows, seed=args.row_sample_seed
                    ),
                    run.model,
                    min(run.training.max_length, args.max_length),
                    label_span=run.training.label_span,
                    answer_marker=str(marker) if marker else None,
                    rank=args.layer_energy_rank,
                    seed=args.sketch_seed,
                )
            )
    finally:
        try:
            session.unload()
        except Exception:
            pass
    record = {
        **_frozen_measurement_record(args, run, available_rows, rows),
        "distinct_rows": distinct_rows,
        **measured,
    }
    write_json(args.out, record)
    print(json.dumps(record, indent=2))
    return 0


def _relative_information_report(args: argparse.Namespace) -> int:
    from fineqcomp.relative_info import write_relative_information_report

    summary = write_relative_information_report(
        Path(args.results),
        Path(args.runs_root),
        Path(args.out),
        permutations=args.permutations,
        skip_missing_r_star=args.skip_missing_r_star,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _rank_frontier(args: argparse.Namespace) -> int:
    from fineqcomp.rank_frontier import frontier, sweep_run

    rows = sweep_run(
        Path(args.run),
        Path(args.config),
        Path(args.prepared_root),
        Path(args.out),
        ranks=tuple(args.ranks),
        force=args.force,
    )
    best = frontier(rows)
    print(
        json.dumps(
            {
                "cells": len(rows),
                "frontier_points": len(best),
                "ranks_on_frontier": sorted({row["rank"] for row in best}),
                "best_bits_saved_per_megabyte": max(
                    row["heldout_bits_saved"] / (row["file_bits"] / 8e6)
                    for row in rows
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _rank_frontier_report(args: argparse.Namespace) -> int:
    from fineqcomp.rank_frontier import write_rank_frontier_report

    summary = write_rank_frontier_report(
        Path(args.results), Path(args.runs_root), Path(args.out)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _relative_law(args: argparse.Namespace) -> int:
    from fineqcomp.relative_validation import write_rate_law_report

    grid = {
        rows: Path(root)
        for rows, root in zip((64, 128), args.row_grid or [], strict=False)
    }
    result = write_rate_law_report(
        Path(args.results),
        Path(args.runs_root),
        Path(args.out),
        row_grid_roots=grid or None,
        prospective_arms=(
            Path(args.prospective_arms) if args.prospective_arms else None
        ),
        permutations=args.permutations,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


def _relative_information_diagnose(args: argparse.Namespace) -> int:
    from fineqcomp.relative_validation import write_measure_diagnostics_report

    result = write_measure_diagnostics_report(
        Path(args.cells),
        Path(args.out),
        permutations=args.permutations,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0




def _relative_information_channel_validate(args: argparse.Namespace) -> int:
    from fineqcomp.relative_validation import (
        write_fixed_channel_prospective_report,
    )

    result = write_fixed_channel_prospective_report(
        Path(args.development_results),
        Path(args.prospective_results),
        Path(args.prospective_manifest),
        Path(args.runs_root),
        Path(args.lock),
        Path(args.out),
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "arms"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "passed" else 1





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
                    hessian_batch_size=args.hessian_batch_size,
                    recompute_activations=args.recompute_activations,
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
        "prepared_dataset_cells": validate_prepared(
            campaign, runs, args.prepared_root
        ),
        "partition": describe_partition(runs, args.shards),
        "rationale_token_contracts": validate_rationale_tokenization(
            campaign, runs, args.prepared_root
        ),
        "answer_retention": validate_answer_retention(
            campaign, runs, args.prepared_root, args.min_answer_retention
        ),
    }
    if args.tokenizers:
        report["label_token_ids"] = validate_tokenizers(campaign)
    if args.model_smoke:
        report["model_smoke"] = model_smoke(
            campaign,
            runs,
            args.prepared_root,
            model_key=args.model_smoke_model,
            dataset_key=args.model_smoke_dataset,
        )
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

    relative = subparsers.add_parser(
        "relative-information",
        help="measure ten ways the frozen model sees a supervised dataset",
    )
    relative.add_argument("--config", required=True)
    relative.add_argument("--manifest", required=True)
    relative.add_argument("--run-id", required=True)
    relative.add_argument("--prepared-root", default="prepared")
    relative.add_argument("--out", required=True)
    relative.add_argument("--split", default="train", choices=["train", "calibration"])
    relative.add_argument("--rows", type=int, default=256)
    relative.add_argument(
        "--row-sampling", choices=("uniform", "head"), default="uniform"
    )
    relative.add_argument("--row-sample-seed", type=int, default=271_828)
    relative.add_argument("--max-length", type=int, default=1024)
    relative.add_argument("--micro-batch-size", type=int, default=1)
    relative.add_argument("--hidden-dim", type=int, default=64)
    relative.add_argument("--residual-dim", type=int, default=32)
    relative.add_argument("--response-tokens", type=int, default=32)
    relative.add_argument("--sketch-seed", type=int, default=1729)
    relative.add_argument("--ntk-ridge", type=float, default=0.1)
    relative.add_argument(
        "--layer-energy-rows",
        type=int,
        default=64,
        help="rows for the zero-adapter backward pass; zero skips it",
    )
    relative.add_argument("--layer-energy-rank", type=int, default=16)
    relative.set_defaults(func=_relative_information)

    retrieval = subparsers.add_parser(
        "trace-retrieval-load",
        help="measure the frozen model's bounded problem-trace retrieval load",
    )
    retrieval.add_argument("--config", required=True)
    retrieval.add_argument("--manifest", required=True)
    retrieval.add_argument("--run-id", required=True)
    retrieval.add_argument("--prepared-root", default="prepared")
    retrieval.add_argument("--out", required=True)
    retrieval.add_argument("--split", default="train", choices=["train", "calibration"])
    retrieval.add_argument("--rows", type=int, default=128)
    retrieval.add_argument(
        "--row-sampling", choices=("uniform", "head"), default="uniform"
    )
    retrieval.add_argument("--row-sample-seed", type=int, default=271_828)
    retrieval.add_argument("--micro-batch-size", type=int, default=1)
    retrieval.add_argument("--candidates", type=int, default=8)
    retrieval.set_defaults(func=_trace_retrieval)

    generation_smoke = subparsers.add_parser(
        "generation-smoke",
        help="measure answer and EOS survival at one frozen generation limit",
    )
    generation_smoke.add_argument("--config", required=True)
    generation_smoke.add_argument("--manifest", required=True)
    generation_smoke.add_argument("--run-id", required=True)
    generation_smoke.add_argument("--prepared-root", default="prepared")
    generation_smoke.add_argument("--out", required=True)
    generation_smoke.add_argument("--split", default="test", choices=["test"])
    generation_smoke.add_argument("--rows", type=int, default=8)
    generation_smoke.add_argument(
        "--row-sampling", choices=("uniform", "head"), default="uniform"
    )
    generation_smoke.add_argument("--row-sample-seed", type=int, default=271_828)
    generation_smoke.add_argument("--micro-batch-size", type=int, default=1)
    generation_smoke.add_argument("--max-new-tokens", type=int, required=True)
    generation_smoke.set_defaults(func=_generation_smoke)

    relative_report = subparsers.add_parser(
        "relative-information-report",
        help="rank frozen-model information measures against finished R* runs",
    )
    relative_report.add_argument(
        "--results", default="reports/relative_information"
    )
    relative_report.add_argument("--runs-root", default="runs")
    relative_report.add_argument(
        "--out",
        default="results/1_rate_behaviour_frontier/relative_information_candidates",
    )
    relative_report.add_argument("--permutations", type=int, default=50_000)
    relative_report.add_argument(
        "--skip-missing-r-star",
        action="store_true",
        help="drop measured cells whose run has no bracketed R*",
    )
    relative_report.set_defaults(func=_relative_information_report)

    rank = subparsers.add_parser(
        "rank-frontier",
        help="sweep rank against rate on a finished adapter; no training",
    )
    rank.add_argument("--run", required=True)
    rank.add_argument("--config", default="configs/campaign.yaml")
    rank.add_argument("--prepared-root", default="prepared")
    rank.add_argument("--out", required=True)
    rank.add_argument(
        "--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16]
    )
    rank.add_argument("--force", action="store_true")
    rank.set_defaults(func=_rank_frontier)

    rank_report = subparsers.add_parser(
        "rank-frontier-report",
        help="join swept adapters and score truncation against the random mask",
    )
    rank_report.add_argument("--results", default="reports/rank_frontier")
    rank_report.add_argument("--runs-root", default="runs")
    rank_report.add_argument(
        "--out", default="results/1_rate_behaviour_frontier/rank_frontier"
    )
    rank_report.set_defaults(func=_rank_frontier_report)

    relative_law = subparsers.add_parser(
        "relative-law",
        help="audit the rate law: candidate gate, forward model, receiver test",
    )
    relative_law.add_argument("--results", default="reports/relative_information_channel")
    relative_law.add_argument("--runs-root", default="runs")
    relative_law.add_argument(
        "--out", default="results/1_rate_behaviour_frontier/rate_law_audit"
    )
    relative_law.add_argument(
        "--row-grid",
        nargs=2,
        metavar=("ROOT64", "ROOT128"),
        help="measurement roots at 64 and 128 rows, for the ceiling check",
    )
    relative_law.add_argument(
        "--prospective-arms",
        help="arm CSV from a finished campaign to score with the same prefit",
    )
    relative_law.add_argument("--permutations", type=int, default=20_000)
    relative_law.set_defaults(func=_relative_law)

    relative_diagnose = subparsers.add_parser(
        "relative-information-diagnose",
        help="post-mortem one measure panel: resolution, receiver share, controls",
    )
    relative_diagnose.add_argument("--cells", required=True)
    relative_diagnose.add_argument("--out", required=True)
    relative_diagnose.add_argument("--permutations", type=int, default=20_000)
    relative_diagnose.set_defaults(func=_relative_information_diagnose)



    channel_validate = subparsers.add_parser(
        "relative-information-channel-validate",
        help="apply one locked correction-spectrum test to untouched cells",
    )
    channel_validate.add_argument("--development-results", required=True)
    channel_validate.add_argument("--prospective-results", required=True)
    channel_validate.add_argument("--prospective-manifest", required=True)
    channel_validate.add_argument("--runs-root", default="runs")
    channel_validate.add_argument("--lock", required=True)
    channel_validate.add_argument("--out", required=True)
    channel_validate.set_defaults(func=_relative_information_channel_validate)




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
    profile.add_argument(
        "--hessian-batch-size", type=int, default=1,
        help="batch size for the gradient and Hessian passes; double backward"
        " needs far less than training did",
    )
    profile.add_argument(
        "--recompute-activations", action="store_true",
        help="checkpoint activations during the Hessian pass; needed at long"
        " sequence lengths, where the attention matrices do not fit twice",
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
    preflight.add_argument("--min-answer-retention", type=float, default=None)
    preflight.add_argument("--model-smoke", action="store_true")
    preflight.add_argument(
        "--model-smoke-model",
        help="model key to use for the two-row model smoke",
    )
    preflight.add_argument(
        "--model-smoke-dataset",
        help="dataset key to use for the two-row model smoke",
    )
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
