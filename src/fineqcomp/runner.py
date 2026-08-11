"""Restart-safe execution of prepared campaign records."""

from __future__ import annotations

import math
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from fineqcomp.adapters import adapter_tensors, apply_adapter_tensors
from fineqcomp.artifacts import claim_run, read_json, run_complete, write_json
from fineqcomp.codec import decode_tensor_map, encode_tensor_map
from fineqcomp.config import RunSpec
from fineqcomp.data import (
    Example,
    controlled_data_dir,
    load_ifeval,
    load_natural_dataset,
    read_jsonl,
    synthetic_data_dir,
    validate_controlled_dataset,
    validate_synthetic_dataset,
)
from fineqcomp.evaluation import (
    evaluate_ifeval,
    evaluate_natural,
    evaluate_synthetic,
    write_predictions,
)
from fineqcomp.modeling import ModelSession, validate_single_token_labels
from fineqcomp.training import causal_nll, train_adapter


def estimate_run_cost(run: RunSpec) -> float:
    """Return a stable relative GPU cost used only for worker partitioning."""
    size = 14.0 if "14b" in run.model.key else 8.0 if "8b" in run.model.key else 7.0
    if run.kind == "synthetic":
        examples = 16_384
        evaluation = float(run.family_count or 1) * 16
    elif run.kind == "controlled":
        examples = 8_192
        evaluation = float(run.binding_count or 1) * len(run.precisions)
    elif run.dataset_key == "gsm8k":
        examples, evaluation = 7_000, 1_319 * len(run.precisions)
    else:
        examples, evaluation = 374, 500 * len(run.precisions)
    train = examples * run.training.epochs * math.sqrt(run.training.max_length / 96)
    return size * (train + evaluation * 16)


def partition_runs(runs: list[RunSpec], shards: int) -> list[list[RunSpec]]:
    if shards < 1:
        raise ValueError("shards must be positive")
    partitions: list[list[RunSpec]] = [[] for _ in range(shards)]
    costs = [0.0] * shards
    for run in sorted(runs, key=lambda item: (-estimate_run_cost(item), item.run_id)):
        target = min(range(shards), key=lambda index: (costs[index], index))
        partitions[target].append(run)
        costs[target] += estimate_run_cost(run)
    for partition in partitions:
        partition.sort(key=lambda run: (run.model.key, run.model.backbone, run.run_id))
    return partitions


def _wait_for_json(path: Path, timeout_seconds: float = 3600.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file():
            return read_json(path)
        time.sleep(1.0)
    raise TimeoutError(f"timed out waiting for shared artifact: {path}")


class RunEngine:
    def __init__(
        self,
        campaign: dict[str, Any],
        prepared_root: str | Path = "prepared",
        runs_root: str | Path = "runs",
    ) -> None:
        self.campaign = campaign
        self.prepared_root = Path(prepared_root)
        self.runs_root = Path(runs_root)
        self._data_cache: dict[tuple[str, int], dict[str, list[Example]]] = {}
        self._ifeval: tuple[list[Example], list[dict[str, Any]]] | None = None

    def _load_data(
        self, run: RunSpec
    ) -> tuple[dict[str, list[Example]], dict[str, Any]]:
        if run.kind == "synthetic":
            if run.family_count is None:
                raise ValueError("synthetic run lacks family_count")
            root = synthetic_data_dir(self.prepared_root, run.family_count, run.seed)
            metadata = validate_synthetic_dataset(root)
            data = {
                split: read_jsonl(root / f"{split}.jsonl")
                for split in ("train", "calibration", "test")
            }
            return data, metadata
        if run.kind == "controlled":
            if run.binding_count is None:
                raise ValueError("controlled run lacks binding_count")
            root = controlled_data_dir(self.prepared_root, run.binding_count, run.seed)
            metadata = validate_controlled_dataset(root)
            data = {
                split: read_jsonl(root / f"{split}.jsonl")
                for split in ("train", "calibration", "test")
            }
            return data, metadata
        if run.dataset_key is None:
            raise ValueError("natural run lacks dataset_key")
        key = (run.dataset_key, run.seed)
        if key not in self._data_cache:
            self._data_cache[key] = load_natural_dataset(
                self.campaign, run.dataset_key, run.seed, self.prepared_root
            )
        return self._data_cache[key], {"dataset_key": run.dataset_key}

    def _load_ifeval(self) -> tuple[list[Example], list[dict[str, Any]]]:
        if self._ifeval is None:
            self._ifeval = load_ifeval(self.campaign, self.prepared_root)
        return self._ifeval

    def _baseline_key(self, run: RunSpec) -> str:
        if run.kind == "synthetic":
            data = f"k{run.family_count}-seed{run.seed}"
        elif run.kind == "controlled":
            data = f"paws-n{run.binding_count}-seed{run.seed}"
        else:
            data = str(run.dataset_key)
        return f"{run.model.key}__{run.model.backbone}__{data}"

    def _screening(self, run: RunSpec, baseline: dict[str, Any]) -> dict[str, Any]:
        if run.kind != "natural" or run.dataset_key is None:
            return {}
        spec = self.campaign["datasets"][run.dataset_key]
        metric = str(spec["screening_metric"])
        maximum = float(spec["maximum_baseline_score"])
        score = float(baseline[metric])
        examples = int(baseline["examples"])
        z = 1.959963984540054
        denominator = 1.0 + z**2 / examples
        center = (score + z**2 / (2 * examples)) / denominator
        radius = (
            z
            * math.sqrt(score * (1 - score) / examples + z**2 / (4 * examples**2))
            / denominator
        )
        ci_low = max(0.0, center - radius)
        ci_high = min(1.0, center + radius)
        return {
            "metric": metric,
            "baseline_score": score,
            "maximum_usable_baseline_score": maximum,
            "headroom": 1.0 - score,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "status": "too_easy" if ci_low > maximum else "usable",
        }

    def ensure_baseline(
        self,
        session: ModelSession,
        run: RunSpec,
        data: dict[str, list[Example]],
        data_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        baseline_dir = self.runs_root / "baselines" / self._baseline_key(run)
        if (baseline_dir / "metrics.json").is_file():
            return read_json(baseline_dir / "metrics.json")
        lock = baseline_dir.with_name(baseline_dir.name + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock.mkdir()
        except FileExistsError:
            return _wait_for_json(baseline_dir / "metrics.json")
        try:
            baseline_dir.mkdir(parents=True, exist_ok=True)
            if run.kind in {"synthetic", "controlled"}:
                labels = list(self.campaign["datasets"]["synthetic_codebook"]["labels"])
                metrics, predictions = evaluate_synthetic(
                    session.model,
                    session.tokenizer,
                    data["test"],
                    run.model,
                    labels,
                    batch_size=run.training.micro_batch_size * 4,
                )
                write_predictions(baseline_dir / "predictions.jsonl", predictions)
                output = {
                    "kind": run.kind,
                    "model": run.model.name,
                    "model_key": run.model.key,
                    "backbone": run.model.backbone,
                    **data_metadata,
                    **metrics,
                }
            else:
                metrics, predictions = evaluate_natural(
                    session.model,
                    session.tokenizer,
                    data["test"],
                    run.model,
                    str(run.dataset_key),
                    batch_size=run.training.micro_batch_size,
                )
                write_predictions(baseline_dir / "predictions.jsonl", predictions)
                output = {
                    "kind": "natural",
                    "model": run.model.name,
                    "model_key": run.model.key,
                    "backbone": run.model.backbone,
                    "dataset_key": run.dataset_key,
                    **metrics,
                    "heldout_nll": causal_nll(
                        session.model,
                        session.tokenizer,
                        data["test"],
                        run.model,
                        run.training.max_length,
                        run.training.micro_batch_size,
                    )["nll"],
                }
                output["screening"] = self._screening(run, output)
            write_json(baseline_dir / "metrics.json", output)
            return output
        finally:
            lock.rmdir()

    def ensure_ifeval_baseline(
        self, session: ModelSession, run: RunSpec
    ) -> dict[str, Any]:
        key = f"ifeval__{run.model.key}__{run.model.backbone}"
        baseline_dir = self.runs_root / "baselines" / key
        if (baseline_dir / "metrics.json").is_file():
            return read_json(baseline_dir / "metrics.json")
        lock = baseline_dir.with_name(baseline_dir.name + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock.mkdir()
        except FileExistsError:
            return _wait_for_json(baseline_dir / "metrics.json")
        try:
            baseline_dir.mkdir(parents=True, exist_ok=True)
            examples, evaluator_rows = self._load_ifeval()
            metrics, predictions = evaluate_ifeval(
                session.model,
                session.tokenizer,
                examples,
                evaluator_rows,
                run.model,
                batch_size=run.training.micro_batch_size,
            )
            write_predictions(baseline_dir / "predictions.jsonl", predictions)
            output = {
                "kind": "ifeval",
                "model": run.model.name,
                "model_key": run.model.key,
                "backbone": run.model.backbone,
                **metrics,
            }
            write_json(baseline_dir / "metrics.json", output)
            return output
        finally:
            lock.rmdir()

    def _calibration_score(
        self,
        session: ModelSession,
        run: RunSpec,
        examples: list[Example],
    ) -> tuple[float, dict[str, Any]]:
        if run.kind in {"synthetic", "controlled"}:
            labels = list(self.campaign["datasets"]["synthetic_codebook"]["labels"])
            metrics, _ = evaluate_synthetic(
                session.model,
                session.tokenizer,
                examples,
                run.model,
                labels,
                batch_size=run.training.micro_batch_size * 4,
            )
            return float(metrics["label_nll"]), metrics
        metrics = causal_nll(
            session.model,
            session.tokenizer,
            examples,
            run.model,
            run.training.max_length,
            run.training.micro_batch_size,
        )
        return float(metrics["nll"]), metrics

    def _evaluate_codec(
        self,
        session: ModelSession,
        run: RunSpec,
        run_dir: Path,
        data: dict[str, list[Example]],
        raw_tensors: dict[str, torch.Tensor],
        bits: int,
    ) -> dict[str, Any]:
        codec_path = run_dir / "codecs" / f"adapter_b{bits}.fqcb"
        candidates = (100.0,) if bits == 16 else run.clip_percentiles
        trials = []
        best: tuple[float, int, float] | None = None
        for clip in candidates:
            candidate = run_dir / "codecs" / f".candidate_b{bits}_p{clip:g}.fqcb"
            storage = encode_tensor_map(
                raw_tensors,
                candidate,
                bits,
                clip,
                metadata={
                    "run_id": run.run_id,
                    "adapter_method": run.adapter.method,
                    "adapter_seed": run.seed,
                    "adapter_key": run.adapter.key,
                },
            )
            _, decoded = decode_tensor_map(candidate)
            apply_adapter_tensors(session.model, decoded)
            score, calibration = self._calibration_score(
                session, run, data["calibration"]
            )
            trials.append(
                {
                    **{key: value for key, value in storage.items() if key != "path"},
                    "calibration": calibration,
                }
            )
            choice = (score, int(storage["file_bits"]), float(clip))
            if best is None or choice < best:
                best = choice
            candidate.unlink()
            apply_adapter_tensors(session.model, raw_tensors)
        if best is None:
            raise RuntimeError("codec calibration produced no candidate")
        selected_clip = best[2]
        storage = encode_tensor_map(
            raw_tensors,
            codec_path,
            bits,
            selected_clip,
            metadata={
                "run_id": run.run_id,
                "adapter_method": run.adapter.method,
                "adapter_seed": run.seed,
                "adapter_key": run.adapter.key,
            },
        )
        _, decoded = decode_tensor_map(codec_path)
        apply_adapter_tensors(session.model, decoded)
        if run.kind in {"synthetic", "controlled"}:
            labels = list(self.campaign["datasets"]["synthetic_codebook"]["labels"])
            task_metrics, predictions = evaluate_synthetic(
                session.model,
                session.tokenizer,
                data["test"],
                run.model,
                labels,
                batch_size=run.training.micro_batch_size * 4,
            )
            extra: dict[str, Any] = {}
        else:
            task_metrics, predictions = evaluate_natural(
                session.model,
                session.tokenizer,
                data["test"],
                run.model,
                str(run.dataset_key),
                batch_size=run.training.micro_batch_size,
            )
            task_metrics["heldout_nll"] = causal_nll(
                session.model,
                session.tokenizer,
                data["test"],
                run.model,
                run.training.max_length,
                run.training.micro_batch_size,
            )["nll"]
            ifeval_examples, evaluator_rows = self._load_ifeval()
            ifeval_metrics, ifeval_predictions = evaluate_ifeval(
                session.model,
                session.tokenizer,
                ifeval_examples,
                evaluator_rows,
                run.model,
                batch_size=run.training.micro_batch_size,
            )
            write_predictions(
                run_dir / "predictions" / f"ifeval_b{bits}.jsonl",
                ifeval_predictions,
            )
            extra = {"ifeval": ifeval_metrics}
        write_predictions(run_dir / "predictions" / f"task_b{bits}.jsonl", predictions)
        apply_adapter_tensors(session.model, raw_tensors)
        return {
            "bits": bits,
            "selected_clip_percentile": selected_clip,
            "storage": storage,
            "calibration_trials": trials,
            "task": task_metrics,
            **extra,
        }

    def run_one(
        self,
        session: ModelSession,
        run: RunSpec,
        force: bool = False,
    ) -> str:
        run_dir = self.runs_root / run.run_id
        if not force and run_complete(run_dir, run.precisions):
            return "skipped"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "config.json", run.to_dict())
        data, data_metadata = self._load_data(run)
        if run.kind in {"synthetic", "controlled"}:
            validate_single_token_labels(
                session.tokenizer,
                list(self.campaign["datasets"]["synthetic_codebook"]["labels"]),
            )
        baseline = self.ensure_baseline(session, run, data, data_metadata)
        screening = self._screening(run, baseline) if baseline else {}
        if screening.get("status") == "too_easy":
            write_json(run_dir / "screening.json", screening)
            write_json(
                run_dir / "status.json",
                {
                    "state": "screened_out",
                    "stage": "baseline",
                    "reason": "base model exceeds the fixed saturation threshold",
                    "updated_at": time.time(),
                },
            )
            return "screened_out"
        if run.kind == "natural":
            self.ensure_ifeval_baseline(session, run)
        write_json(
            run_dir / "status.json",
            {"state": "running", "stage": "adapter", "updated_at": time.time()},
        )
        session.attach(run.adapter, run.seed)
        raw_path = run_dir / "raw_channel.pt"
        training_path = run_dir / "training_metrics.json"
        if raw_path.is_file() and training_path.is_file() and not force:
            raw_tensors = torch.load(raw_path, map_location="cpu", weights_only=True)
            apply_adapter_tensors(session.model, raw_tensors)
            training_metrics = read_json(training_path)
        else:
            training_metrics = train_adapter(
                session.model,
                session.tokenizer,
                data["train"],
                data["calibration"],
                run.model,
                run.training,
                run.seed,
                run_dir / "logs" / "training.jsonl",
            )
            raw_tensors = adapter_tensors(session.model, run.adapter.method)
            torch.save(raw_tensors, raw_path)
            write_json(training_path, training_metrics)
        write_json(
            run_dir / "status.json",
            {"state": "running", "stage": "codecs", "updated_at": time.time()},
        )
        codec_metrics = []
        for bits in run.precisions:
            metric_path = run_dir / "codec_metrics" / f"b{bits}.json"
            if metric_path.is_file() and not force:
                codec_metrics.append(read_json(metric_path))
                continue
            metrics = self._evaluate_codec(
                session, run, run_dir, data, raw_tensors, bits
            )
            write_json(metric_path, metrics)
            codec_metrics.append(metrics)
        output = {
            "run_id": run.run_id,
            "study": run.study,
            "kind": run.kind,
            "model": run.model.name,
            "model_key": run.model.key,
            "model_revision": run.model.revision,
            "backbone": run.model.backbone,
            "adapter": run.adapter.key,
            "adapter_method": run.adapter.method,
            "seed": run.seed,
            "family_count": run.family_count,
            "binding_count": run.binding_count,
            "dataset_key": run.dataset_key,
            "baseline_screening": screening,
            "data": data_metadata,
            "training": training_metrics,
            "nominal_channel_values": sum(
                value.numel() for value in raw_tensors.values()
            ),
            "nominal_channel_bits_bf16": sum(
                value.numel() for value in raw_tensors.values()
            )
            * 16,
            "codecs": codec_metrics,
        }
        write_json(run_dir / "metrics.json", output)
        write_json(
            run_dir / "status.json",
            {"state": "complete", "stage": "done", "updated_at": time.time()},
        )
        return "completed"

    def run_many(
        self, runs: list[RunSpec], force: bool = False, limit: int | None = None
    ) -> dict[str, int]:
        counts = {"completed": 0, "skipped": 0, "screened_out": 0, "failed": 0}
        grouped: dict[tuple[str, str], list[RunSpec]] = defaultdict(list)
        for run in runs[:limit]:
            grouped[(run.model.key, run.model.backbone)].append(run)
        for group in grouped.values():
            session = ModelSession.load(group[0].model)
            try:
                for run in group:
                    run_dir = self.runs_root / run.run_id
                    try:
                        with claim_run(run_dir) as claimed:
                            if not claimed:
                                counts["skipped"] += 1
                                continue
                            result = self.run_one(session, run, force=force)
                        counts[result] += 1
                    except Exception as error:
                        counts["failed"] += 1
                        write_json(
                            run_dir / "status.json",
                            {
                                "state": "failed",
                                "error": repr(error),
                                "traceback": traceback.format_exc(),
                                "updated_at": time.time(),
                            },
                        )
                    finally:
                        if hasattr(session.model, "unload"):
                            try:
                                session.unload()
                            except Exception:
                                pass
            finally:
                del session
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return counts

    def screen_natural(
        self, runs: list[RunSpec], shard: int = 0, shards: int = 1
    ) -> list[dict[str, Any]]:
        """Evaluate each base-model/task cell before any natural-task training."""
        if shards < 1 or not 0 <= shard < shards:
            raise ValueError("invalid screening shard")
        unique: dict[tuple[str, str, str], RunSpec] = {}
        for run in runs:
            if run.kind == "natural" and run.dataset_key is not None:
                key = (run.model.key, run.model.backbone, run.dataset_key)
                unique.setdefault(key, run)
        grouped: dict[tuple[str, str], list[RunSpec]] = defaultdict(list)
        for run in unique.values():
            grouped[(run.model.key, run.model.backbone)].append(run)
        records = []
        selected_groups = [
            group
            for index, (_, group) in enumerate(sorted(grouped.items()))
            if index % shards == shard
        ]
        for group in selected_groups:
            session = ModelSession.load(group[0].model)
            try:
                for run in sorted(group, key=lambda item: str(item.dataset_key)):
                    data, metadata = self._load_data(run)
                    baseline = self.ensure_baseline(session, run, data, metadata)
                    screening = self._screening(run, baseline)
                    records.append(
                        {
                            "model": run.model.name,
                            "model_key": run.model.key,
                            "backbone": run.model.backbone,
                            "dataset_key": run.dataset_key,
                            **screening,
                        }
                    )
            finally:
                del session
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return records


def describe_partition(runs: list[RunSpec], shards: int) -> list[dict[str, Any]]:
    descriptions = []
    for index, partition in enumerate(partition_runs(runs, shards)):
        descriptions.append(
            {
                "shard": index,
                "runs": len(partition),
                "relative_cost": sum(estimate_run_cost(run) for run in partition),
                "models": sorted({run.model.key for run in partition}),
            }
        )
    return descriptions
