"""Restart-safe execution of prepared campaign records."""

from __future__ import annotations

import hashlib
import json
import math
import time
import traceback
from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from fineqcomp.adapters import adapter_tensors, apply_adapter_tensors
from fineqcomp.artifacts import claim_run, read_json, run_complete, write_json
from fineqcomp.codec import (
    decode_adapter_tensor_map,
    encode_loraquant_tensor_map,
    encode_tensor_map,
)
from fineqcomp.config import CodecSpec, RunSpec
from fineqcomp.data import Example, load_natural_dataset
from fineqcomp.evaluation import evaluate_natural, write_predictions
from fineqcomp.modeling import ModelSession, validate_single_token_labels
from fineqcomp.training import causal_nll, train_adapter


def estimate_run_cost(run: RunSpec) -> float:
    """Return a stable relative GPU cost used only for worker partitioning."""
    size = 14.0 if "14b" in run.model.key else 8.0 if "8b" in run.model.key else 7.0
    sizes = {
        "gsm8k": (6_961, 1_319),
        "commonsense_qa": (9_229, 1_221),
        "arc_challenge": (1_119, 1_172),
        "openbookqa": (4_957, 500),
        "metamath": (395_000, 6_319),
        "magicoder": (109_500, 164),
        "xsum": (204_045, 11_334),
    }
    examples, test_rows = sizes.get(str(run.dataset_key), (1_000, 1_000))
    evaluation = test_rows * len(run.codecs)
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


def partition_runs_weighted(
    runs: list[RunSpec], weights: list[float]
) -> list[list[RunSpec]]:
    """Assign costly runs in proportion to fixed worker speed weights."""
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError("worker weights must be positive")
    partitions: list[list[RunSpec]] = [[] for _ in weights]
    costs = [0.0] * len(weights)
    for run in sorted(runs, key=lambda item: (-estimate_run_cost(item), item.run_id)):
        run_cost = estimate_run_cost(run)
        target = min(
            range(len(weights)),
            key=lambda index: (
                (costs[index] + run_cost) / weights[index],
                costs[index] / weights[index],
                index,
            ),
        )
        partitions[target].append(run)
        costs[target] += run_cost
    for partition in partitions:
        partition.sort(key=lambda run: (run.model.key, run.model.backbone, run.run_id))
    return partitions


@contextmanager
def _claim_or_read_baseline(
    baseline_dir: Path,
    timeout_seconds: float = 3600.0,
    poll_seconds: float = 1.0,
):
    """Hold the baseline lock, or read the artifact another holder wrote."""
    metrics_path = baseline_dir / "metrics.json"
    deadline = time.monotonic() + timeout_seconds
    while True:
        metrics = read_json(metrics_path)
        if metrics is not None:
            yield False, metrics
            return
        with claim_run(baseline_dir) as claimed:
            if claimed:
                # The file may have appeared between the read and the lock.
                metrics = read_json(metrics_path)
                yield metrics is None, metrics
                return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for shared artifact: {metrics_path}")
        time.sleep(poll_seconds)


def _span_key(part: str, span: str) -> str:
    """Where a span's numbers live inside an information record."""
    return part if span == "all" else f"{part}_{span}"


class RunEngine:
    def __init__(
        self,
        campaign: dict[str, Any],
        prepared_root: str | Path = "prepared",
        runs_root: str | Path = "runs",
        pilot_rows: int | None = None,
    ) -> None:
        self.campaign = campaign
        self.prepared_root = Path(prepared_root)
        self.runs_root = Path(runs_root)
        if pilot_rows is not None and pilot_rows < 1:
            raise ValueError("pilot_rows must be positive")
        self.pilot_rows = pilot_rows
        self._data_cache: dict[tuple[str, int], dict[str, list[Example]]] = {}

    def _limit_data(self, data: dict[str, list[Example]]) -> dict[str, list[Example]]:
        if self.pilot_rows is None:
            return data
        limited = {
            split: rows[: self.pilot_rows]
            for split, rows in data.items()
            if split != "test"
        }
        test_groups: dict[str, list[Example]] = defaultdict(list)
        for example in data["test"]:
            evaluator = str(example.metadata.get("evaluator", "task"))
            if len(test_groups[evaluator]) < self.pilot_rows:
                test_groups[evaluator].append(example)
        limited["test"] = [
            example for group in test_groups.values() for example in group
        ]
        return limited

    def _load_data(
        self, run: RunSpec
    ) -> tuple[dict[str, list[Example]], dict[str, Any]]:
        if run.dataset_key is None:
            raise ValueError("natural run lacks dataset_key")
        key = (run.dataset_key, run.seed)
        if key not in self._data_cache:
            self._data_cache[key] = load_natural_dataset(
                self.campaign, run.dataset_key, run.seed, self.prepared_root
            )
        return self._limit_data(self._data_cache[key]), {"dataset_key": run.dataset_key}

    @staticmethod
    def _calibration_key(spec: Mapping[str, Any]) -> str:
        """Fingerprint the held-out split a baseline's calibration NLL is on.

        Everything that changes those rows belongs here: the source itself, how
        many rows are held out, and any rewrite applied to them. Row selection
        inside the training set does not, which is what lets the diversity arms
        share one baseline.
        """
        source = dict(spec.get("train_source") or {})
        # `group_field` names the column the diversity lever groups training
        # rows by. It never touches the held-out split, so including it forked
        # the key and sent 36 diversity runs off to recompute baselines the
        # panel had already written -- and then to contend over them.
        source.pop("group_field", None)
        transform = spec.get("response_transform")
        payload = {
            "source": {key: str(source[key]) for key in sorted(source)},
            "validation_rows": spec.get("validation_rows"),
            "validation_split": spec.get("validation_split"),
            # "plain" names the untransformed target, so it has to fingerprint
            # the same as an absent transform or every old baseline orphans.
            "response_transform": (
                None if transform is None or str(transform) == "plain" else str(transform)
            ),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        )
        return digest.hexdigest()[:8]

    def _baseline_key(self, run: RunSpec) -> str:
        """Identify a baseline by what it was scored on, not by the training set.

        The test-set size has to be in the key: without it, changing
        `test_rows` reuses a baseline scored on different problems and every
        retained-gain figure silently compares two populations.

        Keying on the evaluation rather than the dataset name lets arms that
        differ only in training data share one measurement. The compressibility
        arms are exactly that case, and they also share a calibration split and
        the same first `information_rows` training rows, so the stored
        information block is identical too.
        """
        spec = self.campaign["datasets"][str(run.dataset_key)]
        evaluations = "-".join(
            str(item["key"]) for item in spec.get("evaluations", [])
        ) or str(run.dataset_key)
        rows = spec.get("test_rows", "all")
        # The record holds two measurements with different sharing rules. The
        # task score depends only on the evaluation, so arms that differ in
        # training data may share it. The calibration NLL is the base model's
        # bits per token on the held-out split of the *training* corpus, so it
        # may only be shared by arms whose calibration split is identical.
        # Fingerprinting the split covers both cases at once: a response
        # transform rewrites the held-out targets, and a different corpus
        # replaces them outright. Keying on the evaluation alone had both
        # failures -- the behavioural grid shared one baseline across five
        # transforms, and hh-rlhf and Alpaca were gated against MetaMathQA's
        # 0.899 bits per token because all three list gsm8k.
        evaluations = f"{evaluations}-cal{self._calibration_key(spec)}"
        # A study that splits the response by span needs baseline numbers for
        # each span. Those go in a key of their own rather than growing the
        # shared record in place, so a half-written rewrite can never be
        # handed to a job already waiting on the old file.
        split = "-split" if self._split_spans(run) else ""
        return (
            f"{run.model.key}__{run.model.backbone}__"
            f"{evaluations}-seed{run.seed}-n{rows}{split}"
        )

    def _evaluate_natural(
        self,
        session: ModelSession,
        run: RunSpec,
        examples: list[Example],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return evaluate_natural(
            session.model,
            session.tokenizer,
            examples,
            run.model,
            str(run.dataset_key),
            # Generation dominates a codec sweep: one evaluation per codec, and
            # every batch runs to the longest sequence in it. Eight rows barely
            # occupies an 80 GB card.
            batch_size=int(self.campaign.get("evaluation_batch_size", 8)),
            multiple_choice_labels=list(
                map(str, self.campaign.get("multiple_choice_labels", []))
            ),
        )

    def _screening(self, run: RunSpec, baseline: dict[str, Any]) -> dict[str, Any]:
        if run.dataset_key is None:
            return {}
        spec = self.campaign["datasets"][run.dataset_key]
        metric = str(spec["screening_metric"])
        maximum = float(spec["maximum_baseline_score"])
        score = float(baseline[metric])
        examples = int(baseline["examples"])
        if spec.get("screening_binomial", True):
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
        else:
            ci_low = ci_high = score
        return {
            "metric": metric,
            "baseline_score": score,
            "maximum_usable_baseline_score": maximum,
            "headroom": 1.0 - score,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "status": "too_easy" if ci_low > maximum else "usable",
        }

    def _learning_gate(
        self,
        run: RunSpec,
        baseline: dict[str, Any],
        raw_metrics: dict[str, Any],
        span: str = "all",
    ) -> dict[str, Any]:
        spec = self.campaign["datasets"][str(run.dataset_key)]
        if "bits_per_token" in raw_metrics:
            metric = "validation_bits_per_token"
            # An arm taught only the final answer barely moves the NLL of the
            # working, so its gate has to read the span it was trained on or
            # it would be thrown out for learning nothing.
            baseline_score = float(
                baseline["information"][_span_key("heldout", span)]["bits_per_token"]
            )
            raw_score = float(raw_metrics["bits_per_token"])
            minimum = float(spec["minimum_validation_nll_gain_bits_per_token"])
            gain = baseline_score - raw_score
        else:
            metric = str(spec["screening_metric"])
            baseline_score = float(baseline[metric])
            raw_score = float(raw_metrics[metric])
            minimum = float(spec["minimum_raw_gain"])
            gain = raw_score - baseline_score
        return {
            "metric": metric,
            "span": span,
            "baseline_validation_score": baseline_score,
            "raw_adapter_validation_score": raw_score,
            "gain": gain,
            "minimum_gain": minimum,
            "status": "usable" if gain + 1e-12 >= minimum else "no_learning",
        }

    def _information_measure(
        self,
        session: ModelSession,
        run: RunSpec,
        data: dict[str, list[Example]],
        parts: tuple[str, ...] = ("train", "heldout"),
    ) -> dict[str, Any]:
        rows = int(self.campaign.get("information_rows", 256))
        marker = self._answer_marker(run)
        from_end = self._answer_marker_from_end(run)

        def measure(split: str, span: str) -> dict[str, Any]:
            return causal_nll(
                session.model,
                session.tokenizer,
                data[split][:rows],
                run.model,
                run.training.max_length,
                run.training.micro_batch_size,
                span,
                marker,
                from_end,
            )

        split_for = {"train": "train", "heldout": "calibration"}
        measured = {
            part: measure(split_for[part], "all")
            for part in parts
        }
        # With a marker the same rows are scored again on each half of the
        # response, which is what separates bits spent on the working from
        # bits spent on the answer.
        for span in self._split_spans(run):
            for part in parts:
                measured[f"{part}_{span}"] = measure(split_for[part], span)
                if not measured[f"{part}_{span}"]["nll_tokens"]:
                    raise ValueError(
                        f"{run.run_id}: the {part} {span} span scored no tokens;"
                        " check the dataset answer_marker"
                    )
        return measured

    def _answer_marker(self, run: RunSpec) -> str | None:
        marker = self.campaign["datasets"][str(run.dataset_key)].get("answer_marker")
        return str(marker) if marker else None

    def _answer_marker_from_end(self, run: RunSpec) -> bool:
        # `The answer is:` is the last occurrence; a code fence is the first.
        # Datasets that do not say keep the original behaviour.
        spec = self.campaign["datasets"][str(run.dataset_key)]
        return bool(spec.get("answer_marker_from_end", True))

    def _split_spans(self, run: RunSpec) -> tuple[str, ...]:
        return ("reasoning", "answer") if self._answer_marker(run) else ()

    def _span_writes(
        self, run: RunSpec, baseline: dict[str, Any], tuned: dict[str, Any]
    ) -> dict[str, dict[str, float]]:
        """Bits saved on each half of the response, measured separately."""
        return {
            span: self._behavioral_write(
                {part: baseline[_span_key(part, span)] for part in ("train", "heldout")},
                {part: tuned[_span_key(part, span)] for part in ("train", "heldout")},
            )
            for span in self._split_spans(run)
        }

    @staticmethod
    def _behavioral_write(
        baseline: dict[str, Any], tuned: dict[str, Any]
    ) -> dict[str, float]:
        # Signed on purpose. Clamping at zero made "the adapter left this span
        # alone" and "the adapter destroyed this span" the same number, and the
        # chain-of-thought arms are exactly where that mattered: answer-only
        # supervision reads as 0.0 bits saved on the reasoning span when it is
        # really 4.6 bits per token worse than the base model.
        train_saved = (
            float(baseline["train"]["total_bits"])
            - float(tuned["train"]["total_bits"])
        )
        heldout_saved = (
            float(baseline["heldout"]["total_bits"])
            - float(tuned["heldout"]["total_bits"])
        )
        train_rate = (
            float(baseline["train"]["bits_per_token"])
            - float(tuned["train"]["bits_per_token"])
        )
        heldout_rate = (
            float(baseline["heldout"]["bits_per_token"])
            - float(tuned["heldout"]["bits_per_token"])
        )
        return {
            "train_bits_saved": train_saved,
            "heldout_bits_saved": heldout_saved,
            "train_bits_saved_per_token": train_rate,
            "heldout_bits_saved_per_token": heldout_rate,
            "excess_train_bits_per_token": train_rate - heldout_rate,
        }

    def _retained_gain(
        self,
        run: RunSpec,
        baseline: dict[str, Any],
        raw_task: dict[str, Any],
        codec_task: dict[str, Any],
    ) -> dict[str, Any]:
        metric = str(
            self.campaign["datasets"][str(run.dataset_key)]["screening_metric"]
        )
        baseline_score = float(baseline[metric])
        raw_score = float(raw_task[metric])
        codec_score = float(codec_task[metric])
        raw_gain = raw_score - baseline_score
        return {
            "metric": metric,
            "baseline_score": baseline_score,
            "raw_adapter_score": raw_score,
            "codec_score": codec_score,
            "raw_gain": raw_gain,
            "retained_gain": (
                (codec_score - baseline_score) / raw_gain
                if abs(raw_gain) > 1e-12
                else None
            ),
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
        # A directory used as a mutex is not released when the holder is
        # killed, so one cancelled job leaves every later job waiting an hour
        # and then failing. `claim_run` holds an flock, which the kernel drops
        # as soon as the process dies, stale or not.
        #
        # A waiter must also retry the lock, not just poll for the file. If the
        # holder fails, its lock drops at once and the next waiter takes over.
        with _claim_or_read_baseline(baseline_dir) as (claimed, existing):
            if not claimed:
                return existing
            baseline_dir.mkdir(parents=True, exist_ok=True)
            metrics, predictions = self._evaluate_natural(session, run, data["test"])
            write_predictions(baseline_dir / "predictions.jsonl", predictions)
            if (
                self.campaign["datasets"][str(run.dataset_key)].get("task_type")
                == "multiple_choice"
            ):
                calibration_metrics, calibration_predictions = self._evaluate_natural(
                    session, run, data["calibration"]
                )
                write_predictions(
                    baseline_dir / "calibration_predictions.jsonl",
                    calibration_predictions,
                )
            else:
                calibration_metrics = causal_nll(
                    session.model,
                    session.tokenizer,
                    data["calibration"],
                    run.model,
                    run.training.max_length,
                    run.training.micro_batch_size,
                )
            output = {
                "kind": "natural",
                "model": run.model.name,
                "model_key": run.model.key,
                "backbone": run.model.backbone,
                "dataset_key": run.dataset_key,
                "seed": run.seed,
                **metrics,
                "calibration": calibration_metrics,
                "information": self._information_measure(session, run, data),
            }
            output["screening"] = self._screening(run, output)
            write_json(baseline_dir / "metrics.json", output)
            return output

    def _calibration_score(
        self,
        session: ModelSession,
        run: RunSpec,
        examples: list[Example],
    ) -> tuple[float, dict[str, Any]]:
        dataset_spec = self.campaign["datasets"][str(run.dataset_key)]
        if dataset_spec.get("task_type") == "multiple_choice":
            metrics, _ = self._evaluate_natural(session, run, examples)
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
        codec: CodecSpec,
        baseline: dict[str, Any],
        raw_task: dict[str, Any],
    ) -> dict[str, Any]:
        codec_path = run_dir / "codecs" / f"adapter_{codec.key}.fqcb"
        metadata = {
            "run_id": run.run_id,
            "adapter_method": run.adapter.method,
            "adapter_seed": run.seed,
            "adapter_key": run.adapter.key,
            "codec_key": codec.key,
            "codec_method": codec.method,
        }
        if codec.method == "loraquant":
            storage = encode_loraquant_tensor_map(
                raw_tensors,
                codec_path,
                high_bits=int(codec.high_bits or 0),
                variance_ratio=float(codec.variance_ratio or 0.0),
                group_size=codec.group_size,
                optimize_steps=codec.optimize_steps,
                metadata=metadata,
            )
        else:
            storage = encode_tensor_map(
                raw_tensors,
                codec_path,
                int(codec.bits or 0),
                codec.quantizer,
                metadata=metadata,
                blend=codec.blend,
            )
        _, decoded = decode_adapter_tensor_map(codec_path)
        apply_adapter_tensors(session.model, decoded)
        score, calibration = self._calibration_score(
            session, run, data["calibration"]
        )
        trials = [
            {
                **{key: value for key, value in storage.items() if key != "path"},
                "calibration": calibration,
                "score": score,
            }
        ]
        task_metrics = None
        if codec.score_task:
            task_metrics, predictions = self._evaluate_natural(
                session, run, data["test"]
            )
            write_predictions(
                run_dir / "predictions" / f"task_{codec.key}.jsonl", predictions
            )
        information = self._information_measure(session, run, data)
        apply_adapter_tensors(session.model, raw_tensors)
        return {
            "codec_key": codec.key,
            "codec_method": codec.method,
            "bits": codec.bits,
            "high_bits": codec.high_bits,
            "low_bits": codec.low_bits,
            "variance_ratio": codec.variance_ratio,
            "quantizer": codec.quantizer if codec.method == "uniform" else None,
            "blend": codec.blend if codec.method == "uniform" else None,
            "storage": storage,
            "calibration_trials": trials,
            "task": task_metrics,
            "retained_gain": (
                self._retained_gain(run, baseline, raw_task, task_metrics)
                if task_metrics is not None
                else None
            ),
            "information": information,
            "behavioral_write": self._behavioral_write(
                baseline["information"], information
            ),
            "behavioral_write_spans": self._span_writes(
                run, baseline["information"], information
            ),
        }

    def run_one(
        self,
        session: ModelSession,
        run: RunSpec,
        force: bool = False,
    ) -> str:
        run_dir = self.runs_root / run.run_id
        codec_keys = tuple(codec.key for codec in run.codecs)
        if not force and run_complete(run_dir, codec_keys):
            return "skipped"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "config.json", run.to_dict())
        data, data_metadata = self._load_data(run)
        if (
            self.campaign["datasets"][str(run.dataset_key)].get("task_type")
            == "multiple_choice"
        ):
            validate_single_token_labels(
                session.tokenizer,
                list(map(str, self.campaign["multiple_choice_labels"])),
            )
        baseline = self.ensure_baseline(session, run, data, data_metadata)
        dataset_spec = self.campaign["datasets"][str(run.dataset_key)]
        if dataset_spec.get("distinct_source_problems") is not None:
            # These arms share held-out rows and task tests, but select
            # different training rows. Reuse the costly shared baseline while
            # measuring base-model train bits on this arm's actual rows.
            baseline = {
                **baseline,
                "information": {
                    **baseline["information"],
                    **self._information_measure(
                        session, run, data, parts=("train",)
                    ),
                },
            }
        screening = self._screening(run, baseline) if baseline else {}
        if screening.get("status") == "too_easy" and self.pilot_rows is None:
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
        if screening.get("status") == "too_easy":
            screening["pilot_bypassed"] = True
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
                self._answer_marker(run),
                self._answer_marker_from_end(run),
            )
            raw_tensors = adapter_tensors(session.model, run.adapter.method)
            torch.save(raw_tensors, raw_path)
            write_json(training_path, training_metrics)
        raw_information = self._information_measure(session, run, data)
        gate_span = run.training.label_span
        learning_gate = self._learning_gate(
            run,
            baseline,
            raw_information[_span_key("heldout", gate_span)],
            gate_span,
        )
        write_json(run_dir / "learning_gate.json", learning_gate)
        if learning_gate["status"] == "no_learning" and self.pilot_rows is None:
            write_json(
                run_dir / "status.json",
                {
                    "state": "no_learning",
                    "stage": "raw_adapter",
                    "reason": "raw adapter misses the fixed validation-gain gate",
                    "updated_at": time.time(),
                },
            )
            return "no_learning"
        if learning_gate["status"] == "no_learning":
            learning_gate["pilot_bypassed"] = True
        raw_task, raw_predictions = self._evaluate_natural(session, run, data["test"])
        write_predictions(run_dir / "predictions" / "raw_task.jsonl", raw_predictions)
        write_json(
            run_dir / "status.json",
            {"state": "running", "stage": "codecs", "updated_at": time.time()},
        )
        codec_metrics = []
        for codec in run.codecs:
            metric_path = run_dir / "codec_metrics" / f"{codec.key}.json"
            if metric_path.is_file() and not force:
                codec_metrics.append(read_json(metric_path))
                continue
            metrics = self._evaluate_codec(
                session,
                run,
                run_dir,
                data,
                raw_tensors,
                codec,
                baseline,
                raw_task,
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
            "dataset_key": run.dataset_key,
            "baseline_screening": screening,
            "learning_gate": learning_gate,
            "raw_task": raw_task,
            "raw_information": raw_information,
            "raw_behavioral_write_spans": self._span_writes(
                run, baseline["information"], raw_information
            ),
            "raw_behavioral_write": self._behavioral_write(
                baseline["information"], raw_information
            ),
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
        counts = {
            "completed": 0,
            "skipped": 0,
            "screened_out": 0,
            "no_learning": 0,
            "failed": 0,
        }
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
            if run.dataset_key is not None:
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
