from __future__ import annotations

import json

import pytest

from fineqcomp.analysis import _paired_stats, analyze, rate_bound
from fineqcomp.artifacts import write_json


def _prediction(path, correct):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"correct": value}) + "\n" for value in correct))


def _synthetic_run(root, run_id, family_count, accuracy):
    run = root / run_id
    write_json(
        run / "metrics.json",
        {
            "run_id": run_id,
            "study": "exact_seeded",
            "kind": "synthetic",
            "model": "Qwen/Qwen3-8B-Base",
            "model_key": "qwen3_8b_base",
            "backbone": "nf4",
            "adapter": "seeded_last4_r16",
            "adapter_method": "seeded_b",
            "seed": 11,
            "family_count": family_count,
            "dataset_key": None,
            "data": {
                "task_entropy_bits": 64 * family_count,
                "source_symbols": 16 * family_count,
                "train_zlib_bits": 1_000,
            },
            "training": {"elapsed_seconds": 2.0, "peak_memory_bytes": 100},
            "codecs": [
                {
                    "bits": 4,
                    "selected_clip_percentile": 99.9,
                    "storage": {"file_bits": 8_000, "raw_payload_bits": 6_000},
                    "task": {
                        "accuracy": accuracy,
                        "distortion": 1 - accuracy,
                        "label_nll": 0.4,
                    },
                }
            ],
        },
    )
    _prediction(run / "predictions" / "task_b4.jsonl", [True, False, True])


def _natural_run(root, dataset, metric, value):
    run_id = f"natural-{dataset}"
    run = root / run_id
    write_json(
        run / "metrics.json",
        {
            "run_id": run_id,
            "study": "placement_control",
            "kind": "natural",
            "model": "Qwen/Qwen3-8B",
            "model_key": "qwen3_8b_instruct",
            "backbone": "nf4",
            "adapter": "seeded_last4_r16",
            "adapter_method": "seeded_b",
            "seed": 11,
            "family_count": None,
            "dataset_key": dataset,
            "baseline_screening": {
                "baseline_score": value - 0.1,
                "headroom": 1.1 - value,
                "status": "usable",
            },
            "data": {"dataset_key": dataset},
            "training": {},
            "codecs": [
                {
                    "codec_key": "uniform4",
                    "codec_method": "uniform",
                    "bits": 4,
                    "selected_clip_percentile": 100.0,
                    "storage": {"file_bits": 12_000, "raw_payload_bits": 10_000},
                    "task": {metric: value, "heldout_nll": 0.5},
                    "retained_gain": {"retained_gain": 0.8},
                    "behavioral_write": {
                        "train_bits_saved": 200.0,
                        "heldout_bits_saved": 100.0,
                        "excess_train_bits_per_token": 0.1,
                    },
                    "ifeval": {"prompt_level_strict_accuracy": 0.75},
                }
            ],
        },
    )
    _prediction(run / "predictions" / "task_uniform4.jsonl", [True, True, False])
    write_json(
        root
        / "baselines"
        / f"qwen3_8b_instruct__nf4__{dataset}-seed11"
        / "metrics.json",
        {
            "kind": "natural",
            "model": "Qwen/Qwen3-8B",
            "model_key": "qwen3_8b_instruct",
            "backbone": "nf4",
            "dataset_key": dataset,
            metric: value - 0.1,
            "screening": {
                "metric": metric,
                "baseline_score": value - 0.1,
                "maximum_usable_baseline_score": 0.8,
                "headroom": 1.1 - value,
                "status": "usable",
            },
        },
    )
    write_json(
        root / "baselines" / "ifeval__qwen3_8b_instruct__nf4" / "metrics.json",
        {
            "kind": "ifeval",
            "model_key": "qwen3_8b_instruct",
            "backbone": "nf4",
            "prompt_level_strict_accuracy": 0.8,
        },
    )


def _controlled_run(root):
    run = root / "controlled-n256"
    write_json(
        run / "metrics.json",
        {
            "run_id": "controlled-n256",
            "study": "controlled_transfer",
            "kind": "controlled",
            "model": "Qwen/Qwen3-8B-Base",
            "model_key": "qwen3_8b_base",
            "backbone": "nf4",
            "adapter": "seeded_last4_r16",
            "adapter_method": "seeded_b",
            "seed": 11,
            "family_count": None,
            "binding_count": 256,
            "dataset_key": None,
            "data": {
                "task_entropy_bits": 1024,
                "source_symbols": 256,
                "train_zlib_bits": 2_000,
            },
            "training": {},
            "codecs": [
                {
                    "bits": 4,
                    "selected_clip_percentile": 100.0,
                    "storage": {"file_bits": 9_000, "raw_payload_bits": 7_000},
                    "task": {
                        "accuracy": 0.7,
                        "distortion": 0.3,
                        "label_nll": 0.6,
                    },
                }
            ],
        },
    )
    _prediction(run / "predictions" / "task_b4.jsonl", [True, False, True])


def test_rate_bound_endpoints():
    assert rate_bound(100, 0) == 400
    assert rate_bound(100, 15 / 16) == pytest.approx(0, abs=1e-12)


def test_paired_stats_use_matched_predictions():
    stats = _paired_stats([0, 1, 0, 1], [1, 1, 1, 0])

    assert stats["paired_gain"] == 0.25
    assert stats["wrong_to_right"] == 2
    assert stats["right_to_wrong"] == 1


def test_analysis_writes_fixed_tables_and_figures(tmp_path):
    runs = tmp_path / "runs"
    reports = tmp_path / "reports"
    _synthetic_run(runs, "synthetic-k32", 32, 0.9)
    _synthetic_run(runs, "synthetic-k1024", 1024, 0.8)
    _controlled_run(runs)
    _natural_run(runs, "gsm8k", "exact_match", 0.6)
    _natural_run(runs, "mbpp", "pass_at_1", 0.4)

    summary = analyze(runs, reports)

    assert summary == {
        "complete_runs": 5,
        "codec_rows": 5,
        "baselines": 3,
        "status_counts": {},
    }
    for name in (
        "summary.csv",
        "efficiency.csv",
        "accuracy_targets.csv",
        "rate_distortion.png",
        "bits_vs_information.png",
        "rate_allocation.png",
        "controlled_transfer.png",
        "model_checks.png",
        "natural_pareto.png",
        "quantization_retention.png",
        "behavioral_write.png",
        "placement_control.png",
        "ifeval_retention.png",
        "baseline_screening.csv",
        "learning_gates.csv",
    ):
        assert (reports / name).stat().st_size > 0
