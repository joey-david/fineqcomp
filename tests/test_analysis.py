from __future__ import annotations

import json

import pytest

from fineqcomp.analysis import _paired_stats, analyze, collect_rows
from fineqcomp.artifacts import write_json


def _prediction(path, correct):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"correct": value}) + "\n" for value in correct))


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
                    "retained_gain": {
                        "metric": metric,
                        "baseline_score": value - 0.1,
                        "retained_gain": 0.8,
                    },
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


def test_paired_stats_use_matched_predictions():
    stats = _paired_stats([0, 1, 0, 1], [1, 1, 1, 0])

    assert stats["paired_gain"] == 0.25
    assert stats["wrong_to_right"] == 2
    assert stats["right_to_wrong"] == 1


def test_codec_record_supplies_hashed_baseline_score(tmp_path):
    runs = tmp_path / "runs"
    _natural_run(runs, "gsm8k", "exact_match", 0.6)
    baseline = next((runs / "baselines").glob("*/metrics.json"))
    baseline.unlink()

    rows, _ = collect_rows(runs)

    assert rows[0]["baseline_test_score"] == pytest.approx(0.5)
    assert rows[0]["test_gain"] == pytest.approx(0.1)


def test_analysis_writes_fixed_tables_and_figures(tmp_path):
    runs = tmp_path / "runs"
    reports = tmp_path / "reports"
    _natural_run(runs, "gsm8k", "exact_match", 0.6)
    _natural_run(runs, "humaneval", "pass_at_1", 0.4)

    summary = analyze(runs, reports)

    assert summary == {
        "complete_runs": 2,
        "codec_rows": 2,
        "baselines": 2,
        "status_counts": {},
    }
    for name in (
        "summary.csv",
        "natural_pareto.png",
        "quantization_retention.png",
        "behavioral_write.png",
        "placement_control.png",
        "baseline_screening.csv",
        "learning_gates.csv",
    ):
        assert (reports / name).stat().st_size > 0
