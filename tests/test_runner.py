from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

from fineqcomp import runner
from fineqcomp.campaign import expand_campaign
from fineqcomp.config import load_campaign
from fineqcomp.data import Example
from fineqcomp.runner import RunEngine


def test_baseline_waiter_takes_lock_after_failed_holder(tmp_path, monkeypatch):
    attempts = iter((False, True))

    @contextmanager
    def claim_on_second_attempt(unused):
        yield next(attempts)

    monkeypatch.setattr(runner, "claim_run", claim_on_second_attempt)
    monkeypatch.setattr(runner.time, "sleep", lambda unused: None)

    with runner._claim_or_read_baseline(tmp_path, timeout_seconds=1) as result:
        assert result == (True, None)


def test_diversity_configs_reuse_panel_baselines():
    panel = load_campaign("configs/high_gain_panel.yaml")
    for config, panel_dataset in (
        ("configs/diversity_sql.yaml", "text_to_sql"),
        ("configs/diversity_xbrl.yaml", "xbrl_tags"),
    ):
        diversity = load_campaign(config)
        campaign = {
            **panel,
            "datasets": {**panel["datasets"], **diversity["datasets"]},
        }
        engine = RunEngine(campaign)
        panel_run = next(
            run
            for run in expand_campaign(panel)
            if run.dataset_key == panel_dataset and run.seed == 11
        )
        expected = engine._baseline_key(panel_run)
        diversity_runs = [
            run
            for run in expand_campaign(diversity)
            if run.model.key == panel_run.model.key and run.seed == panel_run.seed
        ]

        assert diversity_runs
        assert {engine._baseline_key(run) for run in diversity_runs} == {expected}


def test_generation_limit_gets_its_own_baseline_key():
    campaign = load_campaign("configs/reasoning_native_smoke.yaml")
    run = expand_campaign(campaign)[0]
    original = RunEngine(campaign)._baseline_key(run)
    changed = {
        **campaign,
        "datasets": {
            **campaign["datasets"],
            "cot_math": {
                **campaign["datasets"]["cot_math"],
                "evaluation_max_new_tokens": 2048,
            },
        },
    }

    revised = RunEngine(changed)._baseline_key(run)

    assert original != revised
    assert "gen2048" in revised
    assert expand_campaign(changed)[0].run_id != run.run_id


def test_saturated_natural_cell_stops_before_adapter_training(tmp_path, monkeypatch):
    campaign = load_campaign("configs/campaign.yaml")
    run = next(
        run
        for run in expand_campaign(campaign)
        if run.kind == "natural" and run.dataset_key == "metamath"
    )
    engine = RunEngine(campaign, prepared_root=tmp_path, runs_root=tmp_path / "runs")
    monkeypatch.setattr(engine, "_load_data", lambda unused: ({}, {}))
    monkeypatch.setattr(
        engine,
        "ensure_baseline",
        lambda *unused: {"exact_match": 0.90, "examples": 1_319},
    )
    session = SimpleNamespace()

    assert engine.run_one(session, run) == "screened_out"
    run_dir = tmp_path / "runs" / run.run_id
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "screened_out"
    assert not (run_dir / "raw_channel.pt").exists()


def test_learning_gate_requires_fixed_validation_bit_gain(tmp_path):
    campaign = load_campaign("configs/campaign.yaml")
    run = next(
        run
        for run in expand_campaign(campaign)
        if run.kind == "natural" and run.dataset_key == "metamath"
    )
    engine = RunEngine(campaign, runs_root=tmp_path)
    baseline = {"information": {"heldout": {"bits_per_token": 1.0}}}

    assert (
        engine._learning_gate(run, baseline, {"bits_per_token": 0.97})["status"]
        == "usable"
    )
    assert (
        engine._learning_gate(run, baseline, {"bits_per_token": 0.99})["status"]
        == "no_learning"
    )


def test_behavioral_write_reports_damage_as_negative():
    """A span the adapter made worse must not read as zero bits saved.

    These were clamped at zero, which made "the adapter left this span alone"
    and "the adapter destroyed this span" the same number. The chain-of-thought
    arms are where that mattered: answer-only supervision reads as 0.0 bits
    saved on the reasoning span when it is really 4.6 bits per token worse than
    the base model, so an arm that had wrecked the model looked inert.
    """
    baseline = {
        "train": {"total_bits": 10.0, "bits_per_token": 1.0},
        "heldout": {"total_bits": 10.0, "bits_per_token": 1.0},
    }
    tuned = {
        "train": {"total_bits": 12.0, "bits_per_token": 1.2},
        "heldout": {"total_bits": 8.0, "bits_per_token": 0.8},
    }

    result = RunEngine._behavioral_write(baseline, tuned)

    assert result["train_bits_saved"] == -2.0
    assert abs(result["train_bits_saved_per_token"] + 0.2) < 1e-12
    assert result["heldout_bits_saved"] == 2.0
    assert abs(result["excess_train_bits_per_token"] + 0.4) < 1e-12


def test_pilot_limit_keeps_rows_from_each_evaluator(tmp_path):
    engine = RunEngine({}, runs_root=tmp_path, pilot_rows=2)
    data = {
        "train": [Example(str(i), "p", "r", {}) for i in range(5)],
        "calibration": [Example(str(i), "p", "r", {}) for i in range(5)],
        "test": [
            Example(f"a{i}", "p", "r", {"evaluator": "a"})
            for i in range(4)
        ]
        + [
            Example(f"b{i}", "p", "r", {"evaluator": "b"})
            for i in range(4)
        ],
    }

    limited = engine._limit_data(data)

    assert len(limited["train"]) == 2
    assert len(limited["calibration"]) == 2
    assert [row.example_id for row in limited["test"]] == ["a0", "a1", "b0", "b1"]
