from __future__ import annotations

import json
from types import SimpleNamespace

from fineqcomp.campaign import expand_campaign
from fineqcomp.config import load_campaign
from fineqcomp.runner import RunEngine


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


def test_behavioral_write_clips_negative_savings():
    baseline = {
        "train": {"total_bits": 10.0, "bits_per_token": 1.0},
        "heldout": {"total_bits": 10.0, "bits_per_token": 1.0},
    }
    tuned = {
        "train": {"total_bits": 12.0, "bits_per_token": 1.2},
        "heldout": {"total_bits": 8.0, "bits_per_token": 0.8},
    }

    result = RunEngine._behavioral_write(baseline, tuned)

    assert result["train_bits_saved"] == 0.0
    assert result["train_bits_saved_per_token"] == 0.0
    assert result["heldout_bits_saved"] == 2.0
