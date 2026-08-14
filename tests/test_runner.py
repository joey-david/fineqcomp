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
        if run.kind == "natural" and run.dataset_key == "gsm8k"
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


def test_learning_gate_requires_fixed_five_point_calibration_gain(tmp_path):
    campaign = load_campaign("configs/campaign.yaml")
    run = next(
        run
        for run in expand_campaign(campaign)
        if run.kind == "natural" and run.dataset_key == "gsm8k"
    )
    engine = RunEngine(campaign, runs_root=tmp_path)
    baseline = {"calibration": {"exact_match": 0.40, "examples": 512}}

    assert engine._learning_gate(run, baseline, {"exact_match": 0.46})["status"] == "usable"
    assert (
        engine._learning_gate(run, baseline, {"exact_match": 0.44})["status"]
        == "no_learning"
    )
