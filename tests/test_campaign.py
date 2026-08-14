from __future__ import annotations

from collections import Counter

import pytest

from fineqcomp.campaign import (
    expand_campaign,
    read_manifest,
    validate_manifest,
    write_manifest,
)
from fineqcomp.config import load_campaign
from fineqcomp.runner import RunEngine, estimate_run_cost, partition_runs


def test_campaign_expands_to_fixed_grid(tmp_path):
    runs = expand_campaign(load_campaign("configs/campaign.yaml"))

    assert len(runs) == 48
    assert Counter(run.study for run in runs) == {
        "real_tasks_qwen": 24,
        "real_tasks_mistral": 24,
    }
    assert len({run.run_id for run in runs}) == len(runs)
    assert {run.model.name for run in runs} == {
        "Qwen/Qwen3-8B",
        "mistralai/Mistral-7B-Instruct-v0.3",
    }

    manifest = write_manifest(runs, tmp_path / "manifest.jsonl")
    assert read_manifest(manifest) == runs


def test_stale_manifest_is_rejected():
    campaign = load_campaign("configs/campaign.yaml")
    runs = expand_campaign(campaign)

    with pytest.raises(ValueError, match="prepare again"):
        validate_manifest(runs[:-1], campaign)


def test_two_shards_are_disjoint_and_cost_balanced():
    runs = expand_campaign(load_campaign("configs/campaign.yaml"))
    shards = partition_runs(runs, 2)

    assert {run.run_id for run in shards[0]}.isdisjoint(run.run_id for run in shards[1])
    assert sum(map(len, shards)) == len(runs)
    costs = [sum(estimate_run_cost(run) for run in shard) for shard in shards]
    assert max(costs) / min(costs) < 1.02


def test_four_shards_match_the_two_host_layout():
    runs = expand_campaign(load_campaign("configs/campaign.yaml"))
    shards = partition_runs(runs, 4)

    assert sum(map(len, shards)) == 48
    assert [len(shard) for shard in shards] == [12] * 4
    assert len({run.run_id for shard in shards for run in shard}) == len(runs)
    costs = [sum(estimate_run_cost(run) for run in shard) for shard in shards]
    assert max(costs) / min(costs) < 1.02


def test_natural_screening_uses_fixed_dataset_ceiling(tmp_path):
    campaign = load_campaign("configs/campaign.yaml")
    run = next(
        run
        for run in expand_campaign(campaign)
        if run.kind == "natural" and run.dataset_key == "gsm8k"
    )
    engine = RunEngine(campaign, runs_root=tmp_path)

    assert (
        engine._screening(
            run,
            {"calibration": {"exact_match": 0.90, "examples": 512}},
        )["status"]
        == "too_easy"
    )
    assert (
        engine._screening(
            run,
            {"calibration": {"exact_match": 0.80, "examples": 512}},
        )["status"]
        == "usable"
    )
