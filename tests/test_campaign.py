from __future__ import annotations

from collections import Counter

from fineqcomp.campaign import expand_campaign, read_manifest, write_manifest
from fineqcomp.config import load_campaign
from fineqcomp.runner import RunEngine, estimate_run_cost, partition_runs


def test_campaign_expands_to_fixed_grid(tmp_path):
    runs = expand_campaign(load_campaign("configs/campaign.yaml"))

    assert len(runs) == 105
    assert Counter(run.study for run in runs) == {
        "exact_seeded": 48,
        "exact_full": 6,
        "family_check": 12,
        "scale_check": 4,
        "backbone_check": 2,
        "controlled_transfer": 9,
        "natural_qwen": 18,
        "natural_mistral": 6,
    }
    assert len({run.run_id for run in runs}) == len(runs)
    assert {run.model.name for run in runs} >= {
        "Qwen/Qwen3-8B-Base",
        "Qwen/Qwen3-14B-Base",
        "mistralai/Mistral-7B-v0.3",
    }

    manifest = write_manifest(runs, tmp_path / "manifest.jsonl")
    assert read_manifest(manifest) == runs


def test_two_shards_are_disjoint_and_cost_balanced():
    runs = expand_campaign(load_campaign("configs/campaign.yaml"))
    shards = partition_runs(runs, 2)

    assert {run.run_id for run in shards[0]}.isdisjoint(run.run_id for run in shards[1])
    assert sum(map(len, shards)) == len(runs)
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
        engine._screening(run, {"exact_match": 0.90, "examples": 1_319})["status"]
        == "too_easy"
    )
    assert (
        engine._screening(run, {"exact_match": 0.80, "examples": 1_319})["status"]
        == "usable"
    )
