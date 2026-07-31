from __future__ import annotations

from collections import Counter

from fineqcomp.campaign import expand_campaign, read_manifest, write_manifest
from fineqcomp.config import load_campaign
from fineqcomp.runner import estimate_run_cost, partition_runs


def test_campaign_expands_to_fixed_grid(tmp_path):
    runs = expand_campaign(load_campaign("configs/campaign.yaml"))

    assert len(runs) == 96
    assert Counter(run.study for run in runs) == {
        "exact_seeded": 48,
        "exact_full": 6,
        "family_check": 12,
        "scale_check": 4,
        "backbone_check": 2,
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
