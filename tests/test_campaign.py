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
from fineqcomp.runner import (
    RunEngine,
    estimate_run_cost,
    partition_runs,
    partition_runs_weighted,
)


def test_campaign_expands_to_fixed_grid(tmp_path):
    runs = expand_campaign(load_campaign("configs/campaign.yaml"))

    assert len(runs) == 24
    assert Counter(run.study for run in runs) == {
        "main_rate_distortion": 18,
        "placement_control": 6,
    }
    assert len({run.run_id for run in runs}) == len(runs)
    assert {run.model.name for run in runs} == {
        "Qwen/Qwen2.5-7B",
        "mistralai/Mistral-7B-v0.1",
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

    assert sum(map(len, shards)) == 24
    assert len({run.run_id for shard in shards for run in shard}) == len(runs)
    costs = [sum(estimate_run_cost(run) for run in shard) for shard in shards]
    assert max(costs) / min(costs) < 1.07


def test_weighted_workers_give_a100s_three_times_more_work():
    runs = expand_campaign(load_campaign("configs/campaign.yaml"))
    weights = [3.0, 3.0, 1.0, 1.0]
    partitions = partition_runs_weighted(runs, weights)
    normalized = [
        sum(estimate_run_cost(run) for run in partition) / weight
        for partition, weight in zip(partitions, weights, strict=True)
    ]

    assert [len(partition) for partition in partitions] == [9, 9, 3, 3]
    # Twelve cells over four weighted workers cannot split evenly; the two
    # matched-rate mid-tread controls raise per-run evaluation cost and widen
    # the best achievable gap to about 1.10.
    assert max(normalized) / min(normalized) < 1.12


def test_natural_screening_uses_fixed_dataset_ceiling(tmp_path):
    campaign = load_campaign("configs/campaign.yaml")
    run = next(
        run
        for run in expand_campaign(campaign)
        if run.kind == "natural" and run.dataset_key == "metamath"
    )
    engine = RunEngine(campaign, runs_root=tmp_path)

    assert (
        engine._screening(
            run,
            {"exact_match": 0.90, "examples": 512},
        )["status"]
        == "too_easy"
    )
    assert (
        engine._screening(
            run,
            {"exact_match": 0.80, "examples": 512},
        )["status"]
        == "usable"
    )


def test_run_id_tracks_dataset_size_caps(tmp_path):
    """Two configs differing only in row caps must not share a run id.

    They otherwise collide, and the second config silently reuses the first's
    trained adapter and codec metrics rather than recomputing them.
    """
    def metamath_ids(rows: int) -> set[str]:
        raw = load_campaign("configs/campaign.yaml")
        raw["datasets"]["metamath"]["train_rows"] = rows
        return {
            run.run_id
            for run in expand_campaign(raw)
            if run.dataset_key == "metamath"
        }

    small, large = metamath_ids(8000), metamath_ids(32000)

    assert small and large
    assert len(small) == len(large)
    assert not (small & large)
