from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest
import yaml

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


def test_conditional_trace_control_matches_the_finished_aligned_grid():
    control = load_campaign("configs/conditional_trace_rate.yaml")
    cot = load_campaign("configs/cot_panel.yaml")
    llama = load_campaign("configs/external_llama_panel.yaml")
    runs = expand_campaign(control)

    assert len(runs) == 9
    assert {run.seed for run in runs} == {11, 22, 33}
    assert {run.model.key for run in runs} == {
        "mistral_7b_base", "qwen25_7b_base", "llama31_8b_base",
    }
    control_training = control["training"]["cot_full_1ep"]
    control_adapter = control["adapters"]["all_linear_r16"]
    control_dataset = control["datasets"]["cot_math_permuted"]
    assert control_dataset["rationale_control"] == "permuted"
    for aligned in (cot, llama):
        assert aligned["training"]["cot_full_1ep"] == control_training
        assert aligned["adapters"]["all_linear_r16"] == control_adapter
        dataset = aligned["datasets"]["cot_math"]
        for key in (
            "train_source", "answer_marker", "validation_rows", "train_rows",
            "test_rows", "evaluations",
        ):
            assert dataset[key] == control_dataset[key]
    for model in control["models"]:
        aligned = llama if model == "llama31_8b_base" else cot
        assert control["models"][model] == aligned["models"][model]

    lock = json.loads(Path(
        "results/3_chain_of_thought_under_compression/"
        "conditional_trace_rate_lock.json"
    ).read_text())
    assert set(lock["arms"]["permuted"]["run_ids"]) == {
        run.run_id for run in runs
    }
    for arm in lock["arms"]["aligned"]["runs"]:
        campaign = load_campaign(arm["config"])
        assert set(arm["run_ids"]) == {
            run.run_id for run in expand_campaign(campaign)
            if run.study == arm["study"] and run.model.key == arm["model"]
        }


def test_native_reasoning_smoke_is_one_seed_with_a_wide_rate_grid():
    campaign = load_campaign("configs/reasoning_native_smoke.yaml")
    runs = expand_campaign(campaign)

    assert len(runs) == 9
    assert {run.seed for run in runs} == {11}
    assert {run.dataset_key for run in runs} == {"cot_math"}
    assert all(run.model.chat and not run.model.disable_thinking for run in runs)
    assert {
        run.model.generation_profile
        for run in runs
        if run.model.key.startswith("qwen3_")
    } == {"qwen3_thinking"}
    assert {
        run.model.generation_profile
        for run in runs
        if run.model.key.startswith("r1_")
    } == {"deepseek_r1"}
    assert {run.model.key for run in runs} == set(campaign["models"])
    rates = {
        float(codec["bits"]) + float(codec.get("blend", 0.0))
        for codec in campaign["codecs"].values()
    }
    assert rates == {
        0.0625, 0.125, 0.25, 0.5, 0.75,
        1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 8.0, 16.0,
    }


def test_reasoning_data_smoke_caps_every_new_trace_source():
    campaign = load_campaign("configs/reasoning_data_smoke.yaml")
    runs = expand_campaign(campaign)

    assert len(runs) == 2
    assert {run.dataset_key for run in runs} == {
        "numina_math_cot_smoke", "openr1_math_smoke"
    }
    for spec in campaign["datasets"].values():
        assert spec["train_rows"] == 128
        assert spec["validation_rows"] == 64
        assert spec["test_rows"] == 64


def test_reasoning_scaling_lock_matches_the_smoke_panel_and_rate_grid():
    lock = json.loads(Path(
        "results/3_chain_of_thought_under_compression/"
        "reasoning_scaling_battery_lock.json"
    ).read_text())
    campaign = load_campaign(lock["model_panel"]["smoke_config"])

    assert set(lock["model_panel"]["smoke_models"]) == set(campaign["models"])
    rates = sorted(
        float(codec["bits"]) + float(codec.get("blend", 0.0))
        for codec in campaign["codecs"].values()
    )
    assert rates == lock["rate_grid"]["coarse_target_bits_per_value"]
    assert lock["grpo_extension"]["implementation_owner"] == (
        "src/fineqcomp/grpo_control.py"
    )


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


def test_compressibility_arms_are_compute_matched():
    """Every arm must train for the same steps on the same number of samples.

    The arms vary only in how many of those samples are distinct. If steps
    scale with row count instead, duplication and training volume move
    together and the experiment cannot separate them — which is what a first
    smoke config did by flattening every arm to one epoch.
    """
    for path in ("configs/compressibility.yaml", "configs/compressibility_smoke.yaml"):
        raw = load_campaign(path)
        seen = set()
        for name, study in raw["studies"].items():
            dataset = raw["datasets"][study["datasets"][0]]
            training = raw["training"][study["training"]]
            rows = int(dataset["train_rows"])
            steps = rows // int(training["effective_batch_size"]) * int(
                training["epochs"]
            )
            seen.add((steps, rows * int(training["epochs"])))
        assert len(seen) == 1, f"{path}: arms differ in compute: {sorted(seen)}"


def test_run_id_does_not_repeat_a_study_named_after_its_dataset():
    raw = load_campaign("configs/compressibility.yaml")
    for run in expand_campaign(raw):
        assert run.run_id.count(run.study.replace("_", "-")) == 1


def test_information_pilot_uses_a_cached_model_and_dense_curve():
    campaign = load_campaign("configs/campaign.yaml")
    info = yaml.safe_load(Path("configs/information_scaling.yaml").read_text())

    assert info["model"] == "mistral_7b_base"
    assert info["model"] in campaign["models"]
    assert info["labels"] == list("ABCDEFGHIJKLMNOP")
    rates = {
        (int(codec["bits"]), float(codec.get("blend", 0.0)))
        for codec in info["adapter_codecs"]
    }
    assert (0, 0.0625) in rates
    assert (1, 0.0) in rates
    assert (1, 0.75) in rates
    assert (2, 0.0) in rates
