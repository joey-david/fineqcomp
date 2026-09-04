from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from fineqcomp.budget_probe import (
    budget_at,
    check_gates,
    find_targets,
    training_contract,
    pair_budgets,
    scaled_adapter,
    score_probe,
    update_direction,
)
from fineqcomp.campaign import expand_campaign
from fineqcomp.config import load_campaign


CONFIG = load_campaign(Path("configs/bit_budget.yaml"))


def _cell(rank: int, file_bits: float, saved: float, ceiling: float = 100.0):
    return {
        "rank": rank,
        "file_bits": file_bits,
        "tensor_values": 1000 * rank,
        "heldout_bits_saved": saved,
        "ceiling_heldout_bits_saved": ceiling,
        "model_key": "m",
        "dataset_key": "d",
        "seed": 11,
    }


def test_budget_interpolates_between_the_two_envelope_points_that_bracket_it():
    cells = [
        _cell(4, 1000.0, 50.0),
        _cell(8, 2000.0, 80.0),
        _cell(16, 3000.0, 100.0),
    ]
    budget = budget_at(cells, 0.90)
    assert budget["bracketed"]
    # 90 bits sits halfway between the 80 and 100 rungs, so the file does too.
    assert budget["file_bits"] == pytest.approx(2500.0)
    assert budget["bits_per_value"] == pytest.approx(2500.0 / 16000)
    assert budget["rank"] == 16


def test_a_budget_the_sweep_never_reached_is_reported_unbracketed():
    cells = [_cell(4, 1000.0, 50.0), _cell(8, 2000.0, 80.0)]
    budget = budget_at(cells, 0.90)
    assert not budget["bracketed"]
    assert math.isnan(budget["file_bits"])


def test_a_cheaper_file_can_win_the_budget_from_a_bigger_container():
    """The point of the surface: rank is a way of spending bits, not a setting.

    A small rank-4 file beats a large rank-16 one at the crossing, so the
    budget is read off the rank-4 cell even though a bigger container exists.
    The rate is still quoted against the full container, which is what keeps
    it comparable with every bits-per-value number already recorded.
    """
    cells = [
        _cell(1, 200.0, 40.0),
        _cell(4, 800.0, 95.0),
        _cell(16, 3000.0, 100.0),
    ]
    budget = budget_at(cells, 0.90)
    assert budget["bracketed"]
    assert budget["rank"] == 4
    assert 200.0 < budget["file_bits"] < 800.0
    assert budget["container_values"] == 16000


def test_a_grid_whose_smallest_file_already_passes_is_not_bracketed():
    """Nothing on the grid fails, so the true budget is below what was measured."""
    budget = budget_at([_cell(4, 800.0, 95.0), _cell(16, 3000.0, 100.0)], 0.90)
    assert not budget["bracketed"]
    assert budget["file_bits"] == pytest.approx(800.0)


def test_the_update_direction_is_the_difference_and_scaling_is_linear_in_it():
    initial = {"a": torch.zeros(2, 2), "b": torch.ones(2, 2)}
    trained = {"a": torch.ones(2, 2), "b": torch.ones(2, 2) * 3}
    update = update_direction(initial, trained)
    assert torch.equal(update["a"], torch.ones(2, 2))
    assert torch.equal(update["b"], torch.ones(2, 2) * 2)
    assert torch.equal(scaled_adapter(initial, update, 1.0)["b"], trained["b"])
    assert torch.equal(
        scaled_adapter(initial, update, 0.0)["b"], initial["b"]
    )
    assert torch.equal(
        scaled_adapter(initial, update, 2.0)["a"], torch.ones(2, 2) * 2
    )


def test_update_direction_refuses_mismatched_containers():
    with pytest.raises(KeyError):
        update_direction({"a": torch.zeros(1)}, {"b": torch.zeros(1)})


def _budget_row(model, dataset, kind, bits, seed=11, updates=None):
    return {
        "run_id": f"{model}-{dataset}-{kind}-{seed}",
        "model_key": model,
        "dataset_key": dataset,
        "seed": seed,
        "kind": kind,
        "probe_updates": updates,
        "cells": 45,
        "budget_bracketed": True,
        "budget_file_bits": bits * 1000,
        "budget_bits_per_value": bits,
        "budget_rank": 16,
        "budget_ceiling_heldout_bits_saved": 500.0,
    }


def test_probes_pair_only_with_a_finished_run_on_the_same_cell():
    rows = [
        _budget_row("a", "code", "trained", 0.8),
        _budget_row("a", "code", "probe", 0.7, updates=1),
        _budget_row("b", "code", "probe", 0.5, updates=1),
    ]
    paired = pair_budgets(rows)
    assert len(paired) == 1
    assert paired[0]["model_key"] == "a"
    assert paired[0]["probe_bits_per_value"] == pytest.approx(0.7)
    assert paired[0]["target_bits_per_value"] == pytest.approx(0.8)


def test_an_unbracketed_budget_never_reaches_the_pairing():
    rows = [
        {**_budget_row("a", "code", "trained", 0.8), "budget_bracketed": False},
        _budget_row("a", "code", "probe", 0.7, updates=1),
    ]
    assert pair_budgets(rows) == []


def test_the_score_is_the_identity_line_and_not_a_correlation():
    """A probe that ranks perfectly but is offset must not score as accurate.

    This is the failure the campaign's earlier measures hid behind: a fitted
    receiver intercept turns a constant offset into a perfect result. Here the
    offset is the error.
    """
    paired = [
        {
            "model_key": model,
            "dataset_key": "code",
            "probe_updates": 1,
            "probe_bits_per_value": value + 1.0,
            "target_bits_per_value": value,
        }
        for model, value in (("a", 0.4), ("b", 0.6), ("c", 0.8))
    ]
    score = score_probe(paired, 1)
    assert score["spearman"] == pytest.approx(1.0)
    assert score["identity_rmse"] == pytest.approx(1.0)
    assert score["receiver_pair_sign_accuracy"] == pytest.approx(1.0)


def test_gates_reject_a_probe_that_ranks_without_predicting():
    rows = [_budget_row("a", "code", "trained", 0.8)]
    paired = [
        {
            "model_key": model,
            "dataset_key": "code",
            "probe_updates": 1,
            "probe_bits_per_value": value + 1.0,
            "target_bits_per_value": value,
        }
        for model, value in (("a", 0.4), ("b", 0.6), ("c", 0.8))
    ]
    report = check_gates(rows, paired, CONFIG["gates"])
    assert report["p1_budgets_are_bracketed"]
    assert report["p2_probe_orders_the_arms"]
    assert not report["p3_identity_beats_the_corpus_mean"]
    assert not report["passed"]


def test_gates_accept_a_probe_that_lands_on_the_identity_line():
    rows = [_budget_row("a", "code", "trained", 0.8)]
    paired = [
        {
            "model_key": model,
            "dataset_key": "code",
            "probe_updates": 1,
            "probe_bits_per_value": value + 0.01,
            "target_bits_per_value": value,
        }
        for model, value in (("a", 0.4), ("b", 0.6), ("c", 0.8))
    ]
    report = check_gates(rows, paired, CONFIG["gates"])
    assert report["passed"]


def test_every_run_in_the_panel_is_a_capped_probe():
    """The budget pays for probes and sweeps, never for a fine-tune.

    Every target this panel predicts is an adapter the campaign already
    trained, so a run in this config that was not capped would be an hour of
    training nobody asked for.
    """
    runs = list(expand_campaign(CONFIG))
    assert runs, "the panel is empty"
    assert all(run.training.max_updates is not None for run in runs)
    by_study = {run.study: run for run in runs}
    assert by_study["probe1_core"].training.max_updates == 1
    assert by_study["probe8_ladder"].training.max_updates == 8
    # A probe that restored its best checkpoint would not be the state at k
    # updates, and a warmed-up first step would be no step at all.
    assert by_study["probe1_core"].training.restore_best is False
    assert by_study["probe1_core"].training.warmup_ratio == 0.0


def test_every_probe_cell_has_a_finished_run_to_predict():
    """A probe with nothing to compare against is wasted GPU time.

    This panel trains no targets: it predicts adapters other configs already
    paid for. So the check is that every probe cell names a receiver, a corpus
    and a seed that some campaign config in the repository already covers.
    """
    trained = set()
    for path in sorted(Path("configs").glob("*.yaml")):
        if path.name == "bit_budget.yaml":
            continue
        try:
            raw = load_campaign(path)
        except Exception:
            continue
        if "studies" not in raw:
            continue
        try:
            runs = expand_campaign(raw)
        except Exception:
            continue
        for run in runs:
            trained.add((run.model.key, run.dataset_key, run.seed))
    # A cell a campaign declared but screened out was never trained, so it can
    # never be a target. The panel must not ask for one: the cluster preflight
    # cannot tell "the adapter was cleaned up" from "the adapter was never
    # made", and it stops the chain either way.
    for path in Path("results").rglob("screened*.csv"):
        for row in csv.DictReader(path.open()):
            trained.discard(
                (row["model_key"], row["dataset_key"], int(row["seed"]))
            )
    orphans = sorted(
        {
            (run.model.key, str(run.dataset_key), run.seed)
            for run in expand_campaign(CONFIG)
            if (run.model.key, run.dataset_key, run.seed) not in trained
        }
    )
    assert orphans == []


def test_the_config_gates_are_the_registered_ones():
    raw = yaml.safe_load(Path("configs/bit_budget.yaml").read_text())
    assert raw["gates"] == {
        "bracketed_share": 0.90,
        "probe_spearman": 0.60,
        "identity_over_corpus_mean": 0.75,
        "receiver_sign_accuracy": 0.70,
    }


def _write_run(root: Path, name: str, *, updates, model="m", dataset="d", seed=11,
               adapter="all_linear_r16", with_adapter=True):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    spec = {
        "run_id": name,
        "study": "s",
        "kind": "natural",
        "dataset_key": dataset,
        "seed": seed,
        "model": {"key": model, "name": model, "revision": "r", "backbone": "bf16"},
        "adapter": {
            "key": adapter,
            "method": "full_lora",
            "rank": 16,
            "target_modules": ["q_proj"],
            "last_n_layers": None,
            "alpha": 32,
            "dropout": 0.0,
        },
        "codecs": [{"key": "binary", "method": "uniform", "bits": 1}],
        "training": {
            "epochs": 1,
            "learning_rate": 0.0002,
            "effective_batch_size": 16,
            "micro_batch_size": 4,
            "max_length": 512,
            "max_updates": updates,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(spec))
    if with_adapter:
        (run_dir / "raw_channel.pt").write_bytes(b"")
    return run_dir


def test_a_probe_is_paired_with_the_finished_run_on_its_own_cell(tmp_path):
    from fineqcomp.config import RunSpec

    _write_run(tmp_path, "target", updates=None)
    _write_run(tmp_path, "other-seed", updates=None, seed=22)
    _write_run(tmp_path, "other-corpus", updates=None, dataset="e")
    probe_dir = _write_run(tmp_path, "probe", updates=1)
    probe = RunSpec.from_dict(json.loads((probe_dir / "config.json").read_text()))
    assert [path.name for path, _ in find_targets([probe], tmp_path)] == ["target"]


def test_a_finished_run_without_its_adapter_is_not_a_target(tmp_path):
    """Only the weights make a run swept-able, and old runs get cleaned up."""
    from fineqcomp.config import RunSpec

    _write_run(tmp_path, "target", updates=None, with_adapter=False)
    probe_dir = _write_run(tmp_path, "probe", updates=1)
    probe = RunSpec.from_dict(json.loads((probe_dir / "config.json").read_text()))
    assert find_targets([probe], tmp_path) == []


def test_targets_trained_under_different_budgets_are_not_interchangeable(tmp_path):
    """Two runs on one cell must agree on what they spent, or the receiver
    axis is carrying a difference in training recipe instead."""
    from fineqcomp.config import RunSpec

    one = _write_run(tmp_path, "four-epochs", updates=None)
    two = _write_run(tmp_path, "eight-epochs", updates=None, model="n")
    spec = json.loads((two / "config.json").read_text())
    spec["training"]["epochs"] = 8
    (two / "config.json").write_text(json.dumps(spec))
    contracts = {
        training_contract(RunSpec.from_dict(json.loads((path / "config.json").read_text())))
        for path in (one, two)
    }
    assert len(contracts) == 2
