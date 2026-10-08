"""The Jean-Zay planner and collector for the reparameterization ablation."""

from __future__ import annotations

import importlib.util
import json
import os
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pipeline(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "reparameterization_pipeline", ROOT / "scripts/jean_zay/reparameterization.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The models live on Jean-Zay's $DSDIR; here every snapshot counts as readable.
    monkeypatch.setattr(module, "model_problem", lambda path: None)
    return module


def _counts(state: Path) -> dict[str, str]:
    lines = (state / "lists/counts.env").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines)


def _listed(state: Path, prefix: str) -> list[str]:
    return [run for path in sorted((state / "lists").glob(f"{prefix}_*.txt"))
            for run in path.read_text().split()]


def test_plan_lists_every_stage_and_skips_unreadable_models(pipeline, tmp_path, monkeypatch):
    state = pipeline.State(tmp_path)
    counts = pipeline.plan(state)
    assert (counts["N_RUNS"], counts["N_TRAIN_E1"], counts["N_TRAIN_E2"]) == (33, 9, 24)
    assert (counts["N_SWEEP"], counts["N_PROFILE"], counts["N_SMOKE"]) == (9, 27, 3)
    assert counts["SMOKE_NEEDED"] == 1 and _counts(tmp_path)["EXCLUDED_MODELS"] == ""
    sweep = _listed(tmp_path, "sweep")
    assert sweep == _listed(tmp_path, "train_e1")
    assert all("__s11__" in run and "-r16__" in run for run in sweep)
    arrays = [line.split() for line in (tmp_path / "lists/arrays.tsv").read_text().splitlines()]
    assert sorted((stage, corpus) for stage, _, corpus, _ in arrays) == sorted(
        (stage, corpus) for stage in ("train", "train", "sweep")
        for corpus in ("kind_code", "panel_math", "xbrl_tags"))
    assert sum(int(count) for *_, count in arrays) == 9 + 24 + 9
    # Each corpus's sweep follows that corpus's seed-11 training array.
    for stage, name, corpus, count in arrays:
        assert (tmp_path / "lists" / name).read_text().count("\n") == int(count)
        assert all(corpus.replace("_", "-") in run
                   for run in (tmp_path / "lists" / name).read_text().split())
    assert (tmp_path / "lists/gauges.txt").read_text().split() == list(pipeline.GAUGES)
    # The gauge smoke runs on the last pilot, the init variant.
    assert "conditioned" in (tmp_path / "lists/smoke.txt").read_text().split()[-1]

    monkeypatch.setattr(pipeline, "model_problem",
                        lambda path: "permission denied" if "Llama" in path else None)
    counts = pipeline.plan(state)
    assert counts["EXCLUDED_MODELS"] == "llama31_8b_base"
    assert (counts["N_USABLE"], counts["N_TRAIN_E1"], counts["N_SWEEP"], counts["N_SMOKE"]) == (30, 6, 6, 2)
    assert not any("llama" in run for run in _listed(tmp_path, "train_e1"))


def test_plan_resubmits_only_unfinished_work(pipeline, tmp_path):
    state = pipeline.State(tmp_path)
    pipeline.plan(state)
    finished = _listed(tmp_path, "train_e1")[0]
    screened = _listed(tmp_path, "train_e2")[0]
    for run_id, record in ((finished, {"state": "complete"}), (screened, {"state": "screened_out"})):
        (state.runs / run_id).mkdir(parents=True)
        (state.runs / run_id / "status.json").write_text(json.dumps(record))
    # A complete run needs its metrics and every codec file, not just a status.
    pipeline.plan(state)
    assert finished in _listed(tmp_path, "train_e1")
    assert screened not in _listed(tmp_path, "train_e2")

    swept = _listed(tmp_path, "sweep")[1]
    rows = [{"run_id": swept, "gauge": gauge} for gauge in pipeline.GAUGES]
    pipeline._write_csv(state.sweep / swept / "gauge_0.csv", rows[:-1])
    pipeline.plan(state)
    assert swept in _listed(tmp_path, "sweep")
    pipeline._write_csv(state.sweep / swept / "gauge_0.csv", rows)
    counts = pipeline.plan(state)
    assert swept not in _listed(tmp_path, "sweep")
    assert counts["N_SWEEP"] == 8


def test_collect_writes_an_archive_even_with_nothing_run(pipeline, tmp_path):
    state = pipeline.State(tmp_path)
    pipeline.plan(state)
    archive = pipeline.collect(state)
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert "reparameterization/report/summary.md" in names
    assert "reparameterization/lists/plan.json" in names
    summary = (state.report / "summary.md").read_text()
    assert "pending: 33" in summary


def test_model_problem_reads_the_snapshot_files(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "reparameterization_pipeline_raw", ROOT / "scripts/jean_zay/reparameterization.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "missing" in module.model_problem(str(tmp_path / "absent"))
    for name in ("config.json", "tokenizer.json", "model-00001.safetensors"):
        (tmp_path / name).write_text("{}")
    assert module.model_problem(str(tmp_path)) is None
    if os.geteuid() != 0:
        (tmp_path / "model-00001.safetensors").chmod(0)
        try:
            assert "permission denied" in module.model_problem(str(tmp_path))
        finally:
            (tmp_path / "model-00001.safetensors").chmod(0o644)
