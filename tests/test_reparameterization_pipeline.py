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


def _arrays(state: Path) -> list[list[str]]:
    return [line.split() for line in (state / "lists/arrays.tsv").read_text().splitlines()]


def test_plan_lists_every_stage_and_skips_unreadable_models(pipeline, tmp_path, monkeypatch):
    state = pipeline.State(tmp_path)
    counts = pipeline.plan(state)
    assert (counts["N_RUNS"], counts["N_TRAIN"], counts["N_SWEEP"]) == (45, 45, 9)
    assert (counts["N_FRONTIER"], counts["N_PROFILE"], counts["N_SMOKE"]) == (21, 39, 3)
    assert counts["SMOKE_NEEDED"] == 1 and _counts(tmp_path)["EXCLUDED_MODELS"] == ""
    sweep = _listed(tmp_path, "sweep")
    assert sorted(sweep) == sorted(_listed(tmp_path, "train_e1"))
    assert all("__s11__" in run and "-r16__" in run for run in sweep)
    assert (tmp_path / "lists/gauges.txt").read_text().split() == list(pipeline.GAUGES)
    # The gauge smoke runs on the last pilot, the init variant; the truncation
    # smoke on the math pilot.
    assert "conditioned" in (tmp_path / "lists/smoke.txt").read_text().split()[-1]
    assert "panel-math" in (tmp_path / "lists/smoke_frontier.txt").read_text()

    arrays = _arrays(tmp_path)
    trains = {name for stage, name, *_ in arrays if stage == "train"}
    assert sum(int(tasks) for stage, _, tasks, *_ in arrays if stage == "train") == 45
    for stage, name, tasks, minutes, needs in arrays:
        listed = (tmp_path / "lists" / name).read_text().split()
        assert len(listed) == (int(tasks) if stage != "profile" else 39)
        # One corpus and one walltime per array; waits only on training arrays.
        if stage != "profile":
            assert len({run.split("__")[2] for run in listed}) == 1
        assert needs == "-" or set(needs.split(",")) <= trains
        if stage in ("sweep", "frontier"):
            assert needs != "-"
    # The 14B receiver asks for longer than the 0.5B one on the same corpus.
    by_size = {name: int(minutes) for stage, name, _, minutes, _ in arrays if stage == "train"
               and "scale" in name and "kind_code" in name}
    assert max(by_size.values()) > 2 * min(by_size.values())

    monkeypatch.setattr(pipeline, "model_problem",
                        lambda path: "permission denied" if "Llama" in path else None)
    counts = pipeline.plan(state)
    assert counts["EXCLUDED_MODELS"] == "llama31_8b_base"
    assert (counts["N_USABLE"], counts["N_SWEEP"], counts["N_SMOKE"]) == (42, 6, 2)
    assert not any("llama" in run for run in _listed(tmp_path, "train"))


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
    assert "pending: 45" in summary


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


def test_gates_score_the_rule_against_what_was_measured(pipeline):
    runs = [
        {"study": "init_math", "init": "rotated", "rule_r_star": 0.50, "r_star": 0.55, "bracketed": True},
        {"study": "scale_code", "init": "default", "rule_r_star": 0.90, "r_star": 0.80, "bracketed": True},
        {"study": "gauge_code", "init": "default", "rule_r_star": 9.00, "r_star": 0.10, "bracketed": True},
    ]

    def row(run_id, gauge, r_star, delta, rule, rule_delta):
        return {"run_id": run_id, "gauge": gauge, "gauge_kind": gauge.split("-")[0], "bracketed": True,
                "r_star": r_star, "delta_r_star": delta, "rule_r_star": rule, "rule_delta_r_star": rule_delta}

    gauges = [
        row("a", "identity", 0.60, 0.0, 0.60, 0.0),
        row("a", "permutation-d0", 0.61, 0.01, 0.60, 0.0),
        row("a", "orthogonal-d0", 0.70, 0.10, 0.68, 0.08),
        row("a", "svd", 0.55, -0.05, 0.62, 0.02),
        row("a", "conditioned-k1000-d0", 1.20, 0.60, 0.80, 0.20),
    ]
    result = pipeline.gates(runs, gauges)
    g1, g2, g3, ill = result.values()
    # The default-init panel cell is a reproduction, not a new adapter.
    assert g1["n"] == 2 and g1["value"] == pytest.approx(((0.05**2 + 0.10**2) / 2) ** 0.5)
    assert g1["verdict"] == "PASS"
    # Permutation, orthogonal and svd are well conditioned; k1000 is not.
    assert g2["n"] == 3 and g2["verdict"] == "PASS"
    # Shifts beyond the permutation floor (0.01): orthogonal agrees, svd does
    # not, k1000 agrees -> 2 of 3.
    assert g3["n"] == 3 and g3["value"] == pytest.approx(2 / 3) and g3["verdict"] == "FAIL"
    assert ill["n"] == 1 and ill["bias"] == pytest.approx(-0.40)
