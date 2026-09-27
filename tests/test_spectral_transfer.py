from dataclasses import replace
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest
import torch

from fineqcomp.artifacts import write_json
from fineqcomp.config import load_campaign
from fineqcomp.data import Example, _convert_svamp
from fineqcomp.evaluation import write_predictions
from fineqcomp.studies.generalisation import spectral_slice
from fineqcomp.studies.spectral_transfer import (
    collect, disjoint, grid, materialize, selected_conditions, shortlist,
)
from fineqcomp.studies import spectral_transfer as study
from fineqcomp.campaign import expand_campaign

A = "model.q_proj.lora_A.default.weight"
B = A.replace(".lora_A.", ".lora_B.")


@pytest.fixture
def raw():
    generator = torch.Generator().manual_seed(2)
    return {A: torch.randn(16, 32, generator=generator),
            B: torch.randn(24, 16, generator=generator)}


@pytest.fixture
def cfg():
    return load_campaign(Path("configs/recovery/spectral_transfer.yaml"))["spectral_transfer"]


def test_prepared_lock_detects_a_changed_study_module(tmp_path, monkeypatch):
    config = Path("configs/recovery/spectral_transfer.yaml")
    groups = {name: [Example(name, f"Question: {name}", "#### 1", {})]
              for name in ("train", "search", "selection", "gsm8k")}

    def prepare_rows(campaign, key, seed, root):
        study._write_jsonl(root / "natural" / key / f"seed{seed}" / "train.jsonl",
                          groups["train"])

    monkeypatch.setattr(study, "prepare_natural_dataset", prepare_rows)
    monkeypatch.setattr(study, "materialize_probes", lambda *args: {})
    monkeypatch.setattr(study, "decontaminate_training", lambda *args: 0)
    monkeypatch.setattr(study, "study_rows", lambda *args: groups)
    lock = study.prepare(config, tmp_path)
    source = Path("src/fineqcomp/studies/spectral_transfer.py")
    assert str(source) in lock["implementation"]
    assert study.load_lock(config, tmp_path)["implementation"] == lock["implementation"]

    digest = study.sha256
    monkeypatch.setattr(study, "sha256", lambda path: "changed" if path == source else digest(path))
    with pytest.raises(ValueError, match="implementation changed after prepare"):
        study.load_lock(config, tmp_path)


def test_band_is_dense_update_svd_and_recombines(raw):
    u, s, vh = torch.linalg.svd(raw[B] @ raw[A], full_matrices=False)
    updates = []
    for low, high in ((0, 2), (2, 8), (8, 16)):
        band = spectral_slice(raw, low, high)
        actual = band[B] @ band[A]
        torch.testing.assert_close(actual, (u[:, low:high] * s[low:high]) @ vh[low:high],
                                   atol=4e-5, rtol=5e-5)
        updates.append(actual)
    torch.testing.assert_close(sum(updates), raw[B] @ raw[A], atol=4e-5, rtol=5e-5)


def test_grid_and_selection_are_fixed(cfg):
    candidates = grid(cfg, 16)
    assert len(candidates) == 131
    assert len({c["key"] for c in candidates}) == len(candidates)
    assert {c["family"] for c in candidates} == {"filter", "codec", "band_codec", "shrinkage"}
    rows = [{"condition": c, "accuracy": 0.5, "file_bits": i + 1} for i, c in enumerate(candidates)]
    finalists = shortlist(rows, 2)
    assert len(finalists) == 8
    winners = {family: next(c for c in finalists if c["family"] == family)
               for family in {c["family"] for c in finalists}}
    conditions = selected_conditions(winners, cfg)
    assert len(conditions) == len({c["key"] for c in conditions}) == 18
    assert len([c for c in conditions if c["family"] == "fixed_band"]) == 6


@pytest.mark.parametrize("edges", [[], [0, 2, 1, 16], [0, 1, 1, 16], [1, 16], [0, 17]])
def test_bad_grid_rejected(cfg, edges):
    with pytest.raises(ValueError):
        grid({**cfg, "edges": edges}, 16)


def test_every_code_roundtrips_with_native_rank_and_norm_control(raw, cfg, tmp_path):
    tensors = {"clean": raw, "permuted": raw}
    for name, code in cfg["codes"].items():
        candidate = {"key": name, "kind": "coded", "arm": "permuted",
                     "low": 2, "high": 8, "code": code}
        decoded, geometry = materialize(candidate, tensors, {}, 16, {}, tmp_path)
        assert decoded[A].shape == raw[A].shape
        assert decoded[B].shape == raw[B].shape
        assert torch.isfinite(decoded[A]).all()
        assert geometry["file_bits"] > 0
        assert decoded[A][6:].count_nonzero() == 0
        for scope in ("full", "band"):
            control = {"kind": "matched_norm", "arm": "permuted", "scope": scope, "target": candidate}
            _, matched = materialize(control, tensors, {}, 16, {}, tmp_path)
            assert matched["update_norm"] == pytest.approx(geometry["update_norm"], rel=1e-5)


def test_mechanism_arms_split_the_code_into_reweighting_and_rotation(raw, tmp_path):
    tensors = {"clean": raw, "permuted": raw}
    full = spectral_slice(raw, 0, 16)
    torch.testing.assert_close(study.direction_gains(full, spectral_slice(raw, 0, 4))[A],
                               torch.tensor([1.0] * 4 + [0.0] * 12), atol=1e-4, rtol=0)
    conditions = {c["key"]: c for c in study.mechanism_conditions(16)}
    assert len(conditions) == 3 + 5 * 5 + 4
    arms = {key: materialize(conditions[key], tensors, {}, 16, {}, tmp_path)
            for key in ("band00_04_binary", "profile_04", "profile_04_lead",
                        "profile_04_rest", "restored_04", "flat_04")}
    dense = {key: arm[0][B] @ arm[0][A] for key, arm in arms.items()}
    u, s, vh = torch.linalg.svd(raw[B] @ raw[A], full_matrices=False)
    along = lambda update: torch.diagonal(u.T @ update @ vh.T)[:16]  # the update's own span
    coded = along(dense["band00_04_binary"])
    # The profile carries the code's component on every original direction and nothing else.
    torch.testing.assert_close(along(dense["profile_04"]), coded, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(dense["profile_04_lead"] + dense["profile_04_rest"], dense["profile_04"])
    # Restoring direction 0 puts it back at full strength and leaves the rest as coded.
    restored = along(dense["restored_04"])
    torch.testing.assert_close(restored[0], s[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(restored[1:], coded[1:], atol=1e-4, rtol=1e-4)
    assert arms["flat_04"][1]["update_norm"] == pytest.approx(arms["profile_04"][1]["update_norm"], rel=1e-5)


def test_overlap_uses_question_not_dataset_id():
    a = Example("gsm8k-1", "Solve.\nQuestion: How many apples?\nAnswer:", "#### 3", {})
    b = replace(a, example_id="svamp-1", prompt="Other instruction.\nProblem: How many apples?\nAnswer:")
    with pytest.raises(ValueError, match="share 1 questions"):
        disjoint({"train": [a], "test": [b]})


def test_svamp_uses_existing_numeric_contract():
    row = _convert_svamp([{"ID": "a", "Body": "Three apples.", "Question": "How many?", "Answer": 3}], "test")[0]
    assert row.example_id == "svamp-test-a"
    assert row.response.strip() == "#### 3"
    assert "Three apples. How many?" in row.prompt


def test_collect_requires_exact_complete_rows_and_condition(tmp_path):
    c = {"key": "base", "kind": "base"}
    path = tmp_path / "base/gsm8k/rows00000_00002.json"
    write_json(path, {"condition": c, "file_bits": 0, "update_norm": 0})
    write_predictions(path.with_suffix(".jsonl"), [
        {"example_id": "a", "correct": True, "hit_generation_limit": False},
        {"example_id": "b", "correct": False, "hit_generation_limit": True}])
    row, _ = collect(tmp_path, c, "gsm8k", ["a", "b"])
    assert row["accuracy"] == row["cap_fraction"] == 0.5
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        collect(tmp_path, c, "gsm8k", ["a", "b", "c"])
    with pytest.raises(ValueError, match="wrong condition"):
        collect(tmp_path, {**c, "kind": "raw"}, "gsm8k", ["a", "b"])


def test_search_select_test_report_resume_contract(raw, tmp_path, monkeypatch):
    config = load_campaign("configs/recovery/spectral_transfer.yaml")
    config["spectral_transfer"].update(finalists_per_family=1, bootstrap_draws=20)
    cfg = config["spectral_transfer"]
    all_candidates = grid(cfg, 16)
    candidates = [next(c for c in all_candidates if c["family"] == family)
                  for family in ("filter", "codec", "band_codec", "shrinkage")]
    run = expand_campaign(config)[0].to_dict()
    lock = {"config": config, "runs": {"clean": run, "permuted": run},
            "grid": candidates, "claim_boundary": "test"}
    data = {split: [Example(f"{split}-{i}", str(i), str(i), {}) for i in range(3)]
            for split in ("search", "selection", "gsm8k", "svamp", "math500")}
    monkeypatch.setattr(study, "study_rows", lambda *args: data)
    monkeypatch.setattr(study, "load_adapters", lambda *args: {"clean": raw, "permuted": raw})
    monkeypatch.setattr(study, "sha256", lambda path: "fixed")
    monkeypatch.setattr(study, "adapter_tensors", lambda *args: {A: raw[A], B: raw[B] * 0})
    monkeypatch.setattr(study, "apply_adapter_tensors", lambda *args: None)

    class Session:
        model, tokenizer = None, None

        def attach(self, *args):
            pass

        def unload(self):
            pass

    monkeypatch.setattr(study.ModelSession, "load", lambda *args: Session())
    seen = []

    def evaluate(model, tokenizer, examples, *args, **kwargs):
        seen.extend(x.example_id for x in examples)
        return {"examples": len(examples), "exact_match": 1.0}, [
            {"example_id": x.example_id, "correct": True, "hit_generation_limit": False}
            for x in examples]

    monkeypatch.setattr(study, "evaluate_natural", evaluate)
    for shard in range(2):
        study.evaluate_conditions(lock, tmp_path, candidates + study.references(), "search",
                                  tmp_path / "search", shard=shard, shards=2)
    finalists = study.search_finalists(lock, tmp_path)["conditions"] + study.references()
    for shard in (2, 0, 1):
        study.evaluate_conditions(lock, tmp_path, finalists, "selection",
                                  tmp_path / "selection", shard=shard, shards=3)
    before_selection = len(seen)
    frozen = study.select(lock, tmp_path)
    assert len(seen) == before_selection
    assert all(i.startswith(("search-", "selection-")) for i in seen)
    assert study.select(lock, tmp_path) == frozen
    assert study.report(lock, tmp_path)["status"] == "partial"
    core = study.test_group(frozen["conditions"], "core")
    controls = study.test_group(frozen["conditions"], "controls")
    assert len(core) == 7 and len(controls) == 11
    assert not {c["key"] for c in core} & {c["key"] for c in controls}
    assert {c["key"] for c in core + controls} == {c["key"] for c in frozen["conditions"]}
    # A bounded smoke can resume into the full core without changing results.
    study.evaluate_conditions(lock, tmp_path, core, "test", tmp_path / "test", max_tasks=1)
    assert study.report(lock, tmp_path, "core")["status"] == "partial"
    study.evaluate_conditions(lock, tmp_path, core, "test", tmp_path / "test")
    early = study.report(lock, tmp_path, "core")
    assert early["status"] == "complete" and early["scope"] == "core"
    assert not early["full_study_complete"]
    assert len(early["rows"]) == 21
    assert study.report(lock, tmp_path)["status"] == "partial"
    core_report_bytes = (tmp_path / "summary_core.json").read_bytes()
    study.evaluate_conditions(lock, tmp_path, controls, "test", tmp_path / "test")
    result = study.report(lock, tmp_path)
    assert result["status"] == "complete"
    assert result["full_study_complete"]
    assert (tmp_path / "summary_core.json").read_bytes() == core_report_bytes
    assert len(result["rows"]) == 54
    winner = frozen["winners"]["filter"]["key"]
    row = next(r for r in result["rows"] if r["condition"]["key"] == winner)
    assert row[f"ci95_vs_norm_full_{winner}"] == [0.0, 0.0]
    previous = len(seen)
    study.evaluate_conditions(lock, tmp_path, frozen["conditions"], "test", tmp_path / "test")
    assert len(seen) == previous


def test_mechanism_phase_scores_and_reports(raw, tmp_path, monkeypatch):
    config = load_campaign("configs/recovery/spectral_transfer.yaml")
    config["spectral_transfer"].update(bootstrap_draws=20)
    run = expand_campaign(config)[0].to_dict()
    lock = {"config": config, "runs": {"clean": run, "permuted": run}}
    primary = config["spectral_transfer"]["primary"]["key"]
    data = {primary: [Example(f"q-{i}", str(i), str(i), {}) for i in range(3)]}
    monkeypatch.setattr(study, "study_rows", lambda *args: data)
    monkeypatch.setattr(study, "load_adapters", lambda *args: {"clean": raw, "permuted": raw})
    monkeypatch.setattr(study, "adapter_tensors", lambda *args: {A: raw[A], B: raw[B] * 0})
    monkeypatch.setattr(study, "apply_adapter_tensors", lambda *args: None)
    monkeypatch.setattr(study.ModelSession, "load", lambda *args: type(
        "Session", (), {"model": None, "tokenizer": None,
                        "attach": lambda *a: None, "unload": lambda *a: None})())
    monkeypatch.setattr(study, "evaluate_natural", lambda model, tokenizer, examples, *a, **k: (
        {"examples": len(examples), "exact_match": 1.0},
        [{"example_id": x.example_id, "correct": True, "hit_generation_limit": False} for x in examples]))
    conditions = study.mechanism_conditions(16)
    for shard in range(3):
        study.evaluate_conditions(lock, tmp_path, conditions, primary, tmp_path / "mechanism",
                                  shard=shard, shards=3)
    result = study.mechanism_report(lock, tmp_path)
    assert len(result["rows"]) == len(conditions)
    flat = next(r for r in result["rows"] if r["condition"]["key"] == "flat_04")
    assert flat["ci95_vs_band00_04_binary"] == [0.0, 0.0]
    directions = result["directions"]
    assert len(directions["gain_prefix_01"]) == len(directions["own_code_gain"]) == 16
    assert 0 <= directions["off_diagonal_share_prefix_16"] < 1


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("phase", ["screen", "search", "validate", "test", "mechanism"])
def test_batch_launcher_maps_devices_and_propagates_failure(tmp_path, fail, phase):
    executable = tmp_path / "python"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "device = os.environ.get('CUDA_VISIBLE_DEVICES', 'leader')\n"
        "pathlib.Path(os.environ['TEST_LOG'], device).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.exit(13 if os.environ['TEST_FAIL'] == '1' and device == '7' else 0)\n")
    executable.chmod(0o755)
    srun = tmp_path / "srun"
    srun.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess, sys\n"
        "args = sys.argv[1:]\n"
        "assert '--gpus-per-task=1' in args and '--gpu-bind=single:1' in args\n"
        "command = args[args.index('bash'):]\n"
        "codes = [subprocess.run(command, env={**os.environ, 'SLURM_PROCID': str(i), "
        "'CUDA_VISIBLE_DEVICES': device}).returncode for i, device in enumerate(('4', '7'))]\n"
        "sys.exit(max(codes))\n")
    srun.chmod(0o755)
    env = {**os.environ, "PYTHON": str(executable), "SLURM_NTASKS": "2",
           "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
           "TOTAL_SHARDS": "4", "SHARD_OFFSET": "2", "TEST_LOG": str(tmp_path),
           "TEST_FAIL": str(int(fail))}
    result = subprocess.run(["bash", "scripts/jean_zay/spectral.sbatch",
                             phase, "--out", "study", "--group", "core"], env=env, capture_output=True)
    assert result.returncode == (13 if fail else 0), result.stderr
    for device, shard in (("4", "2"), ("7", "3")):
        args = json.loads((tmp_path / device).read_text())
        assert args[args.index("--shard") + 1] == shard
        assert args[args.index("--shards") + 1] == "4"
    leader = tmp_path / "leader"
    assert leader.exists() == (not fail and phase != "search")
    if leader.exists():
        args = json.loads(leader.read_text())
        expected = {"screen": "recommend_screen", "validate": "select", "test": "report",
                    "mechanism": "mechanism_report"}
        assert args[args.index("--phase") + 1] == expected[phase]


# --- scaled study: fluency screening and the wider probe panel ---------------

SCALE = Path("configs/recovery/spectral_scale.yaml")


def fluent(**overrides):
    metrics = {"exact_match": 0.45, "hit_generation_limit_fraction": 0.01,
               "answer_extracted_fraction": 1.0, "mean_repeated_8gram_fraction": 0.02,
               "mean_completion_tokens": 140.0}
    return {**metrics, **overrides}


@pytest.fixture
def gates():
    return load_campaign(SCALE)["screen"]["gates"]


def test_screen_rejects_the_pilot_failure_before_it_reads_the_score(gates):
    """The Mistral base arm: 74% capped, 0.102. Fluency must fail it first."""
    result = study.verdict(
        fluent(exact_match=0.102, hit_generation_limit_fraction=0.74,
               mean_repeated_8gram_fraction=0.44), gates)
    assert result["status"] == "degenerate"
    assert "hits the generation cap too often" in result["reasons"]


@pytest.mark.parametrize("metrics,expected", [
    (fluent(), "usable"),
    (fluent(exact_match=0.95), "too_easy"),
    (fluent(exact_match=0.02), "too_hard"),
    (fluent(answer_extracted_fraction=0.5), "degenerate"),
    (fluent(mean_repeated_8gram_fraction=0.9), "degenerate"),
])
def test_screen_verdicts(gates, metrics, expected):
    assert study.verdict(metrics, gates)["status"] == expected


def test_screen_prefers_a_receiver_usable_across_the_family():
    """A model usable only on the training task cannot show dataset agnosticism."""
    cfg = load_campaign(SCALE)["screen"]
    rows = []
    for model, statuses in {
        "broad": ["usable", "usable", "usable", "usable", "too_easy", "usable", "usable"],
        "narrow": ["usable", "too_easy", "too_easy", "too_easy", "too_easy", "usable", "usable"],
        "broken": ["degenerate"] * 7,
    }.items():
        for (probe, source), status in zip(cfg["probes"].items(), statuses, strict=True):
            rows.append({"model": model, "probe": probe, "family": source["family"],
                         "status": status, "headroom_to_ceiling": 0.3,
                         "generation_limit_fraction": 0.5 if model == "broken" else 0.01})
    chosen = study.recommend(rows, cfg)
    assert chosen["chosen"] == "broad"
    assert chosen["ranking"][-1]["model"] == "broken"


def test_recommend_will_not_choose_a_model_too_easy_on_the_primary_probe():
    """The exact bug this guards against: a receiver that never fails fluency
    on any probe was chosen even though its untrained GSM8K score (0.88) was
    already past the screen's own 0.80 ceiling -- training on it would have
    left clean/permuted/coded scores clustered near the ceiling with no room
    to show corruption hurting or coding recovering. A narrower model whose
    primary probe sits mid-range must win instead.
    """
    cfg = load_campaign(SCALE)["screen"]
    rows = []
    profiles = {
        # Fluent everywhere, but its primary probe (first, gsm8k) is too_easy.
        "ceiling_saturated": ["too_easy", "too_easy", "too_easy", "too_easy",
                              "usable", "too_easy", "too_easy"],
        # Usable on its primary probe, degenerate on two off-family siblings.
        "primary_usable": ["usable", "too_easy", "too_easy", "usable",
                           "degenerate", "degenerate", "too_easy"],
    }
    for model, statuses in profiles.items():
        for (probe, source), status in zip(cfg["probes"].items(), statuses, strict=True):
            rows.append({"model": model, "probe": probe, "family": source["family"],
                         "status": status, "headroom_to_ceiling": 0.3,
                         "generation_limit_fraction": 0.05})
    result = study.recommend(rows, cfg)
    assert result["chosen"] == "primary_usable"
    top = result["ranking"][0]
    assert top["model"] == "primary_usable"
    ceiling_row = next(r for r in result["ranking"] if r["model"] == "ceiling_saturated")
    assert ceiling_row["primary_status"] == "too_easy"


def test_recommend_reports_no_choice_when_nothing_clears_the_primary_probe():
    cfg = load_campaign(SCALE)["screen"]
    rows = [{"model": "m", "probe": probe, "family": source["family"],
            "status": "too_easy", "headroom_to_ceiling": 0.1,
            "generation_limit_fraction": 0.02}
           for probe, source in cfg["probes"].items()]
    result = study.recommend(rows, cfg)
    assert result["chosen"] is None
    assert result["chosen_but_primary_probe_unusable"] == "m"


def test_scale_probe_families_cover_both_transfer_claims():
    cfg = load_campaign(SCALE)["spectral_transfer"]
    families = {spec["family"] for spec in cfg["transfer"].values()}
    assert families == {"math", "code", "mc_science"}
    # Dataset agnosticism needs several same-family probes from other corpora.
    assert sum(s["family"] == "math" for s in cfg["transfer"].values()) >= 4
    assert cfg["primary"]["key"] not in cfg["transfer"]


def test_every_probe_declares_an_evaluator_and_the_primary_resolves():
    cfg = load_campaign(SCALE)["spectral_transfer"]
    for name in cfg["transfer"]:
        assert study.probe_spec(cfg, name)["evaluator"]
    for name in ("search", "selection", cfg["primary"]["key"]):
        assert study.probe_spec(cfg, name) == {"metric": "exact_match", **cfg["primary"]}
    with pytest.raises(ValueError, match="no evaluator declared"):
        study.probe_spec(cfg, "some_untracked_split")


def test_mc_letter_reads_a_standalone_letter_only():
    from fineqcomp.evaluation import _first_choice_letter
    labels = ["A", "B", "C", "D"]
    assert _first_choice_letter(" C", labels) == "C"
    assert _first_choice_letter("The answer is D.", labels) == "D"
    # A bare word must not be mined for letters.
    assert _first_choice_letter("Chlorophyll absorbs light", labels) is None
    assert _first_choice_letter("no letter here", labels) is None


def test_asdiv_keeps_only_rows_the_numeric_scorer_can_mark():
    from fineqcomp.data import _convert_asdiv
    numeric = {"body": "A has 2.", "question": "How many?", "answer": "9 (apples)"}
    named = {"body": "B and C ran.", "question": "Who won?", "answer": "Mrs. Hilt"}
    clock = {"body": "It is 3:00.", "question": "When?", "answer": "3:30 p.m."}
    converted = _convert_asdiv([numeric] * 18 + [named, clock], "test")
    assert len(converted) == 18
    assert converted[0].response.strip() == "#### 9"
    assert converted[0].metadata["unscoreable_rows_excluded"] == 2
    # Enough unscoreable rows means the answer format moved, not that the probe
    # should quietly shrink.
    with pytest.raises(ValueError, match="numeric"):
        _convert_asdiv([numeric] * 5 + [named] * 5, "test")


def test_gsm_symbolic_ids_separate_template_from_instance():
    from fineqcomp.data import _convert_gsm_symbolic
    rows = [{"id": "7", "instance": "3", "question": "q", "answer": "work\n#### 5"}]
    converted = _convert_gsm_symbolic(rows, "test")
    assert converted[0].example_id == "gsm-symbolic-test-7-3"
    assert converted[0].metadata["evaluator"] == "gsm8k"


def test_probe_cap_samples_instead_of_taking_a_prefix(tmp_path, monkeypatch):
    """GSM-Symbolic ships fifty instances of each template in order."""
    from fineqcomp.data import read_jsonl
    rows = [{"id": str(index // 50), "instance": str(index % 50),
             "question": f"q{index}", "answer": "w\n#### 1"} for index in range(500)]
    stub = type(sys)("datasets")
    stub.load_dataset = lambda path, name=None, revision=None, split=None: rows
    monkeypatch.setitem(sys.modules, "datasets", stub)
    source = {"path": "apple/GSM-Symbolic", "name": "main", "revision": "r",
              "split": "test", "converter": "gsm_symbolic", "evaluator": "gsm8k",
              "rows": 20}
    counts = study.materialize_probes({"gsm_symbolic": source}, tmp_path, 11)
    assert counts == {"gsm_symbolic": 20}
    sampled = read_jsonl(tmp_path / "gsm_symbolic.jsonl")
    templates = {example.example_id.split("-")[3] for example in sampled}
    assert len(templates) > 5, "a prefix would have covered at most one template"
    # A second call must reuse the frozen file rather than resample it.
    assert study.materialize_probes({"gsm_symbolic": source}, tmp_path, 99) == counts


def _setup_fake_screen(monkeypatch, num_models=4, num_probes=2):
    campaign = load_campaign(SCALE)
    probes = dict(list(campaign["screen"]["probes"].items())[:num_probes])
    models = dict(list(campaign["models"].items())[:num_models])
    config = {**campaign, "models": models,
              "screen": {**campaign["screen"], "probes": probes, "rows": 4}}
    monkeypatch.setattr(study, "load_campaign", lambda path: config)
    monkeypatch.setattr(study, "materialize_probes",
                        lambda sources, directory, seed: _fake_probes(sources, directory))

    loaded, scored = [], []

    class Session:
        def __init__(self, key):
            self.key = key
        def unload(self):
            pass

    monkeypatch.setattr(study.ModelSession, "load",
                        classmethod(lambda cls, spec: loaded.append(spec.key) or Session(spec.key)))

    def fake_score(session, spec, seed, examples, cfg, path, **kwargs):
        scored.append((spec.key, path.stem))
        # A weak receiver rambles; the others answer cleanly.
        weak = spec.key == "mistral_7b"
        metrics = {"examples": len(examples), "exact_match": 0.10 if weak else 0.45,
                   "hit_generation_limit_fraction": 0.74 if weak else 0.01,
                   "answer_extracted_fraction": 1.0,
                   "mean_repeated_8gram_fraction": 0.44 if weak else 0.02,
                   "mean_completion_tokens": 430.0 if weak else 130.0, **kwargs["details"]}
        write_json(path, metrics)
        return metrics

    monkeypatch.setattr(study, "score", fake_score)
    return models, probes, loaded, scored


def test_screen_shards_by_model_and_caches_cells(tmp_path, monkeypatch):
    """Each shard scores only its own slice of models, and writes no summary."""
    models, probes, loaded, scored = _setup_fake_screen(monkeypatch)

    first = study.screen(SCALE, tmp_path, shard=0, shards=2)
    assert first == {"shard": 0, "status": "complete"}
    second = study.screen(SCALE, tmp_path, shard=1, shards=2)
    assert second == {"shard": 1, "status": "complete"}
    assert not (tmp_path / "screen.json").exists(), "screen() never aggregates"
    assert sorted(loaded) == sorted(models)
    assert len(scored) == len(models) * len(probes)

    # Rerunning either shard must not regenerate or reload a model.
    study.screen(SCALE, tmp_path, shard=0, shards=2)
    assert len(scored) == len(models) * len(probes)
    assert sorted(loaded) == sorted(models)


def test_recommend_screen_only_aggregates_and_never_generates(tmp_path, monkeypatch):
    """The bug this guards against: an earlier `screen()` did both jobs in one
    function, so the unsharded barrier call (shard=0, shards=1 by default)
    re-walked every model and silently redid a full generation pass on one
    GPU -- the smoke job that exposed it ran 40 minutes and hit the wall-time
    limit. The aggregation phase must read cached cells only.
    """
    models, probes, loaded, scored = _setup_fake_screen(monkeypatch)
    study.screen(SCALE, tmp_path, shard=0, shards=len(models))
    for shard in range(1, len(models)):
        study.screen(SCALE, tmp_path, shard=shard, shards=len(models))
    generated_before = len(scored)
    loaded_before = list(loaded)

    result = study.recommend_screen(SCALE, tmp_path)

    assert loaded == loaded_before, "aggregation must never load a model"
    assert len(scored) == generated_before, "aggregation must never call score()"
    assert result["complete"]
    assert json.loads((tmp_path / "screen.json").read_text())["rows"] == result["rows"]
    assert "mistral_7b" not in result["usable_by_model"]
    assert result["recommendation"]["chosen"] != "mistral_7b"
    assert result["recommendation"]["ranking"][-1]["model"] == "mistral_7b"
    weak = next(r for r in result["rows"] if r["model"] == "mistral_7b")
    assert weak["status"] == "degenerate"


def test_recommend_screen_reports_incomplete_before_every_shard_lands(tmp_path, monkeypatch):
    _setup_fake_screen(monkeypatch, num_models=4)
    study.screen(SCALE, tmp_path, shard=0, shards=2)  # only half the models
    result = study.recommend_screen(SCALE, tmp_path)
    assert not result["complete"]


def _fake_probes(sources, directory):
    from fineqcomp.data import _write_jsonl
    counts = {}
    for name in sources:
        examples = [Example(example_id=f"{name}-{index}", prompt="q", response="1",
                            metadata={"evaluator": "gsm8k"}) for index in range(4)]
        _write_jsonl(directory / f"{name}.jsonl", examples)
        counts[name] = len(examples)
    return counts


def _screen_result(chosen, rows_for):
    return {"recommendation": {"chosen": chosen},
            "rows": [{"model": model, "probe": probe, "status": status}
                     for model, probe, status in rows_for]}


def test_prepare_refuses_a_receiver_the_screen_rejected(tmp_path):
    config = load_campaign(SCALE)
    receiver = config["studies"]["scale_clean"]["models"][0]
    write_json(tmp_path / "screen.json",
               _screen_result("qwen3_32b", [(receiver, "gsm8k", "degenerate")]))
    with pytest.raises(ValueError, match="the screen recommends 'qwen3_32b'"):
        study.check_receiver(config, tmp_path)


def test_prepare_accepts_the_recommended_receiver_or_a_declared_override(tmp_path):
    config = load_campaign(SCALE)
    receiver = config["studies"]["scale_clean"]["models"][0]
    write_json(tmp_path / "screen.json",
               _screen_result(receiver, [(receiver, "gsm8k", "usable")]))
    study.check_receiver(config, tmp_path)

    write_json(tmp_path / "screen.json",
               _screen_result("qwen3_32b", [(receiver, "gsm8k", "too_easy")]))
    overridden = {**config, "spectral_transfer": {
        **config["spectral_transfer"], "override_screen": "32B does not fit the budget"}}
    study.check_receiver(overridden, tmp_path)


def test_prepare_runs_before_any_screen_exists(tmp_path):
    study.check_receiver(load_campaign(SCALE), tmp_path)



def _claim_rows(base, permuted, coded_by_dataset, families, primary="gsm8k"):
    rows = []
    for dataset, coded in coded_by_dataset.items():
        family = families[dataset]
        for key, accuracy in (("base", base), ("permuted_raw", permuted), ("winner", coded)):
            row = {"dataset": dataset, "family": family, "accuracy": accuracy,
                   "condition": {"key": key}}
            if key == "winner":
                # A half-point interval either side, so the sign is the claim.
                row["ci95_vs_base"] = [coded - base - 0.02, coded - base + 0.02]
            rows.append(row)
    return {"rows": rows}


@pytest.fixture
def claim_cfg():
    return load_campaign(SCALE)["spectral_transfer"]


FAMILIES = {"gsm8k": "training_task", "svamp": "math", "asdiv": "math",
            "gsm_symbolic": "math", "math500": "math",
            "humaneval": "code", "arc_challenge": "mc_science"}


def test_claims_hold_when_coding_recovers_across_the_family_only(claim_cfg):
    coded = {"gsm8k": 0.72, "svamp": 0.70, "asdiv": 0.69, "gsm_symbolic": 0.66,
             "math500": 0.61, "humaneval": 0.40, "arc_challenge": 0.41}
    summary = _claim_rows(0.60, 0.40, coded, FAMILIES)
    # Off-family probes must sit on the frozen model, so pin them there.
    for row in summary["rows"]:
        if row["family"] in {"code", "mc_science"} and row["condition"]["key"] == "winner":
            row["accuracy"] = 0.60
            row["ci95_vs_base"] = [-0.02, 0.02]
    verdict = study.claims(summary, claim_cfg, {"band_codec": {"key": "winner"}})
    assert verdict["ordering_on_training_task"]
    assert verdict["dataset_agnostic_within_family"]
    assert verdict["task_specific_off_family"]
    assert verdict["all_claims_hold"]


def test_a_gain_on_every_task_refutes_task_specificity(claim_cfg):
    """A coded adapter that lifts code and science too is not task-specific."""
    coded = dict.fromkeys(FAMILIES, 0.72)
    verdict = study.claims(_claim_rows(0.60, 0.40, coded, FAMILIES), claim_cfg,
                           {"band_codec": {"key": "winner"}})
    assert verdict["ordering_on_training_task"]
    assert not verdict["task_specific_off_family"]
    assert not verdict["all_claims_hold"]


def test_a_gain_only_on_the_training_task_refutes_dataset_agnosticism(claim_cfg):
    coded = {name: (0.72 if name == "gsm8k" else 0.60) for name in FAMILIES}
    summary = _claim_rows(0.60, 0.40, coded, FAMILIES)
    for row in summary["rows"]:
        if row["dataset"] != "gsm8k" and row["condition"]["key"] == "winner":
            row["ci95_vs_base"] = [-0.02, 0.02]
    verdict = study.claims(summary, claim_cfg, {"band_codec": {"key": "winner"}})
    assert verdict["ordering_on_training_task"]
    assert not verdict["dataset_agnostic_within_family"]


def test_a_harmless_corruption_refutes_the_ordering(claim_cfg):
    """If the permuted arm does not hurt, there is nothing to recover."""
    coded = dict.fromkeys(FAMILIES, 0.72)
    verdict = study.claims(_claim_rows(0.60, 0.59, coded, FAMILIES), claim_cfg,
                           {"band_codec": {"key": "winner"}})
    assert not verdict["ordering_on_training_task"]
    assert not verdict["all_claims_hold"]


def test_claims_report_incomplete_rather_than_guessing(claim_cfg):
    summary = _claim_rows(0.60, 0.40, {"gsm8k": 0.72}, FAMILIES)
    summary["rows"] = [r for r in summary["rows"] if r["condition"]["key"] != "base"]
    verdict = study.claims(summary, claim_cfg, {"band_codec": {"key": "winner"}})
    assert verdict["per_dataset"]["gsm8k"] == {"status": "incomplete"}
    assert not verdict["all_claims_hold"]


def test_verdict_reads_the_probes_own_metric_name(gates):
    """gates["metric"] is a global default; a probe like HumanEval overrides it.

    Without the per-probe override, verdict() looks for "exact_match" in a
    HumanEval metrics dict, which only ever has "pass_at_1", and crashes the
    screen's aggregation the moment a HumanEval cell finishes.
    """
    metrics = fluent(exact_match=0.05)
    metrics.pop("exact_match")
    metrics["pass_at_1"] = 0.45
    result = study.verdict(metrics, gates, metric="pass_at_1")
    assert result["status"] == "usable"
    assert result["score"] == 0.45
    with pytest.raises(KeyError):
        study.verdict(metrics, gates)


def test_score_checks_finiteness_of_the_named_metric(tmp_path, monkeypatch):
    """score() used to hard-code exact_match; a pass_at_1-only evaluator raised
    KeyError on that check, and again on its own progress print, before the
    caller ever saw its own metric name."""
    from fineqcomp.config import ModelSpec

    class Session:
        model = object()
        tokenizer = object()

    examples = [Example("e-0", "q", "a", {"evaluator": "humaneval"})]

    def fake_evaluate(model, tokenizer, examples, model_spec, dataset, batch_size,
                      multiple_choice_labels=None, max_new_tokens=None, generation_seed=None):
        return ({"examples": 1, "pass_at_1": 1.0, "hit_generation_limit_fraction": 0.0,
                "answer_extracted_fraction": 1.0, "mean_repeated_8gram_fraction": 0.0,
                "mean_completion_tokens": 4.0},
               [{"example_id": "e-0", "correct": True}])

    monkeypatch.setattr(study, "evaluate_natural", fake_evaluate)
    cfg = {"generation_batch_size": 1, "max_new_tokens": 8}
    result = study.score(Session(), ModelSpec("m", "m", "r", "bf16"), 11, examples, cfg,
                         tmp_path / "out.json", dataset="humaneval", metric="pass_at_1")
    assert result["pass_at_1"] == 1.0


def test_verdict_does_not_require_an_extraction_step_that_does_not_exist(gates):
    """HumanEval grades the whole completion directly; it never sets
    answer_extracted_fraction. Its absence must not fail the fluency gate."""
    metrics = fluent(exact_match=0.45)
    del metrics["answer_extracted_fraction"]
    assert study.verdict(metrics, gates)["status"] == "usable"


def test_sweep_design_grows_linearly_and_covers_each_direction():
    """all_contiguous is quadratic: rank 64 would ask for 2,080 windows."""
    base = {"codes": {"fp16": {"bits": 16}}, "scales": [0.3]}
    sweep = {**base, "band_design": "sweep", "sweep_step": 1}
    sizes = {rank: len(study.band_intervals(sweep, rank)) for rank in (16, 32, 64)}
    assert sizes == {16: 45, 32: 93, 64: 189}
    assert len(study.band_intervals({**base, "edges": list(range(65))}, 64)) == 2080

    windows = study.band_intervals(sweep, 16)
    # Every direction is measured alone, and consecutive suffixes differ by one
    # direction so their difference is that direction's marginal value.
    assert all((i, i + 1) in windows for i in range(16))
    suffixes = sorted(w for w in windows if w[1] == 16)
    assert [lo for lo, _ in suffixes] == list(range(16))
    assert all((0, k) in windows for k in range(1, 17))


def test_sweep_step_coarsens_without_leaving_gaps():
    base = {"codes": {"fp16": {"bits": 16}}, "scales": [0.3]}
    windows = study.band_intervals({**base, "band_design": "sweep", "sweep_step": 8}, 64)
    singles = sorted(w for w in windows if w[1] - w[0] == 8 and w[0] % 8 == 0)
    assert [lo for lo, _ in singles] == list(range(0, 64, 8))
    assert (0, 64) in windows
    with pytest.raises(ValueError, match="sweep_step"):
        study.band_intervals({**base, "band_design": "sweep", "sweep_step": 0}, 64)
    with pytest.raises(ValueError, match="unknown band_design"):
        study.band_intervals({**base, "band_design": "spiral"}, 64)


def test_leave_one_out_keeps_every_direction_but_one(raw):
    """The one question a contiguous window cannot ask."""
    from fineqcomp.studies.generalisation import spectral_subset, spectral_slice
    full = spectral_slice(raw, 0, 16)
    dropped = spectral_subset(raw, [i for i in range(16) if i != 3])
    a = next(n for n in full if ".lora_A." in n)
    assert full[a].shape[0] == 16 and dropped[a].shape[0] == 15
    # Direction 3 is the only one missing: rows 0-2 and 4-15 are untouched.
    torch.testing.assert_close(dropped[a][:3], full[a][:3])
    torch.testing.assert_close(dropped[a][3:], full[a][4:])
    with pytest.raises(ValueError, match="indices must lie inside"):
        spectral_subset(raw, [99])


def test_grid_emits_one_leave_one_out_condition_per_direction():
    cfg = {"edges": list(range(17)), "codes": {"fp16": {"bits": 16}},
           "scales": [0.3], "leave_one_out": True}
    conditions = grid(cfg, 16)
    loo = [c for c in conditions if c["family"] == "leave_one_out"]
    assert len(loo) == 16
    assert sorted(c["dropped"] for c in loo) == list(range(16))
    for c in loo:
        assert c["dropped"] not in c["keep"] and len(c["keep"]) == 15
    # Without the flag the grid is unchanged.
    assert not [c for c in grid({**cfg, "leave_one_out": False}, 16)
                if c["family"] == "leave_one_out"]


def test_subset_condition_materialises_like_a_band(raw, tmp_path):
    condition = {"key": "drop03", "kind": "subset", "arm": "permuted",
                 "keep": [i for i in range(16) if i != 3], "dropped": 3,
                 "family": "leave_one_out"}
    base = {k: torch.zeros_like(v) for k, v in raw.items()}
    decoded, extra = materialize(condition, {"permuted": raw, "clean": raw},
                                 base, 16, {}, tmp_path)
    a = next(n for n in decoded if ".lora_A." in n)
    assert decoded[a].shape[0] == 16, "padded back to the attached rank"
    assert extra["update_norm"] > 0


def _q(text):
    return Example(text[:12], f"Question: {text}\nAnswer:", "x", {})


def test_disjoint_guards_the_splits_that_decide_things():
    """A selecting split sharing a question with anything is a leak."""
    shared = _q("what is two plus two")
    with pytest.raises(ValueError, match="share 1 questions"):
        study.disjoint({"train": [shared], "search": [shared]})
    with pytest.raises(ValueError, match="share 1 questions"):
        study.disjoint({"selection": [shared], "gsm8k": [shared]})
    with pytest.raises(ValueError, match="share 1 questions"):
        study.disjoint({"train": [shared], "mmlu_anatomy": [shared]})


def test_two_probes_may_overlap_and_the_count_is_reported():
    """MMLU ships some questions under two subjects; both stay valid probes."""
    shared, unique = _q("which bone is longest"), _q("unrelated question")
    overlap, protected = study.disjoint({
        "search": [_q("held out entirely")],
        "mmlu_clinical_knowledge": [shared, unique],
        "mmlu_college_medicine": [shared],
    })
    assert overlap == {"mmlu_clinical_knowledge|mmlu_college_medicine": 1}
    assert protected == ["search"]


def test_every_breadth_probe_names_the_metric_its_evaluator_reports():
    # The breadth test died on HumanEval: the probe declared its evaluator but
    # not pass_at_1, and scoring looked up exact_match.
    cfg = load_campaign("configs/recovery/spectral_breadth.yaml")["spectral_transfer"]
    metrics = {name: study.probe_spec(cfg, name)["metric"] for name in cfg["transfer"]}
    assert metrics["humaneval"] == "pass_at_1"
    assert {m for n, m in metrics.items() if n != "humaneval"} == {"exact_match"}


def test_likelihood_reads_no_row_that_decides_its_outcome(raw, tmp_path, monkeypatch):
    config = load_campaign("configs/recovery/spectral_transfer.yaml")
    cfg = config["spectral_transfer"]
    candidates = grid(cfg, 16)[:3]
    run = expand_campaign(config)[0].to_dict()
    lock = {"config": config, "runs": {"clean": run, "permuted": run}, "grid": candidates}
    probes = [cfg["primary"]["key"], *cfg["transfer"]]
    data = {split: [Example(f"{split}-{i}", str(i), str(i), {}) for i in range(10)]
            for split in ("search", "selection", *probes)}
    monkeypatch.setattr(study, "study_rows", lambda *args: data)
    monkeypatch.setattr(study, "load_adapters", lambda *args: {"clean": raw, "permuted": raw})
    monkeypatch.setattr(study, "adapter_tensors", lambda *args: {A: raw[A], B: raw[B] * 0})
    monkeypatch.setattr(study, "apply_adapter_tensors", lambda *args: None)

    class Session:
        model, tokenizer = None, None

        def attach(self, *args):
            pass

        def unload(self):
            pass

    monkeypatch.setattr(study.ModelSession, "load", lambda *args: Session())
    seen = []

    def nll(model, tokenizer, examples, *args):
        seen.append([x.example_id for x in examples])
        return {"nll": 1.0, "bits_per_token": 1.44, "total_bits": 1.0, "nll_tokens": 5}

    monkeypatch.setattr(study, "causal_nll", nll)
    for shard in range(3):
        study.predict_likelihood(lock, tmp_path, shard=shard, shards=3)
    conditions = candidates + study.references()
    assert len(seen) == len(conditions) * (1 + len(probes))
    ids = {i for batch in seen for i in batch}
    assert not any(i.startswith("search-") for i in ids)
    # Probes contribute only their first half; the second half is the outcome.
    assert {i for i in ids if i.startswith(f"{probes[0]}-")} == {f"{probes[0]}-{i}" for i in range(5)}
    written = sorted((tmp_path / "likelihood").rglob("*.json"))
    assert len(written) == len(seen)
    assert study.read_json(written[0])["condition"]["key"] in {c["key"] for c in conditions}
    study.predict_likelihood(lock, tmp_path)
    assert len(seen) == len(conditions) * (1 + len(probes))  # resumes, never rescores


def test_a_grid_without_codecs_selects_only_the_families_it_has():
    # The fp16-only fine sweep died in select: it demanded a codec winner.
    rows = [{"condition": {"key": k, "family": f}, "accuracy": a, "file_bits": None}
            for k, f, a in [("b1", "filter", 0.5), ("b2", "filter", 0.6),
                            ("s1", "shrinkage", 0.4), ("d1", "leave_one_out", 0.9),
                            ("base", "reference", 0.3)]]
    assert study.families_present(rows) == ["filter", "shrinkage"]
    assert [c["key"] for c in study.shortlist(rows, 1)] == ["b2", "s1"]
