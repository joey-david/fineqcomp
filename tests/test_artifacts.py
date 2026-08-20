from __future__ import annotations

from fineqcomp.artifacts import claim_run, run_complete, write_json


def test_run_completion_requires_every_reloadable_result(tmp_path):
    write_json(tmp_path / "status.json", {"state": "complete"})
    write_json(tmp_path / "metrics.json", {})
    assert not run_complete(tmp_path, ("uniform4",))

    write_json(
        tmp_path / "codec_metrics" / "uniform4.json", {"task": {"exact_match": 0.5}}
    )
    (tmp_path / "codecs").mkdir()
    (tmp_path / "codecs" / "adapter_uniform4.fqcb").write_bytes(b"x")
    assert not run_complete(tmp_path, ("uniform4",))

    (tmp_path / "predictions").mkdir()
    (tmp_path / "predictions" / "task_uniform4.jsonl").write_text("{}\n")
    assert run_complete(tmp_path, ("uniform4",))


def test_bits_only_rungs_need_no_predictions(tmp_path):
    """A rung that skips the test pass still counts as finished.

    Below one bit the ladder is scored on held-out bits saved alone, so there
    are no generated answers to write and demanding them would make the run
    look unfinished for ever.
    """
    write_json(tmp_path / "status.json", {"state": "complete"})
    write_json(tmp_path / "metrics.json", {})
    write_json(tmp_path / "codec_metrics" / "sub0_125.json", {"task": None})
    (tmp_path / "codecs").mkdir()
    (tmp_path / "codecs" / "adapter_sub0_125.fqcb").write_bytes(b"x")
    assert run_complete(tmp_path, ("sub0_125",))


def test_fixed_gates_are_terminal_results(tmp_path):
    for state in ("screened_out", "no_learning"):
        write_json(tmp_path / "status.json", {"state": state})
        assert run_complete(tmp_path, ("uniform2", "uniform4", "fp16"))


def test_claim_run_excludes_second_owner(tmp_path):
    with claim_run(tmp_path) as first:
        with claim_run(tmp_path) as second:
            assert first
            assert not second
    with claim_run(tmp_path) as claimed_again:
        assert claimed_again
