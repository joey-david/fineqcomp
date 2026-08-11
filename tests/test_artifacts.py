from __future__ import annotations

from fineqcomp.artifacts import claim_run, run_complete, write_json


def test_run_completion_requires_every_reloadable_result(tmp_path):
    write_json(tmp_path / "status.json", {"state": "complete"})
    write_json(tmp_path / "metrics.json", {})
    assert not run_complete(tmp_path, (4,))

    write_json(tmp_path / "codec_metrics" / "b4.json", {})
    (tmp_path / "codecs").mkdir()
    (tmp_path / "codecs" / "adapter_b4.fqcb").write_bytes(b"x")
    (tmp_path / "predictions").mkdir()
    (tmp_path / "predictions" / "task_b4.jsonl").write_text("{}\n")
    assert run_complete(tmp_path, (4,))


def test_claim_run_excludes_second_owner(tmp_path):
    with claim_run(tmp_path) as first:
        with claim_run(tmp_path) as second:
            assert first
            assert not second
    with claim_run(tmp_path) as claimed_again:
        assert claimed_again
