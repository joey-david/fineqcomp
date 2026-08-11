from __future__ import annotations

import sys
from types import SimpleNamespace

from fineqcomp.preflight import cache_models


def test_cache_models_downloads_exact_revisions_and_checks_weights(
    tmp_path, monkeypatch
):
    calls = []

    def snapshot_download(model, revision, token):
        calls.append((model, revision, token))
        target = tmp_path / revision
        target.mkdir()
        (target / "model.safetensors").write_bytes(b"weights")
        return target

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    campaign = {
        "models": {
            "small": {"name": "org/model", "revision": "abc123"},
        }
    }

    report = cache_models(campaign)

    assert calls == [("org/model", "abc123", None)]
    assert report["small"]["weight_files"] == 1
    assert report["small"]["weight_bytes"] == 7
