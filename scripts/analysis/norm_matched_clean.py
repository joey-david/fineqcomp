"""Score the clean adapter shrunk to the compressed adapter's update norm on every breadth probe.

The breadth study's controls norm-match the corrupted adapter only. This adds
the same control for the clean one: all 16 clean directions, update scaled so
its Frobenius norm equals that of the selected compressed condition (corrupted
direction 0 at 1 bit). It asks whether the clean adapter, made as small as the
compressed one, also stops hurting the far probes. It reuses the study's
matched_norm condition and evaluator, so rows land in the study's own test/
directory beside the other arms.

On a GPU node, one process per GPU:
    python norm_matched_clean.py SHARD SHARDS
"""
from __future__ import annotations

import sys
from pathlib import Path

from fineqcomp.studies.spectral_transfer import (collect, evaluate_conditions, load_lock,
                                         read_json, study_rows, write_predictions)

CONFIG = Path("configs/recovery/spectral_breadth.yaml")
OUT = Path(".cache/reports/spectral_breadth_v1")
TARGET = "band00_01_binary"


def condition() -> dict:
    selected = read_json(OUT / "selected.json")
    target = next(c for c in selected["conditions"] if c["key"] == TARGET)
    return {"key": f"norm_full_clean_{TARGET}", "kind": "matched_norm", "arm": "clean",
            "scope": "full", "target": target, "family": "norm_control"}


def main() -> None:
    lock = load_lock(CONFIG, OUT)
    if sys.argv[1:] == ["join"]:
        # Only the probes `test` scores: the primary one and the transfer panel.
        cfg = lock["config"]["spectral_transfer"]
        rows = study_rows(lock, OUT)
        for name in [cfg["primary"]["key"], *cfg["transfer"]]:
            _, predictions = collect(OUT / "test", condition(), name,
                                     [x.example_id for x in rows[name]])
            write_predictions(OUT / "joined" / name / f"{condition()['key']}.jsonl", predictions)
        return
    shard, shards = int(sys.argv[1]), int(sys.argv[2])
    evaluate_conditions(lock, OUT, [condition()], "test", OUT / "test", shard=shard, shards=shards)


if __name__ == "__main__":
    main()
