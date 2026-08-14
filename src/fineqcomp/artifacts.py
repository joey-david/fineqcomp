"""Atomic run-state and JSON artifact helpers."""

from __future__ import annotations

import json
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_NB, LOCK_UN, flock
from pathlib import Path
from typing import Any, Iterator


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    return target


def read_json(path: str | Path, default: Any = None) -> Any:
    source = Path(path)
    return json.loads(source.read_text()) if source.is_file() else default


def run_complete(run_dir: str | Path, precisions: tuple[int, ...]) -> bool:
    root = Path(run_dir)
    status = read_json(root / "status.json", {})
    if status.get("state") in {"screened_out", "no_learning"}:
        return True
    if status.get("state") != "complete" or not (root / "metrics.json").is_file():
        return False
    return all(
        (root / "codec_metrics" / f"b{bits}.json").is_file()
        and (root / "codecs" / f"adapter_b{bits}.fqcb").is_file()
        and (root / "predictions" / f"task_b{bits}.jsonl").is_file()
        for bits in precisions
    )


@contextmanager
def claim_run(run_dir: str | Path) -> Iterator[bool]:
    """Claim one shared run until this process finishes or exits."""
    lock_path = Path(run_dir) / ".lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as stream:
        try:
            flock(stream.fileno(), LOCK_EX | LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            flock(stream.fileno(), LOCK_UN)
