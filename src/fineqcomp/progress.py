"""Job-level tqdm progress reporting for campaign execution."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from tqdm.auto import tqdm

import fineqcomp.runner as runner

_ORIGINAL_RUN_MANY = runner.RunEngine.run_many
_INSTALLED = False


def _run_many_with_progress(
    self: runner.RunEngine,
    runs: list[runner.RunSpec],
    force: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    selected = runs[:limit]
    original_claim_run = runner.claim_run

    with tqdm(
        total=len(selected),
        desc="fineQComp jobs",
        unit="job",
        dynamic_ncols=True,
    ) as progress:

        @contextmanager
        def claim_run_with_progress(path: object) -> Iterator[bool]:
            try:
                with original_claim_run(path) as claimed:
                    yield claimed
            finally:
                progress.update(1)

        runner.claim_run = claim_run_with_progress
        try:
            return _ORIGINAL_RUN_MANY(self, selected, force=force, limit=None)
        finally:
            runner.claim_run = original_claim_run


def install_progress() -> None:
    """Install job-level progress reporting once for CLI execution."""
    global _INSTALLED
    if _INSTALLED:
        return
    runner.RunEngine.run_many = _run_many_with_progress
    _INSTALLED = True
