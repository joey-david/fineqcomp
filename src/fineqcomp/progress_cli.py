"""CLI entrypoint with fineQComp job progress enabled."""

from __future__ import annotations

from fineqcomp.cli import main as _main
from fineqcomp.progress import install_progress


def main(argv: list[str] | None = None) -> None:
    install_progress()
    _main(argv)
