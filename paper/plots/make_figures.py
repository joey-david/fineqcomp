"""Rebuild the available paper figures and breadth table from saved data."""
import argparse
import sys
from pathlib import Path

import style

style.apply_style()
import main_figures  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=style.ROOT / "paper/manuscript",
                   help="write figures/ and tables/ under this directory")
    p.add_argument("--preview")
    p.add_argument("only", nargs="*", help="figure function names to run, e.g. f2_frontier")
    args = p.parse_args()
    style.OUT = args.out / "figures"
    style.PREVIEW = args.preview
    figs = list(main_figures.ALL)
    import appendix_figures
    import directions
    appendix_figures.TABLES = args.out / "tables"
    figs += appendix_figures.ALL + [directions.direction_windows]
    unknown = set(args.only) - {f.__name__ for f in figs}
    if unknown:
        p.error(f"unknown figure names: {sorted(unknown)}")
    for f in figs:
        if not args.only or f.__name__ in args.only:
            f()
            print(f.__name__, file=sys.stderr)
