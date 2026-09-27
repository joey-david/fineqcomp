"""Rebuild paper plots and rate-prediction scores from tracked measurements."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from analysis.adapter_spectrum_score import (
    attenuation_report,
    audit_attenuation_report,
    independent_attenuation_report,
)


def check_recorded(actual, expected, path="scores"):
    """Reject changed scores, allowing small floating-point roundoff."""
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"{path}: fields differ from the recorded result")
        for key in expected:
            check_recorded(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{path}: row count differs from the recorded result")
        for index, (a, e) in enumerate(zip(actual, expected)):
            check_recorded(a, e, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{path}: {actual} differs from recorded {expected}")
    elif actual != expected:
        raise ValueError(f"{path}: {actual!r} differs from recorded {expected!r}")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root / ".cache/reproduction",
                        help="output directory (default: .cache/reproduction)")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary_path.unlink(missing_ok=True)
    source = root / "results/rate/adapter_spectrum/attenuation_predictor"
    print("Checking rate-prediction scores...", flush=True)
    scores = {}
    for name, directory in (("original", source), ("replication", source / "replication")):
        scores[name] = attenuation_report(
            directory / "profiles.csv", directory / "targets.csv", out / "results" / name)
        check_recorded(scores[name], json.loads((directory / "scores.json").read_text()), name)
    scores["audit"] = audit_attenuation_report(
        source / "audit/profiles.csv", source / "audit/targets.csv", out / "results/audit")
    check_recorded(scores["audit"], json.loads((source / "audit/scores.json").read_text()), "audit")
    # This check used the original seed-11 predictions, not the later replication.
    scores["independent"] = independent_attenuation_report(
        out / "results/original/predictions.csv", source / "independent/validation.csv",
        out / "results/independent")
    check_recorded(scores["independent"],
                   json.loads((source / "independent/scores.json").read_text()), "independent")
    print("Rebuilding figures and the breadth table...", flush=True)
    log_path = out / "figures.log"
    with log_path.open("w") as log:
        built = subprocess.run([sys.executable, str(root / "paper/plots/make_figures.py"),
                                "--out", str(out), "--preview", str(out / "previews")],
                               cwd=root, stdout=log, stderr=log)
    if built.returncode:
        raise SystemExit(f"Figure build failed; see {log_path}")
    preserved = ["F4_layer_location.pdf", "information_intro.tex"]
    for name in preserved:
        shutil.copy2(root / "paper/manuscript/figures" / name, out / "figures" / name)
    summary = {
        "scope": "Plots, breadth table, and four predictor score reports from saved measurements; "
                 "no training, model evaluation, or final manuscript build.",
        "scores_match_recorded": True,
        "preserved_without_rebuilding": preserved,
        "results": scores,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Reproduced plots and table; all four score reports match the recorded results.\n"
          f"Outputs: {out}\n"
          "Layer-location PDF and intro diagram source copied; their renders were not rebuilt.")


if __name__ == "__main__":
    main()
