#!/usr/bin/env python3
"""Is repair gated by headroom? Join the ladder and test it two ways.

The across-model test is the obvious one: bigger model, more repair. It is also
the weak one, because model size covaries with everything.

The within-model test is the real claim and needs no scaling assumption at all.
Inside a single model, split the items by *that model's own* solo pass rate and
ask whether the damage done by corrupting a depended-on value shrinks on the
items it finds easy. If repair is gated by headroom, the curve falls with
competence inside every model on the ladder, and the story does not rest on
comparing a 3B to a 27B.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any

CONDITIONS = ("solo", "clean", "corrupt_relevant", "corrupt_distract")


def load(root: Path) -> list[dict[str, Any]]:
    rows = []
    for rows_file in sorted(root.glob("*/rows.json")):
        model = rows_file.parent.name.replace("__", "/")
        for row in json.loads(rows_file.read_text()):
            rows.append({**row, "model": model})
    return rows


def competence(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    """Each item's solo pass rate for one model: its competence on that item."""
    buckets: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for row in rows:
        if row["condition"] == "solo":
            buckets[(row["model"], row["id"])].append(int(row["correct"]))
    return {key: statistics.mean(values) for key, values in buckets.items()}


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by[(row["model"], row["dataset"], row["condition"])].append(row)
    out = []
    models = sorted({r["model"] for r in rows})
    for model in models:
        for dataset in sorted({r["dataset"] for r in rows}):
            cell: dict[str, Any] = {"model": model, "dataset": dataset}
            for condition in CONDITIONS:
                group = by.get((model, dataset, condition), [])
                if not group:
                    continue
                cell[f"{condition}_acc"] = statistics.mean(r["correct"] for r in group)
                cell[f"{condition}_flag"] = statistics.mean(r["flagged"] for r in group)
                cell[f"{condition}_n"] = len(group)
            if "clean_acc" in cell and "corrupt_relevant_acc" in cell:
                cell["relevant_damage"] = cell["clean_acc"] - cell["corrupt_relevant_acc"]
                cell["distract_damage"] = cell["clean_acc"] - cell["corrupt_distract_acc"]
                # Flagging on clean work is the false-positive rate. Detection is
                # only what exceeds it -- a model that cries error on everything
                # has detected nothing, and the first prompt made one do exactly that.
                cell["detection_above_fp"] = (cell["corrupt_relevant_flag"]
                                              - cell["clean_flag"])
            out.append(cell)
    return out


def paired_repair(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    """Repair rate among the items the model actually solves with clean work.

    Absolute damage (clean minus corrupt) is floor-limited: an item the model
    fails either way contributes zero damage and reads as if the corruption did
    no harm, which makes low-competence bins look falsely healthy. Conditioning
    on the clean condition being correct removes that floor -- the denominator
    is only items the model demonstrably can do when the supplied work is
    sound, so what remains is whether the corruption breaks it.
    """
    outcome: dict[tuple[str, str, str], bool] = {}
    flags: dict[tuple[str, str, str], bool] = {}
    for row in rows:
        if row["condition"] in ("clean", "corrupt_relevant", "corrupt_distract"):
            outcome[(row["model"], row["id"], row["condition"])] = bool(row["correct"])
            flags[(row["model"], row["id"], row["condition"])] = bool(row["flagged"])
    out: dict[tuple[str, str], dict[str, float]] = {}
    keys = {(m, i) for (m, i, c) in outcome}
    per_model: dict[str, list[tuple[bool, bool, bool, bool]]] = collections.defaultdict(list)
    for model, item in keys:
        clean = outcome.get((model, item, "clean"))
        relevant = outcome.get((model, item, "corrupt_relevant"))
        distract = outcome.get((model, item, "corrupt_distract"))
        if clean is None or relevant is None or distract is None:
            continue
        per_model[model].append((clean, relevant, distract,
                                 flags.get((model, item, "corrupt_relevant"), False)))
    for model, values in per_model.items():
        solvable = [v for v in values if v[0]]
        if not solvable:
            continue
        out[(model, "all")] = {
            "solvable_items": len(solvable),
            "repair_rate": statistics.mean(v[1] for v in solvable),
            "distract_survival": statistics.mean(v[2] for v in solvable),
            "flag_when_broken": statistics.mean(
                v[3] for v in solvable if not v[1]) if any(not v[1] for v in solvable) else 0.0,
        }
    return out


def headroom_curve(rows: list[dict[str, Any]], comp: dict[tuple[str, str], float],
                   edges=(0.0, 0.25, 0.5, 0.75, 1.0)) -> list[dict[str, Any]]:
    """Damage from a relevant corruption, binned by the model's own competence."""
    out = []
    for model in sorted({r["model"] for r in rows}):
        for low, high in zip(edges, edges[1:]):
            # The top bin is closed so pass rate 1.0 -- the saturated items, the
            # ones the hypothesis is really about -- is not silently dropped.
            def in_bin(item_id: str) -> bool:
                value = comp.get((model, item_id))
                if value is None:
                    return False
                return low <= value < high or (high == edges[-1] and value == high)

            picked = {c: [r for r in rows
                          if r["model"] == model and r["condition"] == c
                          and in_bin(r["id"])]
                      for c in ("clean", "corrupt_relevant", "corrupt_distract")}
            if not picked["clean"] or len(picked["clean"]) < 5:
                continue
            clean = statistics.mean(r["correct"] for r in picked["clean"])
            relevant = statistics.mean(r["correct"] for r in picked["corrupt_relevant"])
            distract = statistics.mean(r["correct"] for r in picked["corrupt_distract"])
            # Paired within the bin, for the same floor reason as above.
            solved = {r["id"] for r in picked["clean"] if r["correct"]}
            kept = [r for r in picked["corrupt_relevant"] if r["id"] in solved]
            repair = statistics.mean(r["correct"] for r in kept) if kept else None
            out.append({
                "model": model, "bin_low": low, "bin_high": high,
                "items": len(picked["clean"]),
                "clean_acc": clean, "corrupt_relevant_acc": relevant,
                "relevant_damage": clean - relevant,
                "distract_damage": clean - distract,
                "detection_above_fp": (
                    statistics.mean(r["flagged"] for r in picked["corrupt_relevant"])
                    - statistics.mean(r["flagged"] for r in picked["clean"])),
                "solvable_items": len(kept),
                "repair_rate": repair,
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("reports/repair_headroom"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    rows = load(args.root)
    if not rows:
        raise SystemExit(f"no rows.json under {args.root}")
    comp = competence(rows)
    summary = {
        "models": sorted({r["model"] for r in rows}),
        "rows": len(rows),
        "cells": summarise(rows),
        "headroom_curve": headroom_curve(rows, comp),
        "paired_repair": {f"{m}": v for (m, _), v in paired_repair(rows).items()},
    }
    target = args.out or (args.root / "summary.json")
    target.write_text(json.dumps(summary, indent=1))

    print(f"{'model':28}{'data':14}{'solo':>6}{'clean':>7}{'rel':>7}{'dist':>7}"
          f"{'relDmg':>8}{'distDmg':>8}{'det-fp':>8}")
    for cell in summary["cells"]:
        if "relevant_damage" not in cell:
            continue
        print(f"{cell['model'][:27]:28}{cell['dataset']:14}"
              f"{cell['solo_acc']:6.2f}{cell['clean_acc']:7.2f}"
              f"{cell['corrupt_relevant_acc']:7.2f}{cell['corrupt_distract_acc']:7.2f}"
              f"{cell['relevant_damage']:8.2f}{cell['distract_damage']:8.2f}"
              f"{cell['detection_above_fp']:8.2f}")
    print(f"\npaired repair rate (items the model solves with clean work)")
    print(f"{'model':28}{'n':>6}{'repair':>8}{'distSurv':>10}{'flagBroken':>12}")
    for model, value in sorted(summary["paired_repair"].items()):
        print(f"{model[:27]:28}{value['solvable_items']:6.0f}{value['repair_rate']:8.2f}"
              f"{value['distract_survival']:10.2f}{value['flag_when_broken']:12.2f}")
    print(f"\nwithin-model headroom curve (damage should fall as competence rises)")
    print(f"{'model':28}{'bin':>12}{'n':>5}{'clean':>7}{'repair':>8}{'det-fp':>8}")
    for point in summary["headroom_curve"]:
        label = f"{point['bin_low']:.2f}-{point['bin_high']:.2f}"
        repair = point["repair_rate"]
        shown = f"{repair:8.2f}" if repair is not None else f"{'--':>8}"
        print(f"{point['model'][:27]:28}{label:>12}{point['items']:5d}"
              f"{point['clean_acc']:7.2f}{shown}{point['detection_above_fp']:8.2f}")
    print(f"\nwrote {target}")


if __name__ == "__main__":
    main()
