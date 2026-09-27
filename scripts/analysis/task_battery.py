"""Which tasks show the compression recovery? One row per arm of the battery.

Reads a `fineqcomp.studies.generalisation` sweep directory (<OUT>/<arm>/seed11/<condition>.jsonl)
and, for each arm, compares every compressed condition with the frozen base on
the same test rows (paired bootstrap, 95%). An arm is a *hit* when the adapter as
trained is below base and at least one compressed condition is above base; a
*repair* when some compression beats the adapter as trained but not base.

    python scripts/analysis/task_battery.py [--prepared ROOT] [--tex PATH] OUT [OUT ...]

Besides the post-hoc best of five, each arm is scored with a fixed rule: the
better of the two filters the paper uses, direction 0 at 1 bit (`rank1_b1`) and
directions 4-15 at full precision (`tail_b16`). `--tex` writes the rows of
tab:task-battery: every corrupted arm where that rule beats the frozen base,
then the three that come closest.

Summaries predate per-row ROUGE in their prediction files; for those the score
is recomputed against the references in ROOT/natural/<dataset>/seed11/test.jsonl.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "results/tasks"
COMPRESSED = ["rank1_b1", "rank1_b16", "scale_head_0p5", "norm_matched_b1", "tail_b16"]
FIXED = ["rank1_b1", "tail_b16"]
NEAR = 3  # arms below base shown for contrast
DRAWS = 2000


def correct(path: Path, references: dict[str, str]) -> list[float]:
    rows = [json.loads(line) for line in path.open()]
    if "correct" in rows[0]:
        return [float(r["correct"]) for r in rows]
    if "rouge_l" in rows[0]:
        return [float(r["rouge_l"]) for r in rows]
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(references[r["example_id"]].strip(), r["response"].strip())["rougeL"].fmeasure
            for r in rows]


def references(prepared: Path | None, dataset: str) -> dict[str, str]:
    path = prepared / "natural" / dataset / "seed11" / "test.jsonl" if prepared else None
    if path is None or not path.is_file():
        return {}
    return {row["example_id"]: row["response"] for row in map(json.loads, path.open())}


def interval(a: list[float], b: list[float], rng: random.Random) -> tuple[float, float, float]:
    n = len(a)
    diffs = sorted(sum(a[i] - b[i] for i in idx) / n
                   for idx in ([rng.randrange(n) for _ in range(n)] for _ in range(DRAWS)))
    return sum(x - y for x, y in zip(a, b)) / n, diffs[int(0.025 * DRAWS)], diffs[int(0.975 * DRAWS)]


def main() -> None:
    rng = random.Random(0)
    args, prepared, tex = sys.argv[1:], None, None
    if args[:1] == ["--prepared"]:
        prepared, args = Path(args[1]), args[2:]
    if args[:1] == ["--tex"]:
        tex, args = Path(args[1]), args[2:]
    table = []
    for out in map(Path, args):
        datasets = {cell["arm"]: cell["dataset"] for cell in json.load(open(out / "lock.json"))["cells"]}
        for root in sorted(out.glob("*/seed11")):
            if not (root / "complete.json").exists():
                continue
            arm = root.parent.name
            refs = references(prepared, datasets[arm])
            first = json.loads((root / "base.jsonl").open().readline())
            if not refs and "correct" not in first and "rouge_l" not in first:
                print(f"skip {arm}: no per-row scores and no references", file=sys.stderr)
                continue
            scores = {key: correct(root / f"{key}.jsonl", refs) for key in ["base", "raw", *COMPRESSED]}
            mean = {key: sum(v) / len(v) for key, v in scores.items()}
            raw_minus_base = interval(scores["raw"], scores["base"], rng)
            best = max(COMPRESSED, key=lambda key: mean[key])
            best_minus_base = interval(scores[best], scores["base"], rng)
            best_minus_raw = interval(scores[best], scores["raw"], rng)
            fixed = max(FIXED, key=lambda key: mean[key])
            fixed_minus_base = interval(scores[fixed], scores["base"], rng)
            fixed_minus_raw = interval(scores[fixed], scores["raw"], rng)
            damaged = raw_minus_base[2] < 0
            verdict = ("hit" if damaged and best_minus_base[1] > 0 else
                       "repair" if best_minus_raw[1] > 0 else "none")
            table.append({"study": out.name, "arm": arm, "verdict": verdict,
                          **{key: round(mean[key], 4) for key in ["base", "raw", *COMPRESSED]},
                          "best": best,
                          "raw_minus_base": [round(x, 4) for x in raw_minus_base],
                          "best_minus_base": [round(x, 4) for x in best_minus_base],
                          "best_minus_raw": [round(x, 4) for x in best_minus_raw],
                          "fixed": fixed,
                          "fixed_minus_base": [round(x, 4) for x in fixed_minus_base],
                          "fixed_minus_raw": [round(x, 4) for x in fixed_minus_raw]})
    with (HERE / "arms.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    if tex is not None:
        write_tex(table, tex)
    for row in table:
        print(f"{row['verdict']:6s} {row['arm']:22s} base {row['base']:.3f} raw {row['raw']:.3f} "
              f"best {row['best']:15s} {row[row['best']]:.3f}  best-base {row['best_minus_base']}")


TASKS = {"agnews": "AG News", "csqa": "CommonsenseQA", "hellaswag": "HellaSwag", "paws": "PAWS",
         "race": "RACE", "sst2": "SST-2", "winogrande": "WinoGrande", "sum": "XSum", "sql": "text-to-SQL",
         "xbrl": "XBRL"}
CORRUPTIONS = {"flipped": "all labels wrong", "permuted": "class $i\\to i{+}1$",
               "shortcut": "length labels", "noisy": "40\\% labels wrong", "7b": "mismatched",
               "14b": "mismatched", "32b": "mismatched"}


def write_tex(table: list[dict], path: Path) -> None:
    """Rows for tab:task-battery: arms where the fixed rule beats base, then the closest misses."""
    headroom = {row["arm"]: row for row in csv.DictReader((HERE / "headroom_grid.csv").open())}
    rows = {row["arm"]: row for row in table}
    for arm, row in headroom.items():
        if arm not in rows and row["best"] in FIXED:  # summaries: taken as scored when run
            rows[arm] = {**row, **{k: float(row[k]) for k in ["base", "raw", *FIXED]}, "fixed": row["best"],
                         "fixed_minus_base": json.loads(row["best_minus_base"]),
                         "fixed_minus_raw": json.loads(row["best_minus_raw"])}
    def order(arm):
        task, _, rest = arm.partition("_")
        return (task in {"sum", "sql", "xbrl"}, list(TASKS).index(task),
                int(rest[:-1]) if rest.endswith("b") and rest[:-1].isdigit() else 0, rest)
    # The recovery effect is compression beating the adapter as trained; on a
    # clean arm any gain over base is the fine-tune itself. Of those arms, keep
    # every one above base and the NEAR closest below it.
    repaired = [arm for arm, r in rows.items()
                if not arm.endswith("_clean") and r["fixed_minus_raw"][1] > 0]
    above = [arm for arm in repaired if rows[arm]["fixed_minus_base"][1] > 0]
    near = sorted((arm for arm in repaired if arm not in above),
                  key=lambda arm: -rows[arm]["fixed_minus_base"][0])[:NEAR]
    lines = []
    for arm in sorted(above, key=order) + near:
        r = rows[arm]
        task, _, rest = arm.partition("_")
        if arm == near[0]:
            lines.append("\\midrule")
        size = f", {rest.upper()}" if task in {"sum", "sql", "xbrl"} else ""
        d, lo, hi = (f"{100 * x:+.1f}".replace("-", "$-$") for x in r["fixed_minus_base"])
        delta = f"\\textbf{{{d}}}" if r["fixed_minus_base"][1] > 0 else d
        cells = [TASKS[task] + size, CORRUPTIONS.get(rest, rest),
                 *(f"{100 * r[k]:.1f}" for k in ["base", "raw", *FIXED]), f"{delta} [{lo}, {hi}]"]
        lines.append(" & ".join(cells) + " \\\\")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
