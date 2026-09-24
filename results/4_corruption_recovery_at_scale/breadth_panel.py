"""Breadth panel: does the compressed corrupted adapter decay slower than the clean one?

Usage: python breadth_panel.py [STUDY CONFIG COMPRESSED_KEY PREFIX]
(default: spectral_breadth_v1 configs/spectral_breadth.yaml band00_01_binary breadth_panel)
Reads the joined per-row predictions of the study. Every probe is
split in half by row order. Whether the clean adapter hurts a probe is decided
on the first half; the clean-versus-compressed comparison uses the second half,
so the selection cannot manufacture the result by regression to the mean.

Distance tiers are averaged family by family, so the 57 MMLU subjects count as a
handful of families, not as 57 votes. The 78 questions MMLU clinical_knowledge
and college_medicine share are scored once, under clinical_knowledge.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fineqcomp.config import load_campaign  # noqa: E402

NAME, CONFIG, COMPRESSED, PREFIX = (sys.argv[1:5] if len(sys.argv) > 4 else
    ("spectral_breadth_v1", "configs/spectral_breadth.yaml", "band00_01_binary", "breadth_panel"))
STUDY = ROOT / ".cache/reports" / NAME
FIGURE_TAG = "" if PREFIX == "breadth_panel" else "b"
_bits = COMPRESSED.split("_")
_low, _high = int(_bits[0][4:]), int(_bits[1])
COMPRESSED_LABEL = ("corrupted, " + (f"direction {_low}" if _high == _low + 1 else f"directions {_low}-{_high - 1}")
                    + f" at {'1 bit' if _bits[2] == 'binary' else _bits[2]}")
TIER_LABELS = (["GSM8K", "math\nword", "hard\nmath", "STEM", "common\nsense", "code,\nhumanities"]
               if PREFIX == "breadth_panel" else
               ["MATH-500", "hard/MC\nmath", "math\nword", "STEM", "common\nsense", "humanities"])
HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
ARMS = {"base": "base", "clean": "clean_raw", "corrupted": "permuted_raw",
        "compressed": COMPRESSED}
DRAWS = 2000
# A probe counts as hurt by the clean adapter when its first half loses at
# least this fraction of base accuracy; a plain "below base" admits noise.
HURT = 0.05


def rows(probe: str, key: str) -> list[dict]:
    return [json.loads(line) for line in (STUDY / "joined" / probe / f"{key}.jsonl").open()]


def prompts(probe: str) -> dict[str, str]:
    return {r["example_id"]: r["prompt"] for r in map(json.loads, (STUDY / "data" / f"{probe}.jsonl").open())}


def load() -> tuple[dict, dict, dict]:
    cfg = load_campaign(ROOT / CONFIG)["spectral_transfer"]
    specs = {cfg["primary"]["key"]: cfg["primary"], **cfg["transfer"]}
    shared = set(prompts("mmlu_clinical_knowledge").values())
    medicine = prompts("mmlu_college_medicine")
    scores, no_answer = {}, {}
    for probe in list(specs):
        per_arm = {arm: rows(probe, key) for arm, key in ARMS.items()}
        # A probe the frozen model scores zero on cannot be read as a fraction
        # of base (Gemma-it's HumanEval: the harness does not strip its fences).
        if not any(r["correct"] for r in per_arm["base"]):
            del specs[probe]
            continue
        ids = [r["example_id"] for r in per_arm["base"]]
        if any([r["example_id"] for r in v] != ids for v in per_arm.values()):
            raise ValueError(f"{probe}: arms scored different rows")
        keep = [i for i, x in enumerate(ids)
                if not (probe == "mmlu_college_medicine" and medicine[x] in shared)]
        scores[probe] = {arm: [float(v[i]["correct"]) for i in keep] for arm, v in per_arm.items()}
        # Rows with no extracted answer at all: the format failure, not a wrong one.
        no_answer[probe] = {arm: mean([float(v[i].get("prediction") in (None, "")) for i in keep])
                            for arm, v in per_arm.items()}
    return specs, scores, no_answer


def mean(values):
    return sum(values) / len(values)


def ratio_ci(scores: dict, arm: str, rng: random.Random) -> tuple[float, float, float]:
    """Accuracy as a fraction of base, with a paired row bootstrap."""
    a, b = scores[arm], scores["base"]
    point = mean(a) / mean(b)
    n, draws = len(a), []
    for _ in range(DRAWS):
        idx = [rng.randrange(n) for _ in range(n)]
        base = sum(b[i] for i in idx)
        if base:
            draws.append(sum(a[i] for i in idx) / base)
    draws.sort()
    return point, draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]


def halves(scores: dict) -> tuple[dict, dict]:
    n = len(scores["base"]) // 2
    return ({a: v[:n] for a, v in scores.items()}, {a: v[n:] for a, v in scores.items()})


def tier_curve(specs, scores, arm, rng):
    """Family-weighted fraction of base per distance tier, bootstrapped over rows."""
    tiers = defaultdict(lambda: defaultdict(list))
    for probe, spec in specs.items():
        tiers[spec["distance"]][spec["family"]].append(probe)

    def value(sample):
        out = {}
        for tier, families in tiers.items():
            fams = [mean([mean(sample[p][arm]) / max(mean(sample[p]["base"]), 1e-9) for p in probes])
                    for probes in families.values()]
            out[tier] = mean(fams)
        return out

    point = value(scores)
    draws = defaultdict(list)
    for _ in range(300):
        sample = {}
        for p, s in scores.items():
            idx = [rng.randrange(len(s["base"])) for _ in s["base"]]
            sample[p] = {a: [s[a][i] for i in idx] for a in (arm, "base")}
        for tier, v in value(sample).items():
            draws[tier].append(v)
    return {t: (point[t], sorted(d)[7], sorted(d)[292]) for t, d in draws.items()}


def main() -> None:
    rng = random.Random(0)
    specs, scores, no_answer = load()
    table = []
    for probe, spec in specs.items():
        first, second = halves(scores[probe])
        row = {"probe": probe, "family": spec["family"], "distance": spec["distance"],
               "rows": len(scores[probe]["base"]),
               "clean_hurts_first_half": mean(first["clean"]) < (1 - HURT) * mean(first["base"]),
               "clean_hurts_second_half": mean(second["clean"]) < (1 - HURT) * mean(second["base"]),
               "clean_no_answer": no_answer[probe]["clean"],
               "compressed_no_answer": no_answer[probe]["compressed"]}
        for arm in ARMS:
            row[f"{arm}_accuracy"] = mean(scores[probe][arm])
            row[f"{arm}_second_half"] = mean(second[arm])
        for arm in ("clean", "compressed", "corrupted"):
            row[f"{arm}_frac_base_second_half"], row[f"{arm}_lo"], row[f"{arm}_hi"] = ratio_ci(second, arm, rng)
        table.append(row)
    table.sort(key=lambda r: (r["distance"], r["probe"]))
    with (HERE / f"{PREFIX}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    hurt = [r for r in table if r["clean_hurts_first_half"] and r["distance"] > 0]
    replicated = sum(r["clean_hurts_second_half"] for r in hurt)
    wins = sum(r["compressed_second_half"] > r["clean_second_half"] for r in hurt)
    # Pooled over hurt probes on second halves, family-agnostic, paired by probe.
    gap = [r["compressed_second_half"] - r["clean_second_half"] for r in hurt]
    boot = sorted(mean([gap[rng.randrange(len(gap))] for _ in gap]) for _ in range(DRAWS))
    curves = {arm: tier_curve(specs, scores, arm, rng) for arm in ("clean", "compressed", "corrupted")}
    summary = {"probes": len(table), "clean_hurts_first_half": len(hurt),
               "of_which_still_hurt_on_second_half": replicated,
               "compressed_beats_clean_on_second_half": wins,
               "mean_gap_compressed_minus_clean": mean(gap),
               "gap_ci95_over_probes": [boot[int(0.025 * DRAWS)], boot[int(0.975 * DRAWS)]],
               "tiers": {arm: {str(t): v for t, v in sorted(c.items())} for arm, c in curves.items()}}
    (HERE / f"{PREFIX}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    FIGURES.mkdir(exist_ok=True)
    colours = {"clean": "#4C72B0", "compressed": "#DD8452", "corrupted": "#9A9A9A"}
    names = {"clean": "clean adapter", "compressed": COMPRESSED_LABEL,
             "corrupted": "corrupted adapter"}

    home = next(r for r in table if r["distance"] == 0)
    bars = [home] + hurt
    fig, ax = plt.subplots(figsize=(5.5, 0.26 * len(bars) + 1.2))
    for offset, arm in ((0.2, "clean"), (-0.2, "compressed")):
        ys = [-i + offset for i in range(len(bars))]
        xs = [r[f"{arm}_frac_base_second_half"] for r in bars]
        err = [[x - r[f"{arm}_lo"] for x, r in zip(xs, bars)], [r[f"{arm}_hi"] - x for x, r in zip(xs, bars)]]
        ax.barh(ys, xs, 0.4, color=colours[arm], label=names[arm], xerr=err, capsize=1.2,
                error_kw={"lw": 0.5})
    ax.axvline(1, color="black", lw=0.8)
    ax.axhline(-0.5, color="black", lw=0.5, ls=":")
    ax.set_yticks([-i for i in range(len(bars))])
    ax.set_yticklabels([r["probe"].replace("mmlu_", "").replace("_", " ") + f"  [{r['distance']}]"
                        for r in bars], fontsize=7)
    ax.set_xlabel("accuracy / frozen base, held-out half")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / f"B1{FIGURE_TAG}_where_clean_hurts.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for arm in ("corrupted", "clean", "compressed"):
        tiers = sorted(curves[arm])
        ys = [curves[arm][t][0] for t in tiers]
        ax.fill_between(tiers, [curves[arm][t][1] for t in tiers], [curves[arm][t][2] for t in tiers],
                        color=colours[arm], alpha=0.18, lw=0)
        ax.plot(tiers, ys, "o-", color=colours[arm], label=names[arm], lw=1.6, ms=4)
    ax.axhline(1, color="black", lw=0.8)
    ax.set_xticks(range(6))
    ax.set_xticklabels(TIER_LABELS, fontsize=8)
    ax.set_xlabel("distance from the training task")
    ax.set_ylabel("accuracy as a fraction of the frozen base")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"B2{FIGURE_TAG}_decay_with_distance.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
