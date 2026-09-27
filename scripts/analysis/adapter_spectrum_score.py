"""Score adapter features as predictors of R* on the transposed receiver panel.

Usage: python score.py [--panel DIR --map RUN_MAP_CSV] [FEATURE_CSV ...]
Each feature CSV is keyed by run_id; the map gives each run's model and corpus
(defaults: the transposed panel and its own cells.csv). An arm is predicted
only if its corpus also appears on another receiver, since the corpus term
cannot be fitted otherwise.

Same protocol as transposed_receiver_panel/model_scores.csv: for each receiver
in turn, fit R* ~ corpus + feature on the other receivers' arms and predict the
held-out receiver's arms. Reported: RMSE and MAE over held-out arms, and
receiver-pair sign accuracy -- within a corpus, how often the prediction orders
two receivers' R* the right way. `corpus_only` and the recorded candidates are
re-scored here first, so the protocol is checked against the recorded table.
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2] / "results/rate/adapter_spectrum"
PANEL = HERE.parent / "transposed_receiver_panel"


FEATURES = ("spectrum_", "functional_", "gain_")


def attenuation_estimate(row: dict, target: float = 0.90) -> float:
    """Map the raw update's gain-retaining scale to an expected one-bit pair rate."""
    from fineqcomp.rstar import r_star

    base = float(row["attenuation_loss_0"])
    gain = base - float(row["attenuation_loss_1"])
    projection = float(row["attenuation_binary_projection"])
    if gain <= 0 or projection <= 0:
        raise ValueError("attenuation prediction needs positive raw gain and projection")
    points = [{"scale": scale, "gain": base - float(row[f"attenuation_loss_{scale:g}"])}
              for scale in (0, 0.125, 0.25, 0.5, 1)]
    crossing = r_star(points, target=target, rate_key="scale", value_key="gain", reference=gain)["r_star"]
    return crossing / projection


def attenuation_checks(predictions: Path, runs_root: Path, out: Path) -> dict:
    """Check other retention levels, full sub-bit curves, and file selection."""
    from fineqcomp.rstar import r_star

    levels, curves, selections = [], [], []
    for row in csv.DictReader(predictions.open()):
        record = json.loads((runs_root / row["run_id"] / "metrics.json").read_text())
        raw = record["raw_behavioral_write"]["heldout_bits_saved_per_token"]
        points = [{"effective_bits_per_value": c["storage"]["effective_bits_per_value"],
                   "heldout_bits_saved_per_token": c["behavioral_write"]["heldout_bits_saved_per_token"]}
                  for c in record["codecs"]]
        ceiling = max([raw] + [p["heldout_bits_saved_per_token"] for p in points])
        selected = next(p for p in sorted(points, key=lambda p: p["effective_bits_per_value"])
                        if p["effective_bits_per_value"] >= float(row["direct"]))
        selections.append({"run_id": row["run_id"], "split": row["split"],
                           "selected_rate": selected["effective_bits_per_value"],
                           "retention": selected["heldout_bits_saved_per_token"] / ceiling,
                           "oracle_rate": float(row["first_feasible_rate"]),
                           "oracle_ratio": selected["effective_bits_per_value"] / float(row["first_feasible_rate"])})
        if row["split"] != "heldout":
            continue
        for target in (0.5, 0.75, 0.9, 0.95):
            predicted = attenuation_estimate(row, target)
            for mode, reference in (("raw", raw), ("best_decoded", ceiling)):
                observed = r_star(points, target=target, reference=reference)["r_star"]
                levels.append({"run_id": row["run_id"], "retention": target, "reference": mode,
                               "predicted": predicted, "observed": observed})
        scales = (0, 0.125, 0.25, 0.5, 1)
        losses = np.array([float(row[f"attenuation_loss_{s:g}"]) for s in scales])
        gains = (losses[0] - losses) / (losses[0] - losses[-1])
        gamma = float(row["attenuation_binary_projection"])
        for codec in record["codecs"]:
            if codec["bits"] > 1 or (codec["bits"] == 1 and codec["blend"] > 0):
                continue
            fraction = codec["blend"] if codec["bits"] == 0 else 1.0
            curves.append({"run_id": row["run_id"], "codec": codec["codec_key"], "fraction": fraction,
                           "predicted_retention": float(np.interp(fraction * gamma, scales, gains)),
                           "observed_retention": codec["behavioral_write"]["heldout_bits_saved_per_token"] / raw})
    summary = []
    for target in (0.5, 0.75, 0.9, 0.95):
        for mode in ("raw", "best_decoded"):
            selected = [r for r in levels if r["retention"] == target and r["reference"] == mode]
            summary.append({"retention": target, "reference": mode, "n": len(selected),
                            "rmse": float(np.sqrt(np.mean([(r["predicted"] - r["observed"]) ** 2 for r in selected]))),
                            "predicted_above_one_bit": sum(r["predicted"] > 1 for r in selected)})
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in (("retention_levels", {"scores": summary, "predictions": levels}),
                          ("curve_predictions", curves), ("rung_selection", selections)):
        (out / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    return {"retention_levels": summary, "curve_points": len(curves), "selected_files": len(selections)}


def attenuation_report(profiles: Path, targets: Path, out: Path) -> dict:
    """Score the fixed formula; fit any calibration on development receivers only."""
    features = {r["run_id"]: r for r in csv.DictReader(profiles.open())}
    rows = [{**r, **features[r["run_id"]]} for r in csv.DictReader(targets.open())]
    if any(r["attenuation_split"] != "calibration" for r in rows):
        raise ValueError("the frozen comparison uses calibration profiles")
    for row in rows:
        row["direct"] = attenuation_estimate(row)
    dev = [r for r in rows if r["split"] == "development"]
    coef = np.linalg.lstsq([[1, r["direct"]] for r in dev],
                          [float(r["r_star"]) for r in dev], rcond=None)[0]
    fixed_projection = np.mean([float(r["attenuation_binary_projection"]) for r in dev])
    for row in rows:
        row["calibrated"] = float(coef @ [1, row["direct"]])
        alpha = row["direct"] * float(row["attenuation_binary_projection"])
        row["fixed_projection"] = float(alpha / fixed_projection)
        row["gaussian_projection"] = float(alpha / (4 / np.pi ** 2))

    # Give the old baselines the full audit panel, not only six development cells.
    audit = load([HERE / "adapter_spectrum_archive.csv"],
                 HERE.parent / "rate_law_audit", HERE / "run_map.csv")
    panel = {(r["model_key"], r["dataset_key"]): r for r in
             load([HERE / "adapter_spectrum_transposed.csv"])}
    baselines = {"constant": [], "text_redundancy": ["text_cross_row_redundancy"],
                 "spectrum_tokens": ["spectrum_top1", "log_corpus_tokens"]}
    for name, columns in baselines.items():
        usable = [r for r in audit if all(r.get(c) not in (None, "") for c in columns)]
        design = lambda r: [1] + [float(r[c]) for c in columns]
        fitted = np.linalg.lstsq([design(r) for r in usable],
                                [float(r["r_star_bits_per_value"]) for r in usable], rcond=None)[0]
        for row in rows:
            row[name] = float(fitted @ design(panel[row["model_key"], row["dataset_key"]]))
    summaries = []
    for split in sorted({r["split"] for r in rows}):
        selected = [r for r in rows if r["split"] == split]
        for target in ("r_star", "r_star_raw", "first_feasible_rate"):
            for name in ("direct", "calibrated", "fixed_projection", "gaussian_projection", *baselines):
                errors = np.array([r[name] - float(r[target]) for r in selected])
                by_receiver = {}
                for receiver in sorted({r["model_key"] for r in selected}):
                    err = [r[name] - float(r[target]) for r in selected if r["model_key"] == receiver]
                    by_receiver[receiver] = float(np.sqrt(np.mean(np.square(err))))
                by_seed = {seed: float(np.sqrt(np.mean([
                    (r[name] - float(r[target])) ** 2 for r in selected if r["seed"] == seed])))
                    for seed in sorted({r["seed"] for r in selected})}
                arm_errors = defaultdict(list)
                for r in selected:
                    arm_errors[r["model_key"], r["dataset_key"]].append(r[name] - float(r[target]))
                summaries.append({"split": split, "target": target, "candidate": name,
                                  "n": len(selected), "rmse": float(np.sqrt(np.mean(errors ** 2))),
                                  "mae": float(np.mean(abs(errors))), "receiver_rmse": by_receiver,
                                  "seed_rmse": by_seed, "arms": len(arm_errors),
                                  "arm_mean_rmse": float(np.sqrt(np.mean([np.mean(v) ** 2 for v in arm_errors.values()])))})
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    heldout = [r for r in rows if r["split"] == "heldout"]
    groups = sorted({r["model_key"] for r in heldout})
    errors = np.array([r["direct"] - float(r["r_star"]) for r in heldout])
    baseline_errors = np.array([r["spectrum_tokens"] - float(r["r_star"]) for r in heldout])
    rng, improvements = np.random.default_rng(20260925), []
    for _ in range(10000):
        indices = [i for group in rng.choice(groups, len(groups), replace=True)
                   for i, r in enumerate(heldout) if r["model_key"] == group]
        improvements.append(np.sqrt(np.mean(baseline_errors[indices] ** 2))
                            - np.sqrt(np.mean(errors[indices] ** 2)))
    report = {"scores": summaries, "calibration_coefficients": coef.tolist(),
              "fixed_development_projection": float(fixed_projection),
              "receiver_bootstrap_improvement": {"baseline": "spectrum_tokens", "clusters": len(groups),
                  "draws": 10000, "seed": 20260925,
                  "interval_95": np.quantile(improvements, [0.025, 0.975]).tolist()},
              "boundary": "Five uncompressed evaluations on 64 calibration rows per adapter. "
                          "These rows overlap the historical target evaluation sample. "
                          "Predictions above one bit extrapolate beyond the pair-dropping model."}
    (out / "scores.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def audit_attenuation_report(profiles: Path, targets: Path, out: Path) -> dict:
    """Apply the frozen rule across corpora; report censored targets separately."""
    feature_rows = {r["run_id"]: r for r in csv.DictReader(profiles.open())}
    meta = {(r["model_key"], r["dataset_key"]): r for r in
            load([HERE / "adapter_spectrum_archive.csv"],
                 HERE.parent / "rate_law_audit", HERE / "run_map.csv")}
    columns = ("text_cross_row_redundancy", "spectrum_top1", "log_corpus_tokens", "task_family")
    rows = []
    for target in csv.DictReader(targets.open()):
        row = {**target, **feature_rows[target["run_id"]]}
        row.update({c: meta[row["model_key"], row["dataset_key"]][c] for c in columns})
        row["direct"] = attenuation_estimate(row)
        # Frozen six-adapter calibration, not refitted to this audit.
        row["calibrated"] = 0.040924233011248916 + 1.0067854755049834 * row["direct"]
        rows.append(row)
    summaries = []
    for subset in ("all", "bracketed"):
        selected = [r for r in rows if subset == "all" or r["bracketed"] == "True"]
        for target in ("r_star", "r_star_raw"):
            y = np.array([float(r[target]) for r in selected])
            for name in ("direct", "calibrated"):
                errors = np.array([r[name] for r in selected]) - y
                summaries.append({"subset": subset, "target": target, "candidate": name,
                                  "n": len(selected), "rmse": float(np.sqrt(np.mean(errors ** 2))),
                                  "mae": float(np.mean(abs(errors)))})
            for name, features in {"constant": [], "text_redundancy": [columns[0]],
                                   "spectrum_tokens": list(columns[1:3])}.items():
                for group in ("model_key", "task_family"):
                    errors = []
                    for held in sorted({r[group] for r in selected}):
                        train = [r for r in selected if r[group] != held]
                        valid = [r for r in selected if r[group] == held]
                        design = lambda rr: np.array([[1] + [float(r[c]) for c in features] for r in rr])
                        fitted = np.linalg.lstsq(design(train), [float(r[target]) for r in train], rcond=None)[0]
                        errors.extend(design(valid) @ fitted - np.array([float(r[target]) for r in valid]))
                    summaries.append({"subset": subset, "target": target, "candidate": name,
                                      "held_out": group, "n": len(selected),
                                      "rmse": float(np.sqrt(np.mean(np.square(errors))))})
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"scores": summaries, "boundary": "One archived adapter per arm. Six targets already "
              "reach 90% at the lowest recorded rate; their reported rates are upper bounds. "
              "Report all 64 and the 58 bracketed targets. No attenuation coefficient is refitted."}
    (out / "scores.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def independent_attenuation_report(predictions: Path, validation: Path, out: Path) -> dict:
    """Keep predictions fixed and replace targets with disjoint-example curves."""
    predicted = {r["run_id"]: r for r in csv.DictReader(predictions.open())
                 if r["split"] == "heldout"}
    measured = {r["run_id"]: r for r in csv.DictReader(validation.open())}
    if predicted.keys() != measured.keys():
        raise ValueError("independent validation must cover every held-out adapter exactly once")
    rows = []
    for run_id, row in predicted.items():
        target = measured[run_id]
        probe_ids = set(json.loads(target["probe_example_ids"]))
        target_ids = set(json.loads(target["independent_example_ids"]))
        if probe_ids & target_ids or len(probe_ids) != 64 or len(target_ids) != int(target["independent_rows"]):
            raise ValueError(f"invalid disjoint-example split for {run_id}")
        rows.append({**row, **target})
    summaries = []
    for target in ("independent_r_star", "independent_r_star_raw"):
        observed = np.array([float(r[target]) for r in rows])
        for name in ("direct", "calibrated", "constant", "text_redundancy", "spectrum_tokens"):
            errors = np.array([float(r[name]) for r in rows]) - observed
            receiver_errors = {model: float(np.sqrt(np.mean([
                (float(r[name]) - float(r[target])) ** 2 for r in rows if r["model_key"] == model])))
                for model in sorted({r["model_key"] for r in rows})}
            summaries.append({"target": target, "candidate": name, "n": len(rows),
                              "rmse": float(np.sqrt(np.mean(errors ** 2))),
                              "mae": float(np.mean(abs(errors))), "receiver_rmse": receiver_errors})
    shifts = [float(r["independent_r_star"]) - float(r["r_star"]) for r in rows]
    report = {"scores": summaries, "target_shift_rmse": float(np.sqrt(np.mean(np.square(shifts)))),
              "boundary": "Frozen 64-example predictions versus newly measured targets on the "
                          "remaining calibration examples. No prediction or coefficient refitted. "
                          "The runner also excludes duplicate prompt/response content."}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "scores.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def attenuation_figure(root: Path) -> Path:
    """Plot the frozen rule against each of its three validation targets."""
    import matplotlib.pyplot as plt

    panels = [
        (root / "audit/predictions.csv", "r_star", "Across corpora", lambda r: r["bracketed"] == "True"),
        (root / "replication/predictions.csv", "r_star", "Other receivers (three seeds)", lambda r: r["split"] == "heldout"),
        (root / "independent/predictions.csv", "independent_r_star", "Disjoint evaluation examples", lambda r: True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), sharex=True, sharey=True, layout="constrained")
    for ax, (source, target, title, include) in zip(axes, panels):
        rows = [r for r in csv.DictReader(source.open()) if include(r)]
        observed = np.array([float(r[target]) for r in rows])
        predicted = np.array([float(r["direct"]) for r in rows])
        rmse = np.sqrt(np.mean((observed - predicted) ** 2))
        ax.plot([0, 1.6], [0, 1.6], color="0.6", linestyle="--", linewidth=1, zorder=1)
        ax.scatter(observed, predicted, s=35, c="#286a91", alpha=0.8,
                   edgecolors="white", linewidths=0.5, zorder=2)
        ax.set(title=title, xlabel="Measured R* (bits/value)", xlim=(0, 1.6), ylim=(0, 1.6))
        ax.text(0.04, 0.96, f"n = {len(rows)}\nRMSE = {rmse:.3f}", transform=ax.transAxes,
                va="top", fontsize=10)
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Predicted R* (bits/value)")
    fig.suptitle("Predicting behavioral rate from uncompressed adapter attenuation", fontsize=13)
    destination = root / "prediction_summary.png"
    fig.savefig(destination, dpi=200)
    plt.close(fig)
    return destination


def probe_arms(panel: Path, run_map: Path, runs_root: Path) -> list[dict]:
    """Read a fixed half-bit probe without using the target's decoded ceiling.

    Normalize by the raw checkpoint's gain, available before a codec sweep.
    Recompute a second target after dropping the probe rung and its contribution
    to the ceiling, to check direct coupling between predictor and target.
    """
    from fineqcomp.rstar import from_run, r_star

    arms = list(csv.DictReader((panel / "arms.csv").open()))
    keys = {(a["model_key"], a["dataset_key"]) for a in arms}
    grouped = defaultdict(list)
    for cell in csv.DictReader(run_map.open()):
        key = (cell["model_key"], cell["dataset_key"])
        directory = runs_root / cell["run_id"]
        if key not in keys or not (directory / "metrics.json").exists():
            continue
        record = json.loads((directory / "metrics.json").read_text())
        raw = record.get("raw_behavioral_write") or {}
        gain, train_gain = raw.get("heldout_bits_saved_per_token"), raw.get("train_bits_saved_per_token")
        points = [c for c in record.get("codecs", [])
                  if (c.get("storage") or {}).get("effective_bits_per_value") is not None
                  and (c.get("behavioral_write") or {}).get("heldout_bits_saved_per_token") is not None]
        probe = next((c for c in points if c.get("bits") == 0 and c.get("blend") == 0.5), None)
        if probe is None or gain is None or gain <= 0 or train_gain is None or train_gain <= 0:
            continue
        target = from_run(directory)
        other = [{"effective_bits_per_value": c["storage"]["effective_bits_per_value"],
                  "heldout_bits_saved_per_token": c["behavioral_write"]["heldout_bits_saved_per_token"]}
                 for c in points if not (c.get("bits") == 0 and c.get("blend") == 0.5)]
        ceiling = max([gain] + [c["heldout_bits_saved_per_token"] for c in other])
        drop = r_star(other, reference=ceiling)["r_star"]
        raw_target = r_star(other, reference=gain)["r_star"]
        if any(v is None for v in (target["r_star"], drop, raw_target)):
            continue
        grouped[key].append({
            "run_id": cell["run_id"], "target": target["r_star"],
            "target_without_probe": drop, "target_raw_without_probe": raw_target,
            "half_probe": probe["behavioral_write"]["heldout_bits_saved_per_token"] / gain,
            "train_half_probe": probe["behavioral_write"]["train_bits_saved_per_token"] / train_gain,
            "probe_rate": probe["storage"]["effective_bits_per_value"],
        })
    result = []
    for arm in arms:
        cells = grouped[(arm["model_key"], arm["dataset_key"])]
        if cells:
            result.append({**arm, "cells": cells,
                           **{k: float(np.mean([c[k] for c in cells]))
                              for k in cells[0] if k != "run_id"}})
    return result


def probe_report(runs_root: Path, out: Path) -> dict:
    """Fit on the audit panel; transfer unchanged to the transposed panel."""
    dev = probe_arms(HERE.parent / "rate_law_audit", HERE / "run_map.csv", runs_root)
    test = probe_arms(PANEL, PANEL / "cells.csv", runs_root)
    candidates = {"constant": [], "text_redundancy": ["text_cross_row_redundancy"],
                  "half_probe": ["half_probe"], "train_half_probe": ["train_half_probe"]}
    summaries, predictions = [], []
    for target in ("target", "target_without_probe", "target_raw_without_probe"):
        for name, features in candidates.items():
            def design(rows):
                return np.array([[1.0] + [float(r[f]) for f in features] for r in rows])

            y = np.array([r[target] for r in dev])
            coef = np.linalg.lstsq(design(dev), y, rcond=None)[0]
            cv = {}
            for key in ("model_key", "task_family"):
                errors = []
                for held in sorted({r[key] for r in dev}):
                    train = [r for r in dev if r[key] != held]
                    valid = [r for r in dev if r[key] == held]
                    fitted = np.linalg.lstsq(design(train), [r[target] for r in train], rcond=None)[0]
                    errors.extend(design(valid) @ fitted - np.array([r[target] for r in valid]))
                cv[f"held_out_{key}_rmse"] = float(np.sqrt(np.mean(np.square(errors))))
            estimates = design(test) @ coef
            errors = estimates - np.array([r[target] for r in test])
            unseen = np.array([r["model_key"] not in {d["model_key"] for d in dev} for r in test])
            summaries.append({"candidate": name, "target": target, "development_arms": len(dev),
                              "test_arms": len(test), "coefficients": coef.tolist(), **cv,
                              "transfer_rmse": float(np.sqrt(np.mean(errors ** 2))),
                              "new_receiver_arms": int(unseen.sum()),
                              "new_receiver_rmse": float(np.sqrt(np.mean(errors[unseen] ** 2)))})
            for row, estimate, is_new in zip(test, estimates, unseen):
                predictions.append({"candidate": name, "target": target,
                                    "model_key": row["model_key"], "dataset_key": row["dataset_key"],
                                    "new_receiver": bool(is_new), "observed": row[target],
                                    "predicted": float(estimate)})
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("scores", summaries), ("predictions", predictions)):
        (out / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
    for split, rows in (("development", dev), ("test", test)):
        cells = [{"model_key": r["model_key"], "dataset_key": r["dataset_key"], **c}
                 for r in rows for c in r["cells"]]
        with (out / f"{split}_cells.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
            writer.writeheader()
            writer.writerows(cells)
    return {"scores": summaries, "output": str(out),
            "boundary": "Retrospective; probe and target use the same evaluation examples. "
                        "The probe is normalized by raw gain, never the target's decoded ceiling."}


def load(feature_csvs: list[Path], panel: Path = PANEL, run_map: Path | None = None):
    """Arms of the panel, with each feature CSV averaged over an arm's seeds."""
    arms = list(csv.DictReader((panel / "arms.csv").open()))
    cells = {r["run_id"]: (r["model_key"], r["dataset_key"])
             for r in csv.DictReader((run_map or panel / "cells.csv").open())}
    for path in feature_csvs:
        per_arm = defaultdict(list)
        for row in csv.DictReader(path.open()):
            if row["run_id"] in cells:
                per_arm[cells[row["run_id"]]].append(row)
        for arm in arms:
            rows = per_arm[(arm["model_key"], arm["dataset_key"])]
            for key in (rows[0] if rows else {}):
                if key.startswith(FEATURES):
                    arm[key] = str(np.mean([float(r[key]) for r in rows]))
    return arms


def score(arms, feature):
    receivers = sorted({a["model_key"] for a in arms})
    corpora = sorted({a["dataset_key"] for a in arms})
    usable = [a for a in arms if feature is None or a.get(feature) not in (None, "")]
    predicted = {}
    for held in receivers:
        train = [a for a in usable if a["model_key"] != held]
        seen = {a["dataset_key"] for a in train}
        test = [a for a in usable if a["model_key"] == held and a["dataset_key"] in seen]
        if not test:
            continue

        def design(rows):
            cols = [[float(r["dataset_key"] == c) for c in corpora] for r in rows]
            if feature is not None:
                cols = [c + [float(r[feature])] for c, r in zip(cols, rows)]
            return np.array(cols)

        coef, *_ = np.linalg.lstsq(design(train), np.array([float(r["r_star_bits_per_value"]) for r in train]), rcond=None)
        for row, value in zip(test, design(test) @ coef):
            predicted[(row["model_key"], row["dataset_key"])] = value
    observed = {(a["model_key"], a["dataset_key"]): float(a["r_star_bits_per_value"]) for a in usable}
    keys = sorted(predicted)
    errors = np.array([predicted[k] - observed[k] for k in keys])
    right = total = 0
    for corpus in corpora:
        members = [k for k in keys if k[1] == corpus]
        for a, b in itertools.combinations(members, 2):
            total += 1
            right += np.sign(predicted[a] - predicted[b]) == np.sign(observed[a] - observed[b])
    return {"candidate": feature or "corpus_only", "arms": len(keys),
            "rmse": float(np.sqrt(np.mean(errors ** 2))), "mae": float(np.mean(np.abs(errors))),
            "receiver_pair_sign_accuracy": right / total if total else float("nan")}


def main() -> None:
    args = sys.argv[1:]
    if args[:1] == ["--attenuation-checks"]:
        if len(args) != 4:
            raise SystemExit("usage: score.py --attenuation-checks PREDICTIONS_CSV RUNS_ROOT OUTPUT_DIR")
        print(json.dumps(attenuation_checks(*map(Path, args[1:])), indent=2))
        return
    if args[:1] == ["--attenuation-figure"]:
        if len(args) != 2:
            raise SystemExit("usage: score.py --attenuation-figure RESULTS_ROOT")
        print(attenuation_figure(Path(args[1])))
        return
    if args[:1] == ["--attenuation-independent"]:
        if len(args) != 4:
            raise SystemExit("usage: score.py --attenuation-independent PREDICTIONS_CSV VALIDATION_CSV OUTPUT_DIR")
        print(json.dumps(independent_attenuation_report(*map(Path, args[1:])), indent=2))
        return
    if args[:1] == ["--attenuation-audit"]:
        if len(args) != 4:
            raise SystemExit("usage: score.py --attenuation-audit PROFILES_CSV TARGETS_CSV OUTPUT_DIR")
        print(json.dumps(audit_attenuation_report(*map(Path, args[1:])), indent=2))
        return
    if args[:1] == ["--attenuation"]:
        if len(args) != 4:
            raise SystemExit("usage: score.py --attenuation PROFILES_CSV TARGETS_CSV OUTPUT_DIR")
        print(json.dumps(attenuation_report(*map(Path, args[1:])), indent=2))
        return
    if args[:1] == ["--probe"]:
        if len(args) != 3:
            raise SystemExit("usage: score.py --probe RUNS_ROOT OUTPUT_DIR")
        print(json.dumps(probe_report(Path(args[1]), Path(args[2])), indent=2))
        return
    panel, run_map = PANEL, None
    if args[:1] == ["--panel"]:
        panel, run_map, args = Path(args[1]), Path(args[3]), args[4:]
    sources = [Path(p) for p in args]
    arms = load(sources, panel, run_map)
    features = [None] + [f for f in ("dataset_fisher_log_volume", "text_cross_row_redundancy",
                                      "correction_channel_bits") if f in arms[0]]
    features += sorted(k for k in arms[0] if k.startswith(FEATURES))
    results = [score(arms, f) for f in features]
    for r in sorted(results, key=lambda r: r["rmse"]):
        print(f"{r['candidate']:34s} arms={r['arms']:2d} rmse={r['rmse']:.4f} mae={r['mae']:.4f} sign={r['receiver_pair_sign_accuracy']:.3f}")
    if sources:
        with (HERE / f"scores_{panel.name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)


if __name__ == "__main__":
    main()
