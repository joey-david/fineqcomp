"""Locked discovery selection and untouched correction-spectrum validation."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from fineqcomp.artifacts import write_json
from fineqcomp.relative_info import (
    _correlation,
    _rankdata,
    channel_bits_ceiling,
    analyze_relative_information_cells,
    collect_relative_information_cells,
)


def _pooled_centered_correlation(
    left: list[np.ndarray], right: list[np.ndarray]
) -> float:
    centered_left = np.concatenate([values - values.mean() for values in left])
    centered_right = np.concatenate([values - values.mean() for values in right])
    return _correlation(centered_left, centered_right)


def _fit_prediction(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    candidate: str | None,
    *,
    log_candidate: bool = True,
) -> np.ndarray:
    models = sorted({str(row["model_key"]) for row in train})

    def design(rows: list[dict[str, Any]]) -> np.ndarray:
        columns = [np.ones(len(rows))]
        for model in models[1:]:
            columns.append(
                np.asarray([row["model_key"] == model for row in rows], dtype=float)
            )
        if candidate is not None:
            values = np.asarray([row[candidate] for row in rows], dtype=float)
            columns.append(
                np.log(np.maximum(values, 1e-12))
                if log_candidate
                else values
            )
        return np.column_stack(columns)

    target = np.asarray(
        [row["r_star_bits_per_value"] for row in train], dtype=float
    )
    coefficients = np.linalg.lstsq(design(train), target, rcond=None)[0]
    return design(test) @ coefficients


def _fixed_channel_arms(
    cells: list[dict[str, Any]], candidate: str
) -> list[dict[str, Any]]:
    keys = (
        candidate,
        "base_codelength_bits_per_token",
        "train_response_tokens",
    )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in cells:
        groups.setdefault(
            (str(row["model_key"]), str(row["dataset_key"])), []
        ).append(row)
    arms = []
    for (model, dataset), rows in sorted(groups.items()):
        arms.append(
            {
                "model_key": model,
                "dataset_key": dataset,
                "seeds": len(rows),
                "r_star_bits_per_value": float(
                    np.mean([row["r_star_bits_per_value"] for row in rows])
                ),
                **{
                    key: float(np.mean([row[key] for row in rows]))
                    for key in keys
                },
            }
        )
    return arms


def _exact_within_model_rank_test(
    arms: list[dict[str, Any]], candidate: str
) -> tuple[float, float, int]:
    groups = [
        [row for row in arms if row["model_key"] == model]
        for model in sorted({str(row["model_key"]) for row in arms})
    ]
    groups = [rows for rows in groups if len(rows) >= 2]
    if not groups:
        raise ValueError("prospective rank test needs a receiver with two arms")
    candidate_ranks = [
        _rankdata(np.asarray([row[candidate] for row in rows], dtype=float))
        for rows in groups
    ]
    target_ranks = [
        _rankdata(
            np.asarray([row["r_star_bits_per_value"] for row in rows], dtype=float)
        )
        for rows in groups
    ]
    observed = _pooled_centered_correlation(candidate_ranks, target_ranks)
    null = (
        _pooled_centered_correlation(
            candidate_ranks,
            [np.asarray(values, dtype=float) for values in permuted],
        )
        for permuted in itertools.product(
            *(itertools.permutations(ranks) for ranks in target_ranks)
        )
    )
    total = 0
    exceed = 0
    for value in null:
        total += 1
        exceed += value >= observed - 1e-12
    return observed, exceed / total, total


def analyze_fixed_channel_prospective(
    development_cells: list[dict[str, Any]],
    prospective_cells: list[dict[str, Any]],
    lock: dict[str, Any],
) -> dict[str, Any]:
    """Apply one fixed correction-spectrum measure to untouched cells."""
    candidate = str(lock["candidate"])
    expected = lock["measurement_contract"]
    if len(prospective_cells) != int(expected["cells"]):
        raise ValueError(
            f"expected {expected['cells']} prospective cells, found"
            f" {len(prospective_cells)}"
        )
    if any(not row.get("r_star_bracketed") for row in prospective_cells):
        raise ValueError("every prospective R* must be bracketed")
    prospective = _fixed_channel_arms(prospective_cells, candidate)
    if len(prospective) != int(expected["arms"]):
        raise ValueError(
            f"expected {expected['arms']} prospective arms, found {len(prospective)}"
        )

    development_analysis = analyze_relative_information_cells(
        development_cells, permutations=100
    )
    fields = (
        candidate,
        "base_codelength_bits_per_token",
        "train_response_tokens",
    )
    development = [
        {
            "model_key": row["model_key"],
            "r_star_bits_per_value": row["r_star_bits_per_value"],
            **{key: row[key] for key in fields},
        }
        for row in development_analysis["arms"]
    ]
    truth = np.asarray(
        [row["r_star_bits_per_value"] for row in prospective], dtype=float
    )
    predictions = {
        "candidate": _fit_prediction(
            development, prospective, candidate, log_candidate=False
        ),
        "receiver_only": _fit_prediction(development, prospective, None),
        "base_codelength": _fit_prediction(
            development, prospective, "base_codelength_bits_per_token"
        ),
        "train_response_tokens": _fit_prediction(
            development, prospective, "train_response_tokens"
        ),
    }
    rmse = {
        key: float(np.sqrt(np.mean((values - truth) ** 2)))
        for key, values in predictions.items()
    }
    rho, p_value, exact_permutations = _exact_within_model_rank_test(
        prospective, candidate
    )

    cell_groups: dict[tuple[str, str], list[float]] = {}
    for row in prospective_cells:
        cell_groups.setdefault(
            (str(row["model_key"]), str(row["dataset_key"])), []
        ).append(float(row[candidate]))
    cvs = [
        float(np.std(values, ddof=1) / np.mean(values))
        for values in cell_groups.values()
        if len(values) > 1 and np.mean(values)
    ]
    median_cv = float(np.median(cvs))

    rank_gates = lock["primary_test"]["gates"]
    prediction_gates = lock["prediction_gates"]
    validity_gates = lock["validity_gates"]
    gates = {
        "all_cells_have_finite_measure": all(
            np.isfinite(float(row[candidate])) for row in prospective_cells
        ),
        "all_cells_have_locked_measurement_rows": all(
            row.get("measured_rows") == int(expected["rows"])
            for row in prospective_cells
        ),
        "all_cells_use_locked_row_sampling": all(
            row.get("row_sampling") == expected["row_sampling"]
            and row.get("row_sample_seed") == int(expected["row_sample_seed"])
            for row in prospective_cells
        ),
        "rank_rho_at_least_locked_value": rho >= float(rank_gates["rho_at_least"]),
        "rank_p_below_locked_value": p_value
        < float(rank_gates["one_sided_p_below"]),
        "rmse_20pct_below_receiver_only": rmse["candidate"]
        <= float(
            prediction_gates["prefit_rmse_at_most_fraction_of_receiver_only"]
        )
        * rmse["receiver_only"],
        "rmse_below_base_codelength": rmse["candidate"]
        < rmse["base_codelength"],
        "rmse_below_train_response_tokens": rmse["candidate"]
        < rmse["train_response_tokens"],
        "all_arms_have_locked_seed_count": all(
            row["seeds"] == int(expected["seeds_per_arm"])
            for row in prospective
        ),
        "median_measure_cv_below_locked_value": median_cv
        < float(validity_gates["median_within_arm_measure_cv_below"]),
    }
    if validity_gates.get("llama_xbrl_diversity_direction_matches_R_star"):
        llama_diversity = {
            row["dataset_key"]: row
            for row in prospective
            if row["model_key"] == "llama31_8b_base"
            and row["dataset_key"] in {"xbrl_div_10", "xbrl_div_30"}
        }
        if set(llama_diversity) != {"xbrl_div_10", "xbrl_div_30"}:
            raise ValueError("missing locked Llama XBRL diversity arms")
        low = llama_diversity["xbrl_div_10"]
        high = llama_diversity["xbrl_div_30"]
        gates["llama_xbrl_diversity_direction_matches"] = (
            high[candidate] > low[candidate]
            and high["r_star_bits_per_value"] > low["r_star_bits_per_value"]
        )
    for index, row in enumerate(prospective):
        row.update(
            {
                "prefit_prediction": float(predictions["candidate"][index]),
                "receiver_only_prediction": float(
                    predictions["receiver_only"][index]
                ),
                "base_codelength_prediction": float(
                    predictions["base_codelength"][index]
                ),
                "train_response_tokens_prediction": float(
                    predictions["train_response_tokens"][index]
                ),
            }
        )
    return {
        "status": "passed" if all(gates.values()) else "failed",
        "candidate": candidate,
        "prospective_cells": len(prospective_cells),
        "prospective_arms": len(prospective),
        "within_model_rank_rho": rho,
        "within_model_exact_permutation_p": p_value,
        "exact_permutations": exact_permutations,
        "prefit_rmse": rmse,
        "median_measure_cv": median_cv,
        "gates": gates,
        "arms": prospective,
    }


def write_fixed_channel_prospective_report(
    development_root: Path,
    prospective_root: Path,
    prospective_manifest: Path,
    runs_root: Path,
    lock_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    lock = json.loads(Path(lock_path).read_text())
    if lock.get("status") not in {
        "locked_before_results",
        "locked_before_replacement_results",
    }:
        raise ValueError("prospective lock is not active")
    run_ids = {
        str(json.loads(line)["run_id"])
        for line in Path(prospective_manifest).read_text().splitlines()
        if line.strip()
    }
    development = collect_relative_information_cells(
        development_root, runs_root, skip_missing_r_star=True
    )
    prospective = [
        row
        for row in collect_relative_information_cells(
            prospective_root, runs_root, skip_missing_r_star=True
        )
        if str(row["run_id"]) in run_ids
    ]
    development = [row for row in development if str(row["run_id"]) not in run_ids]
    result = analyze_fixed_channel_prospective(development, prospective, lock)
    out_dir = Path(out_dir)
    write_json(
        out_dir / "prospective_summary.json",
        {key: value for key, value in result.items() if key != "arms"},
    )
    _write_csv(out_dir / "prospective_arms.csv", result["arms"])
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    fieldnames.extend(
        key
        for row in rows[1:]
        for key in row
        if key not in fieldnames
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


IDENTIFIER_COLUMNS = frozenset(
    {
        "run_id",
        "study",
        "model_key",
        "dataset_key",
        "seed",
        "adapter_rank",
        "training_epochs",
        "label_span",
        "available_rows",
        "distinct_rows",
        "measured_rows",
        "row_sampling",
        "row_sample_seed",
        "r_star_bits_per_value",
        "r_star_bracketed",
    }
)

# Read off the same retention curve as R*, so they are alternative targets and
# never predictors of it.
RETENTION_TARGET_COLUMNS = frozenset(
    {"r50", "r75", "r90", "r95", "shape_r90_over_r50", "ceiling_heldout_bits_saved"}
)


def read_cells_csv(path: Path) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Read a written cells.csv back into the shape the analysis helpers use."""
    with Path(path).open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    if not raw:
        raise SystemExit(f"no cells in {path}")
    # Later measurement passes added columns the earlier cells never carried,
    # so only keep the ones every cell has a finite value for.
    candidates = tuple(
        key
        for key in raw[0]
        if key
        and key not in IDENTIFIER_COLUMNS
        and all(_finite_float(row[key]) is not None for row in raw)
    )
    cells = []
    for row in raw:
        cell: dict[str, Any] = dict(row)
        cell["r_star_bits_per_value"] = float(row["r_star_bits_per_value"])
        for key in candidates:
            cell[key] = float(row[key])
        cells.append(cell)
    return cells, candidates


def _finite_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _variance_shares(
    arms: list[dict[str, Any]], key: str
) -> tuple[float, float, float]:
    """Split arm variance into dataset, receiver, and interaction shares.

    Only datasets measured on more than one receiver can separate the two main
    effects, so the split is taken on that subset.
    """
    counts: dict[str, set[str]] = {}
    for arm in arms:
        counts.setdefault(str(arm["dataset_key"]), set()).add(str(arm["model_key"]))
    shared = [arm for arm in arms if len(counts[str(arm["dataset_key"])]) > 1]
    if len(shared) < 4:
        return (float("nan"),) * 3
    values = np.array([float(arm[key]) for arm in shared])
    grand = float(values.mean())
    dataset_effect = np.array(
        [
            float(
                np.mean(
                    [
                        float(other[key])
                        for other in shared
                        if other["dataset_key"] == arm["dataset_key"]
                    ]
                )
            )
            - grand
            for arm in shared
        ]
    )
    model_effect = np.array(
        [
            float(
                np.mean(
                    [
                        float(other[key])
                        for other in shared
                        if other["model_key"] == arm["model_key"]
                    ]
                )
            )
            - grand
            for arm in shared
        ]
    )
    total = float(values.var())
    if total <= 0:
        return (float("nan"),) * 3
    residual = values - grand - dataset_effect - model_effect
    return (
        float(dataset_effect.var() / total),
        float(model_effect.var() / total),
        float(residual.var() / total),
    )


def _receiver_pairs(
    arms: list[dict[str, Any]], candidates: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Every same-dataset, different-receiver comparison available."""
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        by_dataset.setdefault(str(arm["dataset_key"]), []).append(arm)
    pairs = []
    for dataset, group in sorted(by_dataset.items()):
        for left, right in itertools.combinations(
            sorted(group, key=lambda row: str(row["model_key"])), 2
        ):
            record = {
                "dataset_key": dataset,
                "model_key_a": left["model_key"],
                "model_key_b": right["model_key"],
                "delta_r_star": float(left["r_star_bits_per_value"])
                - float(right["r_star_bits_per_value"]),
            }
            for key in candidates:
                record["delta_" + key] = float(left[key]) - float(right[key])
            pairs.append(record)
    return pairs


def _cross_receiver_test(
    arms: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    key: str,
    *,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    """Does the measure order receivers correctly on a fixed dataset?

    The null shuffles R* between the receivers that share a dataset, which
    leaves both the dataset and the receiver marginals in place and tests only
    the interaction the model-relative hypothesis claims.
    """
    if len(pairs) < 4:
        return {
            "cross_receiver_spearman": float("nan"),
            "cross_receiver_sign_agreement": float("nan"),
            "cross_receiver_p": float("nan"),
        }
    measure = np.array([pair["delta_" + key] for pair in pairs])
    target = np.array([pair["delta_r_star"] for pair in pairs])
    observed = _correlation(_rankdata(measure), _rankdata(target))
    by_dataset: dict[str, list[int]] = {}
    for index, arm in enumerate(arms):
        by_dataset.setdefault(str(arm["dataset_key"]), []).append(index)
    shared = [indices for indices in by_dataset.values() if len(indices) > 1]
    r_star = np.array([float(arm["r_star_bits_per_value"]) for arm in arms])
    index_of = {
        (str(arm["model_key"]), str(arm["dataset_key"])): index
        for index, arm in enumerate(arms)
    }
    lookup = [
        (
            index_of[(str(pair["model_key_a"]), str(pair["dataset_key"]))],
            index_of[(str(pair["model_key_b"]), str(pair["dataset_key"]))],
        )
        for pair in pairs
    ]
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(permutations):
        shuffled = r_star.copy()
        for indices in shared:
            shuffled[indices] = rng.permutation(r_star[indices])
        drawn = np.array([shuffled[a] - shuffled[b] for a, b in lookup])
        if abs(_correlation(_rankdata(measure), _rankdata(drawn))) >= abs(observed):
            hits += 1
    return {
        "cross_receiver_spearman": float(observed),
        "cross_receiver_sign_agreement": float(
            np.mean(np.sign(measure) == np.sign(target))
        ),
        "cross_receiver_p": float((hits + 1) / (permutations + 1)),
    }


def _partial_within_model(
    arms: list[dict[str, Any]], key: str, controls: tuple[str, ...]
) -> float:
    """Within-receiver rank correlation with R* after removing the controls."""
    residual_measure: list[float] = []
    residual_target: list[float] = []
    by_model: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        by_model.setdefault(str(arm["model_key"]), []).append(arm)
    for group in by_model.values():
        if len(group) < len(controls) + 3:
            continue
        design = np.column_stack(
            [np.ones(len(group))]
            + [_rankdata(np.array([float(a[c]) for a in group])) for c in controls]
        )
        for target, sink in (
            (key, residual_measure),
            ("r_star_bits_per_value", residual_target),
        ):
            values = _rankdata(np.array([float(a[target]) for a in group]))
            fit = np.linalg.lstsq(design, values, rcond=None)[0]
            sink.extend(values - design @ fit)
    if len(residual_measure) < 4:
        return float("nan")
    measure = np.array(residual_measure)
    # A candidate the controls reproduce exactly leaves nothing to correlate,
    # and the ratio of two rounding errors is not a partial correlation.
    if float(np.std(measure)) < 1e-9:
        return 0.0
    return _correlation(measure, np.array(residual_target))


def analyze_measure_diagnostics(
    cells: list[dict[str, Any]],
    candidates: tuple[str, ...],
    *,
    controls: tuple[str, ...] = (
        "text_cross_row_redundancy",
        "train_response_tokens",
    ),
    permutations: int = 20_000,
    seed: int = 20260826,
) -> dict[str, Any]:
    """Ask what a corpus measure still explains once the cheap parts are gone.

    Three questions the ranking table does not answer. Does the measure vary
    as much as R* does? Does any of its variation come from the receiver
    rather than the corpus? And does it beat a text statistic and a token
    count on the corpus axis, and order receivers on a fixed corpus?
    """
    from fineqcomp.relative_info import _aggregate_dataset_arms

    arms = _aggregate_dataset_arms(cells, candidates)
    pairs = _receiver_pairs(arms, candidates)
    target_cv = float(
        np.std([float(a["r_star_bits_per_value"]) for a in arms])
        / abs(np.mean([float(a["r_star_bits_per_value"]) for a in arms]))
    )
    rows = []
    for key in candidates:
        values = np.array([float(arm[key]) for arm in arms])
        mean = float(values.mean())
        dataset_share, model_share, residual_share = _variance_shares(arms, key)
        row = {
            "candidate": key,
            "within_model_spearman": _mean_or_nan(
                [
                    _spearman_group(group, key)
                    for group in _by_model(arms).values()
                    if len(group) >= 4
                ]
            ),
            "measure_cv": float(values.std() / abs(mean)) if mean else float("nan"),
            "cv_ratio_to_r_star": (
                float(values.std() / abs(mean) / target_cv) if mean else float("nan")
            ),
            "dataset_variance_share": dataset_share,
            "receiver_variance_share": model_share,
            "interaction_variance_share": residual_share,
            "partial_within_model_spearman": _partial_within_model(
                arms, key, tuple(c for c in controls if c != key)
            ),
        }
        row.update(
            _cross_receiver_test(
                arms, pairs, key, permutations=permutations, seed=seed
            )
        )
        rows.append(row)
    rows.sort(key=lambda row: -abs(row["within_model_spearman"]))
    return {
        "arms": arms,
        "pairs": pairs,
        "diagnostics": rows,
        "summary": {
            "arms": len(arms),
            "cells": len(cells),
            "receivers": len({str(a["model_key"]) for a in arms}),
            "cross_receiver_pairs": len(pairs),
            "shared_datasets": len(
                {str(p["dataset_key"]) for p in pairs}
            ),
            "r_star_cv": target_cv,
            "controls": list(controls),
            "permutations": permutations,
        },
    }


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _by_model(arms: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        grouped.setdefault(str(arm["model_key"]), []).append(arm)
    return grouped


def _spearman_group(group: list[dict[str, Any]], key: str) -> float:
    return _correlation(
        _rankdata(np.array([float(a[key]) for a in group])),
        _rankdata(np.array([float(a["r_star_bits_per_value"]) for a in group])),
    )


def write_measure_diagnostics_report(
    cells_csv: Path, out_dir: Path, *, permutations: int = 20_000
) -> dict[str, Any]:
    """Write the post-mortem of a corpus measure against its own panel."""
    cells, candidates = read_cells_csv(cells_csv)
    result = analyze_measure_diagnostics(
        cells, candidates, permutations=permutations
    )
    out_dir = Path(out_dir)
    _write_csv(out_dir / "measure_diagnostics.csv", result["diagnostics"])
    _write_csv(out_dir / "cross_receiver_pairs.csv", result["pairs"])
    write_json(out_dir / "diagnostics_summary.json", result["summary"])
    return result


RATE_LAW_CONTROLS = ("text_cross_row_redundancy", "train_response_tokens")


def _log2(value: float) -> float:
    return float(np.log2(max(float(value), 1e-12)))


def join_retention_profiles(
    cells: list[dict[str, Any]], runs_root: Path
) -> list[dict[str, Any]]:
    """Attach the whole rate-retention ladder to every measured cell.

    R*(0.90) is one crossing of a curve whose shape is not fixed: the ratio
    r90/r50 runs from 1.0 to 5.3 across the campaign, so the arms are not a
    one-parameter family and a single threshold cannot stand for the curve.
    """
    from fineqcomp.rstar import retention_profile

    joined = []
    for cell in cells:
        profile = retention_profile(Path(runs_root) / str(cell["run_id"]))
        if not profile.get("points"):
            continue
        row = dict(cell)
        row.update(
            {
                "ceiling_heldout_bits_saved": profile["ceiling_heldout_bits_saved"],
                "shape_r90_over_r50": profile["shape_r90_over_r50"],
                **{
                    key: profile.get(key)
                    for key in ("r50", "r75", "r90", "r95")
                },
            }
        )
        row["_ladder"] = profile["points"]
        joined.append(row)
    return joined


def add_tokenizer_fertility(arms: list[dict[str, Any]]) -> None:
    """Split the supervised token count into corpus size and receiver fit.

    The same corpus tokenizes to different lengths on different receivers, so
    `train_response_tokens` mixes a corpus quantity with a model quantity and
    the two point opposite ways: within a receiver more tokens means a larger
    R*, between receivers on one corpus more tokens means a smaller one.  The
    corpus mean carries the size; the ratio to it carries the receiver, and is
    the only purely model-relative term available on this panel.
    """
    totals: dict[str, list[float]] = {}
    for arm in arms:
        totals.setdefault(str(arm["dataset_key"]), []).append(
            float(arm["train_response_tokens"])
        )
    for arm in arms:
        seen = totals[str(arm["dataset_key"])]
        corpus_tokens = float(np.mean(seen))
        arm["corpus_tokens"] = corpus_tokens
        arm["log_corpus_tokens"] = _log2(corpus_tokens)
        arm["tokenizer_fertility_relative"] = (
            float(arm["train_response_tokens"]) / corpus_tokens
            if corpus_tokens
            else float("nan")
        )
        arm["receivers_sharing_corpus"] = len(seen)


def forward_model_predictions(
    cells: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    *,
    target: float = 0.90,
) -> list[dict[str, Any]]:
    """Predict R* from the corpus spectrum with the codec setting the floor.

    No fitted constant and no receiver term: the rate at which the Wiener
    retention of correction energy reaches `target` is a prediction of R* in
    bits per value, comparable to the measurement on the identity line.
    """
    import torch

    from fineqcomp.relative_info import codec_referenced_retention

    rows = []
    for cell in cells:
        record = records.get(str(cell["run_id"]))
        ladder = cell.get("_ladder")
        if not record or not ladder:
            continue
        spectrum = (record.get("spectra") or {}).get("correction")
        if not spectrum:
            continue
        eigenvalues = torch.as_tensor(spectrum, dtype=torch.float64)
        curve = [
            (
                float(point["rate_bits_per_value"]),
                codec_referenced_retention(
                    eigenvalues, float(point["relative_rmse"])
                ),
            )
            for point in ladder
            if point.get("relative_rmse") is not None
        ]
        if len(curve) < 4:
            continue
        curve.sort()
        rows.append(
            {
                "run_id": cell["run_id"],
                "model_key": cell["model_key"],
                "dataset_key": cell["dataset_key"],
                "observed_r90": cell.get("r90"),
                "predicted_r90": _crossing_of(curve, target),
                "predicted_retention_at_lowest_rate": curve[0][1],
                "predicted_retention_at_highest_rate": curve[-1][1],
                "observed_retention_at_lowest_rate": ladder[0]["retained_fraction"],
                "observed_retention_at_highest_rate": ladder[-1][
                    "retained_fraction"
                ],
            }
        )
    return rows


def _crossing_of(curve: list[tuple[float, float]], target: float) -> float:
    for index in range(1, len(curve)):
        left, right = curve[index - 1], curve[index]
        if left[1] < target <= right[1]:
            span = right[1] - left[1]
            if span <= 0:
                return right[0]
            return left[0] + (target - left[1]) * (right[0] - left[0]) / span
    return float("nan")


def candidate_gate(
    arms: list[dict[str, Any]],
    candidates: tuple[str, ...],
    *,
    controls: tuple[str, ...] = RATE_LAW_CONTROLS,
) -> list[dict[str, Any]]:
    """Rank candidates on what they add to the controls, not on raw agreement.

    The discovery gate that chose `correction_channel_bits` ranked on raw
    within-receiver correlation with R*, which rewards whichever candidate best
    reproduces a text statistic and a token count.  Scoring the residual
    instead makes a candidate eligible only if it carries something those two
    do not.
    """
    rows = []
    for key in candidates:
        raw = _mean_or_nan(
            [
                _spearman_group(group, key)
                for group in _by_model(arms).values()
                if len(group) >= 4
            ]
        )
        partial = _partial_within_model(
            arms, key, tuple(c for c in controls if c != key)
        )
        rows.append(
            {
                "candidate": key,
                "within_model_spearman": raw,
                "partial_within_model_spearman": partial,
                "adds_over_controls": bool(
                    np.isfinite(partial) and abs(partial) >= 0.30
                ),
                "control": key in controls,
            }
        )
    rows.sort(
        key=lambda row: -abs(
            row["partial_within_model_spearman"]
            if np.isfinite(row["partial_within_model_spearman"])
            else 0.0
        )
    )
    return rows


def _fit_with_receiver_effects(
    arms: list[dict[str, Any]], predictors: tuple[str, ...], target: str
) -> tuple[np.ndarray, list[str]]:
    receivers = sorted({str(arm["model_key"]) for arm in arms})
    design = np.column_stack(
        [
            np.array(
                [1.0 if str(arm["model_key"]) == name else 0.0 for arm in arms]
            )
            for name in receivers
        ]
        + [np.array([float(arm[key]) for arm in arms]) for key in predictors]
    )
    values = np.array([float(arm[target]) for arm in arms])
    return np.linalg.lstsq(design, values, rcond=None)[0], receivers


def evaluate_rate_models(
    arms: list[dict[str, Any]],
    model_specs: dict[str, tuple[str, ...]],
    *,
    target: str = "r_star_bits_per_value",
    minimum_held_out_arms: int = 4,
) -> list[dict[str, Any]]:
    """Leave one receiver out: fit on the others, predict with no term for it.

    A receiver fixed effect fitted on the held-out receiver is not a
    prediction.  Dropping it is the only version of the test that answers the
    question the campaign asks, which is whether a corpus measured against a
    frozen model says anything about a model the fit has never seen.
    """
    receivers = sorted({str(arm["model_key"]) for arm in arms})
    usable = [
        name
        for name in receivers
        if sum(str(arm["model_key"]) == name for arm in arms)
        >= minimum_held_out_arms
    ]
    rows = []
    for name, predictors in model_specs.items():
        errors, correlations = [], []
        for held in usable:
            train = [a for a in arms if str(a["model_key"]) != held]
            test = [a for a in arms if str(a["model_key"]) == held]
            observed = np.array([float(a[target]) for a in test])
            if predictors:
                beta, seen = _fit_with_receiver_effects(train, predictors, target)
                intercept = float(np.mean(beta[: len(seen)]))
                slopes = beta[len(seen) :]
                predicted = intercept + np.column_stack(
                    [np.array([float(a[key]) for a in test]) for key in predictors]
                ) @ slopes
            else:
                predicted = np.full(
                    len(test), float(np.mean([float(a[target]) for a in train]))
                )
            errors.append(float(np.sqrt(np.mean((predicted - observed) ** 2))))
            correlations.append(
                _correlation(_rankdata(predicted), _rankdata(observed))
                if len(test) > 2
                else float("nan")
            )
        rows.append(
            {
                "model": name,
                "predictors": "+".join(predictors) if predictors else "receiver mean",
                "held_out_receivers": len(usable),
                "receiver_held_out_rmse": float(np.mean(errors)),
                "receiver_held_out_spearman": _mean_or_nan(
                    [value for value in correlations if np.isfinite(value)]
                ),
            }
        )
    rows.sort(key=lambda row: row["receiver_held_out_rmse"])
    return rows


def score_recorded_arms(
    development: list[dict[str, Any]],
    prospective: list[dict[str, Any]],
    model_specs: dict[str, tuple[str, ...]],
    *,
    target: str = "r_star_bits_per_value",
) -> list[dict[str, Any]]:
    """Fit on development arms, score arms recorded by an earlier campaign.

    The receiver is known here, so its fitted effect is used rather than the
    average: this is the prefit the locked prospective test applied, repeated
    for several predictors on the same arms.  Once those arms have been looked
    at the comparison stops being prospective, and the table says so.
    """
    receivers = sorted({str(arm["model_key"]) for arm in development})
    observed = np.array([float(arm[target]) for arm in prospective])
    rows = []
    for name, predictors in model_specs.items():
        beta, seen = (
            _fit_with_receiver_effects(development, predictors, target)
            if predictors
            else (
                np.array(
                    [
                        float(
                            np.mean(
                                [
                                    float(arm[target])
                                    for arm in development
                                    if str(arm["model_key"]) == receiver
                                ]
                            )
                        )
                        for receiver in receivers
                    ]
                ),
                receivers,
            )
        )
        effects = {name_: beta[index] for index, name_ in enumerate(seen)}
        predicted = np.array(
            [effects.get(str(arm["model_key"]), 0.0) for arm in prospective]
        )
        if predictors:
            predicted = predicted + np.column_stack(
                [
                    np.array([float(arm[key]) for arm in prospective])
                    for key in predictors
                ]
            ) @ beta[len(seen) :]
        rows.append(
            {
                "model": name,
                "predictors": "+".join(predictors) if predictors else "receiver mean",
                "arms": len(prospective),
                "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
                "spearman": _correlation(_rankdata(predicted), _rankdata(observed)),
            }
        )
    rows.sort(key=lambda row: row["rmse"])
    return rows


def channel_ceiling_check(
    records_by_rows: dict[int, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Does the channel rate track its own white-noise ceiling as rows grow?

    The measure's maximum is `0.5 * rows * log2(1 + gamma)`, so a statistic
    that were mostly sampling noise would grow in proportion to the row count.
    Comparing the measured value at 64, 128, and 256 rows against that bound
    says how much of the number is structure and how much is the bound.
    """
    from fineqcomp.relative_info import _channel_bits, channel_bits_ceiling

    import torch

    rows = []
    shared = set.intersection(
        *(set(records.keys()) for records in records_by_rows.values())
    )
    for run_id in sorted(shared):
        entry: dict[str, Any] = {"run_id": run_id}
        for count, records in sorted(records_by_rows.items()):
            spectrum = (records[run_id].get("spectra") or {}).get("correction")
            if not spectrum:
                entry = {}
                break
            value = _channel_bits(torch.as_tensor(spectrum, dtype=torch.float64))
            ceiling = channel_bits_ceiling(len(spectrum))
            entry[f"bits_{count}"] = value
            entry[f"ceiling_{count}"] = ceiling
            entry[f"share_of_ceiling_{count}"] = value / ceiling
        if entry:
            rows.append(entry)
    return rows


def _load_records(root: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for path in sorted(Path(root).glob("*.json")):
        record = json.loads(path.read_text())
        records[str(record["run_id"])] = record
    return records


def analyze_rate_law(
    result_root: Path,
    runs_root: Path,
    *,
    row_grid_roots: dict[int, Path] | None = None,
    prospective_arms: Path | None = None,
    permutations: int = 20_000,
) -> dict[str, Any]:
    """One pass over the panel that answers the six questions the audit raised.

    Which candidates add anything to a text statistic and a token count; what
    the codec-referenced forward model predicts; how much of the channel rate
    is its own white-noise ceiling; whether the corpus and the receiver enter
    the token count with opposite signs; and what the retention curve looks
    like away from the single 90% crossing.
    """
    from fineqcomp.relative_info import (
        _aggregate_dataset_arms,
        collect_relative_information_cells,
    )

    cells = collect_relative_information_cells(
        Path(result_root), Path(runs_root), skip_missing_r_star=True
    )
    cells = join_retention_profiles(cells, Path(runs_root))
    records = _load_records(Path(result_root))
    forward = forward_model_predictions(cells, records)
    candidates = tuple(
        key
        for key in cells[0]
        if key not in IDENTIFIER_COLUMNS
        and key not in RETENTION_TARGET_COLUMNS
        and not key.startswith("_")
        and isinstance(cells[0][key], (int, float))
        and all(np.isfinite(float(cell.get(key, float("nan")))) for cell in cells)
    )
    targets = tuple(sorted(RETENTION_TARGET_COLUMNS))
    complete = [
        {key: value for key, value in cell.items() if not key.startswith("_")}
        for cell in cells
        if all(
            cell.get(key) is not None and np.isfinite(float(cell[key]))
            for key in targets
        )
    ]
    arms = _aggregate_dataset_arms(complete, candidates + targets)
    add_tokenizer_fertility(arms)
    gate = candidate_gate(arms, candidates)
    for arm in arms:
        arm["log_train_response_tokens"] = _log2(arm["train_response_tokens"])
        arm["log_channel_deficit"] = _log2(
            arm.get("correction_channel_deficit_bits", float("nan"))
        )
    specs: dict[str, tuple[str, ...]] = {
        "receiver mean only": (),
        "log tokens": ("log_train_response_tokens",),
        "channel bits (incumbent)": ("correction_channel_bits",),
        "log channel deficit": ("log_channel_deficit",),
        "log tokens + log channel deficit": (
            "log_train_response_tokens",
            "log_channel_deficit",
        ),
        "text redundancy": ("text_cross_row_redundancy",),
    }
    comparison = evaluate_rate_models(arms, specs)
    pairs = _receiver_pairs(
        arms, candidates + ("tokenizer_fertility_relative", "log_channel_deficit")
    )
    cross = []
    for key in candidates + ("tokenizer_fertility_relative", "log_channel_deficit"):
        cross.append(
            {
                "candidate": key,
                **_cross_receiver_test(
                    arms, pairs, key, permutations=permutations, seed=20260826
                ),
            }
        )
    cross.sort(
        key=lambda row: -abs(
            row["cross_receiver_spearman"]
            if np.isfinite(row["cross_receiver_spearman"])
            else 0.0
        )
    )
    recorded = []
    if prospective_arms is not None:
        with Path(prospective_arms).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        needed = {"correction_channel_bits", "train_response_tokens"}
        usable = [row for row in rows if needed <= set(row)]
        for row in usable:
            row["correction_channel_deficit_bits"] = channel_bits_ceiling(
                256
            ) - float(row["correction_channel_bits"])
            row["log_channel_deficit"] = _log2(
                row["correction_channel_deficit_bits"]
            )
            row["log_train_response_tokens"] = _log2(row["train_response_tokens"])
        if usable:
            # An arm table written by an earlier campaign carries only the
            # predictors that campaign cared about; score what it supports.
            available = {
                name: predictors
                for name, predictors in specs.items()
                if all(key in usable[0] for key in predictors)
            }
            recorded = score_recorded_arms(arms, usable, available)
    ceiling = (
        channel_ceiling_check(
            {rows: _load_records(root) for rows, root in row_grid_roots.items()}
        )
        if row_grid_roots
        else []
    )
    shapes = [
        {
            "model_key": arm["model_key"],
            "dataset_key": arm["dataset_key"],
            **{key: arm.get(key) for key in ("r50", "r75", "r90", "r95")},
            "shape_r90_over_r50": arm.get("shape_r90_over_r50"),
        }
        for arm in arms
    ]
    finite_shapes = [
        float(row["shape_r90_over_r50"])
        for row in shapes
        if row["shape_r90_over_r50"] is not None
        and np.isfinite(float(row["shape_r90_over_r50"]))
    ]
    return {
        "arms": [
            {key: value for key, value in arm.items() if not key.startswith("_")}
            for arm in arms
        ],
        "candidate_gate": gate,
        "model_comparison": comparison,
        "cross_receiver": cross,
        "forward_model": forward,
        "recorded_arms": recorded,
        "channel_ceiling": ceiling,
        "retention_shape": shapes,
        "summary": {
            "cells": len(cells),
            "arms": len(arms),
            "receivers": len({str(arm["model_key"]) for arm in arms}),
            "cross_receiver_pairs": len(pairs),
            "candidates_screened": len(candidates),
            "candidates_adding_over_controls": sum(
                1 for row in gate if row["adds_over_controls"] and not row["control"]
            ),
            "best_receiver_held_out": comparison[0]["model"],
            "best_receiver_held_out_rmse": comparison[0]["receiver_held_out_rmse"],
            "receiver_mean_rmse": next(
                row["receiver_held_out_rmse"]
                for row in comparison
                if row["model"] == "receiver mean only"
            ),
            "forward_model_arms": len(forward),
            "recorded_arms_scored": len(recorded),
            "retention_shape_min": min(finite_shapes) if finite_shapes else None,
            "retention_shape_max": max(finite_shapes) if finite_shapes else None,
        },
    }


def write_rate_law_report(
    result_root: Path,
    runs_root: Path,
    out_dir: Path,
    *,
    row_grid_roots: dict[int, Path] | None = None,
    prospective_arms: Path | None = None,
    permutations: int = 20_000,
) -> dict[str, Any]:
    """Write every table the rate-law audit produces."""
    result = analyze_rate_law(
        result_root,
        runs_root,
        row_grid_roots=row_grid_roots,
        prospective_arms=prospective_arms,
        permutations=permutations,
    )
    out_dir = Path(out_dir)
    for name, key in (
        ("arms.csv", "arms"),
        ("candidate_gate.csv", "candidate_gate"),
        ("model_comparison.csv", "model_comparison"),
        ("cross_receiver.csv", "cross_receiver"),
        ("forward_model.csv", "forward_model"),
        ("channel_ceiling.csv", "channel_ceiling"),
        ("retention_shape.csv", "retention_shape"),
        ("recorded_arms.csv", "recorded_arms"),
    ):
        _write_csv(out_dir / name, result[key])
    write_json(out_dir / "summary.json", result["summary"])
    return result
