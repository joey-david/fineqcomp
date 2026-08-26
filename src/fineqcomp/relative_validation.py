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
    SPECTRAL_CANDIDATES,
    _correlation,
    _rankdata,
    analyze_relative_information_cells,
    combine_correction_information_area,
    collect_relative_information_cells,
    spectral_candidates_from_record,
)


def select_locked_candidate(analysis: dict[str, Any]) -> dict[str, Any]:
    """Apply the recorded discovery gates and return one spectral candidate."""
    ranking = {row["candidate"]: row for row in analysis["ranking"]}
    base_rmse = ranking["base_codelength_bits_per_token"]["task_heldout_rmse"]
    slopes = {
        (row["measure"], row["task"], row["model_key"]): row[
            "slope_per_doubling"
        ]
        for row in analysis["diversity_slopes"]
    }
    evaluated = []
    for candidate in SPECTRAL_CANDIDATES:
        row = ranking[candidate]
        xbrl = [
            value
            for (measure, task, _), value in slopes.items()
            if measure == candidate and task == "xbrl"
        ]
        sql = [
            value
            for (measure, task, _), value in slopes.items()
            if measure == candidate and task == "sql"
        ]
        gates = {
            "adjusted_p_below_0_05": row["selection_adjusted_p"] < 0.05,
            "beats_model_only": (
                row["task_heldout_rmse"] < row["model_only_rmse"]
            ),
            "beats_base_codelength": row["task_heldout_rmse"] < base_rmse,
            "positive_fixed_effect_relation": (
                row["model_and_task_residual_r"] > 0
            ),
            "positive_xbrl_both_models": len(xbrl) == 2 and min(xbrl) > 0,
            "near_flat_sql_both_models": (
                len(sql) == 2 and max(abs(value) for value in sql) <= 0.02
            ),
        }
        evaluated.append(
            {
                "candidate": candidate,
                "eligible": all(gates.values()),
                "gates": gates,
                "metrics": row,
                "xbrl_slopes": xbrl,
                "sql_slopes": sql,
            }
        )
    eligible = [row for row in evaluated if row["eligible"]]
    if not eligible:
        return {
            "status": "no_candidate_passed",
            "candidate": None,
            "evaluated": evaluated,
        }
    best_rmse = min(row["metrics"]["task_heldout_rmse"] for row in eligible)
    contenders = [
        row
        for row in eligible
        if row["metrics"]["task_heldout_rmse"] <= best_rmse + 0.02
    ]
    rate_distortion = {
        "correction_rd90_bits",
        "angular_correction_rd90_bits",
        "centered_correction_rd90_bits",
        "centered_angular_correction_rd90_bits",
    }
    chosen = max(
        contenders,
        key=lambda row: (
            row["metrics"]["within_model_mean_spearman"],
            row["candidate"] in rate_distortion,
        ),
    )
    return {
        "status": "locked",
        "candidate": chosen["candidate"],
        "selection_rule": (
            "lowest eligible task-heldout RMSE; within 0.02 choose higher"
            " within-model Spearman, then a 90%-rate-distortion form"
        ),
        "chosen": chosen,
        "evaluated": evaluated,
    }


def write_discovery_lock(
    result_root: Path,
    runs_root: Path,
    out_dir: Path,
    *,
    permutations: int = 50_000,
) -> dict[str, Any]:
    """Analyze discovery cells and write the candidate choice before validation."""
    cells = collect_relative_information_cells(result_root, runs_root)
    analysis = analyze_relative_information_cells(cells, permutations=permutations)
    lock = select_locked_candidate(analysis)
    out_dir = Path(out_dir)
    write_json(out_dir / "discovery_summary.json", analysis["summary"])
    write_json(out_dir / "candidate_lock.json", lock)
    _write_csv(out_dir / "discovery_ranking.csv", analysis["ranking"])
    _write_csv(out_dir / "discovery_diversity_slopes.csv", analysis["diversity_slopes"])
    return lock


def _prospective_block(study: str, model: str, adapter_rank: int) -> str | None:
    if adapter_rank != 16:
        return None
    if study in {"sup_full", "sup_reasoning", "sup_answer"}:
        return "cot_mistral"
    if study in {"qwen_sup_full", "qwen_sup_reasoning", "qwen_sup_answer"}:
        return "cot_qwen"
    if study.startswith(("plain_", "preamble_", "numbered_", "shouted_", "symbolic_")):
        return "response_form"
    if study in {"div_400", "div_1200", "div_3600"}:
        return "fixed_row_content"
    if study in {"arm_a", "arm_b", "arm_c", "arm_d", "arm_e"}:
        return "fixed_compute_rows"
    return None


def _prospective_arms(
    cells: list[dict[str, Any]], candidate_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in cells:
        block = _prospective_block(
            str(row["study"]), str(row["model_key"]), int(row["adapter_rank"])
        )
        if block is None:
            continue
        groups.setdefault(
            (block, str(row["model_key"]), str(row["study"])), []
        ).append(row)
    arms = []
    for (block, model, study), rows in sorted(groups.items()):
        arms.append(
            {
                "block": block,
                "model_key": model,
                "study": study,
                "dataset_key": rows[0]["dataset_key"],
                "seeds": len(rows),
                "r_star_bits_per_value": float(
                    np.mean([row["r_star_bits_per_value"] for row in rows])
                ),
                **{
                    key: float(np.mean([row[key] for row in rows]))
                    for key in candidate_keys
                },
            }
        )
    return arms


def _block_rank_vectors(
    arms: list[dict[str, Any]], candidate: str
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    candidate_ranks = []
    target_ranks = []
    for block in sorted({str(row["block"]) for row in arms}):
        rows = [row for row in arms if row["block"] == block]
        candidate_ranks.append(
            _rankdata(np.asarray([row[candidate] for row in rows], dtype=float))
        )
        target_ranks.append(
            _rankdata(
                np.asarray([row["r_star_bits_per_value"] for row in rows], dtype=float)
            )
        )
    return candidate_ranks, target_ranks


def _pooled_centered_correlation(
    left: list[np.ndarray], right: list[np.ndarray]
) -> float:
    centered_left = np.concatenate([values - values.mean() for values in left])
    centered_right = np.concatenate([values - values.mean() for values in right])
    return _correlation(centered_left, centered_right)


def _blockwise_permutation_test(
    arms: list[dict[str, Any]], candidate: str, permutations: int, seed: int
) -> tuple[float, float]:
    left, right = _block_rank_vectors(arms, candidate)
    observed = _pooled_centered_correlation(left, right)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        permuted = [rng.permutation(values) for values in right]
        exceed += _pooled_centered_correlation(left, permuted) >= observed
    return observed, float((exceed + 1) / (permutations + 1))


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


def analyze_locked_prospective(
    discovery_cells: list[dict[str, Any]],
    prospective_cells: list[dict[str, Any]],
    candidate: str,
    *,
    permutations: int = 100_000,
    seed: int = 1729,
) -> dict[str, Any]:
    """Test one locked candidate without ranking alternatives."""
    candidate_keys = (candidate, "base_codelength_bits_per_token")
    prospective = _prospective_arms(prospective_cells, candidate_keys)
    if len(prospective) != 24:
        raise ValueError(f"expected 24 prospective arms, found {len(prospective)}")

    # Use the discovery reduction already defined for selection, including its
    # exact-data deduplication, but carry only the locked and baseline measures.
    discovery_analysis = analyze_relative_information_cells(
        discovery_cells, permutations=100, seed=seed
    )
    discovery = [
        {
            "model_key": row["model_key"],
            "r_star_bits_per_value": row["r_star_bits_per_value"],
            candidate: row[candidate],
            "base_codelength_bits_per_token": row[
                "base_codelength_bits_per_token"
            ],
        }
        for row in discovery_analysis["arms"]
    ]
    truth = np.asarray(
        [row["r_star_bits_per_value"] for row in prospective], dtype=float
    )
    prediction = _fit_prediction(
        discovery,
        prospective,
        candidate,
        log_candidate=candidate
        not in {"dataset_fisher_log_volume", "correction_information_area"},
    )
    base_prediction = _fit_prediction(
        discovery, prospective, "base_codelength_bits_per_token"
    )
    model_prediction = _fit_prediction(discovery, prospective, None)
    rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
    base_rmse = float(np.sqrt(np.mean((base_prediction - truth) ** 2)))
    model_rmse = float(np.sqrt(np.mean((model_prediction - truth) ** 2)))
    rho, p_value = _blockwise_permutation_test(
        prospective, candidate, permutations, seed
    )
    blocks = []
    for block in sorted({str(row["block"]) for row in prospective}):
        rows = [row for row in prospective if row["block"] == block]
        x = np.log(np.maximum([row[candidate] for row in rows], 1e-12))
        y = np.asarray([row["r_star_bits_per_value"] for row in rows])
        slope = 0.0 if np.ptp(x) <= 1e-12 else float(np.polyfit(x, y, 1)[0])
        blocks.append(
            {
                "block": block,
                "arms": len(rows),
                "slope": slope,
                "spearman": _correlation(_rankdata(x), _rankdata(y)),
            }
        )
    leave_one_out = {}
    for omitted in sorted({str(row["block"]) for row in prospective}):
        rows = [row for row in prospective if row["block"] != omitted]
        left, right = _block_rank_vectors(rows, candidate)
        leave_one_out[omitted] = _pooled_centered_correlation(left, right)
    gates = {
        "primary_rho_at_least_0_60": rho >= 0.60,
        "primary_p_below_0_01": p_value < 0.01,
        "rmse_20pct_below_model_only": rmse <= 0.80 * model_rmse,
        "rmse_20pct_below_base_codelength": rmse <= 0.80 * base_rmse,
        "positive_slope_every_block": min(row["slope"] for row in blocks) > 0,
        "positive_after_removing_any_block": min(leave_one_out.values()) > 0,
    }
    return {
        "status": "passed" if all(gates.values()) else "failed",
        "candidate": candidate,
        "prospective_arms": len(prospective),
        "blockwise_rank_rho": rho,
        "blockwise_permutation_p": p_value,
        "prospective_rmse": rmse,
        "model_only_rmse": model_rmse,
        "base_codelength_rmse": base_rmse,
        "gates": gates,
        "blocks": blocks,
        "leave_one_block_out_rho": leave_one_out,
        "arms": prospective,
    }


def analyze_external_receiver(
    discovery_cells: list[dict[str, Any]],
    external_cells: list[dict[str, Any]],
    candidate: str,
    *,
    min_natural_arms: int = 5,
) -> dict[str, Any]:
    """Apply a fixed discovery fit to one unseen model and its eight arms."""
    if min_natural_arms < 1:
        raise ValueError("min_natural_arms must be positive")
    discovery_analysis = analyze_relative_information_cells(
        discovery_cells, permutations=100
    )
    discovery = [
        {
            "model_key": row["model_key"],
            "r_star_bits_per_value": row["r_star_bits_per_value"],
            candidate: row[candidate],
            "base_codelength_bits_per_token": row[
                "base_codelength_bits_per_token"
            ],
        }
        for row in discovery_analysis["arms"]
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in external_cells:
        groups.setdefault(
            (str(row["study"]), str(row["dataset_key"])), []
        ).append(row)
    arms = []
    for (study, dataset), rows in sorted(groups.items()):
        arms.append(
            {
                "study": study,
                "dataset_key": dataset,
                "model_key": rows[0]["model_key"],
                "seeds": len(rows),
                "r_star_bits_per_value": float(
                    np.mean([row["r_star_bits_per_value"] for row in rows])
                ),
                candidate: float(np.mean([row[candidate] for row in rows])),
                "base_codelength_bits_per_token": float(
                    np.mean(
                        [row["base_codelength_bits_per_token"] for row in rows]
                    )
                ),
            }
        )
    natural_studies = sorted(
        {str(row["study"]) for row in arms if str(row["study"]).endswith("_natural")}
    )
    if len(natural_studies) != 1:
        raise ValueError(
            "expected one receiver natural study, found "
            f"{natural_studies}"
        )
    receiver_prefix = natural_studies[0][:-len("_natural")]
    natural = [row for row in arms if row["study"] == natural_studies[0]]
    cot = [
        row
        for row in arms
        if str(row["study"]).startswith(f"{receiver_prefix}_cot_")
    ]
    if len(natural) < min_natural_arms:
        raise ValueError(
            f"expected at least {min_natural_arms} natural arms, found"
            f" {len(natural)}"
        )

    truth = np.asarray([row["r_star_bits_per_value"] for row in natural])
    candidate_prediction = _fit_prediction(
        discovery, natural, candidate, log_candidate=False
    )
    base_prediction = _fit_prediction(
        discovery, natural, "base_codelength_bits_per_token"
    )
    candidate_rmse = float(
        np.sqrt(np.mean((candidate_prediction - truth) ** 2))
    )
    base_rmse = float(np.sqrt(np.mean((base_prediction - truth) ** 2)))
    candidate_ranks = _rankdata(
        np.asarray([row[candidate] for row in natural], dtype=float)
    )
    target_ranks = _rankdata(truth)
    rho = _correlation(candidate_ranks, target_ranks)
    null = [
        _correlation(candidate_ranks, np.asarray(permutation, dtype=float))
        for permutation in itertools.permutations(target_ranks)
    ]
    rank_p = float(np.mean(np.asarray(null) >= rho))
    cot_rho = None
    if len(cot) == 3:
        cot_rho = _correlation(
            _rankdata(np.asarray([row[candidate] for row in cot], dtype=float)),
            _rankdata(
                np.asarray(
                    [row["r_star_bits_per_value"] for row in cot], dtype=float
                )
            ),
        )
    gates = {
        "natural_spearman_at_least_0_70": rho >= 0.70,
        "prefit_rmse_beats_base_codelength": candidate_rmse < base_rmse,
    }
    for row, prediction, base in zip(
        natural, candidate_prediction, base_prediction, strict=True
    ):
        row["prefit_prediction"] = float(prediction)
        row["base_codelength_prediction"] = float(base)
    return {
        "status": "passed" if all(gates.values()) else "failed",
        "candidate": candidate,
        "natural_arms_available": len(natural),
        "natural_arms_required": min_natural_arms,
        "natural_spearman": rho,
        "natural_exact_permutation_p": rank_p,
        "natural_prefit_rmse": candidate_rmse,
        "natural_base_codelength_rmse": base_rmse,
        "cot_spearman": cot_rho,
        "cot_arms_available": len(cot),
        "gates": gates,
        "arms": arms,
    }


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


def write_external_receiver_report(
    discovery_root: Path,
    external_root: Path,
    runs_root: Path,
    out_dir: Path,
    candidate: str,
    *,
    min_natural_arms: int = 5,
) -> dict[str, Any]:
    discovery = collect_relative_information_cells(discovery_root, runs_root)
    external = collect_relative_information_cells(
        external_root, runs_root, skip_missing_r_star=True
    )
    result = analyze_external_receiver(
        discovery,
        external,
        candidate,
        min_natural_arms=min_natural_arms,
    )
    out_dir = Path(out_dir)
    write_json(
        out_dir / "external_summary.json",
        {key: value for key, value in result.items() if key != "arms"},
    )
    _write_csv(out_dir / "external_arms.csv", result["arms"])
    return result


def analyze_measurement_stability(
    settings: dict[str, dict[str, float]], reference_setting: str
) -> dict[str, Any]:
    """Check magnitude and arm ranks across row and sketch settings."""
    if reference_setting not in settings:
        raise ValueError(f"missing reference setting {reference_setting}")
    run_ids = set(settings[reference_setting])
    if not run_ids or any(set(values) != run_ids for values in settings.values()):
        raise ValueError("every stability setting must contain the same runs")
    ordered = sorted(run_ids)
    reference = np.asarray(
        [settings[reference_setting][run_id] for run_id in ordered]
    )
    setting_rows = []
    for setting, values in sorted(settings.items()):
        current = np.asarray([values[run_id] for run_id in ordered])
        setting_rows.append(
            {
                "setting": setting,
                "runs": len(ordered),
                "rank_spearman_vs_reference": _correlation(
                    _rankdata(current), _rankdata(reference)
                ),
            }
        )
    arm_rows = []
    for run_id in ordered:
        values = np.asarray(
            [setting[run_id] for setting in settings.values()], dtype=float
        )
        arm_rows.append(
            {
                "run_id": run_id,
                "mean": float(values.mean()),
                "coefficient_of_variation": float(
                    values.std(ddof=1) / values.mean()
                ),
            }
        )
    median_cv = float(
        np.median([row["coefficient_of_variation"] for row in arm_rows])
    )
    minimum_rank = min(
        row["rank_spearman_vs_reference"] for row in setting_rows
    )
    gates = {
        "median_arm_cv_below_0_05": median_cv < 0.05,
        "minimum_rank_spearman_at_least_0_95": minimum_rank >= 0.95,
    }
    return {
        "status": "passed" if all(gates.values()) else "failed",
        "settings": len(settings),
        "arms": len(ordered),
        "reference_setting": reference_setting,
        "median_arm_cv": median_cv,
        "minimum_rank_spearman": minimum_rank,
        "gates": gates,
        "setting_rows": setting_rows,
        "arm_rows": arm_rows,
    }


def write_measurement_stability_report(
    stability_root: Path,
    reference_root: Path,
    out_dir: Path,
    candidate: str,
) -> dict[str, Any]:
    settings: dict[str, dict[str, float]] = {}
    for directory in sorted(Path(stability_root).glob("r*")):
        values = {}
        for path in sorted(directory.glob("*.json")):
            measured = json.loads(path.read_text())
            values[str(measured["run_id"])] = spectral_candidates_from_record(
                measured
            )[candidate]
        if values:
            settings[directory.name] = values
    selected_ids = set().union(*(set(values) for values in settings.values()))
    reference = {}
    for path in Path(reference_root).glob("*.json"):
        measured = json.loads(path.read_text())
        run_id = str(measured["run_id"])
        if run_id in selected_ids:
            reference[run_id] = spectral_candidates_from_record(measured)[candidate]
    settings["r256_s1729"] = reference
    result = analyze_measurement_stability(settings, "r256_s1729")
    out_dir = Path(out_dir)
    write_json(
        out_dir / "stability_summary.json",
        {
            key: value
            for key, value in result.items()
            if key not in {"setting_rows", "arm_rows"}
        },
    )
    _write_csv(out_dir / "stability_settings.csv", result["setting_rows"])
    _write_csv(out_dir / "stability_arms.csv", result["arm_rows"])
    return result


def collect_correction_information_area(
    roots: dict[int, Path],
    runs_root: Path,
    *,
    skip_missing_r_star: bool = False,
) -> list[dict[str, Any]]:
    return combine_correction_information_area(
        {
            rows: collect_relative_information_cells(
                root,
                runs_root,
                skip_missing_r_star=skip_missing_r_star,
            )
            for rows, root in roots.items()
        }
    )


def write_information_area_report(
    discovery_roots: dict[int, Path],
    evaluation_roots: dict[int, Path],
    runs_root: Path,
    out_dir: Path,
    mode: str,
    *,
    permutations: int = 100_000,
) -> dict[str, Any]:
    discovery = collect_correction_information_area(
        discovery_roots, runs_root
    )
    evaluation = collect_correction_information_area(
        evaluation_roots,
        runs_root,
        skip_missing_r_star=mode == "development",
    )
    candidate = "correction_information_area"
    if mode == "development":
        result = analyze_locked_prospective(
            discovery,
            evaluation,
            candidate,
            permutations=permutations,
        )
    elif mode == "external":
        result = analyze_external_receiver(discovery, evaluation, candidate)
    else:
        raise ValueError("mode must be development or external")
    out_dir = Path(out_dir)
    write_json(
        out_dir / f"{mode}_area_summary.json",
        {key: value for key, value in result.items() if key != "arms"},
    )
    _write_csv(out_dir / f"{mode}_area_arms.csv", result["arms"])
    return result


def write_locked_prospective_report(
    discovery_root: Path,
    prospective_root: Path,
    runs_root: Path,
    lock_path: Path,
    out_dir: Path,
    *,
    permutations: int = 100_000,
) -> dict[str, Any]:
    lock = json.loads(Path(lock_path).read_text())
    if lock.get("status") != "locked" or not lock.get("candidate"):
        raise ValueError("candidate lock did not select a measure")
    discovery = collect_relative_information_cells(discovery_root, runs_root)
    prospective = collect_relative_information_cells(
        prospective_root, runs_root, skip_missing_r_star=True
    )
    result = analyze_locked_prospective(
        discovery,
        prospective,
        str(lock["candidate"]),
        permutations=permutations,
    )
    out_dir = Path(out_dir)
    write_json(out_dir / "prospective_summary.json", {k: v for k, v in result.items() if k != "arms"})
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
