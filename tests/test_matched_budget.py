from __future__ import annotations

import pytest
import torch

from fineqcomp.codec import decode_adapter_tensor_map
from fineqcomp.matched_budget import (
    budget_units,
    candidate_grid,
    candidate_key,
    check,
    encode_candidate,
    matched_sets,
    scratch_versus_compressed,
)


def _pair(out_features: int, in_features: int, rank: int, seed: int = 0):
    torch.manual_seed(seed)
    weights = torch.linspace(4.0, 1.0, rank)
    left = torch.linalg.qr(torch.randn(out_features, rank))[0] * weights[None, :]
    right = torch.linalg.qr(torch.randn(in_features, rank))[0].T
    return {
        "layer.lora_A.default.weight": right,
        "layer.lora_B.default.weight": left,
    }


def _row(rank: int, bits: int, accuracy: float | None, file_bits: int | None = None):
    return {
        "key": candidate_key(rank, bits),
        "rank": rank,
        "bits": bits,
        "budget_units": budget_units(rank, bits),
        "file_bits": file_bits if file_bits is not None else rank * bits * 1000,
        "gsm8k_accuracy": accuracy,
    }


def test_grid_drops_the_ranks_an_adapter_does_not_have():
    """A rank above the trained one is the same tensors under another name.

    `truncate_lora_rank` clamps, so asking a rank-4 adapter for ranks 8 and 16
    returns its own factors three times over. Scoring the repeats would spend
    two extra full GSM8K passes to re-measure one number, so they are removed
    here rather than de-duplicated after the GPU time is gone.
    """
    grid = candidate_grid(4, [1, 2, 4, 8, 16], [1, 16])
    assert grid == [(1, 1), (1, 16), (2, 1), (2, 16), (4, 1), (4, 16)]
    assert candidate_grid(16, [1, 2, 4, 8, 16], [1, 2, 4, 8, 16]) == sorted(
        (rank, bits) for rank in (1, 2, 4, 8, 16) for bits in (1, 2, 4, 8, 16)
    )


def test_grid_rejects_bit_widths_the_codec_cannot_write():
    """A rung the codec refuses must fail at prepare time, not eight hours in."""
    with pytest.raises(ValueError):
        candidate_grid(16, [1, 4], [1, 5])
    with pytest.raises(ValueError):
        candidate_grid(16, [1, 4], [0])


def test_equal_budgets_buy_equal_payloads(tmp_path):
    """The diagonal is only a fair comparison if its cells really are the same size.

    Rank-4 at four bits and rank-16 at one bit both spend sixteen units, and the
    experiment reads any accuracy difference between them as an effect of how
    the bits were split. That reading needs the two files to hold the same
    number of coded bits, which is what this measures: the coded payload is
    equal across the diagonal, while the whole file differs by the container --
    per-row scales and a header that do not shrink with rank.
    """
    tensors = _pair(64, 48, 16)
    payload, files = {}, {}
    for rank, bits in [(1, 16), (2, 8), (4, 4), (8, 2), (16, 1)]:
        storage, _ = encode_candidate(
            tensors, rank, bits, tmp_path / f"r{rank}_b{bits}.fqcb"
        )
        payload[(rank, bits)] = storage["value_bits"]
        files[(rank, bits)] = storage["file_bits"]
    assert len(set(payload.values())) == 1
    assert max(files.values()) / min(files.values()) < 2.5


def test_a_wider_code_at_one_rank_always_costs_more(tmp_path):
    """Precision has to buy bytes for `rank x bits` to name a budget at all."""
    tensors = _pair(64, 48, 16)
    sizes = [
        encode_candidate(tensors, 16, bits, tmp_path / f"b{bits}.fqcb")[0]["file_bits"]
        for bits in (1, 2, 4, 8, 16)
    ]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)


def test_encoded_candidate_is_scored_through_the_file(tmp_path):
    """What the decoder returns is what gets scored, not the pre-encode tensors.

    The claim under test is about what survives a file of a given size, so a
    sweep that scored the truncated-but-unquantized tensors would answer a
    different question. At one bit the reconstruction must differ from the
    input; at sixteen it must not.
    """
    tensors = _pair(32, 24, 8)
    _, coarse = encode_candidate(tensors, 8, 1, tmp_path / "coarse.fqcb")
    _, exact = encode_candidate(tensors, 8, 16, tmp_path / "exact.fqcb")
    name = "layer.lora_A.default.weight"
    assert not torch.allclose(coarse[name], tensors[name], atol=1e-3)
    _, reread = decode_adapter_tensor_map(tmp_path / "exact.fqcb")
    assert torch.allclose(reread[name], exact[name])


def test_matched_sets_report_the_spread_across_ways_of_spending_a_budget():
    """The spread is the experiment's readout, and singletons are not evidence.

    A budget met by one cell cannot say whether the split matters, so it is
    dropped rather than reported with a zero spread, which would read as
    "splitting the bits made no difference".
    """
    rows = [_row(16, 1, 0.83), _row(4, 4, 0.60), _row(1, 16, 0.42), _row(16, 16, 0.24)]
    sets = matched_sets(rows)
    assert [entry["budget_units"] for entry in sets] == [16]
    entry = sets[0]
    assert entry["best_key"] == "r16_b1" and entry["best_rank"] == 16
    assert entry["spread"] == pytest.approx(0.41)
    assert entry["cells"] == ["r1_b16", "r4_b4", "r16_b1"]


def test_matched_sets_carry_the_file_sizes_that_did_not_match():
    """Nominal budgets tie exactly; real files do not, and the gap is reported.

    Per-row scales and the container header do not shrink with rank, so a
    rank-16 one-bit file is not byte-identical to a rank-1 sixteen-bit one.
    Hiding that would let a size difference be read as a coding effect.
    """
    rows = [_row(16, 1, 0.83, file_bits=1000), _row(1, 16, 0.42, file_bits=1600)]
    assert matched_sets(rows)[0]["file_bits_ratio"] == pytest.approx(1.6)


def test_unscored_cells_do_not_become_a_best_cell():
    """A partially swept budget reports what it has without inventing a winner."""
    sets = matched_sets([_row(16, 1, None), _row(1, 16, None)])
    assert sets[0]["best_key"] is None and sets[0]["spread"] is None


def _cell(arm: str, rank: int, seed: int):
    return {
        "arm": arm,
        "trained_rank": rank,
        "slug": f"qwen/{arm}/seed{seed}",
        "run": {"seed": seed, "model": {"key": "qwen"}},
    }


def test_trained_low_rank_is_paired_with_truncation_at_the_same_budget():
    """The comparison the compression sweep cannot make on its own.

    A rank-2 adapter trained at full precision spends 32 units. So does the
    rank-16 adapter re-coded to two bits, and the rank-4 one at eight. Pairing
    them is what separates "this update is small" from "this update was cut
    down", and the pairing must never include the trained cell against itself.
    """
    cells = [_cell("permuted_r2", 2, 11), _cell("permuted_r16", 16, 11)]
    rows = {
        "qwen/permuted_r2/seed11": [_row(2, 16, 0.55), _row(2, 1, 0.70)],
        "qwen/permuted_r16/seed11": [_row(16, 2, 0.81), _row(8, 4, 0.78), _row(16, 16, 0.24)],
    }
    pairs = scratch_versus_compressed(cells, rows)
    assert {pair["compressed_key"] for pair in pairs} == {"r16_b2", "r8_b4"}
    assert all(pair["budget_units"] == 32 for pair in pairs)
    assert all(pair["trained_accuracy"] == pytest.approx(0.55) for pair in pairs)
    best = max(pairs, key=lambda pair: pair["compressed_accuracy"])
    assert best["trained_minus_compressed"] == pytest.approx(0.55 - 0.81)


def test_pairing_skips_a_seed_whose_wide_arm_is_missing():
    """A seed with no rank-16 partner yields no comparison rather than a cross-seed one."""
    cells = [_cell("permuted_r2", 2, 11), _cell("permuted_r16", 16, 22)]
    rows = {
        "qwen/permuted_r2/seed11": [_row(2, 16, 0.55)],
        "qwen/permuted_r16/seed22": [_row(16, 2, 0.81)],
    }
    assert scratch_versus_compressed(cells, rows) == []


def test_check_names_the_adapters_the_study_does_not_have(tmp_path):
    """Missing training is a non-zero exit before a GPU hour, not an hour in."""
    lock = {"cells": [
        {"slug": "qwen/permuted_r2/seed11", "grid": [[2, 16]],
         "run": {"run_id": "present"}},
        {"slug": "qwen/permuted_r4/seed11", "grid": [[4, 16], [4, 1]],
         "run": {"run_id": "absent"}},
    ]}
    (tmp_path / "present").mkdir()
    (tmp_path / "present" / "raw_channel.pt").write_bytes(b"")
    result = check(lock, tmp_path)
    assert result["passed"] is False
    assert result["missing_adapters"] == ["qwen/permuted_r4/seed11"]
    assert result["candidates"] == 3


def _write_split(root, label, split, accuracy, rows=8):
    """A scored split: the metric file and the per-question predictions beside it."""
    import json

    root.mkdir(parents=True, exist_ok=True)
    correct = round(accuracy * rows)
    (root / f"{label}_{split}.json").write_text(json.dumps({"exact_match": accuracy}))
    (root / f"{label}_{split}.jsonl").write_text(
        "\n".join(
            json.dumps({"example_id": f"{split}-{i}", "cluster": f"t{i % 3}",
                        "correct": i < correct})
            for i in range(rows)
        )
        + "\n"
    )


def test_report_joins_a_finished_study_without_a_gpu(tmp_path):
    """Exercise the join the study only reaches after every GPU hour is spent.

    `report` is the one stage that nothing else runs: prepare, check and the
    sweep all execute long before it, so a shape error here would surface at the
    end of the campaign rather than the start. This builds a two-arm, one-seed
    study on disk with the files the sweep would have written and asserts the
    three outputs the write-up reads -- the per-cell rows, the equal-budget
    diagonals, and the trained-versus-truncated pairing.
    """
    import json

    config = {
        "candidate_splits": ["gsm8k"],
        "bootstrap_draws": 200,
        "analysis_seed": 1,
    }
    cells = []
    for arm, rank, rows in (
        ("permuted_r2", 2, [_row(2, 16, 0.55), _row(2, 1, 0.70), _row(1, 16, 0.60)]),
        ("permuted_r16", 16, [_row(16, 2, 0.81), _row(8, 4, 0.78), _row(16, 16, 0.24)]),
    ):
        slug = f"qwen/{arm}/seed11"
        cell = {
            "arm": arm, "trained_rank": rank, "slug": slug,
            "run": {"seed": 11, "model": {"key": "qwen"},
                    "run_id": f"{arm}-s11", "adapter": {"rank": rank}},
        }
        cells.append(cell)
        root = tmp_path / slug
        root.mkdir(parents=True)
        (root / "complete.json").write_text(json.dumps({"complete": True}))
        (root / "candidates.json").write_text(json.dumps(rows))
        for split in ("calibration", "gsm8k", "symbolic"):
            _write_split(tmp_path / "qwen" / "base" / "seed11", "base", split, 0.75)
            _write_split(root, "raw", split, 0.24)
        for row in rows:
            _write_split(root, row["key"], "gsm8k", row["gsm8k_accuracy"])

    from fineqcomp.matched_budget import report

    summary = report({"config": config, "cells": cells}, tmp_path)
    assert summary["status"] == "complete" and summary["missing"] == []
    assert len(summary["rows"]) == 6
    wide = next(r for r in summary["rows"] if r["key"] == "r16_b2")
    assert wide["gsm8k_change_from_base"] == pytest.approx(0.81 - 0.75)
    assert wide["gsm8k_change_from_raw"] == pytest.approx(0.81 - 0.24)
    assert len(wide["gsm8k_change_from_base_ci95"]) == 2
    # The rank-2 arm's own fp16 cell spends 32 units; so do r16 at two bits and
    # r8 at four, which is the pairing the compression sweep cannot make alone.
    assert {p["compressed_key"] for p in summary["scratch_versus_compressed"]} == {
        "r16_b2", "r8_b4"
    }
    budgets = {(d["arm"], d["budget_units"]) for d in summary["matched_budgets"]}
    assert ("permuted_r16", 32) in budgets


def test_an_unswept_wider_arm_does_not_suppress_the_pairing():
    """A declared-but-unreached arm must not silently empty the comparison.

    The lock names every arm the study will ever have. A night that runs out of
    time leaves the widest of them unswept, and choosing the widest *declared*
    rank as the truncation partner then points every pairing at an arm with no
    rows. The result is an empty list, which reads like "measured, no effect"
    when it means "never measured". The partner has to be the widest arm that
    actually has data.
    """
    cells = [_cell("permuted_r2", 2, 11), _cell("permuted_r16", 16, 11),
             _cell("permuted_r64", 64, 11)]
    rows = {
        "qwen/permuted_r2/seed11": [_row(2, 16, 0.55)],
        "qwen/permuted_r16/seed11": [_row(16, 2, 0.81)],
        # permuted_r64 was declared in the lock but never swept.
    }
    pairs = scratch_versus_compressed(cells, rows)
    assert [p["compressed_key"] for p in pairs] == ["r16_b2"]
    assert pairs[0]["compressed_from_rank"] == 16
