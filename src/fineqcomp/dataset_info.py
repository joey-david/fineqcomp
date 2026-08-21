"""Two cheap measures of how much information a training set carries.

The compressibility experiment varies unique rows while holding tokens and
optimizer steps fixed, so the arms differ only in how much of what they see is
new. Labelling them 1x/2x/4x/8x assumes the answer; these measure it.

The two measures disagree by construction, and the disagreement is the point:

  lzma bits          duplication-aware. Eight copies of a 4k set compress to
                     roughly the size of one, so this tracks unique content.
  base-model bits    duplication-blind. A static code charges the second copy
                     of an example exactly what it charged the first, so this
                     tracks tokens processed.

If R* follows the first curve and not the second, adaptation bits track unique
information rather than training volume, which is the hypothesis under test.

It has to be lzma, not zlib. zlib matches within a 32 KB window, and at these
sizes duplicated rows sit tens of kilobytes apart, so it sees nothing: on four
arms holding 500/1000/2000/4000 unique rows in 4000 rows of text it returned
2.79 Mbit for every one of them. lzma's dictionary spans the whole file and
returned 0.32/0.62/1.22/2.41 Mbit, which is the signal being measured. bz2
saturates too, at its 900 KB block size. zlib is still reported, as the control
that shows why window size decides the answer.
"""

from __future__ import annotations

import json
import lzma
import zlib
from pathlib import Path
from typing import Any

import torch

from fineqcomp.data import Example, read_jsonl


def corpus_bits(rows: list[Example]) -> dict[str, Any]:
    """Compressed size of the training text under a long- and short-range coder."""
    text = "\n".join(f"{row.prompt}\n{row.response}" for row in rows).encode()
    long_range = lzma.compress(text, preset=6)
    short_range = zlib.compress(text, level=9)
    distinct = len({(row.prompt, row.response) for row in rows})
    return {
        "rows": len(rows),
        "distinct_rows": distinct,
        "raw_bits": len(text) * 8,
        "lzma_bits": len(long_range) * 8,
        "lzma_bits_per_row": len(long_range) * 8 / max(len(rows), 1),
        "lzma_bits_per_distinct_row": len(long_range) * 8 / max(distinct, 1),
        "zlib_bits": len(short_range) * 8,
        "compression_ratio": len(text) / max(len(long_range), 1),
    }


@torch.no_grad()
def base_model_bits(
    model: Any,
    tokenizer: Any,
    rows: list[Example],
    model_spec: Any,
    max_length: int,
    micro_batch_size: int,
) -> dict[str, Any]:
    """Code length of the training set under the frozen base model.

    This is `sum -log2 p(response | prompt)` with no adaptation, so it says how
    many bits the base model already needs for this data. Duplicated rows are
    charged in full.
    """
    from fineqcomp.training import causal_nll

    measured = causal_nll(
        model, tokenizer, rows, model_spec, max_length, micro_batch_size
    )
    return {
        "rows": len(rows),
        "base_total_bits": float(measured["total_bits"]),
        "base_bits_per_token": float(measured["bits_per_token"]),
        "base_bits_per_row": float(measured["total_bits"]) / max(len(rows), 1),
        "tokens": int(measured["nll_tokens"]),
    }


def describe(prepared_dir: Path, sample_rows: int | None = None) -> dict[str, Any]:
    """zlib statistics for one prepared training split."""
    rows = read_jsonl(prepared_dir / "train.jsonl")
    if sample_rows is not None:
        rows = rows[:sample_rows]
    return {"path": str(prepared_dir), **corpus_bits(rows)}


def arm_keys(runs: list[Any]) -> list[tuple[str, int, int]]:
    """The distinct (dataset, seed, epochs) triples a manifest trains on.

    Arms that differ only in seed still draw different rows, so the measure has
    to be taken per seed rather than once per dataset.
    """
    seen = {
        (str(run.dataset_key), int(run.seed), int(run.training.epochs))
        for run in runs
        if run.dataset_key is not None
    }
    return sorted(seen)


def measure_arms(
    campaign: dict[str, Any],
    runs: list[Any],
    prepared_root: str | Path | None = None,
    session: Any = None,
    sample_rows: int = 1024,
    max_length: int = 1024,
    micro_batch_size: int = 4,
) -> list[dict[str, Any]]:
    """Both information measures for every arm in a manifest.

    The duplication-aware measure compresses the arm's distinct rows: an arm
    holding 2,000 of them carries less unique content than one holding 32,000,
    and lzma prices that content rather than counting rows, which is the whole
    point -- MetaMathQA rows are templated and redundant among themselves, so
    row count and information are not proportional.

    The duplication-blind control is the base model's code length for every
    sample the run processes, copies included. A static code charges the
    sixteenth copy of a row exactly what it charged the first, so this measure
    is flat across compute-matched arms by construction. R* following the first
    and not the second is the result the campaign is after.

    Without a `session` only the compressed measure is taken, which needs no
    GPU and no model.
    """
    from fineqcomp.data import load_natural_dataset

    records = []
    for dataset_key, seed, epochs in arm_keys(runs):
        rows = load_natural_dataset(campaign, dataset_key, seed, prepared_root)["train"]
        record = {
            "dataset_key": dataset_key,
            "seed": seed,
            "epochs": epochs,
            "samples_seen": len(rows) * epochs,
            **corpus_bits(rows),
        }
        if session is not None:
            base = base_model_bits(
                session.model,
                session.tokenizer,
                rows[:sample_rows],
                session.spec,
                max_length,
                micro_batch_size,
            )
            record["base_sample_rows"] = base["rows"]
            record["base_bits_per_row"] = base["base_bits_per_row"]
            record["base_stream_bits"] = base["base_bits_per_row"] * record[
                "samples_seen"
            ]
        records.append(record)
    return records


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="compressed-size statistics for prepared training splits"
    )
    parser.add_argument("prepared_dirs", type=Path, nargs="+")
    parser.add_argument("--rows", type=int, help="cap rows, to mimic a train_rows arm")
    args = parser.parse_args(argv)
    print(json.dumps([describe(d, args.rows) for d in args.prepared_dirs], indent=2))


if __name__ == "__main__":
    main()
