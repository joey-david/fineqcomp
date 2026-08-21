from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from fineqcomp.data import Example
from fineqcomp.dataset_info import arm_keys, corpus_bits, measure_arms
from fineqcomp.rstar import _fit, collect


def _rows(distinct: int) -> list[Example]:
    # Templated text, like MetaMathQA: rows share most of their tokens, which
    # is exactly why counting them is a poor proxy for what they contain.
    return [
        Example(
            example_id=f"r{index}",
            prompt=f"Compute the value of {index} times seven, showing your work.",
            response=f"Seven times {index} is {index * 7}. The answer is: {index * 7}",
            metadata={},
        )
        for index in range(distinct)
    ]


def _run(dataset_key: str, seed: int, epochs: int):
    return SimpleNamespace(
        dataset_key=dataset_key, seed=seed, training=SimpleNamespace(epochs=epochs)
    )


def test_arm_keys_collapse_runs_to_distinct_training_sets():
    runs = [
        _run("arm_a", 11, 8),
        _run("arm_a", 11, 8),
        _run("arm_a", 22, 8),
        _run("arm_d", 11, 1),
        SimpleNamespace(dataset_key=None, seed=11, training=SimpleNamespace(epochs=1)),
    ]
    assert arm_keys(runs) == [("arm_a", 11, 8), ("arm_a", 22, 8), ("arm_d", 11, 1)]


def test_compression_prices_content_while_row_count_only_counts_rows():
    few, many = corpus_bits(_rows(100)), corpus_bits(_rows(800))

    assert few["distinct_rows"] == 100 and many["distinct_rows"] == 800
    # Eight times the rows, but the rows are templated, so the compressed size
    # grows by far less than eight. That gap is the whole reason a row count is
    # the wrong measure of how much a training set carries.
    assert many["lzma_bits"] > few["lzma_bits"]
    assert many["lzma_bits"] < 8 * few["lzma_bits"]


def test_the_two_measures_separate_duplication_from_volume(monkeypatch):
    """Compute-matched arms see the same samples and differ only in repeats."""
    sets = {"dense": _rows(100), "sparse": _rows(800)}
    monkeypatch.setattr(
        "fineqcomp.data.load_natural_dataset",
        lambda campaign, key, seed, root: {"train": sets[key]},
    )
    monkeypatch.setattr(
        "fineqcomp.training.causal_nll",
        lambda *args, **kwargs: {
            "nll": 1.0, "bits_per_token": 1.0,
            "total_bits": 64.0 * len(args[2]), "nll_tokens": 64 * len(args[2]),
        },
    )
    session = SimpleNamespace(model=object(), tokenizer=object(), spec=object())

    records = measure_arms(
        {}, [_run("dense", 11, 8), _run("sparse", 11, 1)], None, session
    )
    dense, sparse = records[0], records[1]

    assert dense["samples_seen"] == sparse["samples_seen"] == 800
    # Duplication-aware: the arm that repeats 100 rows eight times carries less.
    assert dense["lzma_bits"] < sparse["lzma_bits"]
    # Duplication-blind: a static code charges every copy, so it cannot tell
    # the arms apart. This is the control the claim has to beat.
    assert dense["base_stream_bits"] == pytest.approx(sparse["base_stream_bits"])


def test_collect_drops_runs_from_earlier_configs(tmp_path):
    for run_id, arm in (("live", "arm_a"), ("stale", "arm_a")):
        run = tmp_path / run_id
        (run / "codec_metrics").mkdir(parents=True)
        (run / "config.json").write_text(
            json.dumps(
                {
                    "run_id": run_id, "study": arm, "dataset_key": arm,
                    "seed": 11, "training": {"epochs": 8},
                }
            )
        )
        (run / "codec_metrics" / "binary.json").write_text(
            json.dumps(
                {
                    "codec_key": "binary",
                    "storage": {"effective_bits_per_value": 1.0},
                    "behavioral_write": {"heldout_bits_saved_per_token": 0.5},
                }
            )
        )

    assert len(collect(tmp_path)) == 2
    kept = collect(tmp_path, keep={"live"})
    assert [row["run_id"] for row in kept] == ["live"]


def test_the_fit_reports_bits_per_doubling():
    fit = _fit([2.0, 4.0, 8.0, 16.0], [1.0, 2.0, 3.0, 4.0])

    assert fit["slope_bits_per_doubling"] == pytest.approx(1.0)
    assert fit["r_squared"] == pytest.approx(1.0)
    assert _fit([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])["reason"]
    assert _fit([1.0], [1.0])["n"] == 1


class _Rows:
    """Enough of a Hugging Face dataset for the diversity lever."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems

    def __getitem__(self, field: str) -> list[str]:
        assert field == "original_question"
        return self.problems

    def select(self, indices):
        return _Rows([self.problems[index] for index in indices])


def test_the_diversity_lever_holds_rows_fixed_and_varies_content():
    from fineqcomp.data import limit_source_problems

    # Twelve rows spread over four source problems, three augmentations each.
    problems = [f"problem-{index % 4}" for index in range(12)]

    narrow, groups = limit_source_problems(_Rows(problems), "original_question", 2, 6)
    assert groups == 2
    assert len(narrow.problems) == 6
    assert set(narrow.problems) == {"problem-0", "problem-1"}

    broad, groups = limit_source_problems(_Rows(problems), "original_question", 4, 6)
    assert groups == 4
    assert len(broad.problems) == 6
    # Same row count, twice the content. That is the whole lever.
    assert len(set(broad.problems)) > len(set(narrow.problems))


def test_an_arm_that_cannot_reach_its_row_count_fails_loudly():
    from fineqcomp.data import limit_source_problems

    problems = [f"problem-{index % 4}" for index in range(12)]
    # One problem carries three rows; an arm asking for eight is not
    # compute-matched to its neighbours and must not run.
    with pytest.raises(ValueError, match="short of the 8"):
        limit_source_problems(_Rows(problems), "original_question", 1, 8)


def test_the_prequential_code_prices_what_the_dataset_adds():
    from fineqcomp.dataset_info import prequential_code

    # A learner that never improves codes every block at the base rate, so the
    # dataset is measured as carrying nothing.
    flat = prequential_code([(1000, 10.0)], base_bits_per_row=10.0, rows=4000)
    assert flat["information_bits"] == pytest.approx(0.0)

    # Diminishing returns: later blocks are cheaper, and the saving against the
    # free base code is the information, in the same unit as the adapter file.
    curve = [(1000, 8.0), (2000, 6.0)]
    measured = prequential_code(curve, base_bits_per_row=10.0, rows=4000)
    assert measured["base_code_bits"] == pytest.approx(40000.0)
    # 1000*10 (base) + 1000*8 + 2000*6 = 30000
    assert measured["prequential_code_bits"] == pytest.approx(30000.0)
    assert measured["information_bits"] == pytest.approx(10000.0)
    assert [s["right"] for s in measured["segments"]] == [2000, 4000]


def test_the_diversity_lever_is_part_of_a_run_identity():
    from fineqcomp.campaign import expand_campaign

    def campaign(problems):
        dataset = {"train_rows": 8000, "test_rows": 1319}
        if problems is not None:
            dataset["distinct_source_problems"] = problems
        return {
            "models": {"m": {"name": "m", "revision": "r", "backbone": "nf4"}},
            "adapters": {
                "a": {
                    "method": "full_lora", "rank": 16, "target_modules": ["q_proj"],
                    "last_n_layers": None, "alpha": 32,
                }
            },
            "training": {
                "t": {
                    "epochs": 1, "learning_rate": 1e-4, "effective_batch_size": 16,
                    "micro_batch_size": 4, "max_length": 128,
                }
            },
            "codecs": {"binary": {"method": "uniform", "bits": 1}},
            "datasets": {"d": dataset},
            "studies": {
                "s": {
                    "kind": "natural", "models": ["m"], "datasets": ["d"],
                    "seeds": [11], "adapters": ["a"], "training": "t",
                }
            },
        }

    plain = expand_campaign(campaign(None))[0].run_id
    narrow = expand_campaign(campaign(400))[0].run_id
    broad = expand_campaign(campaign(3600))[0].run_id

    # Arms differing only in content diversity must not share a checkpoint.
    assert narrow != broad != plain
    # And an identity minted before the lever existed keeps its name, so the
    # runs already on disk are not orphaned by adding the field.
    assert expand_campaign(campaign(None))[0].run_id == plain
