from __future__ import annotations

from fineqcomp.data import Example
from fineqcomp.dataset_info import corpus_bits


def _rows(n: int, distinct: int) -> list[Example]:
    """`n` rows drawn from `distinct` unique templates, cycled."""
    return [
        Example(
            example_id=f"r{i}",
            prompt=f"Question: solve problem number {i % distinct} carefully.",
            response=f" The answer for case {i % distinct} is {(i % distinct) * 7}.",
            metadata={},
        )
        for i in range(n)
    ]


def test_compression_sees_duplication_that_row_count_hides():
    """Two sets of equal length, one eight times more repetitive."""
    unique = corpus_bits(_rows(800, 800))
    duplicated = corpus_bits(_rows(800, 100))

    assert unique["rows"] == duplicated["rows"] == 800
    # Same row count, but the repetitive set carries far less unique content.
    assert duplicated["lzma_bits"] < unique["lzma_bits"] / 2
    assert duplicated["compression_ratio"] > unique["compression_ratio"]


def test_compression_grows_with_genuinely_new_rows():
    small = corpus_bits(_rows(200, 200))
    large = corpus_bits(_rows(800, 800))
    assert large["lzma_bits"] > small["lzma_bits"]


def test_long_range_coder_is_needed_at_realistic_corpus_size():
    """zlib's 32 KB window cannot see repeats tens of kilobytes apart.

    The compressibility arms hold thousands of rows, so duplicates are far
    apart in the stream. zlib reports the same size for every arm and is
    useless as an information measure; lzma separates them.
    """
    arms = {name: corpus_bits(_rows(4000, distinct))
            for name, distinct in (("A", 500), ("B", 1000), ("C", 2000), ("D", 4000))}

    zlibs = [a["zlib_bits"] for a in arms.values()]
    lzmas = [a["lzma_bits"] for a in arms.values()]

    # zlib barely moves across an eightfold change in unique content.
    assert max(zlibs) / min(zlibs) < 2.0
    # lzma tracks it.
    assert lzmas == sorted(lzmas)
    assert max(lzmas) / min(lzmas) > 4.0
    assert arms["A"]["distinct_rows"] == 500
