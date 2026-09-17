"""Pointer chasing: iterated function composition as a procedural task.

Every other corpus in this project is retrieved and then modified. This one is
generated, because the whole point of it is a knob that no retrieved dataset
has: the difficulty is set exactly, by two integers, independently of the model
being tested. An item shows a map from numbers to numbers, a starting value and
a step count, and asks where you land. Nothing has to be known to answer it and
nothing can be guessed: the only route is to apply the map the requested number
of times.

    Map:
    12 -> 47
    47 -> 83
    ...
    Start: 12
    Steps: 3

    Step 1: 12 -> 47
    Step 2: 47 -> 83
    Step 3: 83 -> 26
    The answer is: 26

Two decisions make this ride the apparatus already built for chain of thought
rather than needing one of its own.

The labels are numbers, so the recorded GSM8K scorer -- the last number in the
response, compared with the last number in the reference -- reads this task
correctly with no change. And the response is a sequence of worked lines
followed by the same answer marker the MetaMathQA corpus uses, so
`permute_rationales` corrupts it in exactly the way the recorded result was
corrupted: the working belongs to a different question, the answer line is
still right.

That makes this a sharp test of the mechanism the chain-of-thought study found.
If what survives compression is a conditional step-segmentation prior -- write
one step per line, then stop and answer -- then a task that is nothing but steps
should show it more clearly than word problems do, not less.
"""

from __future__ import annotations

import random
from typing import Any

from fineqcomp.data import Example

ANSWER_MARKER = "The answer is:"


def _labels(rng: random.Random, count: int, low: int, high: int) -> list[int]:
    """Distinct numeric node labels, one per map entry."""
    available = high - low + 1
    if count > available:
        raise ValueError(
            f"cannot draw {count} distinct labels from [{low}, {high}]"
        )
    return rng.sample(range(low, high + 1), count)


def _build_item(
    rng: random.Random, nodes: int, hops: int, low: int, high: int
) -> tuple[str, str, dict[str, Any]]:
    """One map, one query, and the worked walk that answers it.

    The map is a total function on its own labels, not a permutation: a node may
    have several predecessors and the walk may enter a cycle. That is deliberate.
    A permutation lets a model shortcut by learning that each value appears once
    on each side; a general function does not, and repeated visits are exactly
    the case where a model that is pattern-matching rather than stepping fails.
    """
    labels = _labels(rng, nodes, low, high)
    # Self-loops are excluded: a step that does not move is free to get right
    # and would dilute the difficulty the hop count is supposed to set.
    mapping = {
        label: rng.choice([other for other in labels if other != label])
        for label in labels
    }
    start = rng.choice(labels)

    walk = []
    current = start
    for _ in range(hops):
        nxt = mapping[current]
        walk.append((current, nxt))
        current = nxt
    answer = current

    order = list(labels)
    rng.shuffle(order)
    map_lines = "\n".join(f"{label} -> {mapping[label]}" for label in order)
    prompt = (
        "Follow the pointers through the map. Apply the map once per step, "
        "starting from the start value.\n\n"
        f"Map:\n{map_lines}\n\n"
        f"Start: {start}\n"
        f"Steps: {hops}\n\n"
        "Walk:"
    )
    steps = "\n".join(
        f"Step {index}: {source} -> {target}"
        for index, (source, target) in enumerate(walk, start=1)
    )
    response = f"{steps}\n{ANSWER_MARKER} {answer}"
    metadata = {
        "nodes": nodes,
        "hops": hops,
        "start": start,
        "answer": answer,
        # A walk that revisits a node is the harder case; recording it lets a
        # later analysis split the score without regenerating anything.
        "revisits": len({source for source, _ in walk}) < len(walk),
    }
    return prompt, response, metadata


def build_split(
    split: str,
    rows: int,
    *,
    nodes: int,
    hops: int,
    seed: int,
    label_low: int = 10,
    label_high: int = 99,
) -> list[Example]:
    """Generate one split. Distinct maps per item, distinct streams per split."""
    if rows < 1:
        raise ValueError("a split needs at least one row")
    if hops < 1:
        raise ValueError("a walk needs at least one step")
    if nodes < 2:
        raise ValueError("a map needs at least two labels")
    # The split name enters the seed so train, calibration and test never draw
    # the same maps even when the campaign seed is the same.
    stream = random.Random(f"pointer-chasing|{split}|{seed}|{nodes}|{hops}")
    examples = []
    for index in range(rows):
        prompt, response, metadata = _build_item(
            stream, nodes, hops, label_low, label_high
        )
        examples.append(
            Example(
                example_id=f"pointer-{nodes}n{hops}h-{split}-{index}",
                prompt=prompt,
                response="\n" + response,
                metadata={"split": split, "evaluator": "gsm8k", **metadata},
            )
        )
    return examples


def build_pointer_splits(spec: dict[str, Any], seed: int) -> dict[str, list[Example]]:
    """Build the three splits a campaign dataset needs from a config block."""
    nodes = int(spec["nodes"])
    hops = int(spec["hops"])
    low = int(spec.get("label_low", 10))
    high = int(spec.get("label_high", 99))
    counts = {
        "train": int(spec["train_rows"]),
        "calibration": int(spec["validation_rows"]),
        "test": int(spec["test_rows"]),
    }
    return {
        split: build_split(
            split,
            count,
            nodes=nodes,
            hops=hops,
            # Test rows use a fixed seed, like every other test cap in this
            # project, so all seeds and codecs are scored on the same items.
            seed=0 if split == "test" else seed,
            label_low=low,
            label_high=high,
        )
        for split, count in counts.items()
    }


def generator_identity(spec: dict[str, Any]) -> dict[str, Any]:
    """What stands in for a dataset revision when the data is generated."""
    return {
        "generator": "pointer_chasing",
        "version": 1,
        "nodes": int(spec["nodes"]),
        "hops": int(spec["hops"]),
        "label_low": int(spec.get("label_low", 10)),
        "label_high": int(spec.get("label_high", 99)),
    }
