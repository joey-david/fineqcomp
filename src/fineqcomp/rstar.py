"""Locate R*: the smallest adapter file that still holds a target of the gain.

The rate axis is measured on exact file size. The retention axis defaults to
held-out bits saved rather than task accuracy, because a bits-saved measurement
is a forward pass over a few hundred examples and costs seconds, while a GSM8K
accuracy pass costs about nineteen minutes at the full test set. That gap is
what makes a dense ladder affordable, and a dense ladder is what R* needs: the
whole-bit rungs leave a 0.95-bit hole exactly where the crossing falls.

Accuracy is still measured, at a few anchor rates, to check that the two axes
order the codecs the same way. They are not interchangeable: on the Mistral runs
R*(0.90) came to 0.98/1.00/1.03 bits on bits-saved against 1.30/1.11/1.40 on
accuracy, so bits-saved reaches its ceiling sooner. Comparing arms is therefore
fine — the offset is shared — but an absolute R* must say which axis produced
it. Both of those figures interpolate across the 0.95-bit hole in the whole-bit
ladder, which is what the blended rungs exist to close.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _crossing(points: list[tuple[float, float]], target: float) -> float | None:
    """Rate at which retention first reaches `target`, linearly interpolated.

    Points are (rate, retention). Retention is not guaranteed monotone in rate,
    so this takes the *lowest* rate whose running maximum reaches the target;
    a dip at a higher rate cannot lower the answer.
    """
    ordered = sorted(points)
    best = float("-inf")
    previous: tuple[float, float] | None = None
    for rate, value in ordered:
        running = max(best, value)
        if running >= target:
            if previous is None:
                return rate
            rate0, value0 = previous
            if value <= value0:
                return rate
            span = value - value0
            return rate0 + (target - value0) / span * (rate - rate0)
        best = running
        previous = (rate, running)
    return None


def r_star(
    points: Iterable[dict[str, Any]],
    target: float = 0.90,
    rate_key: str = "effective_bits_per_value",
    value_key: str = "heldout_bits_saved_per_token",
    reference: float | None = None,
) -> dict[str, Any]:
    """R* against a fraction of the raw adapter's own gain on `value_key`.

    `reference` is the raw adapter's value; without it the largest measured
    value stands in, which is what the highest-rate point converges to.
    """
    usable = [
        (float(p[rate_key]), float(p[value_key]))
        for p in points
        if p.get(rate_key) is not None and p.get(value_key) is not None
    ]
    if not usable:
        return {"target": target, "r_star": None, "reason": "no measured points"}
    ceiling = reference if reference is not None else max(v for _, v in usable)
    if ceiling <= 0:
        return {"target": target, "r_star": None, "reason": "no gain to retain"}
    fractions = [(rate, value / ceiling) for rate, value in usable]
    crossing = _crossing(fractions, target)
    return {
        "target": target,
        "value_key": value_key,
        "reference": ceiling,
        "r_star": crossing,
        "bracketed": crossing is not None
        and min(r for r, _ in fractions) < crossing < max(r for r, _ in fractions),
        "points": sorted(fractions),
    }


def from_run(run_dir: Path, target: float = 0.90) -> dict[str, Any]:
    """R* for one finished run, read from its codec metric files."""
    points = []
    for path in sorted((run_dir / "codec_metrics").glob("*.json")):
        metric = json.loads(path.read_text())
        storage = metric.get("storage") or {}
        behaviour = metric.get("behavioral_write") or {}
        if storage.get("effective_bits_per_value") is None:
            continue
        points.append(
            {
                "codec": metric.get("codec_key"),
                "effective_bits_per_value": storage["effective_bits_per_value"],
                "heldout_bits_saved_per_token": behaviour.get(
                    "heldout_bits_saved_per_token"
                ),
            }
        )
    raw = run_dir / "metrics.json"
    reference = None
    if raw.is_file():
        record = json.loads(raw.read_text())
        reference = (record.get("raw_behavioral_write") or {}).get(
            "heldout_bits_saved_per_token"
        )
    return r_star(points, target=target, reference=reference)
