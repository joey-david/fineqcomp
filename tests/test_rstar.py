from __future__ import annotations

from fineqcomp.rstar import _crossing, r_star


def test_crossing_interpolates_between_bracketing_rates():
    # 90% falls between the 1.0 and 2.0 rungs at 0.80 and 1.00.
    assert abs(_crossing([(1.0, 0.80), (2.0, 1.00)], 0.90) - 1.5) < 1e-9


def test_crossing_returns_none_when_target_is_never_reached():
    assert _crossing([(1.0, 0.10), (2.0, 0.40)], 0.90) is None


def test_crossing_ignores_a_dip_at_a_higher_rate():
    """Retention is noisy, so a later dip must not push R* rightward."""
    points = [(1.0, 0.60), (1.5, 0.95), (2.0, 0.88), (3.0, 0.99)]
    assert _crossing(points, 0.90) < 1.5 + 1e-9


def test_r_star_is_measured_against_the_raw_adapter_reference():
    points = [
        {"effective_bits_per_value": 1.0, "heldout_bits_saved_per_token": 0.40},
        {"effective_bits_per_value": 2.0, "heldout_bits_saved_per_token": 0.50},
    ]
    # Against a raw adapter that saved 0.50, the 2.0 point is already 100%.
    tight = r_star(points, target=0.90, reference=0.50)
    assert abs(tight["r_star"] - 1.5) < 1e-9
    assert tight["bracketed"] is True

    # Against a stronger raw adapter the same points never reach 90%.
    loose = r_star(points, target=0.90, reference=1.00)
    assert loose["r_star"] is None


def test_r_star_reports_when_nothing_was_measured():
    assert r_star([], target=0.90)["r_star"] is None
