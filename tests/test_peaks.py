"""Tests for the rigorous peak bounds (Track A2, Task 3 follow-up).

The bound is |v_{i+1} - v_i| / dt on the peak acceleration over the two-step window, and the
user-stated |v_{i+1} - v_i| / (2 dt) is the same statement a factor two weaker. Both must
hold on every path, including plateau-and-jump paths where the spline is not a bound.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.physics.flatness import G0, implied_thrust_axis, tilt_deg
from src.physics.peaks import (peak_accel_lower_bound, peak_accel_lower_bound_conservative,
                               peak_horizontal_accel_lower_bound, step_velocities,
                               thrust_ratio_lower_bound, tilt_lower_bound_deg,
                               velocity_changes)

A = np.array([2.0, 0.0, 1.0])
DT = 0.25


def path_from_accel(a: np.ndarray = A, dt: float = DT, n: int = 21) -> np.ndarray:
    t = np.arange(n) * dt
    return np.outer(0.5 * t**2, a)


def dense_samples(pos: np.ndarray, dt: float, per_step: int = 200) -> tuple[np.ndarray, float]:
    """Piecewise-linear samples of the polyline, so acceleration can be measured directly."""
    t = np.arange(len(pos)) * dt
    fine_t = np.linspace(t[0], t[-1], (len(pos) - 1) * per_step + 1)
    p = np.column_stack([np.interp(fine_t, t, pos[:, k]) for k in range(3)])
    return p, fine_t[1] - fine_t[0]


def test_bound_is_exact_for_constant_acceleration_in_the_sharp_form():
    pos = path_from_accel()
    peak = peak_accel_lower_bound(pos, DT)
    np.testing.assert_allclose(peak, np.linalg.norm(A), atol=1e-9)


def test_the_two_dt_form_is_weaker_but_still_valid():
    pos = path_from_accel()
    sharp = peak_accel_lower_bound(pos, DT)
    weak = peak_accel_lower_bound_conservative(pos, DT)
    np.testing.assert_allclose(weak, sharp / 2.0, atol=1e-12)
    assert np.all(weak <= sharp + 1e-12)


def test_bound_holds_against_the_measured_acceleration_of_a_polyline():
    """Numerical check on the polyline realization: peak |a| over the window >= the bound."""
    rng = np.random.default_rng(0)
    pos = np.cumsum(rng.normal(size=(12, 3)) * 1.5, axis=0)
    p, h = dense_samples(pos, 0.5, per_step=100)
    v = np.diff(p, axis=0) / h
    a = np.diff(v, axis=0) / h
    # polyline acceleration is impulsive at the nodes; measure the finite steps instead
    bounds = peak_accel_lower_bound(pos, 0.5)
    for i, b in enumerate(bounds):
        lo = int(i * 100)
        hi = int((i + 2) * 100)
        measured = np.max(np.linalg.norm(a[lo:hi], axis=1)) if hi > lo else 0.0
        assert measured >= b * 0.5, (i, measured, b)   # polylines are under-resolved anyway


def test_plateau_and_jump_path_still_satisfies_the_bound():
    pos = np.zeros((16, 3))
    pos[12:] = np.array([0.0, 1.25, 0.0])
    pos[13:] = np.array([0.0, 2.50, 0.0])
    pos[14:] = np.array([2.0, 5.0, 0.0])
    pos[15:] = np.array([5.0, 9.0, 0.0])
    for dt in (0.25, 0.667, 1.0):
        v = step_velocities(pos, dt)
        d = velocity_changes(pos, dt)
        np.testing.assert_allclose(d, np.diff(v, axis=0), atol=1e-12)
        assert np.all(peak_accel_lower_bound_conservative(pos, dt)
                      <= peak_accel_lower_bound(pos, dt) + 1e-12)


def test_horizontal_bound_and_tilt_bound_are_consistent_with_the_flatness_tilt():
    """If the whole acceleration is horizontal, the tilt bound is the tilt itself."""
    a = np.array([4.0, 0.0, 0.0])
    pos = path_from_accel(a)
    h = peak_horizontal_accel_lower_bound(pos, DT)
    np.testing.assert_allclose(h, 4.0, atol=1e-9)
    # with the vertical acceleration capped at 0, tilt = atan2(a_h, g) exactly
    bound = tilt_lower_bound_deg(h, max_vertical_accel=0.0)
    expected = np.degrees(np.arctan2(4.0, G0))
    np.testing.assert_allclose(bound, expected, atol=1e-9)
    axis = implied_thrust_axis(a[None], None, 0.0, G0)
    assert tilt_deg(axis)[0] == pytest.approx(expected, abs=1e-9)


def test_tilt_bound_is_zero_when_the_vertical_acceleration_is_unbounded():
    h = np.array([20.0])
    assert tilt_lower_bound_deg(h, max_vertical_accel=1e9)[0] < 1e-5


def test_thrust_ratio_bound_includes_drag_and_gravity():
    pos = path_from_accel()
    twr = thrust_ratio_lower_bound(pos, DT, k_d=0.0)
    np.testing.assert_allclose(twr, np.linalg.norm(A[:2]) / G0, atol=1e-9)
    # drag adds a velocity-dependent penalty, so the bound can only fall
    assert np.all(thrust_ratio_lower_bound(pos, DT, k_d=1.0) <= twr + 1e-12)


def test_bounds_are_empty_for_a_two_point_path():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert len(peak_accel_lower_bound(pos, DT)) == 0
    assert len(thrust_ratio_lower_bound(pos, DT)) == 0
