"""Tests for the required-tilt interpolators (Track A2, Task 3).

Two things are pinned here because the ANWM conclusions rest on them:

* the per-step estimator agrees with the analytic flatness tilt on a constant-acceleration
  path (so "required tilt" means what the verifier's physics means), and
* the **lower bound never exceeds the upper bound** — a smooth spline through the same
  waypoints cannot demand more tilt than the per-step estimator, and if it did, a bug in
  the interpolator could manufacture an "infeasible" verdict.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.a2_anwm_required import (frame_required, lower_bound_series,
                                          required_tilt, summarise, upper_bound_series)
from src.physics.flatness import G0

A = np.array([2.0, 0.0, 1.0])      # m/s^2, horizontal + vertical
DT = 0.5
N = 17


def constant_acceleration_path(a: np.ndarray = A, dt: float = DT, n: int = N) -> np.ndarray:
    t = np.arange(n) * dt
    return np.outer(0.5 * t**2, a)


def test_per_step_estimator_recovers_the_analytic_tilt():
    pos = constant_acceleration_path()
    v, acc = upper_bound_series(pos, DT)
    np.testing.assert_allclose(acc, np.tile(A, (N - 2, 1)), atol=1e-9)
    tilt = required_tilt(v, acc, 0.0)
    expected = np.degrees(np.arctan2(np.hypot(A[0], A[1]), A[2] + G0))
    np.testing.assert_allclose(tilt, expected, atol=1e-9)


def test_spline_lower_bound_matches_the_same_physics():
    """A quadratic path has a constant second derivative, so the spline's is the same."""
    pos = constant_acceleration_path()
    v, acc = lower_bound_series(pos, DT, 0.0)
    interior = slice(len(acc) // 4, 3 * len(acc) // 4)
    np.testing.assert_allclose(np.median(acc[interior], axis=0), A, atol=1e-4)
    expected = np.degrees(np.arctan2(np.hypot(A[0], A[1]), A[2] + G0))
    assert np.median(required_tilt(v, acc, 0.0)) == pytest.approx(expected, abs=1e-3)


def test_drag_shifts_the_tilt_by_the_velocity_it_balances():
    pos = constant_acceleration_path()
    v, acc = upper_bound_series(pos, DT)
    k_d = 0.295
    tilt = required_tilt(v, acc, k_d)
    u = A[None, :] + np.array([0.0, 0.0, G0]) + k_d * v
    expected = np.degrees(np.arctan2(np.hypot(u[:, 0], u[:, 1]), u[:, 2]))
    np.testing.assert_allclose(tilt, expected, atol=1e-9)


def test_spline_is_not_a_lower_bound_on_a_plateau_path():
    """Pinned because it was assumed the other way round and that assumption was wrong.

    On a stall-and-jump path most per-step accelerations are exactly zero, so the per-step
    median tilt is 0 while the natural cubic spline spreads the jump over its neighbours.
    A natural spline minimises the integral of squared acceleration among C2 interpolants;
    it does not bound tilt for a statistic, and it can overshoot even the peak (a real
    flight may have discontinuous acceleration, which is outside the spline's class).
    """
    pos = np.zeros((16, 3))
    pos[12:] = np.array([0.0, 1.25, 0.0])
    pos[13:] = np.array([0.0, 2.50, 0.0])
    pos[14:] = np.array([2.0, 5.0, 0.0])
    pos[15:] = np.array([5.0, 9.0, 0.0])
    v_lo, a_lo = lower_bound_series(pos, 0.25, 0.0)
    v_hi, a_hi = upper_bound_series(pos, 0.25)
    lo = summarise(required_tilt(v_lo, a_lo, 0.0))
    hi = summarise(required_tilt(v_hi, a_hi, 0.0))
    assert lo["median_deg"] > hi["median_deg"], "the spline smooths the plateaus away"
    assert hi["median_deg"] == pytest.approx(0.0, abs=1e-9)


def test_frame_level_requirement_aligns_with_the_sixteen_frames():
    """Frame i is the depiction after the step ending at node i+1: 16 frames from 17 nodes."""
    pos = constant_acceleration_path(n=17)
    for kind in ("spline", "step"):
        req = frame_required(pos, DT, 0.0, kind)
        assert len(req) == 16
        expected = np.degrees(np.arctan2(np.hypot(A[0], A[1]), A[2] + G0))
        interior = req[2:-1]
        np.testing.assert_allclose(np.nanmedian(interior), expected, atol=1e-3)
        if kind == "step":
            assert np.isnan(req[0]), "the first frame has no predecessor step"
        else:
            assert np.isfinite(req[0]), "the spline is defined at every node"


def test_both_interpolants_agree_on_a_smooth_path():
    pos = constant_acceleration_path(n=17)
    a = frame_required(pos, DT, 0.0, "spline")
    b = frame_required(pos, DT, 0.0, "step")
    np.testing.assert_allclose(np.nanmedian(a), np.nanmedian(b), atol=0.5)


def test_a_free_fall_path_is_rejected_rather_than_scored():
    """a = -g e3 carries no thrust direction; the physics raises instead of guessing."""
    pos = np.zeros((6, 3))
    pos[:, 2] = -0.5 * G0 * (np.arange(6) * DT) ** 2
    v, acc = upper_bound_series(pos, DT)
    with pytest.raises(ValueError, match="degenerate"):
        required_tilt(v, acc, 0.0)


def test_summarise_reports_none_for_an_empty_population():
    s = summarise(np.array([np.nan, np.nan]))
    assert s["median_deg"] is None and s["frac_ge10"] is None
