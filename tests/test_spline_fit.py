"""Unit tests for the F2 spline + flatness fit (synthetic tracks only)."""
from __future__ import annotations

import numpy as np
import pytest

from src.pathrec.spline_fit import (G0, drag_velocity, dyn_fit, fit_spline_path,
                                    knot_intervals)

DT = 1 / 15.07


def _synthetic(n=60, k=0.295, amp=(0.25, 0.2), freq=(0.4, 0.3)):
    t = np.arange(n) * DT
    lean = np.column_stack([amp[0] * np.sin(2 * np.pi * freq[0] * t),
                            amp[1] * np.cos(2 * np.pi * freq[1] * t)])
    forcing = G0 * lean
    H, F = drag_velocity(t, forcing, k)
    v0 = np.array([1.2, -0.4])
    v = H[:, None] * v0 + F
    X = np.cumsum(np.vstack([np.zeros(2), 0.5 * (v[1:] + v[:-1]) * DT]), axis=0)
    return t, lean, X


def test_knot_rule_and_cap():
    assert knot_intervals(60, DT, 0.386) == 10
    assert knot_intervals(60, DT, 1.42) == 4
    assert knot_intervals(60, DT, 0.336) == 12
    # cap: at most one interval per three samples
    assert knot_intervals(20, DT, 0.05) <= (20 - 1) // 3


def test_spline_recovers_analytic_derivatives():
    t = np.arange(60) * DT
    X = np.column_stack([np.sin(2 * np.pi * 0.5 * t), 0.5 * np.cos(2 * np.pi * 0.4 * t)])
    a_true = np.column_stack([-(2 * np.pi * 0.5) ** 2 * np.sin(2 * np.pi * 0.5 * t),
                              -0.5 * (2 * np.pi * 0.4) ** 2 * np.cos(2 * np.pi * 0.4 * t)])
    sp = fit_spline_path(t, X, 0.386)
    assert sp.n_intervals == 10
    assert len(sp.t) == 60 - 6
    rel = np.sqrt(np.mean((sp.a - a_true[3:-3]) ** 2)) / np.sqrt(np.mean(a_true[3:-3] ** 2))
    assert rel < 0.10
    fine = fit_spline_path(t, X, 0.386, n_intervals=15)
    rel_fine = np.sqrt(np.mean((fine.a - a_true[3:-3]) ** 2)) / np.sqrt(
        np.mean(a_true[3:-3] ** 2))
    assert rel_fine < rel


def test_drag_velocity_limits():
    n = 10
    t = np.arange(n) * DT
    f = np.tile([0.3, -0.2], (n, 1))
    H, F = drag_velocity(t, f, 0.0)
    assert np.allclose(H, 1.0)
    # drag-free velocity is the cumulative trapezoid of the forcing
    expect = np.cumsum(np.vstack([np.zeros(2), 0.5 * (f[1:] + f[:-1]) * DT]), axis=0)
    assert np.allclose(F, expect)
    H2, F2 = drag_velocity(t, f, 0.5)
    assert H2[1] == pytest.approx(np.exp(-0.5 * DT))
    # a positive k_d damps the forced response relative to the drag-free one
    assert np.linalg.norm(F2[-1]) < np.linalg.norm(F[-1])


def test_real_fit_is_small_and_edits_raise_it():
    t, lean, X = _synthetic()
    sp = fit_spline_path(t, X, 0.386)
    yaw = np.zeros(len(sp.t))
    lean_i = lean[3:-3]
    real = dyn_fit(sp.t, sp.a, yaw, lean_i)
    assert real.ok and real.statistic < 0.15
    assert real.scale > 0
    fixed = dict(scale=real.scale, psi=np.deg2rad(real.psi_deg), v0=np.array(real.v0))
    again = dyn_fit(sp.t, sp.a, yaw, lean_i, **fixed)
    assert again.statistic == pytest.approx(real.statistic, rel=1e-9)
    removed = dyn_fit(sp.t, sp.a, yaw, np.zeros_like(lean_i), **fixed)
    mirror = dyn_fit(sp.t, sp.a, yaw, lean_i * np.array([1.0, -1.0]), **fixed)
    assert removed.statistic > 0.5
    assert mirror.statistic > 0.5


def test_free_scale_cannot_see_removal():
    t, lean, X = _synthetic()
    sp = fit_spline_path(t, X, 0.386)
    free = dyn_fit(sp.t, sp.a, np.zeros(len(sp.t)), np.zeros_like(lean[3:-3]))
    # no positive-scale solution, or one that explains the removal away
    assert (not np.isfinite(free.statistic)) or free.statistic < 0.05


def test_yaw_invariance():
    t, lean, X = _synthetic()
    sp = fit_spline_path(t, X, 0.386)
    ang = np.deg2rad(53.0)
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    sp2 = fit_spline_path(t, X @ R.T, 0.386)
    a = dyn_fit(sp.t, sp.a, np.zeros(len(sp.t)), lean[3:-3])
    b = dyn_fit(sp2.t, sp2.a, np.zeros(len(sp2.t)), lean[3:-3] @ R.T)
    assert b.statistic == pytest.approx(a.statistic, rel=1e-6)
