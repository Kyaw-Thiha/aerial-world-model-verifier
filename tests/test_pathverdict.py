"""Unit tests for the F01 pixel-path statistics (synthetic trajectories only)."""
from __future__ import annotations

import numpy as np
import pytest

from src.pathrec.pathverdict import (G0, apply_rot2, band_coherence, complex_correlation,
                                     drag_response, horizontal_basis, physics_scale_check,
                                     preintegration_fit, shooting_fit, smooth_derivative,
                                     velocity_drag_fit)


def _simulate(lean: np.ndarray, dt: float, k_d: float, v0=(0.0, 0.0), scale=0.37,
              yaw=None) -> tuple[np.ndarray, np.ndarray]:
    """Exact discrete trajectory of the drag model under a known lean, in path units."""
    lean = np.asarray(lean, dtype=np.float64)
    n = len(lean)
    yaw = np.zeros(n) if yaw is None else np.asarray(yaw, dtype=np.float64)
    forcing = G0 * apply_rot2(yaw, lean)
    resp = drag_response(forcing, dt, k_d)
    x = resp.H_x[:, :, 0] * v0[0] + resp.H_x[:, :, 1] * v0[1] + resp.F_x
    # H_x[:, :, 0] is (N,2): response of each component to a unit v0x; same for y.
    return x / scale, lean


def _clip(n=60, dt=1 / 15.072243783239902, amp=0.25, freq=0.45):
    t = np.arange(n) * dt
    lean = np.column_stack([amp * np.sin(2 * np.pi * freq * t),
                            amp * np.cos(2 * np.pi * freq * t * 0.7)])
    return t, lean


def test_drag_response_matches_closed_form():
    dt, k = 0.05, 0.3
    f = np.tile([0.2, -0.1], (4, 1))
    r = drag_response(f, dt, k)
    decay = np.exp(-k * dt)
    a = (1 - decay) / k
    assert np.allclose(r.H_v[1], decay * np.eye(2))
    assert np.allclose(r.F_v[1], a * f[0])
    # position after one step: a * v_prev + (dt/k - a/k) * f
    assert np.allclose(r.F_x[1], (dt / k - a / k) * f[0])
    # drag-free limit
    r0 = drag_response(f, dt, 0.0)
    assert np.allclose(r0.H_v[1], np.eye(2))
    assert np.allclose(r0.F_x[1], 0.5 * dt * dt * f[0])


def test_horizontal_basis_is_orthonormal():
    for up in ([0, 0, 1], [0.3, -0.2, 0.9], [1, 0, 0]):
        e1, e2 = horizontal_basis(np.asarray(up, float))
        u = np.asarray(up, float) / np.linalg.norm(up)
        assert abs(np.dot(e1, e2)) < 1e-12
        assert abs(np.dot(e1, u)) < 1e-12
        assert abs(np.dot(e2, u)) < 1e-12


def test_shooting_recovers_scale_and_explains_path():
    t, lean = _clip()
    dt = float(t[1] - t[0])
    x_path, _ = _simulate(lean, dt, 0.295, v0=(1.2, -0.4), scale=0.37)
    fit = shooting_fit(x_path, np.zeros(len(t)), lean, t, k_d=0.295)
    assert fit.statistic < 0.02
    assert fit.scale == pytest.approx(0.37, rel=0.15)


def test_tilt_removal_and_mirror_raise_the_statistic():
    """Amendment 1: the real-condition fit sets the scale and yaw for every edit."""
    t, lean = _clip(amp=0.3)
    dt = float(t[1] - t[0])
    x_path, _ = _simulate(lean, dt, 0.295, v0=(1.0, 0.5), scale=0.5)
    yaw0 = np.zeros(len(t))
    real = shooting_fit(x_path, yaw0, lean, t)
    rho_real = real.statistic
    removed = shooting_fit(x_path, yaw0, np.zeros_like(lean), t,
                           scale=real.scale, psi=np.deg2rad(real.psi_deg))
    mirror = shooting_fit(x_path, yaw0, lean * np.array([1.0, -1.0]), t,
                          scale=real.scale, psi=np.deg2rad(real.psi_deg))
    assert removed.statistic > 0.3
    assert mirror.statistic > 3 * rho_real
    assert mirror.statistic > 0.1


def test_free_scale_cannot_see_tilt_removal():
    """The finding behind amendment 1: a free scale absorbs the removal (s -> 0)."""
    t, lean = _clip(amp=0.3)
    dt = float(t[1] - t[0])
    x_path, _ = _simulate(lean, dt, 0.295, v0=(1.0, 0.5), scale=0.5)
    free = shooting_fit(x_path, np.zeros(len(t)), np.zeros_like(lean), t)
    assert not np.isfinite(free.statistic) or free.statistic < 0.05


def test_shooting_is_yaw_invariant():
    t, lean = _clip()
    dt = float(t[1] - t[0])
    x_path, _ = _simulate(lean, dt, 0.295, v0=(1.0, 0.2), scale=0.4)
    ang = np.deg2rad(37.0)
    rotated_path = apply_rot2(ang, x_path)
    rotated_lean = apply_rot2(ang, lean)
    a = shooting_fit(x_path, np.zeros(len(t)), lean, t).statistic
    b = shooting_fit(rotated_path, np.zeros(len(t)), rotated_lean, t).statistic
    assert b == pytest.approx(a, rel=1e-6)


def test_preintegration_and_velocity_drag_on_clean_data():
    t, lean = _clip()
    dt = float(t[1] - t[0])
    x_path, _ = _simulate(lean, dt, 0.295, v0=(1.0, -0.3), scale=0.4)
    v_obs = smooth_derivative(x_path, dt, 7, 1)
    fit = preintegration_fit(v_obs, np.zeros(len(t)), lean, t, k_d=0.295)
    assert fit.statistic < 0.05
    assert abs(fit.extra["r_x"]) > 0.9
    # steady flight: lean proportional to velocity
    steady = np.tile([0.06, -0.02], (len(t), 1))
    xs, _ = _simulate(steady, dt, 0.295, v0=(G0 * 0.06 / 0.295, -G0 * 0.02 / 0.295), scale=0.4)
    vs = smooth_derivative(xs, dt, 7, 1)
    vfit = velocity_drag_fit(vs, np.zeros(len(t)), steady, k_d=0.295)
    assert vfit.statistic < 0.05


def test_complex_correlation_is_rotation_and_scale_invariant():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(40, 2))
    b = 2.5 * apply_rot2(np.deg2rad(73.0), a)
    assert complex_correlation(a, b) == pytest.approx(1.0, abs=1e-9)
    assert complex_correlation(a, rng.normal(size=(40, 2))) < 0.5


def test_band_coherence_rejects_removed_tilt():
    t, lean = _clip(amp=0.3, freq=0.4)
    dt = float(t[1] - t[0])
    x_path, _ = _simulate(lean, dt, 0.295, v0=(1.0, 0.2), scale=0.4)
    a_obs = smooth_derivative(x_path, dt, 9, 2)
    real = band_coherence(a_obs, np.zeros(len(t)), lean, dt, 0.05, 1.5)
    removed = band_coherence(a_obs, np.zeros(len(t)), np.zeros_like(lean), dt, 0.05, 1.5)
    assert real["statistic"] < 0.4
    assert removed["statistic"] > real["statistic"]


def test_physics_scale_check_flags_degenerate_scale():
    t, lean = _clip()
    dt = float(t[1] - t[0])
    x_path, _ = _simulate(lean, dt, 0.295, v0=(1.0, 0.0), scale=0.4)
    fit = shooting_fit(x_path, np.zeros(len(t)), lean, t)
    steps = np.linalg.norm(np.diff(x_path, axis=0), axis=1)
    good = physics_scale_check(fit, steps, np.full(len(steps), 4.0))
    assert good["identifiable"] and good["plausible"] and not good["flag"]
    # a scale 1e-6 too small cannot be a metre-per-unit conversion at 4 m depth
    bad = physics_scale_check(fit, steps * 1e-6, np.full(len(steps), 4.0))
    assert bad["flag"]
