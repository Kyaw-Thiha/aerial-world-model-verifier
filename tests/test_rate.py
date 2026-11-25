"""Unit tests for the body-rate consistency machinery (Track P)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.verdict.rate import (E3, align_e3_to, depicted_axis_rate, depicted_axis_rate_literal,
                              filter_gain, gauge_fit, match_sigma, noisy_sequence, residuals,
                              rotation_rates, step_errors, thrust_axis, uniform_path, axis_rate)


def _arc(omega_deg_s: float, tilt_deg: float, n: int, fps: float) -> np.ndarray:
    """Body attitudes whose thrust axis is tilted and whose azimuth spins at omega."""
    t = np.arange(n) / fps
    out = []
    for tk in t:
        yaw = np.deg2rad(omega_deg_s) * tk
        Rz = Rotation.from_rotvec([0, 0, yaw]).as_matrix()
        Ry = Rotation.from_rotvec([0, np.deg2rad(tilt_deg), 0]).as_matrix()
        out.append(Rz @ Ry)
    return np.asarray(out), t


def test_rotation_rates_constant_world_rate():
    R, t = _arc(20.0, 30.0, 200, 50.0)
    w = np.einsum("nij,nj->ni", R, rotation_rates(R, t))  # body -> world
    assert np.nanmax(np.abs(w - np.array([0.0, 0.0, np.deg2rad(20.0)]))) < 1e-9


def test_align_e3_to_maps_third_axis():
    b3 = np.array([[0.0, 0.0, 1.0], [0.5, 0.0, np.sqrt(0.75)], [0.0, 0.0, -1.0]])
    R = align_e3_to(b3)
    assert np.max(np.abs(R @ E3 - b3)) < 1e-12


def test_gauge_fit_recovers_a_cross_product_rate():
    """rd built from the video rotation and the fitted gauge equals omega_world x b3."""
    R_bw, t = _arc(35.0, 25.0, 300, 100.0)
    R_cb = np.array([[0.0, 0.5, np.sqrt(3) / 2], [-1.0, 0.0, 0.0], [0.0, -np.sqrt(3) / 2, 0.5]])
    R_wc = R_bw @ R_cb
    L = Rotation.from_rotvec([0.4, -0.9, 0.2]).as_matrix()   # arbitrary video gauge
    R_est = np.einsum("ij,njk->nik", L, R_wc)
    b3 = R_bw[:, :, 2]
    g = gauge_fit(R_est, b3, R_cb)
    assert np.abs(g["G"] - L.T).max() < 1e-8
    w_cam = rotation_rates(R_est, t)
    rd = depicted_axis_rate(w_cam, R_est, R_cb, g["G"])
    w = np.einsum("nij,nj->ni", R_bw, rotation_rates(R_bw, t))
    ref = np.cross(w, b3)
    ok = np.isfinite(rd).all(1) & np.isfinite(ref).all(1)
    assert np.max(np.abs(rd[ok] - ref[ok])) < 1e-8


def test_literal_substitution_is_not_yaw_invariant():
    """The handoff-literal form picks an arbitrary frame and is wrong by O(|omega| sin psi)."""
    R_bw, t = _arc(35.0, 25.0, 300, 100.0)
    R_cb = np.eye(3)
    c = R_bw @ E3
    b3 = R_bw[:, :, 2]
    w_cam = rotation_rates(R_bw, t)
    literal = depicted_axis_rate_literal(w_cam, b3, R_cb)
    w = np.einsum("nij,nj->ni", R_bw, rotation_rates(R_bw, t))
    ref = np.cross(w, b3)
    ok = np.isfinite(literal).all(1)
    # with the path-yaw convention the heading is not the minimal rotation from e3, so the
    # literal form is off by a first-order amount, never zero
    assert np.max(np.abs(literal[ok] - ref[ok])) > 0.05
    assert np.isfinite(c).all()


def test_body_yaw_leaves_the_thrust_axis_rate_unchanged():
    """A rotation about b3 (the unconstrained yaw) must not enter rd."""
    R_bw, t = _arc(15.0, 30.0, 200, 100.0)
    R_cb = np.eye(3)
    psi = np.deg2rad(40.0) * t
    adds = np.asarray([Rotation.from_rotvec([0, 0, p]).as_matrix() for p in psi])
    R_bw2 = R_bw @ adds
    b3 = R_bw[:, :, 2]
    w0 = rotation_rates(R_bw, t)
    w1 = rotation_rates(R_bw2, t)
    g0 = gauge_fit(R_bw, b3, R_cb)
    g1 = gauge_fit(R_bw2, b3, R_cb)
    rd0 = depicted_axis_rate(w0, R_bw, R_cb, g0["G"])
    rd1 = depicted_axis_rate(w1, R_bw2, R_cb, g1["G"])
    ok = np.isfinite(rd0).all(1) & np.isfinite(rd1).all(1)
    # the residual is the central-difference truncation of the two rate sequences, 2e-4 rad/s
    assert np.max(np.abs(rd0[ok] - rd1[ok])) < 1e-3


def test_uniform_path_and_axis_rate_recover_a_known_path():
    fps = 250.0
    t = np.arange(0.0, 6.0, 1 / fps)
    w = 2 * np.pi / 3.0
    pos = np.column_stack([2 * np.cos(w * t), 2 * np.sin(w * t), 0.5 * t])
    tg, vel, acc, interior = uniform_path(t, pos)
    b3 = thrust_axis(acc, vel, k_d=0.0)
    rb = axis_rate(tg, b3)
    # a constant-speed level circle banks the thrust axis and spins it about the vertical at
    # the orbital rate: |db/dt| = w sin(tilt) with tilt = atan2(w^2 r, g)
    tilt = np.arctan2(w ** 2 * 2.0, 9.81)
    expect = w * np.sin(tilt)
    assert abs(np.nanmedian(np.linalg.norm(rb, axis=1)) / expect - 1.0) < 5e-3


def test_step_errors_zero_for_identical_sequences():
    R, _ = _arc(10.0, 20.0, 50, 30.0)
    assert np.max(step_errors(R, R, None)) == 0.0


def test_match_sigma_recovers_the_injected_value():
    R, _ = _arc(0.0, 30.0, 400, 30.0)
    rng = np.random.default_rng(7)
    est = noisy_sequence(R, 0.05, rng)
    e = step_errors(est, R, None)
    assert 0.03 < match_sigma(e) < 0.075


def test_filter_gain_is_a_low_pass_with_a_stated_cutoff():
    g = filter_gain()
    assert 0.2 < g["cutoff_3db_hz"] < 5.0


def test_residuals_flags_a_rate_mismatch():
    rb = np.tile([1.0, 0.0, 0.0], (10, 1))
    rd = np.tile([0.0, 1.0, 0.0], (10, 1))
    r = residuals(rb, rd)
    assert np.allclose(r["err_deg_s"], np.degrees(np.sqrt(2.0)))
    assert np.allclose(r["cos"], 0.0)
