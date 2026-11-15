"""Unit tests for src/pathrec/align.py and lean.py (synthetic trajectories, no data)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.pathrec.align import align_yaw_scale, both_errors, path_length, trajectory_error, umeyama
from src.pathrec.lean import (differentiate, direction_error_deg, lean_correlation, lean_xy,
                              magnitude_ratio, tilt_deg)


def make_track(n=400, dt=0.01, accel=(0.0, 0.0, 0.0), vel0=(3.0, 0.0, 0.0)):
    t = np.arange(n) * dt
    a = np.asarray(accel, dtype=float)
    p = (np.asarray(vel0) * t[:, None] + 0.5 * a * (t ** 2)[:, None])
    return t, p


def test_umeyama_recovers_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    rot = Rotation.from_euler("zyx", [33.0, -17.0, 8.0], degrees=True).as_matrix()
    scale, trans = 0.37, np.array([4.0, -2.0, 9.0])
    dst = scale * src @ rot.T + trans
    r_hat, s_hat, t_hat = umeyama(src, dst)
    assert np.allclose(r_hat, rot, atol=1e-10)
    assert s_hat == pytest.approx(scale, rel=1e-10)
    assert np.allclose(t_hat, trans, atol=1e-8)


def test_identity_alignment_gives_zero_error():
    _, p = make_track()
    err = trajectory_error(p, p.copy(), with_scale=False)
    assert err["ate_rmse_m"] == pytest.approx(0.0, abs=1e-12)
    assert err["scale"] == pytest.approx(1.0)
    assert err["path_ratio"] == pytest.approx(1.0)


def test_sim3_hides_scale_and_se3_does_not():
    """The whole reason both are reported: a 0.5x track is exact under Sim(3)."""
    _, p = make_track()
    scaled = 0.5 * p
    res = both_errors(scaled, p)
    assert res["sim3"]["ate_rmse_m"] == pytest.approx(0.0, abs=1e-9)
    assert res["sim3"]["scale"] == pytest.approx(2.0)
    assert res["se3"]["path_ratio"] == pytest.approx(0.5, rel=1e-6)
    assert res["se3"]["ate_rmse_m"] > 1.0
    assert res["se3"]["scale"] == 1.0


def test_yaw_scale_fit_recovers_yaw_and_scale():
    _, p = make_track(accel=(0.0, 0.0, 0.0))
    yaw = np.radians(37.0)
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                   [np.sin(yaw), np.cos(yaw), 0.0],
                   [0.0, 0.0, 1.0]])
    est = 0.25 * p @ rz.T + np.array([1.0, 2.0, 0.0])
    rot, scale, _ = align_yaw_scale(est, p)
    assert scale == pytest.approx(4.0, rel=1e-6)
    got = Rotation.from_matrix(rot).as_euler("zyx", degrees=True)[0]
    # the fit maps est -> truth, and est was built with Rz(+yaw), so it returns -yaw
    assert got == pytest.approx(-37.0, abs=1e-6)


def test_path_length_simple():
    p = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 2.0, 0]])
    assert path_length(p) == pytest.approx(3.0)


def test_differentiate_recovers_constant_acceleration():
    t, p = make_track(n=200, dt=0.02, accel=(2.0, -1.0, 0.5), vel0=(1.0, 0.0, 0.0))
    tt, v, a = differentiate(p, dt=0.02, window_s=0.2)
    assert len(tt) == len(v) == len(a)
    # interior samples: parabolic motion is fit exactly by a cubic Savitzky-Golay window
    assert np.allclose(a, np.array([2.0, -1.0, 0.5]), atol=1e-6)
    t_end = tt[-1]
    assert np.allclose(v[-1], np.array([1.0 + 2.0 * t_end, -t_end, 0.5 * t_end]), atol=1e-6)


def test_differentiate_rejects_bad_windows():
    _, p = make_track(n=50)
    with pytest.raises(ValueError):
        differentiate(p, dt=0.01)                      # neither window given
    with pytest.raises(ValueError):
        differentiate(p, dt=0.01, window_s=0.1, window_steps=5)  # both
    with pytest.raises(ValueError):
        differentiate(p, dt=0.0, window_s=0.1)


def test_lean_of_constant_acceleration_is_a_over_g():
    a = np.array([[1.96133, 0.0, 0.0]])   # 0.2 g horizontal
    l = lean_xy(a)
    assert l[0, 0] == pytest.approx(0.2, rel=1e-9)
    assert l[0, 1] == 0.0
    assert np.degrees(np.arctan(np.linalg.norm(l[0]))) == pytest.approx(np.degrees(np.arctan(0.2)))


def test_lean_marks_free_fall_as_nan():
    l = lean_xy(np.array([[0.0, 0.0, -9.80665]]))
    assert np.isnan(l).all()


def test_tilt_with_drag_matches_analytic():
    """Constant velocity v: thrust must beat drag, tilt = atan(k_d v / g)."""
    v = np.tile(np.array([0.0, 10.0, 0.0]), (5, 1))
    a = np.zeros((5, 3))
    k_d = 0.295
    got = tilt_deg(a, v, k_d=k_d)
    want = np.degrees(np.arctan(k_d * 10.0 / 9.80665))
    assert np.allclose(got, want, atol=1e-9)


def test_direction_error_zero_after_constant_yaw():
    rng = np.random.default_rng(1)
    l_true = rng.normal(size=(200, 2)) * 0.1
    yaw = np.radians(63.0)
    r = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    l_est = l_true @ r.T * 3.0        # wrong scale and a constant yaw
    raw = direction_error_deg(l_est, l_true)
    fixed = direction_error_deg(l_est, l_true, remove_constant_yaw=True)
    assert np.median(raw) == pytest.approx(63.0, abs=1e-6)
    assert np.median(fixed) < 1e-6


def test_magnitude_ratio_and_correlation_scale_behaviour():
    rng = np.random.default_rng(2)
    l_true = rng.normal(size=(300, 2)) * 0.1
    for factor in (0.5, 2.0):
        l_est = factor * l_true
        assert magnitude_ratio(l_est, l_true) == pytest.approx(factor, rel=1e-9)
        fit = lean_correlation(l_est, l_true)
        assert fit["slope"] == pytest.approx(1.0 / factor, rel=1e-9)
        assert fit["r2"] == pytest.approx(1.0, abs=1e-12)
        assert fit["corr_x"] == pytest.approx(1.0, abs=1e-9)
