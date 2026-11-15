"""Unit tests for scoring, priors and reliability signals (synthetic only)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.pathrec.evaluate import relative_rotation_deg, score_clip, up_axis
from src.pathrec.priors import (consistent, interval, lean_regression, scale_ratio,
                                time_rescale)
from src.pathrec.reliability import (direction_agreement, jitter_ratio, parallax_to_depth,
                                    rotation_rate_deg)


def circle(n=400, dt=1 / 30, r=3.0, w=0.6, z=1.5):
    t = np.arange(n) * dt
    return np.stack([r * np.cos(w * t), r * np.sin(w * t), np.full(n, z)], 1)


def test_identity_scores_zero():
    p = circle()
    s = score_clip(p, p.copy(), 1 / 30, window_s=0.5)
    assert s["ate_sim3_m"] == pytest.approx(0.0, abs=1e-9)
    assert s["step_dir_err_deg_median"] == pytest.approx(0.0, abs=1e-9)
    assert s["accel_dir_err_deg_median"] == pytest.approx(0.0, abs=1e-9)
    assert s["accel_mag_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert s["lean_r2"] == pytest.approx(1.0, abs=1e-9)
    assert s["tilt_err_deg_median"] == pytest.approx(0.0, abs=1e-6)


def test_scale_error_hidden_by_sim3_but_not_se3():
    p = circle()
    s = score_clip(0.5 * p, p, 1 / 30)
    assert s["ate_sim3_m"] == pytest.approx(0.0, abs=1e-7)
    assert s["scale_sim3"] == pytest.approx(2.0, rel=1e-6)
    assert s["path_ratio_se3"] == pytest.approx(0.5, rel=1e-6)
    assert s["ate_se3_m"] > 0.5


def test_noise_degrades_direction_but_not_necessarily_scale():
    rng = np.random.default_rng(3)
    p = circle()
    noisy = p + rng.normal(scale=0.02, size=p.shape)
    clean = score_clip(p, p, 1 / 30)
    bad = score_clip(noisy, p, 1 / 30)
    assert bad["step_dir_err_deg_median"] > clean["step_dir_err_deg_median"]
    assert bad["accel_dir_err_deg_median"] > clean["accel_dir_err_deg_median"]


def test_up_axis_detection_and_ambiguity():
    p = circle()
    u = up_axis(p)
    assert u["axis"] == 2 and u["unambiguous"]
    flat = np.zeros((50, 3))           # no extent at all: cannot be unambiguous
    assert up_axis(flat)["unambiguous"] is False


def test_relative_rotation_error_zero_and_known():
    ang = np.linspace(0, 1, 50)
    r1 = Rotation.from_rotvec(np.c_[np.zeros_like(ang), np.zeros_like(ang), ang]).as_matrix()
    assert np.allclose(relative_rotation_deg(r1, r1), 0, atol=1e-4)
    ang2 = ang + 0.2                                   # a constant yaw offset
    r2 = Rotation.from_rotvec(np.c_[np.zeros_like(ang2), np.zeros_like(ang2), ang2]).as_matrix()
    err = relative_rotation_deg(r1, r2)
    assert np.median(err) == pytest.approx(0.0, abs=1e-6)  # relative rates coincide


def test_scale_ratio_and_interval_rule():
    assert scale_ratio(2 * circle(), circle()) == pytest.approx(0.5, rel=1e-9)
    iv = interval([0.9, 1.0, 1.1, 50.0], coverage=0.9)
    assert iv["n"] == 4 and iv["low"] <= iv["median"] <= iv["high"]
    assert consistent(1.0, interval([0.9, 1.0, 1.1]))
    assert not consistent(5.0, interval([0.9, 1.0, 1.1]))


def test_lean_regression_reports_apparent_gravity_and_is_scale_free():
    """l = a/g  ->  slope 1/g. An unknown scale on a divides the slope and leaves R^2."""
    g = 9.80665
    rng = np.random.default_rng(4)
    a = rng.normal(size=(500, 2)) * 1.5
    l = a / g
    fit = lean_regression(l, a)
    assert fit["g_app"] == pytest.approx(g, rel=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)
    for f in (0.3, 4.0):
        scaled = lean_regression(l, a * f)
        assert scaled["g_app"] == pytest.approx(g * f, rel=1e-9)
        assert scaled["r2"] == pytest.approx(fit["r2"], abs=1e-12)
        assert scaled["r"] == pytest.approx(fit["r"], abs=1e-12)


def test_time_rescale_is_quadratic_in_acceleration():
    a = np.ones((4, 2))
    assert np.allclose(time_rescale(a, 2.0), 4 * a)
    with pytest.raises(ValueError):
        time_rescale(a, 0.0)


def test_reliability_signals():
    p = circle()
    rng = np.random.default_rng(7)
    assert jitter_ratio(p) < 0.1                                  # smooth, curved flight
    assert jitter_ratio(p + rng.normal(scale=0.02, size=p.shape)) > 0.3
    assert parallax_to_depth(p, 10.0) > 0
    assert np.isnan(parallax_to_depth(p, np.nan))
    assert direction_agreement(p, p) == pytest.approx(0.0, abs=1e-9)
    # 90 deg rotation about z of the whole track changes the step directions by 90 deg
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    assert direction_agreement(p @ rot.T, p) == pytest.approx(90.0, abs=1e-6)
    ang = np.linspace(0, 1, 50)
    r = Rotation.from_rotvec(np.c_[0 * ang, 0 * ang, ang]).as_matrix()
    assert rotation_rate_deg(r) > 0


def test_prereg_guard_detects_untracked_and_clean(tmp_path):
    """The runner must only score against a committed, unmodified pre-registration."""
    import subprocess
    from pathlib import Path

    from experiments.pathrec_bakeoff import prereg_status

    st = prereg_status(Path("configs/pathrec_prereg.yaml"))
    assert st["tracked"] is True and st["modified"] is False

    tmp = tmp_path / "not_tracked.yaml"
    tmp.write_text("version: 1\n")
    st2 = prereg_status(tmp)
    assert st2["tracked"] is False


def test_apparent_gravity_is_scale_free_and_fps_quadratic():
    """Gate 2's statistic: g_app = 1/slope, unchanged by an unknown scale, x4 for 2x fps."""
    from experiments.pathrec_priors_study import analyse_clip
    rng = np.random.default_rng(0)
    n, dt = 300, 1 / 15
    t = np.arange(n) * dt
    a = np.c_[2.0 * np.sin(0.7 * t), 1.5 * np.cos(0.5 * t)]
    pos = np.c_[np.cumsum(np.cumsum(a[:, 0]) * dt ** 2),
                np.cumsum(np.cumsum(a[:, 1]) * dt ** 2), np.zeros(n)]
    q = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    for s in (1.0, 0.2):
        c2w = np.tile(np.eye(4), (n, 1, 1))
        c2w[:, :3, 3] = s * pos
        r = analyse_clip(c2w, pos, q, 15.0)
        assert r["scale_ratio"] == pytest.approx(1.0 / s, rel=1e-6)
        assert r["g_app_oracle_lean"] == pytest.approx(9.80665, rel=1e-3)
        assert r["g_app_fps_x2"] == pytest.approx(4 * 9.80665, rel=1e-2)
        assert r["r2_oracle_lean"] == pytest.approx(1.0, abs=1e-6)
