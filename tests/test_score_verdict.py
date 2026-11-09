"""The verdict's geometry, nuisance fit, abstention and metrics, on synthetic truth."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.perception.frames import camera_to_body_pitched
from src.score import (auroc, conformal_quantile, depicted_axis_world, fit_nuisances,
                       score_clip, spearman)
from src.score.verdict import rotation_between_batch, speed_bin

E3 = np.array([0.0, 0.0, 1.0])
PREREG = {
    "nuisances": {"uptilt_offset_deg": {"bounds": [-7.0, 7.0], "step": 1.0},
                  "lag_ms": {"bounds": [-150.0, 150.0], "step": 25.0}},
    "abstention": {},
}


def _trajectory(n=200, tilt_deg=20.0, seed=0, dt=1 / 12):
    """Smooth banking attitude, its yaw, and the implied (true) thrust axis."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    yaw = 0.4 * np.sin(0.3 * t)
    roll = np.deg2rad(tilt_deg) * np.sin(0.6 * t + 0.4)
    pitch = np.deg2rad(tilt_deg) * 0.5 * np.sin(0.5 * t)
    R = (Rotation.from_rotvec(np.column_stack([np.zeros(n), np.zeros(n), yaw]))
         * Rotation.from_rotvec(np.column_stack([np.zeros(n), pitch, np.zeros(n)]))
         * Rotation.from_rotvec(np.column_stack([roll, np.zeros(n), np.zeros(n)]))).as_matrix()
    imp = np.einsum("nij,j->ni", R, E3)
    return t, np.unique(np.round(Rotation.from_matrix(R).as_euler("ZYX")[:, 0], 12))[:1] if False else t, R, yaw, imp


def test_rotation_between_batch_recovers_the_rotation():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(20, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    R = Rotation.random(20, random_state=2).as_matrix()
    b = np.einsum("nij,nj->ni", R, a)
    Rc = rotation_between_batch(a, b)
    assert np.allclose(np.einsum("nij,nj->ni", Rc, a), b, atol=1e-12)
    assert np.allclose(np.linalg.det(Rc), 1.0)


def test_depicted_axis_is_up_for_a_level_airframe():
    R_cb = camera_to_body_pitched(np.deg2rad(50.0))
    up_cam = np.tile(R_cb.T @ E3, (10, 1))       # gravity straight down the body z axis
    z = depicted_axis_world(up_cam, R_cb, np.zeros(10))
    assert np.allclose(z, np.tile(E3, (10, 1)), atol=1e-9)


def test_nuisance_fit_recovers_a_mount_offset_and_a_lag():
    n = 240
    t = np.arange(n) / 12.0
    yaw = 0.3 * np.sin(0.4 * t)
    roll = np.deg2rad(25.0) * np.sin(0.8 * t)
    pitch = np.deg2rad(10.0) * np.sin(0.5 * t + 1.0)
    R_bw = (Rotation.from_rotvec(np.column_stack([np.zeros(n), np.zeros(n), yaw]))
            * Rotation.from_rotvec(np.column_stack([np.zeros(n), pitch, np.zeros(n)]))
            * Rotation.from_rotvec(np.column_stack([roll, np.zeros(n), np.zeros(n)]))).as_matrix()
    theta_true = 45.0
    R_cb_true = camera_to_body_pitched(np.deg2rad(theta_true))
    # depicted up in camera for the true attitude (exact, no estimator error)
    up_body = np.einsum("nji,j->ni", R_bw, E3)        # world up in body coords
    up_cam = np.einsum("ji,nj->ni", R_cb_true, up_body)  # then in camera coords
    # implied axis is what the path says; a lag of 75 ms means the depicted series
    # corresponds to the path 75 ms earlier, so the fit must find +75 ms
    lag_true = 0.075
    imp = np.einsum("nij,j->ni", R_bw, E3)
    implied_tilt = np.degrees(np.arccos(np.clip(imp[:, 2], -1, 1)))
    mask = implied_tilt > 10.0
    # the recorded mount is 5 deg off the truth
    fit = fit_nuisances(up_cam, imp, yaw, t, theta_true - 5.0, mask, PREREG)
    assert fit["ok"]
    assert fit["uptilt_offset_deg"] == pytest.approx(5.0, abs=1.0)
    assert fit["loss"] < 2.0
    # and with a lag in the data the fit finds it
    imp_lagged = np.column_stack([np.interp(t - lag_true, t, imp[:, k]) for k in range(3)])
    fit2 = fit_nuisances(up_cam, imp_lagged, yaw, t, theta_true - 5.0, mask, PREREG)
    assert fit2["lag_ms"] == pytest.approx(75.0, abs=25.0)


def test_conformal_quantile_and_coverage():
    rng = np.random.default_rng(0)
    s = rng.normal(size=200)
    q = conformal_quantile(s, alpha=0.05)
    assert (s > q).mean() <= 0.05
    assert (s > q).mean() > 0.02
    assert conformal_quantile(np.array([])) != conformal_quantile(np.array([]))  # nan
    assert conformal_quantile(np.array([1.0]), 0.05) == float("inf")


def test_auroc_detect_orientation():
    from src.score import auroc_detect
    real = np.array([2.0, 3.0, 4.0])
    viol = np.array([20.0, 30.0, 40.0])
    assert auroc_detect(real, viol) == 1.0        # violations score higher -> detected
    assert auroc_detect(viol, real) == 0.0


def test_auroc_and_spearman_behave():
    assert auroc(np.array([5.0, 6.0]), np.array([0.0, 1.0])) == 1.0
    assert auroc(np.array([0.0, 1.0]), np.array([5.0, 6.0])) == 0.0
    assert auroc(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.5
    assert spearman(np.arange(10.0), np.arange(10.0)) == pytest.approx(1.0)
    assert spearman(np.arange(10.0), -np.arange(10.0)) == pytest.approx(-1.0)


def test_clip_abstains_without_excitation():
    n = 120
    t = np.arange(n) / 12.0
    R_cb = camera_to_body_pitched(np.deg2rad(30.0))
    up_cam = np.tile(R_cb.T @ E3, (n, 1))
    imp = np.tile(E3, (n, 1))
    prereg = dict(PREREG)
    prereg["abstention"] = {"rule": "not_testable_if_any"}
    out = score_clip(up_cam, imp, np.zeros(n), t, 30.0, np.zeros(n), np.ones(n, bool), prereg)
    assert not out["testable"]
    assert any("signal_frames" in r for r in out["abstain_reasons"])


def test_speed_bin_edges():
    bins = [[0, 2], [2, 5], [20, 99]]
    assert speed_bin(1.0, bins) == "0-2"
    assert speed_bin(2.0, bins) == "2-5"
    assert speed_bin(50.0, bins) == "20-99"
    assert speed_bin(-1.0, bins) == "0-2"
