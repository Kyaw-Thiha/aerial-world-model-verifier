"""Check that the UZH scorer reduces to Track A2's scorer when the mount is a TII uptilt.

Track A2's `score_clip` parameterises the mount by `camera_to_body_pitched(uptilt)` and fits a
1-degree-step offset in [-7, 7]. This test feeds the same synthetic data to both, with the
mount fixed at the same value, and requires the scores to agree to floating-point noise.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.depicted.stat_a2 import score_clip_mount
from src.perception.frames import camera_to_body_pitched
from src.score.verdict import score_clip

PREREG = {
    "nuisances": {"uptilt_offset_deg": {"bounds": [0.0, 0.0], "step": 1.0},
                  "lag_ms": {"bounds": [0.0, 0.0], "step": 25.0}},
    "abstention": {},
}


def _synthetic(n=40, uptilt_deg=50.0, tilt_deg=25.0, seed=0):
    """A body attitude with a real tilt, gravity through the mount, and a matching path."""
    rng = np.random.default_rng(seed)
    R_bw = np.zeros((n, 3, 3))
    from scipy.spatial.transform import Rotation

    for i in range(n):
        yaw = rng.uniform(-np.pi, np.pi)
        pitch = np.deg2rad(tilt_deg)
        R_bw[i] = Rotation.from_euler("ZYX", [yaw, pitch, 0.0]).as_matrix()
    R_cb = camera_to_body_pitched(np.deg2rad(uptilt_deg))
    up_b = np.einsum("nji,j->ni", R_bw, np.array([0.0, 0.0, 1.0]))
    up_cam = np.einsum("ij,nj->ni", R_cb, up_b)
    imp = R_bw[:, :, 2]
    yaw = Rotation.from_matrix(R_bw).as_euler("ZYX")[:, 0]
    t = np.arange(n) * 0.08
    # inject a known residual by perturbing the depicted gravity by 6 deg about camera x
    ang = np.deg2rad(6.0)
    Rx = np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]])
    up_cam = np.einsum("ij,nj->ni", Rx, up_cam)
    implied_tilt = np.full(n, 25.0)
    return up_cam, imp, yaw, t, R_cb, implied_tilt


def test_reduces_to_a2_score_clip():
    up_cam, imp, yaw, t, R_cb, implied_tilt = _synthetic()
    valid = np.ones(len(t), bool)
    a2 = score_clip(up_cam, imp, yaw, t, 50.0, implied_tilt, valid, PREREG,
                    nuisances={"ok": True, "loss": 0.0, "uptilt_offset_deg": 0.0, "lag_ms": 0.0})
    mine = score_clip_mount(up_cam, imp, yaw, t, R_cb, implied_tilt, valid,
                            uptilt_offsets_deg=(0.0,), lag_grid_ms=np.array([0.0]))
    assert mine["testable"] == a2["testable"]
    assert mine["score"] == pytest.approx(a2["score"], abs=1e-9)
    assert mine["n_scored"] == a2["n_scored"]
    assert mine["score"] > 1.0  # the injected 6 deg perturbation must show up


def test_abstention_rule_is_a2_rule():
    up_cam, imp, yaw, t, R_cb, implied_tilt = _synthetic(n=40)
    valid = np.ones(len(t), bool)
    low = np.full(len(t), 5.0)          # implied tilt below the 10 deg signal floor
    out = score_clip_mount(up_cam, imp, yaw, t, R_cb, low, valid)
    assert out["testable"] is False
    assert any("signal_frames" in r for r in out["abstain_reasons"])


def test_mount_rotation_is_the_only_difference():
    up_cam, imp, yaw, t, R_cb, implied_tilt = _synthetic()
    valid = np.ones(len(t), bool)
    good = score_clip_mount(up_cam, imp, yaw, t, R_cb, implied_tilt, valid,
                           lag_grid_ms=np.array([0.0]))
    # a 10 deg mount pitch error must be visible in the score
    from src.depicted.stat_a2 import rotate_mount_about_body_y

    bad_R = rotate_mount_about_body_y(R_cb, np.deg2rad(10.0))
    bad = score_clip_mount(up_cam, imp, yaw, t, bad_R, implied_tilt, valid,
                           lag_grid_ms=np.array([0.0]))
    assert bad["score"] > good["score"] + 3.0


def test_mount_offset_sign_matches_a2_family():
    """Ry(-delta) @ R_cb(theta) must equal R_cb(theta + delta) for TII's mount family."""
    import numpy as np

    from src.depicted.stat_a2 import rotate_mount_about_body_y

    rng = np.random.default_rng(3)
    for theta_deg in (0.0, 30.0, 50.0):
        for delta in (-7.0, -2.0, 3.0, 7.0):
            R0 = camera_to_body_pitched(np.deg2rad(theta_deg))
            R1 = camera_to_body_pitched(np.deg2rad(theta_deg + delta))
            R2 = rotate_mount_about_body_y(np.eye(3), -np.deg2rad(delta)) @ R0
            assert np.allclose(R1, R2, atol=1e-12)
            up_cam = rng.normal(size=(1, 3))
            assert np.allclose(R1 @ up_cam.T, R2 @ up_cam.T, atol=1e-12)
