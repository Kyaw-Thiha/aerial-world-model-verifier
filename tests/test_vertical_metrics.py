"""Unit tests for the Track V statistic (src/vertical/metrics.py)."""
from __future__ import annotations

import numpy as np

from src.vertical import metrics as M


def _truth(z0=12.0, dz=20.0, speed=5.0, dur=10.0, dt=0.02):
    t = np.arange(0, dur + 1e-9, dt)
    pos = np.column_stack([speed * t, np.zeros_like(t), z0 + dz * t / dur])
    return t, pos


def _poses(c):
    P = np.repeat(np.eye(4)[None], len(c), 0)
    P[:, :3, 3] = c
    return P


def test_perfect_depiction_has_zero_error():
    t, pos = _truth()
    scale = 0.3                      # reconstruction units per metre
    h0 = 12.0 * scale
    # camera height in reconstruction units follows the truth exactly
    th = pos[:, 2] * scale
    c = np.column_stack([0.1 * pos[:, 0], np.zeros_like(t), th])
    # the plane is z = 0 in this gauge, normal +z
    rows = M.window_statistics(t, _poses(c), np.array([0, 0, 1.0]), 0.0, t, pos, mode="sim")
    ok = [r for r in rows if r["ok"]]
    assert len(ok) >= 10
    assert max(abs(r["e"]) for r in ok) < 1e-9
    assert h0 > 0


def test_removed_vertical_motion_is_log_of_the_ratio():
    t, pos = _truth()
    scale = 0.3
    h = np.full_like(t, 12.0 * scale)          # camera holds altitude
    c = np.column_stack([0.1 * pos[:, 0], np.zeros_like(t), h])
    rows = M.window_statistics(t, _poses(c), np.array([0, 0, 1.0]), 0.0, t, pos, mode="sim")
    ok = [r for r in rows if r["ok"]]
    for r in ok:
        assert abs(r["e"] - (-np.log(r["ratio_cmd"] + 0.0))) < 1e-9
        assert r["e"] < 0


def test_defect_reference_matches_the_definitions():
    r = 1.8
    assert abs(M.defect_reference(r, "removed") - np.log(1 / r)) < 1e-12
    assert abs(M.defect_reference(r, "scaled_0.5") - np.log(1.4 / r)) < 1e-12
    assert abs(M.defect_reference(r, "scaled_1.5") - np.log(2.2 / r)) < 1e-12
    assert abs(M.defect_reference(r, "reversed") - np.log(0.2 / r)) < 1e-12


def test_real_mode_uses_the_scale_and_commanded_dz():
    t, pos = _truth(z0=2.0, dz=1.0, speed=6.0, dur=8.0)
    s = 4.0                                   # metres per reconstruction unit
    h = (pos[:, 2]) / s                       # perfect depiction in recon units
    c = np.column_stack([pos[:, 0] / s, np.zeros_like(t), h])
    rows = M.window_statistics(t, _poses(c), np.array([0, 0, 1.0]), 0.0, t, pos, mode="real",
                               scale_s=s)
    ok = [r for r in rows if r["ok"]]
    assert ok
    assert max(abs(r["e"]) for r in ok) < 1e-9


def test_sequence_scale_is_rotation_free():
    rng = np.random.default_rng(0)
    pos = np.column_stack([np.linspace(0, 30, 50), rng.normal(size=50) * 3,
                           np.linspace(0, 5, 50)])
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    cams = pos @ R.T
    s = M.sequence_scale(cams, pos)
    # pos = s * (rotated cams) in radius: the ratio of RMS radii is 1/s
    assert abs(s - 1.0) < 1e-9
    assert abs(M.sequence_scale(2 * cams, pos) - 0.5) < 1e-9
