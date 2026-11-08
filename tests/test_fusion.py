"""Tests for the gravity alignment and the complementary attitude filter."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.fusion import (body_z_from_camera, complementary_filter, gravity_align,
                        lean_error_deg, relative_rotations, rotation_between)

E3 = np.array([0.0, 0.0, 1.0])


def _smooth_rotation(n=400, seed=0):
    rng = np.random.default_rng(seed)
    steps = Rotation.from_rotvec(rng.normal(scale=0.01, size=(n, 3)))
    return steps.as_matrix()


def test_rotation_between_is_minimal():
    a = np.array([0.0, -1.0, 0.0])
    b = E3
    R = rotation_between(a, b)
    assert np.allclose(R @ a, b, atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.isclose(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())), 90.0)


def test_gravity_align_recovers_up_and_reports_spread():
    n = 200
    R_true = _smooth_rotation(n)
    R_align_true = rotation_between(E3, np.array([1.0, 0.0, 0.0]))
    R_vo = np.einsum("ij,njk->nik", R_align_true, R_true)
    up_cam = np.tile(np.array([0.0, -0.686, 0.727]), (n, 1))  # 50 deg mount, level
    up_cam = up_cam / np.linalg.norm(up_cam)
    out = gravity_align(R_vo, up_cam)
    # applying the alignment puts the mean measured up on world up
    u = np.einsum("ij,nj->ni", out["R_align"], np.einsum("nij,nj->ni", R_vo, up_cam))
    assert np.allclose(u.mean(axis=0) / np.linalg.norm(u.mean(axis=0)), E3, atol=1e-9)
    assert out["spread_median_deg"] < 1.0     # small steps: near-constant up in VO


def test_complementary_filter_tracks_perfect_inputs():
    n = 150
    R_true = _smooth_rotation(n, seed=1)
    up_cam = np.tile(np.array([0.0, -1.0, 0.0]), (n, 1))
    rel = relative_rotations(R_true)
    out = complementary_filter(rel, up_cam, gain=1.0, R0=R_true[0])
    assert np.abs(out.up_err_after_deg[1:]).max() < 1e-6


def test_gain_zero_follows_vo_only_and_gain_one_snaps_to_measurement():
    n = 60
    R_true = _smooth_rotation(n, seed=2)
    rel = relative_rotations(R_true)
    up_true = np.tile(np.array([0.0, -0.766, 0.643]), (n, 1))
    up_true /= np.linalg.norm(up_true, axis=1, keepdims=True)
    # start the filter with a 20 deg error about the x axis
    R0 = Rotation.from_euler("x", 20.0, degrees=True).as_matrix() @ R_true[0]
    out0 = complementary_filter(rel, up_true, gain=0.0, R0=R0)
    assert out0.up_err_after_deg[0] > 5.0                  # error is never corrected
    assert abs(out0.up_err_after_deg[0] - out0.up_err_after_deg[-1]) < 2.0
    out1 = complementary_filter(rel, up_true, gain=1.0, R0=R0)
    assert np.abs(out1.up_err_after_deg).max() < 1e-6      # snapped on the first frame
    outp = complementary_filter(rel, up_true, gain=0.2, R0=R0)
    e = outp.up_err_after_deg
    assert e[0] > e[3] > e[10] > e[40]                     # monotone decay
    assert e[0] > 20.0                                     # 50 deg mount + 20 deg offset
    assert e[40] < 1.0                                     # converged after ~0.3 s at 120 Hz
    assert e[-1] < e[40]


def test_body_z_and_lean_error():
    R_cb = np.array([[0.0, 0.766, 0.643], [-1.0, 0.0, 0.0], [0.0, -0.643, 0.766]])
    R_cw = np.tile(np.eye(3), (5, 1, 1))
    z = body_z_from_camera(R_cw, R_cb)
    assert np.allclose(z[:, 2], 0.766)          # level airframe, body z up
    assert np.allclose(z[0], [0.0, -0.643, 0.766])   # body z in camera = third row
    tru = np.tile(np.array([0.3, 0.0, 0.95]), (5, 1))
    tru /= np.linalg.norm(tru, axis=1, keepdims=True)
    est = np.tile(np.array([0.0, 0.3, 0.95]), (5, 1))
    est /= np.linalg.norm(est, axis=1, keepdims=True)
    err = lean_error_deg(est, tru, min_tilt_deg=5.0, yaw_align=0.0)
    assert err["n"] == 5
    assert err["median_abs_deg"] == pytest.approx(90.0, abs=1e-6)
