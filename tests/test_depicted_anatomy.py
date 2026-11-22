"""Unit tests for the Track J reader-error anatomy (src/depicted/anatomy.py)."""
from __future__ import annotations

import numpy as np

from src.depicted.anatomy import (
    decompose, dct_split, contiguous_runs, fit_rotation, tail_runs, tangent_error,
    trim_mask,
)
from src.perception.frames import camera_to_body_pitched


def _rot(axis: np.ndarray, ang: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * (k @ k)


def test_tangent_error_is_an_exact_two_axis_split():
    rng = np.random.default_rng(0)
    R_cb = camera_to_body_pitched(np.deg2rad(40.0))
    up = rng.standard_normal((200, 3))
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    pert = up + 0.02 * rng.standard_normal((200, 3))
    e = tangent_error(pert, up, R_cb)
    assert np.allclose(np.hypot(e["pitch"], e["roll"]), e["theta"], atol=1e-9)


def test_tangent_error_axes_are_pure_pitch_and_roll():
    R_cb = camera_to_body_pitched(np.deg2rad(30.0))
    up = (R_cb.T @ np.array([0.0, 0.0, 1.0]))[None, :]   # level: body up == body z
    # rotating the up-vector about the body x axis is a roll of the airframe, and the project's
    # convention is pitch = -asin(u_x), roll = atan2(u_y, u_z), so a positive rotation about
    # +x gives a NEGATIVE roll angle
    R = _rot(np.array([1.0, 0.0, 0.0]), np.deg2rad(3.0))
    e = tangent_error(((R_cb.T @ R @ R_cb) @ up.T).T, up, R_cb)
    assert abs(e["roll"][0] + 3.0) < 1e-6 and abs(e["pitch"][0]) < 1e-9
    # rotating about the body y axis is a pure pitch, likewise with a negative sign
    R = _rot(np.array([0.0, 1.0, 0.0]), np.deg2rad(3.0))
    e = tangent_error(((R_cb.T @ R @ R_cb) @ up.T).T, up, R_cb)
    assert abs(e["pitch"][0] + 3.0) < 1e-6 and abs(e["roll"][0]) < 1e-9


def test_dct_split_is_orthogonal_and_separates_a_slow_ramp():
    dt, n = 1 / 12.0, 600
    t = np.arange(n) * dt
    x = np.sin(2 * np.pi * 0.1 * t) + 0.3 * np.random.default_rng(3).standard_normal(n)
    slow, fast = dct_split(x, 1.0 / dt)
    assert np.isclose(np.mean((slow + fast - x) ** 2), 0.0, atol=1e-20)
    # energy is conserved (the DCT is orthogonal, so this holds on x, not on x - mean: the
    # k = 0 coefficient is kept by the slow part)
    assert np.isclose(np.mean(slow ** 2) + np.mean(fast ** 2), np.mean(x ** 2), rtol=1e-9)
    assert np.isclose(np.mean(fast), 0.0, atol=1e-12)
    # the 0.1 Hz ramp lives in the slow part
    assert np.mean(slow ** 2) > 5 * np.mean(fast ** 2)


def test_decompose_closes_on_a_known_process():
    dt, n = 1 / 12.0, 1200
    t = np.arange(n) * dt
    rng = np.random.default_rng(1)
    x = 2.0 + 1.5 * np.sin(2 * np.pi * 0.05 * t) + 0.5 * rng.standard_normal(n)
    d = decompose(x, t, contiguous_runs(t))
    assert abs(d["closure"]) < 1e-9
    assert np.isclose(d["share_bias"] + d["share_slow"] + d["share_fast"], 1.0, atol=1e-9)
    assert 1.5 < d["bias"] < 2.5


def test_contiguous_runs_breaks_on_a_gap():
    t = np.concatenate([np.arange(0, 10) * 0.1, 5.0 + np.arange(0, 10) * 0.1])
    assert contiguous_runs(t) == [(0, 10), (10, 20)]


def test_trim_mask_drops_the_worst_tenth():
    x = np.arange(100, dtype=float)
    keep = trim_mask(x, 0.10)
    assert keep.sum() == 90 and not keep[-1]


def test_tail_runs_counts_maximal_spans():
    mask = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], bool)
    assert tail_runs(mask, [(0, len(mask))]) == [2, 1, 3]


def test_fit_rotation_recovers_a_known_rotation():
    rng = np.random.default_rng(2)
    src = rng.standard_normal((50, 3))
    src /= np.linalg.norm(src, axis=1, keepdims=True)
    R = _rot(np.array([0.3, -0.2, 0.9]), np.deg2rad(7.0))
    dst = (R @ src.T).T
    R_hat = fit_rotation(src, dst)
    assert np.allclose(R_hat, R, atol=1e-9)
