"""Unit tests for the Track V ground-plane rule (src/vertical/plane.py).

Synthetic scenes with a known plane, in an arbitrary gauge (the reconstruction's frame is not
gravity-aligned, amendment 3), including a competing rooftop plane and a decoy wall.
"""
from __future__ import annotations

import numpy as np

from src.vertical.plane import fit_ground_plane, scene_scale


def _rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _scene(seed: int = 1, n_cam: int = 20, building: bool = True):
    rng = np.random.default_rng(seed)
    # cameras fly along x at 10 m, in a gravity-aligned frame
    t = np.linspace(0, 40, n_cam)
    cams = np.column_stack([t, np.zeros(n_cam), 10 + 0.05 * t])
    ground = np.column_stack([rng.uniform(-50, 90, 6000), rng.uniform(-50, 50, 6000),
                              0.02 * rng.normal(size=6000)])
    pts = [ground]
    if building:
        # a wall along y at x=20, z in [0, 6]
        wall = np.column_stack([20 + 0.02 * rng.normal(size=2000), rng.uniform(-20, 20, 2000),
                                rng.uniform(0, 6, 2000)])
        roof = np.column_stack([rng.uniform(10, 30, 1500), rng.uniform(-20, 20, 1500),
                                6 + 0.02 * rng.normal(size=1500)])
        pts += [wall, roof]
    p = np.concatenate(pts)
    R = _rot(rng)
    s = 0.37
    off = rng.normal(size=3) * 5
    return (p - off) @ R.T * s, (cams - off) @ R.T * s, R, s


def test_finds_the_ground_in_an_arbitrary_gauge():
    pts, cams, R, s = _scene()
    travel = cams[-1] - cams[0]
    fit = fit_ground_plane(pts, cams, travel=travel)
    assert fit["ok"]
    n = np.asarray(fit["plane"].n)
    # the true ground normal in the gauge is R @ e3
    true_n = R @ np.array([0.0, 0.0, 1.0])
    assert abs(float(n @ true_n)) > 0.99
    # heights: the cameras are 10 m above the ground, scaled by s
    h = cams @ n - fit["plane"].d
    assert abs(float(np.mean(h)) - 10.0 * s) < 1.0 * s
    assert fit["coverage_ok"]


def test_rejects_a_plane_that_leaves_cameras_below():
    rng = np.random.default_rng(3)
    n_cam = 12
    cams = np.column_stack([np.linspace(0, 20, n_cam), np.zeros(n_cam), np.full(n_cam, 5.0)])
    ground = np.column_stack([rng.uniform(-20, 40, 4000), rng.uniform(-20, 20, 4000),
                              0.02 * rng.normal(size=4000)])
    # a huge ceiling above the cameras: the largest plane, but the cameras are below it
    ceiling = np.column_stack([rng.uniform(-20, 40, 12000), rng.uniform(-20, 20, 12000),
                               20 + 0.02 * rng.normal(size=12000)])
    pts = np.concatenate([ground, ceiling])
    fit = fit_ground_plane(pts, cams, travel=cams[-1] - cams[0])
    assert fit["ok"]
    n = np.asarray(fit["plane"].n)
    assert abs(float(n @ np.array([0.0, 0.0, 1.0]))) > 0.99
    h = cams @ n - fit["plane"].d
    assert np.all(h > 0)


def test_scene_scale_is_positive_and_rotation_invariant():
    pts, cams, R, s = _scene(seed=5)
    a = scene_scale(pts, cams)
    R2 = _rot(np.random.default_rng(9))
    b = scene_scale(pts @ R2.T, cams @ R2.T)
    assert a > 0 and abs(a / b - 1.0) < 1e-9


def test_hover_window_has_no_motion_condition():
    rng = np.random.default_rng(7)
    cams = np.tile(np.array([[0.0, 0.0, 8.0]]), (10, 1))
    cams[:, 0] += rng.normal(scale=0.01, size=10)
    ground = np.column_stack([rng.uniform(-20, 20, 3000), rng.uniform(-20, 20, 3000),
                              0.02 * rng.normal(size=3000)])
    fit = fit_ground_plane(ground, cams, travel=cams[-1] - cams[0])
    assert fit["ok"]
    assert np.isnan(fit["plane"].facing_deg)
