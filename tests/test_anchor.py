"""Unit tests for the path-anchor helpers (Track N3, :mod:`src.verdict.anchor`)."""
from __future__ import annotations

import numpy as np
import pytest

from src.verdict.anchor import (E3, G_ANCHOR, K_D_ANCHOR, angle_between_deg, find_windows,
                                first_sustained_motion, implied_thrust_axis, implied_tilt_deg,
                                last_sustained_motion, mount_rotated, propagate_from,
                                propagate_from_nearest, select_anchor, select_anchor_first,
                                snap_to_grid, up_cam_at_anchor)


def _grid(n: int = 300, hz: float = 100.0) -> np.ndarray:
    return np.arange(n) / hz


def test_zero_acceleration_and_velocity_is_vertical():
    assert np.allclose(implied_thrust_axis(np.zeros((1, 3)), np.zeros((1, 3))), E3[None])


def test_implied_tilt_matches_the_flatness_angle():
    acc = np.array([[1.0, 0.0, 0.0]])
    expected = np.degrees(np.arctan2(1.0, G_ANCHOR))
    assert implied_tilt_deg(acc, np.zeros((1, 3)))[0] == pytest.approx(expected, abs=1e-12)


def test_drag_enters_with_the_frozen_coefficient():
    vel = np.array([[G_ANCHOR / K_D_ANCHOR, 0.0, 0.0]])  # k_d * v == g -> 45 degrees
    assert implied_tilt_deg(np.zeros((1, 3)), vel)[0] == pytest.approx(45.0, abs=1e-9)


def test_free_fall_is_undefined_not_silently_zero():
    acc = np.array([[0.0, 0.0, -G_ANCHOR]])
    assert np.isnan(implied_tilt_deg(acc, np.zeros((1, 3)))[0])


def test_find_windows_requires_the_declared_duration():
    t = _grid(300, 100.0)  # 3 s
    theta = np.full(300, 10.0)
    theta[100:180] = 1.0   # 0.79 s -> too short for 1.0 s
    assert find_windows(theta, t, 2.0, 1.0) == []
    theta[100:200] = 1.0   # 0.99 s
    assert find_windows(theta, t, 2.0, 1.0) == []
    theta[100:201] = 1.0   # 1.00 s
    wins = find_windows(theta, t, 2.0, 1.0)
    assert len(wins) == 1 and wins[0].start == 100 and wins[0].stop == 201


def test_nan_samples_break_a_window():
    t = _grid(300, 100.0)
    theta = np.ones(300)
    theta[150] = np.nan
    assert len(find_windows(theta, t, 2.0, 1.0)) == 2


def test_selection_prefers_the_lowest_mean_then_the_earliest():
    t = _grid(400, 100.0)
    theta = np.full(400, 10.0)
    theta[10:160] = 1.5
    theta[240:390] = 0.5
    wins = find_windows(theta, t, 2.0, 1.0)
    assert len(wins) == 2
    best = select_anchor(wins)
    assert best.start == 240
    assert select_anchor_first(wins).start == 10


def test_ties_go_to_the_earliest_window():
    t = _grid(400, 100.0)
    theta = np.full(400, 10.0)
    theta[10:160] = 1.0
    theta[240:390] = 1.0
    wins = find_windows(theta, t, 2.0, 1.0)
    assert select_anchor(wins).start == 10


def test_snap_to_grid_picks_the_nearest_earlier_sample():
    t = np.array([0.0, 1.0, 2.0])
    assert snap_to_grid(0.4, t) == 0
    assert snap_to_grid(0.6, t) == 1
    assert snap_to_grid(-5.0, t) == 0


def test_propagation_is_the_identity_at_the_anchor_and_invertible_backwards():
    rng = np.random.default_rng(0)
    R = np.stack([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(10)])
    u0 = np.array([0.0, 0.0, 1.0])
    idx = np.arange(10)
    ups = propagate_from(R, 5, u0, idx)
    assert np.allclose(ups[5], R[5].T @ R[5] @ u0)
    # the angle between the propagated up at k and at the anchor equals the rotation between them
    ang = angle_between_deg(ups[2], ups[5])
    rel = angle_between_deg((R[2].T @ R[5]) @ u0, u0)
    assert ang == pytest.approx(rel, abs=1e-9)


def test_propagation_marks_frames_outside_the_pose_range_as_nan():
    R = np.stack([np.eye(3) for _ in range(4)])
    ups = propagate_from(R, 1, E3, np.array([2, 9, -1]))
    assert np.isfinite(ups[0]).all()
    assert np.isnan(ups[1]).all() and np.isnan(ups[2]).all()


def test_nearest_anchor_switches_at_the_midpoint():
    R = np.stack([np.eye(3) for _ in range(11)])
    ups = propagate_from_nearest(R, [0, 10], E3, np.arange(11))
    assert np.isfinite(ups).all()
    # with identity rotations every frame equals u0, so the switch cannot be seen; use a rotation
    th = np.deg2rad(10.0)
    Rp = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0, 0, 1]])
    R2 = np.stack([np.eye(3), Rp, Rp @ Rp, Rp @ Rp @ Rp])
    ups2 = propagate_from_nearest(R2, [0, 3], E3, np.arange(4))
    assert np.allclose(ups2[1], (R2[1].T @ R2[0]) @ E3)
    assert np.allclose(ups2[2], (R2[2].T @ R2[3]) @ E3)


def test_mount_rotation_changes_the_anchor_direction_by_the_perturbation():
    R_cb = np.array([[0.0, 0.7, 0.7], [-1.0, 0.0, 0.0], [0.0, -0.7, 0.7]])
    u = up_cam_at_anchor(R_cb)
    for deg in (2.0, 5.0):
        u2 = up_cam_at_anchor(mount_rotated(R_cb, deg, "x"))
        assert angle_between_deg(u[None], u2[None])[0] == pytest.approx(deg, abs=1e-9)


def test_sustained_motion_bounds():
    t = _grid(500, 10.0)  # 50 s at 10 Hz
    speed = np.zeros(500)
    speed[100:300] = 5.0
    assert first_sustained_motion(t, speed, 2.0, 0.5) == pytest.approx(t[100])
    assert last_sustained_motion(t, speed, 2.0, 0.5) == pytest.approx(t[299])
    assert first_sustained_motion(t, np.zeros(500)) == float("inf")
    assert last_sustained_motion(t, np.zeros(500)) == float("-inf")


def test_sustained_motion_ignores_a_single_spike():
    t = _grid(200, 10.0)
    speed = np.zeros(200)
    speed[50] = 9.0  # one sample is not 0.5 s
    assert first_sustained_motion(t, speed, 2.0, 0.5) == float("inf")
