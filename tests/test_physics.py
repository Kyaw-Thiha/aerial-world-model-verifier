"""Physics floor tests: differentiation, flatness inversion, drag fitting.

The point of these tests is not coverage for its own sake. Each one pins a
failure mode that would silently invalidate the floor number:
a wrong gravity sign, a wrong body axis, an acceleration that is off by fps^2,
and a drag coefficient that is fitted but not identifiable.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.physics.diff import savgol_derivatives, edge_slice
from src.physics.flatness import (
    G0,
    angle_between_deg,
    fit_kd,
    implied_thrust_axis,
    implied_thrust_axis_body_drag,
    thrust_axis_from_quat,
    tilt_deg,
    valid_thrust_mask,
)


def _quadratic_track(a: np.ndarray, v0: np.ndarray, p0: np.ndarray, dt: float, n: int):
    t = np.arange(n) * dt
    pos = p0 + v0 * t[:, None] + 0.5 * a * (t**2)[:, None]
    vel = v0 + a * t[:, None]
    return t, pos, vel


def _attitude_from_thrust_axis(z_B: np.ndarray, yaw_deg: float = 0.0) -> Rotation:
    """Build a body-to-world rotation whose third column is z_B (any yaw)."""
    z = z_B / np.linalg.norm(z_B)
    yaw = np.radians(yaw_deg)
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    x = np.cross(heading, z)
    if np.linalg.norm(x) < 1e-8:  # thrust axis parallel to the heading: pick another
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return Rotation.from_matrix(np.column_stack([x, y, z]))


def test_known_attitude_constant_thrust_is_recovered_exactly():
    """Known attitude + constant thrust -> constant acceleration -> exact recovery."""
    dt, n, window, poly = 1 / 275.0, 600, 21, 2
    m, thrust = 0.8, 0.0  # kg, then set below
    for tilt in (0.0, 10.0, 35.0, 60.0, 90.0, 150.0):
        z_B = np.array([np.sin(np.radians(tilt)), 0.0, np.cos(np.radians(tilt))])
        thrust = m * (G0 + 3.0)  # arbitrary specific thrust in newtons
        a = (thrust / m) * z_B - np.array([0.0, 0.0, G0])
        _, pos, _ = _quadratic_track(a, np.zeros(3), np.zeros(3), dt, n)
        vel, acc = savgol_derivatives(pos, dt, window, poly)
        sl = edge_slice(n, window)
        np.testing.assert_allclose(acc[sl], np.broadcast_to(a, acc[sl].shape), atol=1e-9)
        implied = implied_thrust_axis(acc[sl], vel[sl])
        np.testing.assert_allclose(implied, np.broadcast_to(z_B, implied.shape), atol=1e-9)
        assert np.allclose(tilt_deg(implied), tilt, atol=1e-7)


def test_known_attitude_fixed_yaw_matches_rotation_column():
    """The body z used as 'truth' agrees with thrust_axis_from_quat."""
    for yaw in (0.0, 45.0, -120.0):
        for tilt in (5.0, 30.0, 80.0):
            z_B = np.array([np.sin(np.radians(tilt)), 0.0, np.cos(np.radians(tilt))])
            R = _attitude_from_thrust_axis(z_B, yaw)
            axis = thrust_axis_from_quat(R.as_quat()[None, :], "xyzw")
            np.testing.assert_allclose(axis[0], z_B, atol=1e-12)


def test_acceleration_scales_with_fps_squared_and_tilt_error_is_quadratic_in_dt():
    """Using the wrong time base inflates acceleration by (dt'/dt)^2."""
    dt_true = 1 / 275.0
    n, window, poly = 2000, 21, 2
    a = np.array([6.0, 3.0, 1.5])
    _, pos, _ = _quadratic_track(a, np.zeros(3), np.zeros(3), dt_true, n)
    sl = edge_slice(n, window)
    _, acc_ok = savgol_derivatives(pos, dt_true, window, poly)

    s = 2.0  # claim the samples are 2x further apart than they are
    _, acc_bad = savgol_derivatives(pos, dt_true * s, window, poly)
    # acceleration is proportional to 1/dt^2, so a wrong (larger) dt shrinks it
    np.testing.assert_allclose(acc_bad[sl], np.broadcast_to(a / s**2, acc_bad[sl].shape),
                               rtol=1e-8, atol=1e-9)

    z_ok = implied_thrust_axis(acc_ok[sl])
    z_bad = implied_thrust_axis(acc_bad[sl])
    err = angle_between_deg(z_ok, z_bad)
    # closed form: tan(theta) = ||a_xy|| / (a_z + g)
    th_ok = np.degrees(np.arctan2(np.linalg.norm(a[:2]), a[2] + G0))
    th_bad = np.degrees(np.arctan2(np.linalg.norm(a[:2]) / s**2, a[2] / s**2 + G0))
    expected = abs(th_bad - th_ok)
    assert np.allclose(err, expected, atol=1e-6), (err[0], expected)
    assert expected > 5.0, "the test needs a visibly wrong tilt to be meaningful"

    # acceleration scales with fps^2: half the claimed rate -> a quarter of the acceleration
    assert np.isclose(np.linalg.norm(acc_bad[sl][0]) / np.linalg.norm(acc_ok[sl][0]), 1.0 / s**2)


def test_drag_coefficient_is_recovered():
    """A trajectory generated with linear drag returns its k_d."""
    k_true, dt, n, window, poly = 0.35, 1 / 275.0, 1500, 21, 2
    a = np.array([-2.0, 4.0, 1.0])
    t, pos, vel = _quadratic_track(a, np.array([3.0, -1.0, 0.5]), np.zeros(3), dt, n)
    true_axis = implied_thrust_axis(a + 0 * vel, vel, k_true)
    vel_sg, acc_sg = savgol_derivatives(pos, dt, window, poly)
    sl = edge_slice(n, window)
    fit = fit_kd(acc_sg[sl], vel_sg[sl], true_axis[sl], bounds=(0.0, 3.0))
    assert fit["k_d"] > 0.0
    assert abs(fit["k_d"] - k_true) <= 0.01 * k_true, fit
    # and the drag-free residual is materially worse than the fitted one
    assert fit["median_deg_fit"] < 0.1 * max(fit["median_deg_free"], 1e-9) + 1e-6


def test_body_drag_diagnostic_reduces_residual_when_drag_is_body_fixed():
    """Body x/y drag coefficients are recovered when that is the true model."""
    k_true = np.array([0.30, 0.10])
    dt, n, window, poly = 1 / 275.0, 1200, 21, 2
    a = np.array([-1.5, 3.0, 0.5])
    t, pos, vel = _quadratic_track(a, np.array([2.0, 1.0, 0.2]), np.zeros(3), dt, n)
    R = _attitude_from_thrust_axis(np.array([0.3, 0.1, 1.0]), yaw_deg=30.0)
    quat = np.broadcast_to(R.as_quat(), (n, 4)).copy()
    true_axis = implied_thrust_axis_body_drag(a + 0 * vel, vel, quat, "xyzw",
                                              (k_true[0], k_true[1]))
    vel_sg, acc_sg = savgol_derivatives(pos, dt, window, poly)
    sl = edge_slice(n, window)
    from src.physics.flatness import fit_body_drag

    fit = fit_body_drag(acc_sg[sl], vel_sg[sl], true_axis[sl], quat[sl], "xyzw")
    assert fit["median_deg"] < 0.05
    assert fit["median_deg"] < 0.5 * fit["median_deg_free"]


def test_angle_between_is_robust_and_correct():
    e3 = np.array([0.0, 0.0, 1.0])
    for eps in (0.0, 1e-12, 1e-9, 1e-6):
        u = np.array([eps, 0.0, 1.0])
        assert angle_between_deg(u, e3) == pytest.approx(np.degrees(eps), abs=1e-9)
    assert angle_between_deg(e3, -e3) == pytest.approx(180.0, abs=1e-9)
    assert angle_between_deg(e3, np.array([1.0, 0.0, 0.0])) == pytest.approx(90.0, abs=1e-9)
    # arccos would lose precision here; arctan2 keeps the small angle
    assert angle_between_deg(e3, np.array([1e-8, 0.0, 1.0])) > 0.0


def test_quat_order_does_not_silently_change_the_axis():
    q_xyzw = Rotation.from_euler("ZYX", [20, -15, 40], degrees=True).as_quat()
    q_wxyz = np.concatenate([q_xyzw[3:], q_xyzw[:3]])
    a = thrust_axis_from_quat(q_xyzw[None, :], "xyzw")[0]
    b = thrust_axis_from_quat(q_wxyz[None, :], "wxyz")[0]
    np.testing.assert_allclose(a, b, atol=1e-12)
    with pytest.raises(ValueError):
        thrust_axis_from_quat(q_xyzw[None, :], "wxyx")


def test_free_fall_raises_instead_of_dividing_by_zero():
    a = np.array([[0.0, 0.0, -G0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="degenerate"):
        implied_thrust_axis(a)
    mask = valid_thrust_mask(a, min_norm=1.0)
    assert mask.tolist() == [False, True]


def test_kd_requires_velocity():
    with pytest.raises(ValueError, match="velocity"):
        implied_thrust_axis(np.zeros((3, 3)), None, k_d=0.2)


def test_savgol_window_validation():
    pos = np.zeros((10, 3))
    with pytest.raises(ValueError, match="odd"):
        savgol_derivatives(pos, 0.01, 10, 2)
    with pytest.raises(ValueError, match="polyorder"):
        savgol_derivatives(pos, 0.01, 3, 2)
    with pytest.raises(ValueError, match="exceeds"):
        savgol_derivatives(pos, 0.01, 11, 2)
