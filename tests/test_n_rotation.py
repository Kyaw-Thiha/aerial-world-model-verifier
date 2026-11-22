"""The Part 0 propagation statistic: yaw is free, lean is not, and the oracle start is exact."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.n01_rotation_pretest import horizon_errors, steps


def _seq(err_deg: float, axis: str, n: int = 11) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True attitude identity; the estimate drifts by err_deg over the first second."""
    t = np.arange(n) / 10.0
    R_true = np.tile(np.eye(3), (n, 1, 1))
    ang = np.radians(err_deg) * np.arange(n) / (n - 1)
    R_est = np.stack([Rotation.from_euler(axis, a).as_matrix() for a in ang])
    return R_est, R_true, t


def test_identity_is_exact() -> None:
    R = Rotation.from_euler("zyx", [10, -20, 30], degrees=True).as_matrix()
    t = np.arange(20) / 15.0
    e = horizon_errors(np.tile(R, (20, 1, 1)), np.tile(R, (20, 1, 1)), t, 1.0, 15.0, None)
    assert e["n"] > 0
    # arccos of a dot product of unit vectors carries ~1e-6 deg of rounding
    assert e["median_deg"] < 1e-4
    assert e["errors_deg"].max() < 1e-4


def test_roll_drift_shows_as_lean_error() -> None:
    R_est, R_true, t = _seq(5.0, "x")
    e = horizon_errors(R_est, R_true, t, 1.0, 10.0, None)
    assert e["n"] >= 1
    assert abs(e["median_deg"] - 5.0) < 1e-6


def test_yaw_drift_is_invisible_to_gravity() -> None:
    R_est, R_true, t = _seq(30.0, "z")
    e = horizon_errors(R_est, R_true, t, 1.0, 10.0, None)
    assert e["n"] >= 1
    assert e["median_deg"] < 1e-6


def test_horizon_needs_a_pair() -> None:
    R = np.tile(np.eye(3), (5, 1, 1))
    t = np.arange(5) / 10.0
    assert horizon_errors(R, R, t, 1.0, 10.0, None)["n"] == 0


def test_frozen_step_detection() -> None:
    R_true = np.stack([Rotation.from_euler("x", 1.0 * k, degrees=True).as_matrix()
                       for k in range(10)])
    R_frozen = np.tile(np.eye(3), (10, 1, 1))
    est, tru = steps(R_frozen, R_true)
    assert est < 0.1 * tru
