import numpy as np
import pytest

from src.generation.poses import camera_axis_change, pose_from_position_yaw, yaw_matrix

R_EC = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # FLU -> RDF


def test_yaw_matrix_is_a_rotation():
    for y in (0.0, 0.7, -2.5):
        R = yaw_matrix(y)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_round_trip_pose_convention():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(9, 3))
    yaw = rng.uniform(-np.pi, np.pi, 9)
    poses = pose_from_position_yaw(pos, yaw, R_EC)
    R_ec, spread = camera_axis_change(poses, yaw)
    assert spread < 1e-12
    np.testing.assert_allclose(R_ec, R_EC, atol=1e-12)
    again = pose_from_position_yaw(pos, yaw, R_ec)
    np.testing.assert_allclose(again, poses, atol=1e-12)


def test_inconsistent_convention_is_rejected():
    pos = np.zeros((5, 3))
    yaw = np.zeros(5)
    poses = pose_from_position_yaw(pos, yaw, R_EC)
    poses[3, :3, :3] = yaw_matrix(0.3) @ R_EC      # one frame with a different axis change
    with pytest.raises(ValueError, match="not constant"):
        camera_axis_change(poses, yaw)


def test_length_mismatch():
    with pytest.raises(ValueError):
        pose_from_position_yaw(np.zeros((3, 3)), np.zeros(4), R_EC)
