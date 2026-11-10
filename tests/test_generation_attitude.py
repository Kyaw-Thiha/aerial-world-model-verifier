import numpy as np
import pytest

from src.generation.attitude import (body_rotation, body_z_from_poses, pose_with_attitude,
                                     release_euler_triple, roll_pitch_from_thrust,
                                     rotation_homography, thrust_from_accel)

R_EC = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # notebook t_ec

ANWM_OK = True
try:
    from src.generation.anwm_env import anwm_modules  # noqa: F401
    anwm_modules()
except Exception:
    ANWM_OK = False


def test_body_rotation_is_intrinsic_zyx():
    yaw, pitch, roll = 0.4, -0.2, 0.15
    R = body_rotation(yaw, pitch, roll)[0]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    # scipy's "zyx" is extrinsic, i.e. Rx @ Ry @ Rz: a different matrix for nonzero tilt
    from scipy.spatial.transform import Rotation
    other = Rotation.from_euler("zyx", [yaw, pitch, roll]).as_matrix()
    assert np.abs(R - other).max() > 1e-3


def test_thrust_round_trip():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(200, 3))
    z[:, 2] = np.abs(z[:, 2]) + 0.3
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    yaw = rng.uniform(-np.pi, np.pi, 200)
    phi, theta = roll_pitch_from_thrust(z, yaw)
    np.testing.assert_allclose(body_rotation(yaw, theta, phi)[:, :, 2], z, atol=1e-12)


def test_pose_carries_thrust_and_matches_the_level_constructor():
    from src.generation.poses import pose_from_position_yaw
    pos = np.array([[1.0, 2.0, 3.0], [5.0, 2.0, 3.5]])
    yaw = np.array([0.3, 0.3])
    z = np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 0.9797958971132712]])
    P = pose_with_attitude(pos, yaw, z, R_EC)
    np.testing.assert_allclose(body_z_from_poses(P, R_EC), z / np.linalg.norm(z, axis=1,
                                                                             keepdims=True),
                               atol=1e-12)
    level = pose_with_attitude(pos, yaw, np.tile([0.0, 0.0, 1.0], (2, 1)), R_EC)
    np.testing.assert_allclose(level, pose_from_position_yaw(pos, yaw, R_EC), atol=1e-12)


def test_thrust_from_accel_is_at_most_g_limited():
    a = np.array([[3.0, 0.0, 0.0], [0.0, -3.0, 0.0]])
    z = thrust_from_accel(a)
    np.testing.assert_allclose(np.degrees(np.arctan2(np.linalg.norm(a, axis=1), 9.81)),
                               np.degrees(np.arccos(z[:, 2])), atol=1e-9)


def test_release_euler_triple_reconstructs_the_matrix():
    from scipy.spatial.transform import Rotation
    yaw = np.linspace(-1.0, 1.0, 7)
    theta = np.linspace(-0.4, 0.4, 7)
    phi = np.linspace(-0.3, 0.3, 7)
    R = body_rotation(yaw, theta, phi)
    y, p, r = release_euler_triple(R)
    np.testing.assert_allclose(Rotation.from_euler("zyx", np.stack([y, p, r], 1)).as_matrix(),
                               R, atol=1e-12)


@pytest.mark.skipif(not ANWM_OK, reason="ANWM's cv2/scipy deps are not installed here")
def test_pure_rotation_matches_K_R_K_inverse():
    """Point-level identity: a rotation-only pose change projects by K R K^-1 exactly."""
    sys_path = None
    import sys

    from src.generation.anwm_env import REPO

    sys_path = str(REPO)
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from anwm.projection import reproject_depth_to_other_pose
    from scipy.spatial.transform import Rotation

    K = np.array([[256.0, 0, 256.0], [0, 256.0, 256.0], [0, 0, 1.0]])
    rng = np.random.default_rng(2)
    H = W = 48
    depth = rng.uniform(2.0, 30.0, size=(H, W))
    img = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
    for R_rel in (Rotation.from_euler("x", 12, degrees=True).as_matrix(),
                  Rotation.from_euler("y", -9, degrees=True).as_matrix()):
        pose_src = np.eye(4)
        pose_dst = np.eye(4)
        pose_dst[:3, :3] = R_rel.T
        pts, _ = reproject_depth_to_other_pose(K, depth, img, pose_src, pose_dst)
        R = pose_dst[:3, :3].T @ pose_src[:3, :3]
        Hm = rotation_homography(K, R)
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        uv = np.stack([u.ravel(), v.ravel(), np.ones(H * W)], axis=1)
        pred = (Hm @ uv.T).T
        pred = pred[:, :2] / pred[:, 2:3]
        got = np.stack([K[0, 0] * pts[:, 0] / pts[:, 2] + K[0, 2],
                        K[1, 1] * pts[:, 1] / pts[:, 2] + K[1, 2]], axis=1)
        np.testing.assert_allclose(got, pred, atol=1e-9)
