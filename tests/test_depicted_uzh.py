"""Direction checks for the camera/body frame plumbing.

The mount is published as `T_cam_imu` (IMU -> camera), so `R_cb = R(T_cam_imu)^T` maps
camera -> body and `up_cam = R_cb^T @ up_body`. Reversing that relation leaves the body-frame
agreement untouched while putting 90-180 deg into every camera-frame angle, so it is asserted
here rather than trusted.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from src.depicted.uzh import optical_axis_elevation_deg, published_mount


def _meta_with_mount(T_cam_imu: np.ndarray) -> dict:
    return {"kalibr_imu_cam0": {"T_cam_imu": T_cam_imu.tolist()}}


def test_mount_is_the_transpose_of_T_cam_imu():
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("y", 45.0, degrees=True).as_matrix()
    R_cb, t = published_mount(_meta_with_mount(T))
    assert np.allclose(R_cb, T[:3, :3].T)
    # Ry(+45) in T_cam_imu puts the camera's optical axis 45 deg *above* the body horizon
    assert abs(optical_axis_elevation_deg(R_cb) + 45.0) < 1e-9
    assert np.allclose(t, T[:3, 3])


def test_up_direction_round_trip_camera_body():
    rng = np.random.default_rng(0)
    for _ in range(50):
        T = np.eye(4)
        T[:3, :3] = Rotation.random(random_state=rng).as_matrix()
        R_cb, _ = published_mount(_meta_with_mount(T))
        up_cam = rng.normal(size=3)
        up_cam /= np.linalg.norm(up_cam)
        up_body = R_cb @ up_cam            # camera -> body
        up_cam_back = R_cb.T @ up_body     # body -> camera
        assert np.allclose(up_cam_back, up_cam)


def test_published_mounts_give_the_named_rig_elevations():
    # the four rigs' T_cam_imu rotations, copied from the Kalibr camchain-imucam files
    rots = {
        "indoor_forward": [[-0.028228787368606456, -0.999601488301944, 1.2175294828553618e-05],
                           [0.014401251861751119, -0.00041887083271471837, -0.9998962088597202],
                           [0.999497743623523, -0.028225682131089447, 0.014407337010089172]],
        "indoor_45": [[-0.0273, -0.9996, 0.0022], [-0.7139, 0.0179, -0.7],
                      [0.6997, -0.0206, -0.7141]],
    }
    expect = {"indoor_forward": -0.83, "indoor_45": 45.57}
    for rig, R in rots.items():
        T = np.eye(4)
        # the Kalibr files are orthonormal to ~1e-9; these copies are rounded to 4 decimals
        T[:3, :3] = Rotation.from_matrix(np.array(R)).as_matrix()
        R_cb, _ = published_mount(_meta_with_mount(T))
        assert abs(optical_axis_elevation_deg(R_cb) - expect[rig]) < 0.5
