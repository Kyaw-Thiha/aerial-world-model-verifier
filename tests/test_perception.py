"""Tests for undistortion, camera frames and the video loader.

Only the calibration-free parts run without data files; the ones that need the
TII release skip when the local copies are absent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.perception import frames as F
from src.perception import undistort as U

DATA = Path("data/raw/tii_video")
HAS_DATA = (DATA / "calibration" / "calib_ap-ellipse-lemniscate.json").exists()


def _calib():
    import json

    return json.loads((DATA / "calibration" / "calib_ap-ellipse-lemniscate.json").read_text())


def test_distortion_radius_round_trip():
    dist = np.array([-0.28, 0.11, 0.0, 0.0, -0.02])
    for r in (0.05, 0.2, 0.5, 0.9):
        rd = float(U.distortion_radius(np.array([r]), dist)[0])
        assert U.inv_distortion_radius(rd, dist) == pytest.approx(r, abs=1e-4)


def test_distortion_beyond_peak_has_no_ray():
    dist = np.array([-0.28, 0.11, 0.0, 0.0, -0.02])
    rd_peak = U.model_diagnostics(np.diag([300.0, 300.0, 1.0]), dist, (640, 480))["rd_peak"]
    assert np.isnan(U.inv_distortion_radius(rd_peak + 0.05, dist))


def test_virtual_camera_identity_on_pinhole():
    # no distortion, and a virtual FOV that exactly matches the source focal
    K = np.array([[300.0, 0.0, 320.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]])
    dist = np.zeros(5)
    fov = np.degrees(2 * np.arctan(320.0 / 300.0))
    vc = U.virtual_camera(K, dist, (640, 480), fov_deg=fov, out_size=(640, 480))
    assert vc.valid_fraction == pytest.approx(1.0, abs=0.02)
    h, w = vc.map_x.shape
    ys, xs = np.mgrid[0:h, 0:w]
    err = np.nanmax(np.hypot(vc.map_x - xs, vc.map_y - ys))
    assert err < 1.0


def test_level_airframe_puts_gravity_on_the_optical_axis():
    # camera pitched up 50 deg relative to a level body: R_cb = rotation about
    # the body y axis (left) that takes camera z into the body x-z plane
    # columns are the camera axes in body coordinates: camera x (right) = -body y,
    # camera y (down) = (sin,0,-cos), camera z (optical axis) = (cos,0,sin)
    th = np.radians(50.0)
    R_cb = F.camera_to_body_pitched(th)
    R_bw = Rotation.identity().as_matrix()[None]
    up_cam = F.up_in_camera(R_bw, R_cb)[0]
    assert np.degrees(np.arccos(abs(up_cam[2]))) == pytest.approx(40.0, abs=1e-6)
    assert F.tilt_deg(F.up_in_body(R_bw))[0] == pytest.approx(0.0, abs=1e-6)
    K = np.array([[300.0, 0.0, 320.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]])
    px, ok = F.project_direction(up_cam, K)
    assert ok[0]
    assert px[0, 1] < K[1, 2]          # up is above the principal point
    assert px[0, 0] == pytest.approx(K[0, 2], abs=1e-6)


def test_tilt_is_rotation_invariant_and_matches_roll_pitch():
    R = Rotation.from_euler("ZYX", [30.0, 12.0, -7.0], degrees=True).as_matrix()[None]
    up_b = F.up_in_body(R)
    # tilt = arccos(cos pitch * cos roll) exactly; hypot(roll, pitch) only to ~2e-3
    assert F.tilt_deg(up_b)[0] == pytest.approx(np.hypot(12.0, 7.0), rel=2e-3)
    roll, pitch = F.roll_pitch_deg(R)
    assert roll[0] == pytest.approx(-7.0, abs=1e-6)
    assert pitch[0] == pytest.approx(12.0, abs=1e-6)


def test_dual_extrinsic_candidates_are_inverse():
    pytest.importorskip("src.datasets.tii_video")
    from src.datasets.tii_video import camera_to_body

    if not HAS_DATA:
        pytest.skip("TII calibration not fetched")
    ext = camera_to_body(DATA / "calibration", "autonomous", "ellipse")
    assert np.allclose(ext["R_json"] @ ext["R_json_T"], np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(ext["R_json"]) - 1) < 1e-9


def test_flight_parts():
    from src.datasets.tii_video import flight_parts

    assert flight_parts("flight-01a-ellipse") == ("autonomous", "ellipse")
    assert flight_parts("flight-16p-trackRATM") == ("piloted", "trackRATM")


@pytest.mark.skipif(not HAS_DATA, reason="TII video metadata not fetched")
def test_frame_table_is_monotone_and_complete():
    from src.datasets.tii_video import frame_table

    t = frame_table(DATA, "flight-01a-ellipse")
    assert (np.diff(t["timestamp"].to_numpy()) > 0).all()
    if (DATA / "autonomous" / "camera_flight-01a-ellipse.zip").exists():
        assert t["member"].notna().all()


def test_body_z_and_optical_axis_are_not_confused():
    th = np.radians(50.0)
    R_cb = F.camera_to_body_pitched(th)
    # body z in camera coords: 40 deg above the optical axis for a 50 deg mount
    zc = F.body_z_in_camera(R_cb)
    assert np.allclose(zc, [0.0, -np.cos(th), np.sin(th)])
    assert np.degrees(np.arccos(abs(zc @ np.array([0, 0, 1.0])))) == pytest.approx(40.0)
    # optical axis in body coords: 50 deg above the body x axis
    ax = F.optical_axis_in_body(R_cb)
    assert np.allclose(ax, [np.cos(th), 0.0, np.sin(th)])
    assert F.expected_gravity_offset_deg(R_cb) == pytest.approx(40.0)
