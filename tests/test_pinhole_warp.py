"""Tests for the pinhole remap used by the generated-frame edit-recovery control."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.negatives.pinhole_warp import applied_tilt_change_deg, pinhole_remap


def test_identity_map_is_the_pixel_grid():
    """Maps are in cv2.remap index coordinates: identity is 0..N-1, not centred."""
    w, h, f = 224, 224, 112.0
    mx, my = pinhole_remap(np.eye(3), w, h, f)
    cols = np.tile(np.arange(w), (h, 1)).astype(np.float32)
    rows = np.tile(np.arange(h)[:, None], (1, w)).astype(np.float32)
    np.testing.assert_allclose(mx, cols, atol=1e-4)
    np.testing.assert_allclose(my, rows, atol=1e-4)
    assert (mx >= 0).all() and (my >= 0).all()


def test_rotation_shifts_the_map_the_expected_way():
    """A positive rotation about camera x moves the source sampling up in the image."""
    w, h, f = 224, 224, 112.0
    M = Rotation.from_euler("x", 10.0, degrees=True).as_matrix()
    mx, my = pinhole_remap(M, w, h, f)
    cy = h // 2
    # for a rotation that pitches the camera down, source rows come from above the target
    assert my[cy, w // 2] < cy
    assert applied_tilt_change_deg(M) == pytest.approx(10.0, abs=1e-6)


def test_rotation_about_the_optical_axis_leaves_only_corners_invalid():
    """An in-plane rotation samples outside the frame at the corners; the interior is fine."""
    w, h, f = 224, 224, 112.0
    M = Rotation.from_euler("z", 20.0, degrees=True).as_matrix()
    mx, my = pinhole_remap(M, w, h, f)
    valid = (mx >= 0) & (my >= 0)
    assert valid.mean() > 0.6, "most of a 20 deg in-plane rotation stays inside"
    assert not valid[0, 0], "the top-left corner samples outside"
    assert valid[h // 2, w // 2], "the centre always maps inside"
    assert applied_tilt_change_deg(M) == pytest.approx(0.0, abs=1e-5)


def test_large_pitch_leaves_unprojectable_corners_marked_invalid():
    w, h, f = 224, 224, 112.0
    M = Rotation.from_euler("x", 80.0, degrees=True).as_matrix()
    mx, my = pinhole_remap(M, w, h, f)
    assert (mx == -1).any() or (my == -1).any()


def test_applied_tilt_change_matches_the_rotation_angle_for_pure_tilt():
    for ang in (5.0, 10.0, 20.0):
        M = Rotation.from_euler("y", ang, degrees=True).as_matrix()
        assert applied_tilt_change_deg(M) == pytest.approx(ang, abs=1e-9)


def test_body_roll_and_pitch_map_to_the_right_camera_axes():
    """Roll is about the optical axis; pitch is about camera x; camera y is yaw (no tilt)."""
    from src.perception.frames import camera_to_body_pitched
    from src.physics.flatness import tilt_deg
    from src.score.verdict import depicted_axis_world

    R_cb = camera_to_body_pitched(0.0)
    level = np.array([[0.0, -1.0, 0.0]])       # world up in a level camera
    for axis, euler in (("roll", "z"), ("pitch", "x")):
        for ang in (5.0, 10.0, 20.0):
            M = Rotation.from_euler(euler, ang, degrees=True).as_matrix()
            up_after = level @ M.T              # rotate the up direction into the new camera
            z = depicted_axis_world(up_after, R_cb, np.zeros(1))
            assert tilt_deg(z)[0] == pytest.approx(ang, abs=1e-6), (axis, ang)
    yaw = Rotation.from_euler("y", 20.0, degrees=True).as_matrix()
    up_after = level @ yaw.T
    assert tilt_deg(depicted_axis_world(up_after, R_cb, np.zeros(1)))[0] == pytest.approx(
        0.0, abs=1e-6)
