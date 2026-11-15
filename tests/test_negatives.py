"""The fisheye warp must be exact at identity and geometrically correct under rotation."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from src.negatives import equidistant_f, pixels_from_rays, rays_from_pixels, warp_fisheye

W, H = 64, 48


def _img(seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    return ((xx * 3 + yy * 5) % 256).astype(np.uint8)


def test_equidistant_focal_matches_declared_field():
    f = equidistant_f(640, 480, 175.0)
    assert np.isclose(f * np.deg2rad(87.5), 320.0)


def test_pixel_ray_round_trip():
    u, v = np.meshgrid(np.linspace(-30, 30, 5), np.linspace(-20, 20, 4))
    f = equidistant_f(W, H)
    x = rays_from_pixels(u, v, f)
    assert np.allclose(np.linalg.norm(x, axis=-1), 1.0)
    u2, v2, _ = pixels_from_rays(x, f)
    assert np.allclose(u2, u, atol=1e-9) and np.allclose(v2, v, atol=1e-9)


def test_identity_rotation_is_exact():
    img = _img()
    out, valid = warp_fisheye(img, np.eye(3))
    assert valid.all()
    assert np.array_equal(out, img)


def test_rotation_moves_content_the_predicted_way():
    """A bright blob is placed at a known direction; after a rotation it must appear at
    the pixel that ``x_source = M @ x_output`` predicts (within a pixel)."""
    f = equidistant_f(W, H)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    d = np.array([0.15, -0.25, 1.0])
    d /= np.linalg.norm(d)
    u_src, v_src, _ = pixels_from_rays(d, f)
    cx_src, cy_src = u_src + (W - 1) / 2, v_src + (H - 1) / 2
    img = np.exp(-(((xx - cx_src) ** 2 + (yy - cy_src) ** 2) / 8.0)) * 255.0
    M = Rotation.from_euler("x", np.deg2rad(12.0)).as_matrix()
    out, valid = warp_fisheye(img.astype(np.uint8), M)
    u_out, v_out, _ = pixels_from_rays(M.T @ d, f)
    cx_out, cy_out = u_out + (W - 1) / 2, v_out + (H - 1) / 2
    cy, cx = np.unravel_index(np.argmax(out.astype(float)), out.shape)
    assert np.isfinite(cx_out) and 0 <= cx_out < W and 0 <= cy_out < H
    assert abs(cx - cx_out) <= 1.5 and abs(cy - cy_out) <= 1.5, (cx, cy, cx_out, cy_out)
    assert valid[cy, cx]


def test_larger_rotation_loses_more_border():
    img = _img()
    v_small = warp_fisheye(img, Rotation.from_euler("y", 5, degrees=True).as_matrix())[1].mean()
    v_big = warp_fisheye(img, Rotation.from_euler("y", 30, degrees=True).as_matrix())[1].mean()
    assert v_big < v_small < 1.0


def test_tilt_edit_attitudes():
    from src.negatives.edits import (camera_frame_map, delta_from_attitudes, mirrored_roll_attitude,
                                     rotation_angle_deg, scaled_tilt_attitude, tilt_of_attitude_deg,
                                     yaw_control_attitude, zyx_parts)
    rng = np.random.default_rng(0)
    yaw = rng.uniform(-np.pi, np.pi, 50)
    pitch = np.deg2rad(rng.uniform(-45, 45, 50))
    roll = np.deg2rad(rng.uniform(-45, 45, 50))
    def axis(ax, ang):
        rv = np.zeros((len(ang), 3))
        rv[:, {"x": 0, "y": 1, "z": 2}[ax]] = ang
        return Rotation.from_rotvec(rv).as_matrix()
    R = np.einsum("nij,njk,nkl->nil", axis("z", yaw), axis("y", pitch), axis("x", roll))

    # k = 0 gives a level attitude with the heading preserved
    lev = scaled_tilt_attitude(R, 0.0)
    assert np.allclose(tilt_of_attitude_deg(lev), 0.0, atol=1e-4)
    assert np.allclose(zyx_parts(lev)[0], yaw, atol=1e-9)
    # k = 1 is the identity edit
    assert np.allclose(scaled_tilt_attitude(R, 1.0), R, atol=1e-12)
    assert np.allclose(delta_from_attitudes(scaled_tilt_attitude(R, 1.0), R), np.eye(3), atol=1e-12)
    # k = 1.5 increases the tilt, monotonically in k on average
    t15 = tilt_of_attitude_deg(scaled_tilt_attitude(R, 1.5))
    t10 = tilt_of_attitude_deg(scaled_tilt_attitude(R, 1.0))
    t05 = tilt_of_attitude_deg(scaled_tilt_attitude(R, 0.5))
    assert np.all(t05 <= t10 + 1e-4) and np.all(t10 <= t15 + 1e-4)
    # mirrored roll flips the sign of roll and keeps the scalar tilt
    mir = mirrored_roll_attitude(R)
    assert np.allclose(zyx_parts(mir)[2], -roll, atol=1e-9)
    assert np.allclose(tilt_of_attitude_deg(mir), tilt_of_attitude_deg(R), atol=1e-4)
    # yaw control leaves the tilt untouched
    yc = yaw_control_attitude(R, np.deg2rad(15.0))
    assert np.allclose(tilt_of_attitude_deg(yc), tilt_of_attitude_deg(R), atol=1e-4)
    # the camera-frame map of a world rotation is a similar matrix
    R_cw = Rotation.random(50, random_state=7).as_matrix()
    D = delta_from_attitudes(scaled_tilt_attitude(R, 0.5), R)
    M = camera_frame_map(D, R_cw)
    assert np.allclose(rotation_angle_deg(M), rotation_angle_deg(D), atol=1e-9)
    assert np.allclose(np.linalg.det(M), 1.0)


def test_remap_arrays_match_numpy_warp():
    """The cv2 fast path must produce the same geometry as the numpy reference."""
    from src.negatives import equidistant_f, remap_arrays
    f = equidistant_f(W, H)
    ident, _ = remap_arrays(np.eye(3), W, H, f), None
    assert np.all(ident[0] >= 0) and np.all(ident[1] >= 0), "identity must keep every pixel"
    M = Rotation.from_euler("y", 8, degrees=True).as_matrix()
    mx, my = remap_arrays(M, W, H, f)
    ref, ref_valid = warp_fisheye(_img(), M)
    # the two validity definitions differ only on the one-pixel boundary ring: the numpy
    # reference needs all four neighbours inside, cv2.remap interpolates at the edge
    remap_valid = mx >= 0  # remap_arrays writes -1 for out-of-image samples
    diff = np.logical_xor(remap_valid, ref_valid)
    # on this 64x48 test frame the ring is ~3% of the pixels; on the 640x480 frames it
    # is under 1%
    assert diff.mean() < 0.06
    # the two masks agree on nearly everything: at least 90% of the reference's valid
    # pixels are also valid in the remap path
    assert (remap_valid & ref_valid).sum() >= 0.9 * ref_valid.sum()
    assert np.isnan(mx).sum() == 0
