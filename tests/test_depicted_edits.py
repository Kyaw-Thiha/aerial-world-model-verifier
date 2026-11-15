"""Unit tests for the UZH edit geometry and the depicted-tilt helpers.

The KB4 model is checked against OpenCV's own implementation on random rays, and a known
rotation is checked to move a projected point exactly where the analytic prediction says.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.depicted.edits import (KB4Camera, kb4_source_coords, pinhole_source_coords,
                                warp)
from src.depicted.metrics import angle_deg, summarize, trimmed_rmse, wilson

cv2 = pytest.importorskip("cv2")

K = np.array([[278.66723066149086, 0.0, 319.75221200593535],
              [0.0, 278.48991409740296, 241.96858910358173],
              [0.0, 0.0, 1.0]])
D = np.array([-0.013721808247486035, 0.020727425669427896,
              -0.012786476702685545, 0.0025242267320687625])


def _kb4() -> KB4Camera:
    return KB4Camera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], D)


def test_kb4_projection_matches_opencv():
    rng = np.random.default_rng(0)
    # rays spread over the field of view (theta up to 60 deg)
    theta = rng.uniform(0.0, np.deg2rad(60.0), 500)
    phi = rng.uniform(0.0, 2.0 * np.pi, 500)
    rays = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                     np.cos(theta)], axis=1)
    cam = _kb4()
    u, v = cam.pixels_from_rays(rays)
    pts = rays.reshape(-1, 1, 3)
    uv, _ = cv2.fisheye.projectPoints(pts, np.zeros(3), np.zeros(3), K, D)
    uv = uv.reshape(-1, 2)
    assert np.max(np.abs(u - uv[:, 0])) < 5e-3
    assert np.max(np.abs(v - uv[:, 1])) < 5e-3


def test_kb4_unprojection_round_trip():
    cam = _kb4()
    u = np.array([[100.0, 200.0, 400.0, 319.75]])
    v = np.array([[90.0, 300.0, 420.0, 241.97]])
    rays = cam.rays_from_pixels(u, v)
    u2, v2 = cam.pixels_from_rays(rays)
    assert np.max(np.abs(u2 - u)) < 5e-3
    assert np.max(np.abs(v2 - v)) < 5e-3
    # OpenCV's own unprojection agrees on the ray direction
    und = cv2.fisheye.undistortPoints(np.stack([u, v], axis=-1).reshape(-1, 1, 2), K, D)
    xy = und.reshape(-1, 2)
    z = 1.0 / np.sqrt(1.0 + (xy ** 2).sum(axis=1))
    ref = np.column_stack([xy[:, 0] * z, xy[:, 1] * z, z])
    assert np.max(np.abs(rays.reshape(-1, 3) - ref)) < 1e-4


def test_pinhole_rotation_moves_points_as_predicted():
    """A rotation about the camera y axis by 10 deg must move x by tan(t)*fx at the centre."""
    th = np.deg2rad(10.0)
    M = np.array([[np.cos(th), 0.0, -np.sin(th)], [0.0, 1.0, 0.0],
                  [np.sin(th), 0.0, np.cos(th)]])
    w, h = 640, 480
    xs, ys, ok = pinhole_source_coords(M, K, w, h)
    # centre pixel: sampling coordinate under the rotation
    cu, cv_ = int(round(K[0, 2])), int(round(K[1, 2]))
    expected_x = K[0, 2] - np.tan(th) * K[0, 0]
    assert abs(xs[cv_, cu] - expected_x) < 1.0
    # within a tenth of a pixel: the sampled row is an integer pixel, the centre is not
    assert abs(ys[cv_, cu] - K[1, 2]) < 0.1


def test_identity_warp_is_pixel_exact():
    img = (np.arange(480 * 640).reshape(480, 640) % 251).astype(np.uint8)
    for geometry, calib in (("pinhole", K), ("kb4", _kb4())):
        out, valid = warp(img, np.eye(3), geometry, calib)
        assert valid.mean() > 0.99
        assert np.max(np.abs(out.astype(int) - img.astype(int))) <= 1
        assert abs(float(np.mean(out[valid])) - float(np.mean(img[valid]))) < 1e-6


def test_kb4_rotation_is_a_rotation():
    """Rotating by M then by M^T must return the original sampling coordinates."""
    cam = _kb4()
    th = np.deg2rad(7.0)
    M = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0],
                  [0.0, 0.0, 1.0]])
    w, h = 640, 480
    xs, ys, ok = kb4_source_coords(M, cam, w, h)
    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    back = cam.rays_from_pixels(xs, ys)
    rays = cam.rays_from_pixels(u, v)
    unrot = np.einsum("ij,hwj->hwi", M.T, back)
    assert np.nanmax(np.abs(unrot - rays)) < 1e-6


def test_angle_and_summaries():
    a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.allclose(angle_deg(a, b), 0.0, atol=1e-6)
    assert abs(angle_deg(np.array([[1.0, 0, 0]]), np.array([[0.0, 1.0, 0]]))[0] - 90.0) < 1e-6
    e = np.arange(1.0, 101.0)
    assert abs(trimmed_rmse(e, 0.10) - np.sqrt(np.mean(np.arange(1.0, 91.0) ** 2))) < 1e-9
    s = summarize(e)
    assert s["n"] == 100 and abs(s["median_deg"] - 50.5) < 1e-9
    lo, hi = wilson(8, 10)
    assert 0.49 < lo < 0.51 and 0.94 < hi < 0.97
