"""Synthetic checks for the H03 classical roll estimator.

These pin the sign convention and the accuracy of ``src.perception.horizon`` on renders
whose roll is known exactly, before it reads any real or generated frame.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.perception.frames import camera_to_body_pitched
from src.perception.horizon import (estimate_roll, line_family_reading, roll_from_up,
                                    sky_line_reading)


def _body_rotation(yaw: float, theta: float, phi: float) -> np.ndarray:
    """ZYX body-to-world, the project's convention (src.generation.attitude)."""
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(theta), np.sin(theta)
    cx, sx = np.cos(phi), np.sin(phi)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx


def _render(phi_deg: float, theta_deg: float, mount_deg: float = 0.0, size: int = 240,
            f: float = 200.0, height: float = 20.0) -> tuple[np.ndarray, float]:
    """Sky/checker-ground pinhole render at a known body attitude; returns (bgr, true roll)."""
    R_bw = _body_rotation(0.0, np.deg2rad(theta_deg), np.deg2rad(phi_deg))
    R_cb = camera_to_body_pitched(np.deg2rad(mount_deg))
    R_wc = R_bw @ R_cb
    up_cam = R_cb.T @ R_bw.T @ np.array([0.0, 0.0, 1.0])
    u = np.arange(size) + 0.5
    v = np.arange(size) + 0.5
    uu, vv = np.meshgrid(u, v)
    d_cam = np.stack([(uu - size / 2) / f, (vv - size / 2) / f, np.ones_like(uu)], axis=-1)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
    d_world = d_cam @ R_wc.T
    dz = d_world[..., 2]
    img = np.zeros((size, size, 3), np.uint8)
    sky = dz > -1e-3
    img[sky] = (230, 190, 140)            # BGR, light blue
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -height / dz
        p = t[..., None] * d_world
    ground = (~sky) & np.isfinite(p[..., 0])
    gx = np.floor(p[..., 0] / 2.0).astype(np.int64)
    gy = np.floor(p[..., 1] / 2.0).astype(np.int64)
    checker = ((gx + gy) % 2).astype(np.uint8)
    img[ground] = np.where(checker[ground][:, None] == 1, (70, 90, 70), (120, 140, 110))
    return img, float(roll_from_up(up_cam))


def test_roll_from_up_sign():
    assert abs(float(roll_from_up(np.array([0.0, -1.0, 0.0])))) < 1e-9
    assert float(roll_from_up(np.array([1.0, 0.0, 0.0]))) == 90.0
    assert float(roll_from_up(np.array([-1.0, 0.0, 0.0]))) == -90.0


def test_sky_horizon_recovers_roll():
    for phi in (0.0, 8.0, -12.0, 20.0):
        img, true_roll = _render(phi, theta_deg=-6.0)
        r = sky_line_reading(img)
        assert r.usable, f"no sky line at phi={phi}"
        assert abs(r.roll_deg - true_roll) < 1.0, (phi, r.roll_deg, true_roll)


def test_sky_horizon_recovers_roll_with_mount():
    img, true_roll = _render(6.0, theta_deg=-4.0, mount_deg=15.0)
    r = sky_line_reading(img)
    assert r.usable
    assert abs(r.roll_deg - true_roll) < 1.0, (r.roll_deg, true_roll)


def test_line_family_recovers_roll():
    size = 320
    for phi in (0.0, 10.0, -17.0):
        img = np.full((size, size, 3), 40, np.uint8)
        ang = np.deg2rad(phi)
        d = np.array([np.sin(ang), -np.cos(ang)])       # projected world-up direction
        for x0 in np.linspace(40, 280, 8):
            for y0 in np.linspace(40, 280, 4):
                p0 = np.array([x0, y0]) - 40 * d
                p1 = np.array([x0, y0]) + 40 * d
                cv2.line(img, tuple(np.round(p0).astype(int)),
                         tuple(np.round(p1).astype(int)), (220, 220, 220), 2)
        r = line_family_reading(img)
        assert r.usable, f"no line family at phi={phi}"
        assert abs(r.roll_deg - phi) < 1.5, (phi, r.roll_deg)


def test_abstains_on_noise():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (240, 240, 3), dtype=np.uint8)
    reading, both = estimate_roll(img)
    assert not reading.usable
