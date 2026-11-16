"""Tests for the simulator track's analytic flights.

The point of these tests is that every downstream number in Track I rests on two claims:
the acceleration is the second derivative of the path (so ``a + g e3 + k_d v`` is the
right thrust axis), and the camera pose really places the world the way the projections
say. Both are checked against an independent computation here.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.perception.frames import up_in_camera
from src.sim import flight as F


def test_derivatives_match_numerical_differentiation():
    for name, kw in [("ellipse", dict(radius=8.0, speed=7.0, duration=3.0)),
                     ("lemniscate", dict(scale=6.0, speed=6.0, duration=3.0)),
                     ("slalom", dict(speed=9.0, amplitude=3.0, wavelength=14.0, duration=4.0)),
                     ("turn", dict(speed=11.0, radius=5.0, arc_deg=180.0))]:
        f = F.build(name, **kw)
        dt = 1.0 / f.gt_hz
        v_num = np.gradient(f.pos, dt, axis=0)
        a_num = np.gradient(f.vel, dt, axis=0)
        # Fourth-order central differences are accurate to ~1e-4 at this rate; compare on
        # the interior so the one-sided end stencils do not dominate.
        sl = slice(5, -5)
        assert np.abs(v_num - f.vel)[sl].max() < 1e-3, name
        assert np.abs(a_num - f.acc)[sl].max() < 2e-2, name


def test_hover_is_level_and_carries_no_signal():
    f = F.hover(duration=2.0)
    assert np.abs(f.tilt_deg()).max() < 1e-9
    assert np.allclose(f.thrust_axis, F.E3)


def test_ellipse_tilt_matches_closed_form():
    r, v = 8.0, 7.0
    f = F.ellipse(radius=r, speed=v, duration=6.0)
    w = v / r
    expected = np.degrees(np.arctan2(w * w * r, F.G0))
    # drag adds a horizontal term of k_d * speed on the tangent; the analytic tilt without
    # it bounds the answer from below, and the difference is what the drag model does.
    got = np.median(f.tilt_deg())
    assert got >= expected - 1e-6
    assert abs(got - expected) < 3.0


def test_attitude_reproduces_thrust_axis_and_heading():
    f = F.turn(speed=14.0, radius=5.0)
    z = f.R_bw[:, :, 2]
    assert np.allclose(z, f.thrust_axis, atol=1e-9)
    # body x projected on the horizontal plane is the heading
    x = f.R_bw[:, :, 0]
    assert np.allclose(np.arctan2(x[:, 1], x[:, 0]), np.unwrap(f.yaw), atol=1e-6)


def test_camera_pose_puts_world_up_where_gravity_should_be():
    f = F.turn(speed=12.0, radius=6.0, arc_deg=90.0)
    uptilt = 0.0
    out = f.camera_poses(uptilt_deg=uptilt)
    poses = out["poses"]
    # up_in_camera is the project's definition; recompute it from the render poses alone
    R_wc = poses[:, :3, :3]
    idx = out["frame_idx"]
    up_b = np.einsum("nji,j->ni", f.camera_R_bw[idx], F.E3)
    up_cam = np.einsum("ji,nj->ni", f.camera_R_cb, up_b)
    up_from_pose = np.einsum("nji,j->ni", R_wc, F.E3)     # R_wc^T e3
    assert np.allclose(up_cam, up_from_pose, atol=1e-12)
    # With a level body the mount pitch is the only gravity offset: world up makes
    # (90 - uptilt) degrees with the optical axis, i.e. it sits 30 deg above the centre row
    # of a 30 deg up-tilted camera.
    for uptilt, expect in ((0.0, 90.0), (30.0, 60.0)):
        level = F.hover(duration=1.0)
        level.camera_poses(uptilt_deg=uptilt)
        up = up_in_camera(level.R_bw, level.camera_R_cb)
        ang = np.degrees(np.arccos(np.clip(up @ F.E3, -1, 1)))
        assert np.allclose(ang, expect, atol=1e-9)


def test_level_camera_removes_the_depicted_tilt():
    f = F.turn(speed=14.0, radius=5.0, arc_deg=60.0)
    phys = f.camera_poses(uptilt_deg=0.0)["poses"]
    level = f.camera_poses(uptilt_deg=0.0, level_camera=True)["poses"]
    # the camera in the level render is level: with a 0 deg mount its optical axis has no
    # vertical component at all
    assert np.abs(level[:, 2, 2]).max() < 1e-9
    up_phys = np.einsum("nji,j->ni", phys[:, :3, :3], F.E3)
    up_lev = np.einsum("nji,j->ni", level[:, :3, :3], F.E3)
    # physically the camera sees the horizon tilted/offset; level it does not
    assert np.median(np.abs(up_phys[:, 2])) > 0.05
    assert np.abs(up_lev[:, 2]).max() < 1e-9


def test_schedule_lines_up_with_gt_grid():
    f = F.ellipse(radius=10.0, speed=8.0)
    out = f.camera_poses(uptilt_deg=0.0)
    idx = out["frame_idx"]
    assert len(idx) == int(np.floor(f.t[-1] * f.video_fps + 1e-9)) + 1
    assert np.all(np.diff(idx) >= 1)
    assert out["poses"].shape == (len(idx), 4, 4)
    # frame times are a uniform 30 fps grid
    assert np.allclose(np.diff(out["frame_t"]), 1.0 / f.video_fps)


def test_projection_round_trip():
    f = F.hover(duration=1.0, pos=(0.0, 0.0, 10.0))
    poses = f.camera_poses(uptilt_deg=0.0)["poses"]
    K = F.pinhole_K(640, 360, 90.0)
    # a point dead ahead along the optical axis lands on the principal point
    T = poses[0]
    R_wc, p_wc = T[:3, :3], T[:3, 3]
    ahead = p_wc + 20.0 * R_wc[:, 2]
    pix = F.project_world_points(poses[:1], K, ahead[None])[0, 0]
    assert np.allclose(pix, [(640 - 1) / 2, (360 - 1) / 2], atol=1e-9)


def test_summary_reports_denominators_and_units():
    f = F.ellipse(radius=10.0, speed=8.0)
    f.camera_poses(uptilt_deg=0.0)
    s = f.summary()
    assert s["gt_hz"] >= 200.0
    assert s["n_gt"] == len(f.t) and s["n_frames"] == len(f.frame_t)
    assert s["tilt_max_deg"] > 10.0


def test_nonuniform_or_ragged_input_is_rejected():
    t = np.array([0.0, 0.004, 0.012])
    p = np.zeros((3, 3))
    with pytest.raises(ValueError):
        F.Flight(name="bad", t=t, pos=p, vel=p, acc=p, yaw=np.zeros(3))
    with pytest.raises(ValueError):
        F.Flight(name="bad", t=np.arange(3) * 0.004, pos=np.zeros((2, 3)),
                 vel=np.zeros((2, 3)), acc=np.zeros((2, 3)), yaw=np.zeros(2))
