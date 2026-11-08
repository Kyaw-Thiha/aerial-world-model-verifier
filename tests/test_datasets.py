"""Loader tests against the real TII mocap files (skipped when data is absent)."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest

from src.datasets.clip import Clip
from src.datasets.tii import TII_RATE_HZ_EXPECTED, load_tii
from src.physics.flatness import thrust_axis_from_quat

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(glob.glob(str(ROOT / "data/raw/tii/*/mocap_*.csv")))
pytestmark = pytest.mark.skipif(not FILES, reason="TII mocap files not downloaded")


def test_rates_timestamps_and_attitude_are_sane():
    for f in FILES:
        clip = load_tii(f)
        assert 250.0 < clip.fps < 300.0, (clip.name, clip.fps)
        assert abs(clip.fps - TII_RATE_HZ_EXPECTED) < 1.0, (clip.name, clip.fps)
        assert np.isfinite(clip.pos).all() and np.isfinite(clip.quat).all()
        assert np.all(np.diff(clip.t) > 0)
        assert clip.t[0] == 0.0
        norms = np.linalg.norm(clip.quat, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)
        assert clip.quat_order == "xyzw"
        R = thrust_axis_from_quat(clip.quat, clip.quat_order)
        np.testing.assert_allclose(np.linalg.norm(R, axis=1), 1.0, atol=1e-9)


def test_body_z_points_up_at_low_speed():
    """The mocap body z-axis must point roughly up when the drone is slow.

    Catches a transposed matrix, a quaternion order flip, or a z-down world.
    """
    for f in FILES:
        clip = load_tii(f)
        speed = np.linalg.norm(np.gradient(clip.pos, clip.t, axis=0), axis=1)
        slow = speed < np.percentile(speed, 5)
        if slow.sum() < 10:
            continue
        z = thrust_axis_from_quat(clip.quat[slow], clip.quat_order)[:, 2]
        assert np.median(z) > 0.8, (clip.name, float(np.median(z)))


def test_uptilt_comes_from_metadata():
    for f in FILES:
        clip = load_tii(f)
        assert clip.cam_uptilt_deg in (30.0, 40.0, 50.0), (clip.name, clip.cam_uptilt_deg)


def test_clip_rejects_bad_input():
    t = np.array([0.0, 0.01, 0.02])
    pos = np.zeros((3, 3))
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (3, 1))
    with pytest.raises(ValueError, match="quat_order"):
        Clip(name="x", t=t, pos=pos, quat=quat, quat_order="abcd", fps=100.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        Clip(name="x", t=np.array([0.0, 0.02, 0.01]), pos=pos, quat=quat,
             quat_order="xyzw", fps=100.0)
    with pytest.raises(ValueError, match="fps"):
        Clip(name="x", t=t, pos=pos, quat=quat, quat_order="xyzw", fps=0.0)
    with pytest.raises(ValueError, match="NaNs"):
        bad = pos.copy(); bad[1, 2] = np.nan
        Clip(name="x", t=t, pos=bad, quat=quat, quat_order="xyzw", fps=100.0)
