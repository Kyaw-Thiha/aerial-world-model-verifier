"""Unit tests for src/pathrec/testbeds.py (no dataset download, no cv2)."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from src.pathrec.testbeds import (PathRecClip, anwm_clip_ids, load_anwm, tii_crop_intrinsics,
                                 uzh_calib, uzh_groundtruth)

CAMCHAIN = """cam0:
  cam_overlaps: [1]
  camera_model: pinhole
  distortion_coeffs: [-0.0137, 0.0207, -0.0128, 0.0025]
  distortion_model: equidistant
  intrinsics: [278.667, 278.489, 319.752, 241.968]
  resolution: [640, 480]
  rostopic: /snappy_cam/stereo_l
"""


def _make_clip(n=5, **over):
    kw = dict(name="c", source="uzh", t=np.arange(n) * 0.1,
              pos=np.zeros((n, 3)), quat=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
              quat_order="xyzw", fps=10.0, intrinsics={"fx": 1.0})
    kw.update(over)
    return PathRecClip(**kw)


def test_clip_validates_shapes_and_time():
    c = _make_clip()
    assert c.duration_s == pytest.approx(0.4)
    with pytest.raises(ValueError):
        _make_clip(pos=np.zeros((4, 3)))
    with pytest.raises(ValueError):
        _make_clip(t=np.array([0.0, 0.2, 0.1, 0.3, 0.4]))
    with pytest.raises(ValueError):
        _make_clip(quat_order="xywz")
    with pytest.raises(ValueError):
        _make_clip(fps=0.0)
    with pytest.raises(ValueError):
        _make_clip(frames=["a.png"])


def test_uzh_calib_and_groundtruth_from_synthetic_zip(tmp_path: Path):
    z = tmp_path / "calib.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("calib/camchain-..x_cam.yaml", CAMCHAIN)
    cal = uzh_calib(z)
    assert cal["fx"] == pytest.approx(278.667)
    assert cal["model"] == "pinhole+equidistant"
    assert cal["w"] == 640 and cal["h"] == 480

    seq = tmp_path / "seq.zip"
    gt = "1.0 0 0 1 0 0 0 1\n0.0 1 2 3 0 0 0 1\n"
    with zipfile.ZipFile(seq, "w") as f:
        f.writestr("groundtruth.txt", gt)
    with zipfile.ZipFile(seq) as f:
        rows = uzh_groundtruth(f)
    assert rows.shape == (2, 8)
    assert rows[0, 0] == 0.0 and rows[1, 0] == 1.0     # sorted by timestamp
    assert np.allclose(rows[0, 1:4], [1, 2, 3])


def test_tii_crop_intrinsics_matches_track_a_virtual_camera():
    k = tii_crop_intrinsics(320, 240, fov_deg=100.0)
    assert k["fx"] == pytest.approx(134.2559, abs=1e-3)   # Track A's virtual pinhole
    assert k["cx"] == pytest.approx(160.0) and k["cy"] == pytest.approx(120.0)


def test_anwm_loader_reads_manifest_frames_and_command(tmp_path: Path):
    d = tmp_path / "A" / "A00"
    (d / "frames").mkdir(parents=True)
    (d / "gt").mkdir()
    for i in range(4):
        (d / "frames" / f"{i:02d}.png").write_bytes(b"x")
        (d / "gt" / f"{i:02d}.png").write_bytes(b"x")
    cmd = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    (d / "manifest.json").write_text(json.dumps(
        {"id": "A00", "regime": "A", "commanded_path": {"positions": cmd}}))
    assert anwm_clip_ids(tmp_path) == ["A/A00"]
    c = load_anwm(tmp_path, "A/A00")
    assert len(c.frames) == 4 and len(c.gt_frames) == 4
    assert c.fps == pytest.approx(1.0)          # 1 s per commanded step, documented assumption
    assert np.allclose(c.pos[:, 0], [0, 1, 2, 3])
    assert c.intrinsics["fx"] == pytest.approx(112.0)
    assert "commanded" in c.meta["pose_semantics"]


def test_anwm_loader_shifts_n_plus_one_command(tmp_path: Path):
    """A 17-row command for 16 frames: frame i lines up with positions[i+1]."""
    d = tmp_path / "A" / "C00"
    (d / "frames").mkdir(parents=True)
    for i in range(3):
        (d / "frames" / f"{i:02d}.png").write_bytes(b"x")
    cmd = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]        # 4 rows for 3 frames
    (d / "manifest.json").write_text(json.dumps(
        {"id": "C00", "regime": "C", "commanded_path": {"positions": cmd}}))
    c = load_anwm(tmp_path, "C00")
    assert np.allclose(c.pos[:, 0], [1, 2, 3])
    assert "level" in c.meta["gt_frames_semantics"]


def test_anwm_loader_pads_short_command(tmp_path: Path):
    d = tmp_path / "A" / "B01"
    (d / "frames").mkdir(parents=True)
    for i in range(3):
        (d / "frames" / f"{i:02d}.png").write_bytes(b"x")
    (d / "manifest.json").write_text(json.dumps(
        {"id": "B01", "regime": "B", "commanded_path": {"positions": [[0, 0, 0]]}}))
    c = load_anwm(tmp_path, "B01")
    assert len(c.pos) == 3 and np.allclose(c.pos[-1], [0, 0, 0])
