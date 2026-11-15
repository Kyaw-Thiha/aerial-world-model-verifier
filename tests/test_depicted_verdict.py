"""End-to-end check of the verdict chain on synthetic clips.

Builds a fake prepared sequence and a set of clips in the shared schema, with the depicted
tilt derived from a known attitude, and checks that:

* a perfect estimator scores near zero and is accepted;
* a tilt-removed clip scores far above the frozen threshold;
* the abstention rule trips when the required tilt is too small;
* the clip files pass the schema validation in `scripts/export_verdict_clips.py`.

This is the T6 check ("scoring script reproducible on any clip in the shared schema") in unit
form, and it touches neither the network nor a GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.depicted.clips import lean_from_up_body, path_side_with_gt  # noqa: E402
from src.depicted.stat_a2 import FROZEN_THRESHOLD_DEG, score_clip_upb  # noqa: E402
from src.depicted.uzh import load_truth  # noqa: E402

T_CAM_IMU = np.eye(4)


def _trajectory(seconds: float = 4.0, fps: float = 15.0):
    """A banked circular flight: real tilt, real speed, no clutter."""
    n = int(seconds * fps)
    t = np.arange(n) / fps
    r, omega = 6.0, 2.0
    pos = np.column_stack([r * np.cos(omega * t), r * np.sin(omega * t), 2.0 + 0 * t])
    v = np.column_stack([-r * omega * np.sin(omega * t), r * omega * np.cos(omega * t), 0 * t])
    yaw = np.arctan2(v[:, 1], v[:, 0])
    return t, pos, yaw


def _zyx_attitude(axis: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Body-to-world attitude in ZYX form for a given thrust axis and heading."""
    R = np.zeros((len(yaw), 3, 3))
    for i in range(len(yaw)):
        z = axis[i] / np.linalg.norm(axis[i])
        c, s_ = np.cos(-yaw[i]), np.sin(-yaw[i])
        zx = c * z[0] - s_ * z[1]
        zy = s_ * z[0] + c * z[1]
        roll = -np.arcsin(np.clip(zy, -1.0, 1.0))
        pitch = np.arcsin(np.clip(zx / max(np.cos(roll), 1e-9), -1.0, 1.0))
        R[i] = Rotation.from_euler("ZYX", [yaw[i], pitch, roll]).as_matrix()
    return R


def _write_sequence(tmp: Path, name: str = "synthetic_forward"):
    """Write a prepared-sequence directory whose truth is the analytic trajectory."""
    t, pos, yaw = _trajectory()
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    t_dense = np.arange(0.0, t[-1] + 1e-9, 0.002)
    pos_d = np.column_stack([np.interp(t_dense, t, pos[:, k]) for k in range(3)])
    quat = Rotation.identity(len(t)).as_quat()
    quat_d = Rotation.identity(len(t_dense)).as_quat()
    gt = np.column_stack([t_dense + 1.0e9, pos_d, quat_d])
    np.savez_compressed(d / "truth.npz", t=t, pos=pos, quat=quat, gt=gt,
                        t_unix=1.0e9 + t, gt_t0=1.0e9)
    meta = {"sequence": name, "rig": "indoor_forward", "environment": "indoor",
            "n": len(t), "fps_measured": 15.0, "n_frames_inside_truth": len(t),
            "clock_offset_applied_s": 0.0, "t_unix": (1.0e9 + t).tolist(),
            "kalibr_imu_cam0": {"T_cam_imu": T_CAM_IMU.tolist(), "timeshift_cam_imu": 0.0,
                                "intrinsics": [278.667, 278.49, 319.752, 241.969],
                                "resolution": [640, 480],
                                "distortion_coeffs": [0.0, 0.0, 0.0, 0.0],
                                "camera_model": "pinhole", "distortion_model": "equidistant",
                                "rostopic": "/snappy_cam/stereo_l"},
            "intrinsics_pinhole": {"fx": 278.667, "fy": 278.49, "cx": 320.0, "cy": 240.0,
                                   "w": 640, "h": 480, "model": "pinhole"},
            "intrinsics_kb4": {"fx": 278.667, "fy": 278.49, "cx": 319.752, "cy": 241.969,
                               "k": [0.0, 0.0, 0.0, 0.0], "w": 640, "h": 480}}
    (d / "meta.json").write_text(json.dumps(meta))
    return d, t, pos, yaw


def _write_clip(d: Path, name: str, cond: str, t, pos, yaw, up_b):
    np.savez_compressed(d / f"{name}.npz", t=t, pos=pos, yaw=yaw,
                        depicted_lean=lean_from_up_body(up_b),
                        depicted_conf=np.full(len(t), 0.01),
                        valid_frac=np.ones(len(t)))
    meta = {"dataset": "uzh_fpv", "sequence": d.name, "clip_id": name, "condition": cond,
            "index": [0, len(t)], "fps": 15.0, "mount_source": "published",
            "mount_rotation_R_cb": np.eye(3).tolist(), "estimator": "synthetic",
            "estimator_variant": "perfect", "rig": "indoor_forward"}
    (d / f"{name}.meta.json").write_text(json.dumps(meta))


def _setup(tmp_path):
    d, t, pos, yaw = _write_sequence(tmp_path)
    truth = load_truth(d.parent, d.name, rig="indoor_forward", env="indoor")
    path = path_side_with_gt(truth, np.load(d / "truth.npz")["gt"], k_d=0.0)
    axis = path["required_axis"]
    R = _zyx_attitude(axis, yaw)
    up_b = np.einsum("nji,j->ni", R, np.array([0.0, 0.0, 1.0]))
    return d, t, pos, yaw, truth, path, up_b


def test_perfect_estimator_scores_near_zero_and_tilt_removal_fails(tmp_path):
    d, t, pos, yaw, truth, path, up_b = _setup(tmp_path)
    up_flat = np.tile(np.array([0.0, 0.0, 1.0]), (len(t), 1))  # level-camera edit
    _write_clip(d, "real", "real", t, pos, yaw, up_b)
    _write_clip(d, "removed", "tilt_removal", t, pos, yaw, up_flat)

    kw = dict(imp_axis=path["required_axis"], yaw=yaw, t=t, R_cb=np.eye(3),
              implied_tilt_deg=path["required_tilt_deg"], valid=np.ones(len(t), bool),
              lag_grid_ms=np.array([0.0]))
    real = score_clip_upb(up_body=up_b, **kw)
    bad = score_clip_upb(up_body=up_flat, **kw)
    assert real["testable"] and bad["testable"]
    assert real["score"] < 1e-6, real["score"]
    assert bad["score"] > FROZEN_THRESHOLD_DEG + 10.0, bad["score"]


def test_abstention_when_required_tilt_is_small(tmp_path):
    d, t, pos, yaw, truth, path, up_b = _setup(tmp_path)
    res = score_clip_upb(up_body=up_b, imp_axis=path["required_axis"], yaw=yaw, t=t,
                         R_cb=np.eye(3), implied_tilt_deg=np.full(len(t), 2.0),
                         valid=np.ones(len(t), bool), lag_grid_ms=np.array([0.0]))
    assert not res["testable"]
    assert any("signal_frames" in r for r in res["abstain_reasons"])


def test_schema_validation_accepts_generated_clips(tmp_path):
    d, t, pos, yaw, truth, path, up_b = _setup(tmp_path)
    _write_clip(d, "real", "real", t, pos, yaw, up_b)
    from scripts.export_verdict_clips import check_clip

    rec = check_clip(d / "real.npz", d / "real.meta.json")
    assert rec["n_frames"] == len(t)
    assert rec["depicted_finite_fraction"] == 1.0
    assert rec["condition"] == "real"


def test_lean_inversion_is_exact_for_physical_tilts():
    from src.depicted.clips import up_body_from_lean

    rng = np.random.default_rng(1)
    for _ in range(200):
        tilt = rng.uniform(0.0, 80.0)
        heading = rng.uniform(-np.pi, np.pi)
        up = np.array([np.sin(np.deg2rad(tilt)) * np.cos(heading),
                       np.sin(np.deg2rad(tilt)) * np.sin(heading), np.cos(np.deg2rad(tilt))])
        back = up_body_from_lean(lean_from_up_body(up[None, :]))[0]
        assert np.allclose(back, up, atol=1e-12)
