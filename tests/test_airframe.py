"""Unit tests for the Track K airframe models and schema v3 (CPU, no data downloads)."""
from __future__ import annotations

import numpy as np
import pytest

from src.airframe.kinematics import Kinematics, load_tii_kinematics, load_uzh_kinematics
from src.airframe.models import (axis_M0, axis_M1, axis_of_lean, lean_of_axis,
                                 lowpass_axis, unit)
from src.physics.flatness import G0, implied_thrust_axis


def _kin(n=64, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.01
    v = rng.normal(size=(n, 3)) * 5
    a = rng.normal(size=(n, 3)) * 2
    j = rng.normal(size=(n, 3))
    yaw = np.linspace(-0.3, 0.4, n)
    R = np.tile(np.eye(3), (n, 1, 1))
    z = R[:, :, 2]
    from src.verdict.lean import world_to_yaw_frame
    lean = world_to_yaw_frame(z[:, :2], yaw)
    return Kinematics("synthetic", "tii", "tii_racing", t, np.zeros((n, 3)), v, a, j,
                      yaw, yaw, z, lean, np.ones(n, bool), 0.01, 5, 0.05)


def test_m0_is_the_frozen_flatness_axis():
    k = _kin()
    got = axis_M0(k)
    want = implied_thrust_axis(k.a, k.v, k_d=0.295, g=G0)
    assert np.allclose(got, want, atol=1e-12)


def test_lean_axis_round_trip():
    k = _kin()
    lean = np.column_stack([0.2 * np.cos(k.yaw), -0.2 * np.sin(k.yaw)])
    ax = axis_of_lean(lean, k.yaw)
    back = lean_of_axis(ax, k.yaw)
    assert np.allclose(lean, back, atol=1e-12)


def test_lowpass_tau_zero_is_identity_and_nan_does_not_poison():
    k = _kin()
    u = axis_M0(k)
    assert np.allclose(lowpass_axis(u, 0.0, k.dt), u)
    masks = u.copy()
    masks[:3] = np.nan
    out = lowpass_axis(masks, 0.02, k.dt)
    assert np.isnan(out[:3]).all()
    assert np.isfinite(out[3:]).all()


def test_m1_with_m0_gains_reproduces_m0_horizontal_drag():
    k = _kin()
    p = {"k_forward": 0.295, "k_lateral": 0.295}
    assert np.allclose(axis_M1(k, p), axis_M0(k), atol=1e-10)


def test_unit_rejects_degenerate_but_passes_nan():
    bad = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        unit(bad)
    nan = np.array([[np.nan, 0.0, 1.0], [0.0, 0.0, 1.0]])
    out = unit(nan)
    assert np.isnan(out[0]).any() and np.isfinite(out[1]).all()


def test_tii_window_is_the_registered_037_seconds():
    k = load_tii_kinematics("flight-01a-ellipse")
    assert k.window_samples == 103          # 0.37 s at 274.73 Hz, nearest odd sample
    assert abs(k.window_s - 0.37) < 0.01
    assert k.valid.sum() == len(k) - 102
    assert abs(k.dt - 1 / 274.73) < 1e-5


def test_uzh_is_truncated_to_the_image_span():
    k = load_uzh_kinematics("indoor_forward_3")
    assert 0 < len(k)
    assert k.window_s < 0.2
    assert np.isfinite(k.lean[k.valid]).all()


def test_m5_with_zero_weights_is_m1_even_where_the_axis_is_inverted():
    from src.airframe.models import axis_M1, axis_M5
    k = _kin()
    p = {"k_forward": 0.4, "k_lateral": 0.4, "W": np.zeros((11, 2)),
         "feat_mean": np.zeros(11), "feat_sd": np.ones(11)}
    # force some inverted required axes through a strong downward acceleration
    k.a[:, 2] = -20.0
    assert np.allclose(axis_M5(k, p), axis_M1(k, p), atol=1e-10)


def test_schema_v3_writes_new_files_and_keeps_v2():
    import json
    import tempfile
    from pathlib import Path
    from src.airframe.schema_v3 import write_v3_clip
    src = next((Path("data/processed/track_d/clips")).glob("*/*__real"))
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "v3"
        n = len(np.load(src / "verdict_clip.npz")["t"])
        lean = np.zeros((n, 2))
        tilt = np.zeros(n)
        axis = np.tile([0.0, 0.0, 1.0], (n, 1))
        before = (src / "verdict_clip.npz").read_bytes()
        write_v3_clip(src, dst, lean, tilt, "M0", {}, "test", "test note", axis=axis)
        assert (src / "verdict_clip.npz").read_bytes() == before
        z = np.load(dst / "verdict_clip.npz")
        assert "axis_req_v3" in z.files and "lean_req_v3" in z.files
        meta = json.loads((dst / "meta.json").read_text())
        assert meta["schema_version"] == 3
