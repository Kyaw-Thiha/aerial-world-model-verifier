"""Unit tests for the gain test (src/verdict).

Synthetic clips where gain, intercept, drag, mirror sign and lag are known.  These tests
pin the geometry conversions and the regression; they do not touch any stored artifact.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.verdict import (Clip, FitOptions, assess, depicted_lean_from_gravity, fit_bands,
                         fit_gain, flag, lean_to_roll_pitch, load_clip,
                         required_lean, roll_pitch_to_lean, save_clip,
                         world_to_yaw_frame)
from src.verdict.schema import validate_clip


# ------------------------------------------------------------------ geometry
def _up_body(roll: float, pitch: float) -> np.ndarray:
    return np.array([-np.sin(pitch), np.sin(roll) * np.cos(pitch),
                     np.cos(roll) * np.cos(pitch)])


@pytest.mark.parametrize("roll,pitch", [
    (0.0, 0.0), (0.2, 0.0), (0.0, 0.2), (0.2, 0.2), (-0.2, 0.15), (0.2, -0.15),
    (0.5, -0.3),
])
def test_lean_round_trip_pure_and_mixed(roll, pitch):
    lean = roll_pitch_to_lean(np.array([roll]), np.array([pitch]))
    r2, p2 = lean_to_roll_pitch(lean)
    assert np.allclose(r2, roll, atol=1e-12)
    assert np.allclose(p2, pitch, atol=1e-12)


@pytest.mark.parametrize("roll,pitch", [(0.0, 0.0), (0.2, 0.0), (0.0, 0.2),
                                        (0.2, 0.2), (-0.2, -0.2)])
def test_depicted_lean_from_gravity_identity_mount(roll, pitch):
    """With R_cb = I, the body-frame up is the camera-frame up; lean is recovered."""
    up_cam = _up_body(roll, pitch)[None, :]
    lean = depicted_lean_from_gravity(up_cam, np.eye(3))
    expect = roll_pitch_to_lean(np.array([roll]), np.array([pitch]))
    assert np.allclose(lean, expect, atol=1e-12)


def test_depicted_lean_yaw_free_under_rotated_mount():
    """A yaw rotation of the camera about body z must not change the depicted lean."""
    roll, pitch = 0.25, -0.18
    up = _up_body(roll, pitch)
    c, s = np.cos(0.7), np.sin(0.7)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])  # body -> camera-ish
    # up_cam = R^T up_b, and R_cb = R^T; the composed mapping still gives up_b
    up_cam = (Rz @ up)[None, :]          # up_b = R_cb up_cam
    lean = depicted_lean_from_gravity(up_cam, Rz.T)
    expect = roll_pitch_to_lean(np.array([roll]), np.array([pitch]))
    assert np.allclose(lean, expect, atol=1e-12)


def test_yaw_frame_rotation():
    vec = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = world_to_yaw_frame(vec, np.array([np.pi / 2, np.pi / 2]))
    assert np.allclose(out[0], [0.0, -1.0], atol=1e-12)
    assert np.allclose(out[1], [1.0, 0.0], atol=1e-12)


def test_required_lean_recovers_known_acceleration():
    """A straight accelerated path with constant yaw: l_req0 = a_h / ||a + g e3||."""
    t = np.arange(0, 4.0, 1 / 12)
    a = np.array([2.0, 0.0, 0.0])
    pos = 0.5 * a[None, :] * (t ** 2)[:, None]
    yaw = np.full(len(t), 0.4)
    req = required_lean(pos, yaw, t, window_s=0.37)
    interior = req.mask & (t > 1.0) & (t < 3.0)
    unit_h = np.array([2.0, 0.0]) / np.linalg.norm(a + np.array([0.0, 0.0, 9.81]))
    expect = world_to_yaw_frame(unit_h[None, :], np.array([0.4]))[0]
    assert np.allclose(req.l_req0[interior], expect, atol=2e-3)


# ------------------------------------------------------------------ synthetic clips
def _synthetic_clip(gain=0.8, drag=0.3, intercept=(0.05, -0.03), mirror_roll=False,
                    noise=0.0, seed=0, n=48, dt=1 / 12):
    t = np.arange(n) * dt
    pos = np.column_stack([
        3.0 * np.sin(1.1 * t) + 0.6 * np.sin(2.3 * t),
        2.0 * np.sin(0.9 * t + 0.4) + 0.4 * np.cos(1.7 * t),
        0.3 * np.sin(1.3 * t),
    ])
    yaw = 0.4 * np.sin(0.8 * t) + 0.2
    req = required_lean(pos, yaw, t, window_s=0.37)
    lean = gain * req.l_req0 + drag * req.velocity_reg + np.asarray(intercept)
    if mirror_roll:
        lean = lean.copy()
        lean[:, 1] = gain * (-req.l_req0[:, 1]) + drag * req.velocity_reg[:, 1] \
            + intercept[1]
    if noise:
        lean = lean + np.random.default_rng(seed).normal(0, noise, lean.shape)
    meta = {"dataset": "synthetic", "clip_id": "syn", "condition": "real",
            "flight": "synthetic-0"}
    return Clip(t=t, pos=pos, yaw=yaw, depicted_lean=lean,
                depicted_conf=np.ones(n), meta=meta)


def _fit(clip, **kwargs):
    opts = FitOptions(lag_search=False, bootstrap=False, **kwargs)
    return fit_gain(clip, opts)


def test_known_gain_intercept_drag_are_recovered():
    clip = _synthetic_clip(gain=0.8, drag=0.3, intercept=(0.05, -0.03))
    r = _fit(clip, drag="fitted", intercept=True)
    assert r.gain == pytest.approx(0.8, abs=1e-6)
    assert r.drag == pytest.approx(0.3, abs=1e-6)
    assert r.intercept[0] == pytest.approx(0.05, abs=1e-6)
    assert r.intercept[1] == pytest.approx(-0.03, abs=1e-6)
    assert r.axis_gains[0] == pytest.approx(0.8, abs=1e-6)
    assert r.axis_gains[1] == pytest.approx(0.8, abs=1e-6)
    assert r.r2 > 0.999


def test_zero_gain_gives_zero():
    r = _fit(_synthetic_clip(gain=0.0, drag=0.3))
    assert abs(r.gain) < 1e-6
    assert abs(r.axis_gains[0]) < 1e-6
    assert abs(r.axis_gains[1]) < 1e-6


def test_mirror_roll_flips_the_roll_axis_gain():
    r = _fit(_synthetic_clip(gain=0.8, drag=0.3, mirror_roll=True))
    assert r.axis_gains[0] == pytest.approx(0.8, abs=1e-6)
    assert r.axis_gains[1] == pytest.approx(-0.8, abs=1e-6)


def test_intercept_absorbs_mount_offset_without_moving_the_gain():
    a = _fit(_synthetic_clip(gain=0.8, intercept=(0.0, 0.0)))
    b = _fit(_synthetic_clip(gain=0.8, intercept=(0.12, -0.08)))
    assert a.gain == pytest.approx(b.gain, abs=1e-9)
    assert b.intercept[0] == pytest.approx(0.12, abs=1e-9)


def test_fixed_drag_variant_subtracts_the_registered_value():
    clip = _synthetic_clip(gain=0.8, drag=0.3)
    r = _fit(clip, drag="fixed")
    assert r.drag == pytest.approx(0.295)
    # with the wrong fixed drag the gain is biased, which is the point of the variant
    assert abs(r.gain - 0.8) < 0.02


def test_drag_free_variant_has_no_velocity_term():
    r = _fit(_synthetic_clip(gain=0.8, drag=0.0), drag="free")
    assert r.drag == 0.0
    assert abs(r.gain - 0.8) < 1e-6
    # on data that does carry drag, omitting the term biases the gain: that is the
    # reason drag is a fitted nuisance in the primary statistic.
    biased = _fit(_synthetic_clip(gain=0.8, drag=0.3), drag="free")
    assert abs(biased.gain - 0.8) > 0.1


def test_lag_search_recovers_a_known_delay():
    clip = _synthetic_clip(gain=0.8, drag=0.3, intercept=(0.0, 0.0))
    req = required_lean(clip.pos, clip.yaw, clip.t)
    ldep = 0.8 * req.l_req0 + 0.3 * req.velocity_reg
    # depicted lags by 75 ms: l_dep(t) = truth(t - 0.075)
    t = clip.t
    ldep_lag = np.column_stack([np.interp(t - 0.075, t, ldep[:, k]) for k in range(2)])
    clip.depicted_lean = ldep_lag
    opts = FitOptions(lag_search=True, bootstrap=False)
    r = fit_gain(clip, opts)
    assert abs(r.lag_ms - (-75.0)) <= 30.0


def test_collinear_inputs_abstain():
    """Straight-line acceleration makes acceleration and velocity collinear."""
    n, dt = 48, 1 / 12
    t = np.arange(n) * dt
    pos = np.column_stack([0.7 * t ** 2, np.zeros(n), np.zeros(n)])
    yaw = np.full(n, 0.2)
    req = required_lean(pos, yaw, t)
    clip = Clip(t=t, pos=pos, yaw=yaw,
                depicted_lean=0.8 * req.l_req0 + 0.3 * req.velocity_reg,
                depicted_conf=np.ones(n),
                meta={"dataset": "synthetic", "clip_id": "line", "condition": "real",
                      "flight": "synthetic-0"})
    r = fit_gain(clip, FitOptions(lag_search=False, bootstrap=False))
    ab = assess(r)
    assert not ab.testable
    assert any("condition_number" in reason for reason in ab.reasons)


def test_noise_does_not_bias_a_clean_gain():
    r = _fit(_synthetic_clip(gain=0.8, drag=0.3, noise=0.002, seed=1))
    assert abs(r.gain - 0.8) < 0.01


# ------------------------------------------------------------------ schema and decisions
def test_schema_round_trip(tmp_path):
    clip = _synthetic_clip()
    save_clip(tmp_path / "c", clip)
    back = load_clip(tmp_path / "c")
    assert np.allclose(back.t, clip.t)
    assert np.allclose(back.depicted_lean, clip.depicted_lean)
    assert back.meta["condition"] == "real"


def test_schema_rejects_non_monotone_time():
    clip = _synthetic_clip()
    clip.t[3] = clip.t[2]
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_clip(clip)


def test_schema_rejects_wrong_shape():
    clip = _synthetic_clip()
    clip.depicted_lean = clip.depicted_lean[:, :1]
    with pytest.raises(ValueError, match="depicted_lean"):
        validate_clip(clip)


def test_extra_npz_fields_are_ignored(tmp_path):
    """No mocap-derived quantity may enter: an unused field must not change the result."""
    clip = _synthetic_clip()
    save_clip(tmp_path / "a", clip)
    import shutil
    shutil.copytree(tmp_path / "a", tmp_path / "b")
    with np.load(tmp_path / "b" / "verdict_clip.npz") as d:
        np.savez_compressed(tmp_path / "b" / "verdict_clip.npz",
                            **{k: d[k] for k in d.files}, implied_axis=np.ones((len(clip.t), 3)))
    a = fit_gain(load_clip(tmp_path / "a"), FitOptions(lag_search=False, bootstrap=False))
    b = fit_gain(load_clip(tmp_path / "b"), FitOptions(lag_search=False, bootstrap=False))
    assert a.gain == b.gain


def test_conformal_band_flags_outliers_and_not_the_body():
    import numpy as np
    from src.verdict import conformal_band
    values = list(np.linspace(0.6, 0.9, 40))
    band = conformal_band(values, alpha=0.05)
    assert not band.flag(0.75)
    assert band.flag(1.5)
    assert band.flag(-0.5)


def test_joint_axis_band_catches_a_mirror():
    from src.verdict import fit_joint_band
    rng = np.random.default_rng(3)
    real = []
    for _ in range(40):
        r = _synthetic_clip(gain=0.8, drag=0.3, noise=0.002, seed=int(rng.integers(1e6)))
        real.append(_fit(r))
    band = fit_joint_band(real, ("gx", "gy"))
    mirrored = _fit(_synthetic_clip(gain=0.8, drag=0.3, mirror_roll=True, noise=0.002))
    assert band.flag(mirrored)
    assert not band.flag(real[0])
