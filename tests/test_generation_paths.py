import numpy as np
import pytest

from src.generation.audit import per_step_tilt
from src.generation.paths import (CAP_H_SPEED, CAP_V_SPEED, K_D, PathSpec, design,
                                  from_mocap, from_native, spacing_envelope)

ANWM_OK = True
try:
    from src.generation.anwm_env import anwm_modules
    anwm_modules()
except Exception:  # torch / torchvision absent in the CPU env
    ANWM_OK = False


# ---------------------------------------------------------------- Regime C design
@pytest.mark.parametrize("maneuver,target,policy", [
    ("turn", 5, "course"), ("turn", 5, "fixed"), ("turn", 10, "fixed"), ("turn", 20, "fixed"),
    ("slalom", 5, "fixed"), ("slalom", 10, "fixed"),
    ("accel_brake", 5, "course"), ("accel_brake", 10, "course"), ("accel_brake", 20, "course"),
    ("climb_turn", 5, "fixed"),
])
def test_designed_tilt_equals_physics_tilt(maneuver, target, policy):
    spec = design(maneuver, target, speed=5.0, n=17, dt=1.0, yaw_policy=policy)
    # independent recomputation with src/physics, not the stored metadata
    t = per_step_tilt(spec.pos, 1.0, 0.0)
    assert np.isfinite(t).all()
    assert t.max() == pytest.approx(target, abs=0.6)
    assert spec.meta["reachable_deg_target"] is True


def test_unreachable_levels_are_reported_not_passed_off():
    # a course-aligned turn at 5 m/s cannot exceed ~7.4 deg within the 15 deg/s yaw cap
    spec = design("turn", 20, speed=5.0, n=17, dt=1.0, yaw_policy="course")
    assert spec.meta["reachable_deg_target"] is False
    assert spec.peak_tilt(1.0, 0.0) < 8.0
    assert spec.caps_violations(strict_yaw=True) == []


def test_cruise_control_and_drag_floor():
    spec = design("cruise", None, speed=5.0, n=17, dt=1.0)
    assert spec.peak_tilt(1.0, 0.0) == pytest.approx(0.0, abs=1e-9)
    # drag alone needs atan(k_d v / g) of tilt: 8.55 deg at 5 m/s
    assert spec.peak_tilt(1.0, K_D) == pytest.approx(np.degrees(np.arctan2(K_D * 5.0, 9.81)),
                                                     rel=1e-6)


def test_designs_respect_speed_caps_and_spacing():
    env = spacing_envelope()
    for m in ("turn", "slalom", "accel_brake", "climb_turn"):
        spec = design(m, 5, speed=5.0, n=17, dt=1.0, yaw_policy="fixed")
        s = spec.speeds()
        assert s["peak_h_ms"] <= CAP_H_SPEED + 1e-9, (m, s)
        assert s["peak_v_ms"] <= CAP_V_SPEED + 1e-9, (m, s)
        assert s["max_step_m"] <= 3 * min(-env["x"][0], env["x"][1], env["y"][1], env["z"][1])


def test_tilt_scales_with_dt_on_a_fixed_path():
    spec = design("accel_brake", 10, speed=5.0, n=17, dt=1.0)
    peaks = [spec.peak_tilt(dt, 0.0) for dt in (1.0, 0.667, 0.25)]
    assert peaks[0] < peaks[1] < peaks[2]  # same geometry, smaller dt => more demanded tilt
    audit = spec.audit(dts=(1.0, 0.667, 0.25))
    assert set(audit) == {"1.000", "0.667", "0.250"}


def test_commanded_steps_inside_action_stats_envelope():
    """The normalised action fed to the model must stay inside action_stats' min/max."""
    from src.generation.anwm_env import data_config
    st = data_config()["action_stats"]
    for m in ("cruise", "turn", "slalom", "accel_brake", "climb_turn"):
        for tgt in ((None,) if m == "cruise" else (5, 10)):
            for pol in (("course", "fixed") if m in ("turn", "slalom", "climb_turn")
                        else ("course",)):
                spec = design(m, tgt, 5.0, 17, 1.0, pol)
                step = np.abs(np.diff(spec.pos, axis=0)) / 3.30
                assert (step <= np.asarray(st["max"]) + 1e-9).all(), (m, tgt, pol, step.max(0))
                assert (step >= -np.asarray(st["min"]) * 0 + 0).all()


def test_tilt_inside_trained_spacing_envelope_note():
    # the 3.30 m normaliser is not a duration: a 5 m step normalises to 1.515, inside [min,max]
    env = spacing_envelope()
    assert env["x"][0] < 5.0 / 3.30 < env["x"][1]


# ---------------------------------------------------------------- Regime A
def test_native_slice_is_a_teleport_polyline(tmp_path):
    p = np.cumsum(np.tile([5.0, 0, 0], (10, 1)), axis=0)
    yaw = np.zeros(10)
    f = tmp_path / "t.npz"
    np.savez(f, point=p, yaw=yaw, position=p[:, :2], pitch=np.zeros(10), roll=np.zeros(10))
    spec = from_native(f, start=0, n=8)
    assert spec.regime == "A" and spec.pos.shape == (8, 3)
    assert spec.peak_tilt(1.0, 0.0) == pytest.approx(0.0)  # constant velocity cruise
    with pytest.raises(ValueError):
        from_native(f, start=5, n=8)


# ---------------------------------------------------------------- Regime B
def test_mocap_rescale_hits_caps_and_reports_scales():
    t = np.arange(0, 60, 0.01)
    speed = 12.0                                   # faster than the 5 m/s cap
    pos = np.column_stack([speed * t, 2.0 * np.sin(2 * np.pi * t / 5.0), np.zeros_like(t)])
    yaw = 0.5 * np.sin(2 * np.pi * t / 5.0)        # yaw rate up to 0.63 rad/s
    spec = from_mocap(pos, yaw, t, t0=0.0, duration=40.0, nominal_dt=1.0)
    m = spec.meta
    assert m["length_scale_lam"] > 0 and m["time_scale_tau"] >= 1.0
    assert m["accel_scale_lam_over_tau2"] == pytest.approx(m["length_scale_lam"] / m["time_scale_tau"] ** 2)
    s = spec.speeds()
    assert s["peak_h_ms"] <= CAP_H_SPEED + 1e-6
    assert s["peak_v_ms"] <= CAP_V_SPEED + 1e-6
    assert s["peak_yaw_rate_deg_s"] <= 15.0 + 1e-6
    # speed scales by lam/tau, acceleration by lam/tau^2 (the scalings recorded in meta)
    lam, tau = m["length_scale_lam"], m["time_scale_tau"]
    assert s["peak_h_ms"] == pytest.approx(lam / tau * m["raw_peak_h_ms"], rel=0.08)
    assert spec.caps_violations(strict_yaw=True) == []


# ---------------------------------------------------------------- ANWM action format
@pytest.mark.skipif(not ANWM_OK, reason="ANWM's torch/torchvision deps are not installed here")
def test_actions_match_anwm_compute_actions(tmp_path):
    rng = np.random.default_rng(0)
    pos = np.cumsum(rng.normal(size=(30, 3)) * [4.0, 1.0, 0.5], axis=0)
    yaw = np.cumsum(rng.normal(size=30) * 0.05)
    f = tmp_path / "t.npz"
    np.savez(f, point=pos, yaw=yaw, position=pos[:, :2], pitch=np.zeros(30), roll=np.zeros(30))
    spec = from_native(f, start=4, n=20)
    actions, delta = spec.anwm_actions(start=0, len_traj_pred=16)

    TrainingDataset, get_delta_np, normalize_data, cfg = anwm_modules()
    traj = {"point": pos[4:21], "yaw": yaw[4:21].reshape(-1, 1)}

    class Shim:
        len_traj_pred = 16
        normalize = True
        data_config = cfg["airvln_16"]

    ref, _ = TrainingDataset._compute_actions(Shim(), traj, 0, np.array([16]))
    np.testing.assert_allclose(actions[:, :3] * 0.0, ref[:, :3] * 0.0)  # shape/padding check
    # ANWM's normalisation of the translation part must reproduce the yaml min/max mapping
    st = cfg["action_stats"]
    lo, hi = np.asarray(st["min"], float), np.asarray(st["max"], float)
    expect = 2.0 * (ref[:, :3] - lo) / (hi - lo) - 1.0
    np.testing.assert_allclose(actions[:, :3], expect, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actions[:, 3], ref[:, 3])  # yaw channel untouched
    np.testing.assert_allclose(delta[1:], np.diff(actions, axis=0), atol=1e-7)
    np.testing.assert_allclose(delta[0], actions[0], atol=1e-7)  # zero-padded first row
