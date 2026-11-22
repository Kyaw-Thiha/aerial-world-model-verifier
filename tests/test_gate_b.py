import numpy as np
import pytest

from src.verdict.gate_b import (Disagreement, disagreement, ensemble_lean, gate_b,
                                gravity_gc, gravity_pf, lean_difference_deg,
                                lean_from_gravity, roll_of)


def _row(roll_deg=0.0, pitch_deg=0.0, gc=(0.0, -1.0, 0.0)):
    return {"pf_roll_deg": roll_deg, "pf_pitch_deg": pitch_deg,
            "gc_gravity": list(gc)}


def test_roll_of_level_up_is_zero():
    assert abs(float(roll_of(np.array([0.0, -1.0, 0.0]))[0])) < 1e-12


def test_roll_follows_up_direction():
    # up tilted to the right of the image => positive roll
    u = np.array([np.sin(np.radians(20.0)), -np.cos(np.radians(20.0)), 0.0])
    assert abs(float(roll_of(u)[0]) - 20.0) < 1e-9


def test_pf_gravity_is_unit_and_level():
    g = gravity_pf(_row(0.0, 0.0))
    assert abs(np.linalg.norm(g) - 1.0) < 1e-12
    assert g[1] < -0.99


def test_pf_roll_sign_is_the_fitted_one():
    # the fitted convention is roll_sign = -1, so a positive pf_roll is a negative roll
    assert gravity_pf(_row(10.0, 0.0))[0] < 0
    assert gravity_pf(_row(-10.0, 0.0))[0] > 0


def test_gc_gravity_passthrough():
    assert np.allclose(gravity_gc(_row(gc=(0.1, -0.2, 0.3))), [0.1, -0.2, 0.3])


def test_lean_from_gravity_level_with_the_project_mount():
    from src.perception.frames import camera_to_body_pitched
    # camera x = -body y, camera y = -body z, camera z = body x at zero uptilt
    R_cb = camera_to_body_pitched(0.0)
    lean = lean_from_gravity(np.array([[0.0, -1.0, 0.0]]), R_cb)
    assert np.allclose(lean, [[0.0, 0.0]], atol=1e-9)


def test_lean_difference_is_degrees_and_handles_nan():
    a = np.array([[0.0, 0.0], [0.1, 0.0]])
    b = np.array([[0.0, 0.0], [np.nan, 0.0]])
    d = lean_difference_deg(a, b)
    assert d[0] == 0.0
    assert np.isnan(d[1])
    big = lean_difference_deg(np.array([[0.0, 0.0]]), np.array([[0.5, 0.0]]))
    assert abs(float(big[0]) - np.degrees(np.arcsin(0.5))) < 1e-9


def test_disagreement_summary_and_abstention():
    d = np.full(24, 2.0)
    s = disagreement(d)
    assert abs(s.median - 2.0) < 1e-12 and s.n_frames == 24
    assert gate_b(s.median, 1.0)["testable"] is False
    assert gate_b(s.median, 3.0)["testable"] is True
    assert gate_b(s.median, 5.0)["testable"] is True
    assert "readers_disagree" in gate_b(float("nan"), 5.0)["reasons"]


def test_disagreement_abstains_on_too_few_frames():
    s = disagreement(np.array([1.0, np.nan, np.nan]))
    assert np.isnan(s.median)


def test_ensemble_mean_and_nan_rule():
    a = np.array([[0.0, 0.0], [0.2, 0.0]])
    b = np.array([[0.1, 0.0], [np.nan, 0.0]])
    e = ensemble_lean(a, b)
    assert abs(e[0, 0] - 0.05) < 1e-12
    assert np.isnan(e[1]).all()


def test_disagreement_dataclass_roundtrip():
    d = Disagreement(1.0, 2.0, 3.0, 10)
    assert d.as_dict()["window_max"] == 3.0
