"""Tests for the simulated set's construction: defects, intervals and labels.

These are the claims the whole downstream evaluation rests on, so they are checked rather than
assumed: a defect must be *exactly* the stated change, over exactly the stated interval, and
the feasible clips must be exactly physical. A silent rotation-convention error here would
look like a verifier failure on 300 clips.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.sim.dataset import build_clip
from src.sim.labels import interval_for

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "configs/sim_splits_frozen.json"


@pytest.fixture(scope="module")
def table():
    if not TABLE.is_file():
        pytest.skip("frozen table not present")
    return json.loads(TABLE.read_text())


def _first(table, **match):
    for r in table["clips"]:
        if all(r.get(k) == v for k, v in match.items()):
            return r
    raise AssertionError(f"no clip matches {match}")


def test_feasible_clip_depicts_exactly_what_the_path_requires(table):
    row = _first(table, defect_family="feasible", maneuver="ellipse")
    clip = build_clip(row)
    assert np.allclose(clip.lean_dep, clip.lean_req, atol=1e-9)
    assert np.allclose(clip.tilt_dep, clip.tilt_req, atol=1e-6)
    assert clip.active.all()
    assert clip.labels["label"] == "feasible"


def test_tilt_removed_whole_flight_leaves_no_tilt(table):
    row = _first(table, defect_family="tilt_removed")
    clip = build_clip(row)
    assert np.nanmax(clip.tilt_dep) < 1e-9
    assert np.median(clip.tilt_req) > 5.0
    assert clip.labels["violation"] == "tilt_removed"
    assert clip.labels["magnitude_deg"] == pytest.approx(np.median(clip.tilt_req), rel=1e-9)


def test_mirrored_roll_flips_only_the_roll_axis(table):
    row = _first(table, defect_family="mirrored_roll")
    clip = build_clip(row)
    assert np.allclose(clip.lean_dep[:, 0], clip.lean_req[:, 0], atol=1e-9)
    assert np.allclose(clip.lean_dep[:, 1], -clip.lean_req[:, 1], atol=1e-9)
    assert np.allclose(clip.tilt_dep, clip.tilt_req, atol=1e-6)   # magnitude unchanged


@pytest.mark.parametrize("scale", [0.5, 1.5])
def test_tilt_scaled_scales_the_body_angles(table, scale):
    """The defect is defined as roll -> s*roll and pitch -> s*pitch, so check the angles.

    The *tilt* is a nonlinear function of the two angles, so a 0.5-scaling of the angles is a
    0.502-scaling of the tilt at 20-30 deg of lean; that is a property of the geometry, not an
    error, and the manifest's magnitude column is measured on the realised tilt.
    """
    from src.verdict.lean import lean_to_roll_pitch

    rows = [r for r in table["clips"]
            if r["defect_family"] == "tilt_scaled" and float(r["magnitude"]) == scale]
    if not rows:
        pytest.skip(f"no x{scale} clip")
    clip = build_clip(rows[0])
    roll_dep, pitch_dep = lean_to_roll_pitch(clip.lean_dep)
    assert np.allclose(roll_dep, scale * clip.flight.roll, atol=1e-9)
    assert np.allclose(pitch_dep, scale * clip.flight.pitch, atol=1e-9)


def test_partial_clip_is_physical_outside_its_interval(table):
    row = _first(table, defect_family="partial")
    clip = build_clip(row)
    t0, t1 = clip.labels["interval_start_s"], clip.labels["interval_end_s"]
    inside = clip.active
    outside = ~inside
    assert 0 < inside.sum() < len(inside)
    assert np.allclose(clip.lean_dep[outside], clip.lean_req[outside], atol=1e-9)
    # and the defect is present inside, whatever kind it is
    assert not np.allclose(clip.lean_dep[inside], clip.lean_req[inside], atol=1e-3)
    dur = clip.labels["duration_s"]
    assert 0.0 <= t0 < t1 <= dur + 1e-9
    assert (t1 - t0) == pytest.approx(clip.labels["interval_fraction"] * dur, rel=1e-6)


def test_interval_placement_stays_inside_the_flight():
    for seed in range(50):
        t0, t1 = interval_for(20.0, 0.5, seed)
        assert 0.0 <= t0 < t1 <= 20.0 + 1e-9
        assert t1 - t0 == pytest.approx(10.0, rel=1e-9)
        assert t0 == pytest.approx(round(t0 * 30) / 30, abs=1e-12)   # frame grid


def test_thrust_limit_clip_is_labelled_on_the_path_not_the_camera(table):
    row = _first(table, defect_family="thrust_limit")
    clip = build_clip(row)
    assert clip.labels["peak_thrust_ratio"] > 2.0
    assert clip.labels["violation"] == "thrust_limit"
    # the camera is honest: depicted equals required even though the path is infeasible
    assert np.allclose(clip.lean_dep, clip.lean_req, atol=1e-9)


def test_every_clip_builds_with_the_declared_frame_count(table):
    for row in table["clips"][:40]:
        clip = build_clip(row)
        assert len(clip.frame_t) == clip.labels["n_frames"]
        assert clip.poses().shape == (clip.labels["n_frames"], 4, 4)
        assert np.isfinite(clip.lean_req).all() and np.isfinite(clip.lean_dep).all()
