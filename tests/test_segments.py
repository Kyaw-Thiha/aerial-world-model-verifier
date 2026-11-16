"""Unit tests for the segment-level detection machinery (report G01)."""
from __future__ import annotations

import numpy as np
import pytest

from src.depicted.clips import up_body_from_lean
from src.depicted.metrics import angle_deg
from src.depicted.stat_a2 import depicted_axis_from_up_body
from src.verdict.lean import lean_to_roll_pitch, roll_pitch_to_lean
from src.verdict.schema import Clip
from src.verdict.segments import (WindowStat, clip_max, clip_max_persistent,
                                  clip_median_whole, conformal_threshold,
                                  flagged_windows, flagged_windows_persistent,
                                  frame_residual_deg, interval_scores, merge_intervals,
                                  window_statistics)


def make_clip(n=40, seed=0, slope=1.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 12.0
    pos = np.column_stack([np.cumsum(np.ones(n) * 0.05), np.zeros(n), np.zeros(n)])
    yaw = 0.05 * np.arange(n)
    lean_req = np.column_stack([0.02 * np.sin(t * 6.0) + slope * 0.01,
                                0.015 * np.cos(t * 4.0)])
    lean_dep = lean_req + rng.normal(0, 0.004, lean_req.shape)
    conf = np.full(n, 1.0)
    tilt = np.degrees(np.arcsin(np.clip(np.linalg.norm(lean_req, axis=1), 0, 1)))
    return Clip(t=t, pos=pos, yaw=yaw, depicted_lean=lean_dep, depicted_conf=conf, meta={},
                lean_req=lean_req, vel_req=0.01 * lean_req, tilt_req_deg=tilt)


def test_residual_matches_frozen_construction():
    """The per-frame residual must equal the frozen median-residual route exactly."""
    clip = make_clip()
    mine = frame_residual_deg(clip.depicted_lean, clip.lean_req, clip.yaw)
    ax_dep = depicted_axis_from_up_body(up_body_from_lean(clip.depicted_lean), clip.yaw)
    ax_req = depicted_axis_from_up_body(up_body_from_lean(clip.lean_req), clip.yaw)
    assert np.allclose(mine, angle_deg(ax_dep, ax_req), equal_nan=True)
    assert np.nanmax(np.abs(mine)) < 180.0


def test_lean_round_trip():
    lean = np.array([[0.1, -0.05], [0.0, 0.0], [0.3, 0.2]])
    r, p = lean_to_roll_pitch(lean)
    assert np.allclose(roll_pitch_to_lean(r, p), lean, atol=1e-9)


def test_windows_cover_the_tail_and_abstain_flags():
    clip = make_clip(n=40)
    stats = window_statistics(clip, length=8, hop=4)
    assert stats[0].start == 0 and stats[-1].stop == 40
    assert all((s.stop - s.start) == 8 for s in stats)
    assert stats[0].n_usable == 8


def test_window_without_signal_abstains():
    clip = make_clip(n=16)
    clip.tilt_req_deg[:] = 2.0          # below the 10 deg signal floor
    stats = window_statistics(clip, length=8, hop=4)
    assert all(s.abstained for s in stats)
    assert np.isnan(clip_max(stats, "residual"))


def test_clip_max_is_the_maximum_and_nan_when_all_abstain():
    clip = make_clip()
    clip.tilt_req_deg[:] = 20.0        # enough signal that no window abstains
    stats = window_statistics(clip, length=8, hop=4)
    vals = [s.residual for s in stats if not s.abstained and np.isfinite(s.residual)]
    assert np.isclose(clip_max(stats, "residual"), max(vals))
    assert np.isclose(clip_median_whole(clip),
                      float(np.nanmedian(frame_residual_deg(clip.depicted_lean, clip.lean_req,
                                                            clip.yaw))))


def test_persistence_ignores_a_single_spike():
    stats = [WindowStat(start=4 * i, stop=4 * i + 8, n_usable=8, n_signal=8,
                        residual=5.0, coherence=0.2, sign=0.1) for i in range(6)]
    stats[2].residual = 100.0
    assert clip_max(stats, "residual") == 100.0
    assert clip_max_persistent(stats, "residual", 2) == 5.0
    mask1 = flagged_windows(stats, "residual", 10.0)
    mask2 = flagged_windows_persistent(stats, "residual", 10.0, 2)
    assert mask1.any() and not mask2.any()


def test_conformal_threshold_is_the_upper_quantile():
    v = np.arange(1, 101, dtype=float)
    assert conformal_threshold(v, 0.05) == pytest.approx(np.quantile(v, 0.95))
    assert np.isnan(conformal_threshold([1.0, 2.0], 0.05))   # too few values


def test_merge_intervals_and_scores():
    mask = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)
    assert merge_intervals(mask) == [(1, 3), (5, 6)]
    truth = np.array([0, 0, 1, 1, 0, 0, 0], dtype=bool)
    sc = interval_scores(mask, truth)
    assert sc["detected"] is True
    assert sc["iou"] == pytest.approx(1 / 4)
    assert sc["onset_err_frames"] == -1.0
    empty = interval_scores(np.zeros(7, bool), truth)
    assert empty["detected"] is False and empty["recall"] == 0.0


def test_partial_edit_outside_the_interval_is_untouched():
    """Composition invariant: outside the injected interval the real clip is unchanged."""
    real = make_clip(seed=1)
    edit_dep = real.depicted_lean * 0.5
    frac, n = 0.25, real.n_frames
    m = max(1, int(round(frac * n)))
    lo = (n - m) // 2
    dep = real.depicted_lean.copy()
    dep[lo:lo + m] = edit_dep[lo:lo + m]
    outside = np.ones(n, bool)
    outside[lo:lo + m] = False
    assert np.allclose(dep[outside], real.depicted_lean[outside])
    assert not np.allclose(dep[~outside], real.depicted_lean[~outside])
