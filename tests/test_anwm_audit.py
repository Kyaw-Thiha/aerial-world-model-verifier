import numpy as np
import pytest

from src.generation.audit import (heading_statistics, per_step_tilt, pooled_stats,
                                  smoothed_tilt)


def test_pooled_stats_consistency_on_bimodal_trajectories():
    # 3 trajectories: two never accelerate, one always does at 30 deg
    tilts = [np.zeros(50), np.zeros(50), np.full(50, 30.0)]
    s = pooled_stats(tilts)
    assert s["median"] == 0.0 and abs(s["frac_gt"] - 1 / 3) < 1e-9
    # the old estimator (mean of per-trajectory medians) would have said 10 deg
    assert np.mean([np.median(t) for t in tilts]) == pytest.approx(10.0)


def test_median_above_threshold_implies_majority_above():
    rng = np.random.default_rng(0)
    for _ in range(50):
        t = rng.uniform(0, 40, 200)
        s = pooled_stats([t])
        assert not (s["median"] > 10 and s["frac_gt"] < 0.5)


def test_per_step_tilt_matches_legacy_audit():
    from experiments.anwm_excitation_audit import per_step_tilt as legacy
    rng = np.random.default_rng(1)
    p = np.cumsum(rng.normal(size=(40, 3)) * [3, 3, 0.5], axis=0)
    for dt in (0.25, 0.667, 1.0):
        for k in (0.0, 0.295):
            np.testing.assert_allclose(per_step_tilt(p, dt, k), legacy(p, dt, k), equal_nan=True)


def test_constant_velocity_has_zero_tilt_and_scales_inverse_dt2():
    t = np.arange(30)[:, None]
    p = np.hstack([t * 3.0, 0 * t, 0 * t]).astype(float)
    assert np.nanmax(smoothed_tilt(p, 1.0)) < 1e-6
    p2 = p.copy()
    p2[15:, 0] += 3.0 * np.arange(15)  # speed doubles at step 15
    a1 = np.nanmax(per_step_tilt(p2, 2.0)); a2 = np.nanmax(per_step_tilt(p2, 1.0))
    assert a2 > a1  # smaller dt, larger tilt


def test_heading_statistics_separates_yaw_from_travel_direction():
    # strafe: travel turns 90 deg in the world while yaw stays constant
    p = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0], [10, 5, 0], [10, 10, 0]], float)
    h = heading_statistics(p, np.zeros(5))
    assert h["yaw_change"].max() == 0.0
    assert h["travel_dir_change"].max() == pytest.approx(90.0)
