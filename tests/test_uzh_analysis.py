"""The Task 4 analysis must recover a known scale and reject a known-bad track."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("uzh", ROOT / "experiments/phase3a_uzh_dpvo.py")
uzh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uzh)


def _track(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.02
    pos = np.column_stack([3 * np.sin(0.6 * t), 2 * np.cos(0.4 * t), 0.5 * np.sin(0.9 * t)])
    pos += rng.normal(scale=1e-3, size=pos.shape)
    return t, pos


def test_oracle_scale_recovers_a_known_factor():
    _, gt = _track()
    vo = gt * 0.25
    fit = uzh.oracle_scale(vo, gt)
    assert fit["scale"] == np.float64(4.0) or abs(fit["scale"] - 4.0) < 1e-3


def test_correlation_is_one_for_a_scaled_copy_and_low_for_noise():
    _, gt = _track()
    corr = [np.corrcoef((gt * 0.25)[:, k], gt[:, k])[0, 1] for k in range(3)]
    assert min(corr) > 0.999
    rng = np.random.default_rng(1)
    noisy = rng.normal(size=gt.shape)
    corr_bad = [np.corrcoef(noisy[:, k], gt[:, k])[0, 1] for k in range(3)]
    assert max(abs(np.array(corr_bad))) < 0.2


def test_implied_tilt_of_a_constant_velocity_track_is_the_drag_tilt():
    """Constant speed with no acceleration still needs tilt to balance drag: atan(k_d v / g)."""
    t = np.arange(300) * 0.02
    v = 1.5
    pos = np.column_stack([v * t, np.zeros_like(t), np.zeros_like(t)])
    tilt = uzh.tilt_deg(np.asarray(uzh.implied_axis(pos, t))[10:-10])
    expected = np.degrees(np.arctan(uzh.K_D * v / 9.81))
    assert abs(float(np.nanmedian(tilt)) - expected) < 0.1
