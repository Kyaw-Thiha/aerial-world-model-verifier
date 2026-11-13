"""Synthetic check of the clock-alignment search.

A known offset (including a case whose two clocks differ by years, which is the indoor-45
rig's situation) must be recovered, and the runtime must stay small enough that fourteen
sequences are a few minutes, not an hour.
"""
from __future__ import annotations

import time

import numpy as np

from src.depicted.sync import estimate_offset


def _signals(tau_true: float, epoch_shift: float = 0.0, dur_imu: float = 60.0,
             dur_gt: float = 20.0, seed: int = 0):
    """|omega| observed twice, on two clocks: random attitude bursts plus noise.

    Deliberately non-periodic — a periodic test signal makes the cross-correlation ambiguous
    and would "fail" a correct estimator.
    """
    rng = np.random.default_rng(seed)
    fs = 500.0
    centres = np.sort(rng.uniform(0.0, dur_imu, 25))
    widths = rng.uniform(0.05, 0.25, 25)
    amps = rng.uniform(0.5, 3.0, 25)

    def om(t):
        t = np.asarray(t, dtype=np.float64)
        out = 0.3 * np.ones_like(t)
        for c, w, a in zip(centres, widths, amps):
            out = out + a * np.exp(-0.5 * ((t - c) / w) ** 2)
        return out

    t_imu = np.arange(0.0, dur_imu, 1 / fs)
    t_gt = np.arange(5.0, 5.0 + dur_gt, 1 / fs)
    imu = om(t_imu) + rng.normal(0, 0.02, len(t_imu))
    gt = om(t_gt - tau_true) + rng.normal(0, 0.02, len(t_gt))
    return t_imu, imu, t_gt + epoch_shift, gt


def test_recovers_known_offset():
    tau = 3.4
    t_imu, imu, t_gt, gt = _signals(tau)
    rec = estimate_offset(t_imu, imu, t_gt, gt)
    assert abs(rec["offset_s"] - tau) < 0.01, rec["offset_s"]
    assert rec["peak_corr"] > 0.5
    assert rec["n_usable_bins"] >= 1
    if rec["drift_span_s"] is not None:
        assert abs(rec["drift_span_s"]) < 0.05


def test_recovers_offset_with_different_epochs():
    """The numeric ranges overlap but the clocks differ by 18 years: still recoverable."""
    tau = 7.25
    t_imu, imu, t_gt, gt = _signals(tau, epoch_shift=18 * 365.25 * 86400.0)
    rec = estimate_offset(t_imu, imu, t_gt, gt)
    assert abs(rec["offset_s"] - tau - 18 * 365.25 * 86400.0) < 0.01


def test_runtime_is_small():
    t_imu, imu, t_gt, gt = _signals(2.0, dur_imu=140.0)
    t0 = time.time()
    rec = estimate_offset(t_imu, imu, t_gt, gt)
    dt = time.time() - t0
    assert dt < 60.0, f"offset search took {dt:.1f} s"
    assert abs(rec["offset_s"] - 2.0) < 0.01
