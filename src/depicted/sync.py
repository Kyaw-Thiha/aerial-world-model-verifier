"""Cross-correlation clock alignment between two inertial signals (Task 1).

The IMU and the Leica prism are bolted to the same airframe, so the *magnitude* of the angular
velocity is one physical signal measured twice, on two clocks. That makes it the right thing to
correlate: no camera, no extrinsics, no pixel estimator in the loop, and no assumption about
which frame the truth refers to.

``tau`` is defined so that ``gt_time = imu_time + tau``. The search covers every offset that
puts some part of the truth span over the IMU span, which is what makes it work when the two
clocks have different epochs (the indoor-45 rig's images are stamped ~18 years away from the
Leica rows while the numeric ranges still overlap).
"""
from __future__ import annotations

import numpy as np


def _corr(grid: np.ndarray, a: np.ndarray, t_gt: np.ndarray, om_gt: np.ndarray, tau: float,
          mask: np.ndarray | None = None) -> float:
    m = np.isfinite(a) if mask is None else (np.isfinite(a) & mask)
    if m.sum() < 50:
        return -1.0
    a_ = a[m]
    b_ = np.interp(grid[m] + tau, t_gt, om_gt, left=np.nan, right=np.nan)
    ok = np.isfinite(b_)
    if ok.sum() < 50:
        return -1.0
    a_, b_ = a_[ok], b_[ok]
    # sqrt compresses the spikes from aggressive attitude changes so a few frames of bang-bang
    # rotation do not set the whole correlation
    a_ = np.sqrt(np.abs(a_))
    b_ = np.sqrt(np.abs(b_))
    a_ = a_ - a_.mean()
    b_ = b_ - b_.mean()
    d = np.sqrt((a_ @ a_) * (b_ @ b_))
    return float(a_ @ b_ / d) if d > 0 else -1.0


def estimate_offset(t_imu: np.ndarray, om_imu: np.ndarray, t_gt: np.ndarray,
                    om_gt: np.ndarray, coarse_s: float = 0.020, fine_s: float = 0.002,
                    drift_bins: int = 3, drift_span_s: float = 0.5) -> dict:
    """Search ``tau`` (coarse then fine), then repeat per time bin to measure clock drift."""
    t_imu = np.asarray(t_imu, dtype=np.float64)
    om_imu = np.asarray(om_imu, dtype=np.float64)
    t_gt = np.asarray(t_gt, dtype=np.float64)
    om_gt = np.asarray(om_gt, dtype=np.float64)
    if len(t_imu) < 100 or len(t_gt) < 100:
        raise ValueError(f"too few samples: {len(t_imu)} imu, {len(t_gt)} gt")

    grid_dt = min(0.002, fine_s)
    lo = t_imu[0] - 1.0
    hi = t_imu[-1] + 1.0
    grid = np.arange(lo, hi + grid_dt, grid_dt)
    a_grid = np.interp(grid, t_imu, om_imu, left=np.nan, right=np.nan)
    if not np.isfinite(a_grid).any():
        raise ValueError("no IMU coverage on the grid")

    tau_lo = float(t_gt[0] - t_imu[-1]) - 2.0
    tau_hi = float(t_gt[-1] - t_imu[0]) + 2.0
    coarse = np.arange(tau_lo, tau_hi, coarse_s)
    cs = np.array([_corr(grid, a_grid, t_gt, om_gt, t) for t in coarse])
    best_c = float(coarse[int(np.argmax(cs))])
    fine = best_c + np.arange(-2 * coarse_s, 2 * coarse_s, fine_s)
    fs = np.array([_corr(grid, a_grid, t_gt, om_gt, t) for t in fine])
    ib = int(np.argmax(fs))
    best = float(fine[ib])

    bins = []
    span = t_imu[-1] - t_imu[0]
    n_bins = max(1, int(drift_bins))
    for k in range(n_bins):
        t0 = t_imu[0] + span * k / n_bins
        t1 = t_imu[0] + span * (k + 1) / n_bins
        mask = (grid >= t0) & (grid <= t1)
        loc = best + np.arange(-drift_span_s, drift_span_s, fine_s)
        vals = np.array([_corr(grid, a_grid, t_gt, om_gt, t, mask) for t in loc])
        peaks = float(vals.max())
        bins.append({"t0_s": float(t0 - t_imu[0]), "t1_s": float(t1 - t_imu[0]),
                     "offset_s": float(loc[int(np.argmax(vals))]),
                     "peak_corr": peaks,
                     # a bin with no truth coverage has no identifiable offset; its peak is
                     # noise, so it must not enter a drift estimate
                     "usable": bool(peaks > 0.2)})
    # a plateau of near-equal correlation means the offset is not identifiable; report the
    # width of the region within 0.5% of the peak as the resolution
    near = fine[fs > fs[ib] - 0.005]
    return {
        "offset_s": best,
        "peak_corr": float(fs[ib]),
        "coarse_peak_corr": float(cs.max()),
        "search_range_s": [tau_lo, tau_hi],
        "resolution_s": float(near.max() - near.min()) if len(near) > 1 else fine_s,
        "drift_bins": bins,
        "drift_span_s": (float(bins[-1]["offset_s"] - bins[0]["offset_s"])
                         if sum(b["usable"] for b in bins) > 1 else None),
        "n_usable_bins": int(sum(b["usable"] for b in bins)),
        "curve_best_lags": fine[np.argsort(-fs)[:5]].tolist(),
    }
