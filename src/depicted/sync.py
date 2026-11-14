"""Cross-correlation clock alignment between two inertial signals (Task 1).

The IMU and the Leica prism are bolted to the same airframe, so the *body-frame angular
velocity* is one physical quantity measured twice, on two clocks. That makes it the right thing
to correlate: no camera, no extrinsics, no pixel estimator in the loop, and no assumption about
which frame the truth refers to.

The correlation is vector-valued (all three gyro axes at once), not on |omega|: the magnitude
only has one degree of freedom and a short overlap can produce a spurious peak, while the full
3-vector has almost no chance of matching by accident. The overlap at the winning offset is
reported and an offset whose overlap is too short is declared unidentifiable rather than
believed.

``tau`` is defined so that ``gt_time = imu_time + tau``. The search covers every offset that
puts some part of the truth span over the IMU span, which is what makes it work when the two
clocks have different epochs.
"""
from __future__ import annotations

import numpy as np

MIN_OVERLAP_S = 3.0


def _corr(grid: np.ndarray, a: np.ndarray, t_gt: np.ndarray, om_gt: np.ndarray, tau: float,
          mask: np.ndarray | None = None) -> tuple[float, float]:
    """Correlation and overlap at offset ``tau`` for (N,3) signals."""
    m = np.isfinite(a).all(axis=1) if mask is None else (np.isfinite(a).all(axis=1) & mask)
    if m.sum() < 50:
        return -1.0, 0.0
    a_ = a[m]
    tq = grid[m] + tau
    b_ = np.column_stack([np.interp(tq, t_gt, om_gt[:, k], left=np.nan, right=np.nan)
                          for k in range(3)])
    ok = np.isfinite(b_).all(axis=1)
    if ok.sum() < 50:
        return -1.0, 0.0
    a_, b_, tq = a_[ok], b_[ok], tq[ok]
    # compress the spikes from aggressive attitude changes with a signed square root
    a_ = np.sign(a_) * np.sqrt(np.abs(a_))
    b_ = np.sign(b_) * np.sqrt(np.abs(b_))
    a_ = a_ - a_.mean(axis=0)
    b_ = b_ - b_.mean(axis=0)
    d = np.sqrt((a_ ** 2).sum() * (b_ ** 2).sum())
    corr = float((a_ * b_).sum() / d) if d > 0 else -1.0
    overlap = float(tq[-1] - tq[0]) if len(tq) > 1 else 0.0
    return corr, overlap


def estimate_offset(t_imu: np.ndarray, om_imu: np.ndarray, t_gt: np.ndarray,
                    om_gt: np.ndarray, coarse_s: float = 0.020, fine_s: float = 0.002,
                    drift_bins: int = 3, drift_span_s: float = 0.5,
                    min_overlap_s: float = MIN_OVERLAP_S) -> dict:
    """Search ``tau`` (coarse then fine), then repeat per fully covered time bin."""
    t_imu = np.asarray(t_imu, dtype=np.float64)
    om_imu = np.atleast_2d(np.asarray(om_imu, dtype=np.float64))
    if om_imu.shape[0] != len(t_imu):
        om_imu = om_imu.T
    t_gt = np.asarray(t_gt, dtype=np.float64)
    om_gt = np.atleast_2d(np.asarray(om_gt, dtype=np.float64))
    if om_gt.shape[0] != len(t_gt):
        om_gt = om_gt.T
    if len(t_imu) < 100 or len(t_gt) < 100:
        raise ValueError(f"too few samples: {len(t_imu)} imu, {len(t_gt)} gt")

    grid_c = np.arange(t_imu[0] - 1.0, t_imu[-1] + 1.0, coarse_s)
    grid_f = np.arange(t_imu[0] - 1.0, t_imu[-1] + 1.0, min(0.002, fine_s))
    a_c = np.column_stack([np.interp(grid_c, t_imu, om_imu[:, k], left=np.nan, right=np.nan)
                           for k in range(3)])
    a_f = np.column_stack([np.interp(grid_f, t_imu, om_imu[:, k], left=np.nan, right=np.nan)
                           for k in range(3)])
    if not np.isfinite(a_f).all(axis=1).any():
        raise ValueError("no IMU coverage on the grid")

    tau_lo = float(t_gt[0] - t_imu[-1]) - 2.0
    tau_hi = float(t_gt[-1] - t_imu[0]) + 2.0
    coarse = np.arange(tau_lo, tau_hi, coarse_s)
    cs = np.array([_corr(grid_c, a_c, t_gt, om_gt, t)[0] for t in coarse])
    best_c = float(coarse[int(np.argmax(cs))])
    fine = best_c + np.arange(-2 * coarse_s, 2 * coarse_s, fine_s)
    res = np.array([_corr(grid_f, a_f, t_gt, om_gt, t) for t in fine])
    fs = res[:, 0]
    ib = int(np.argmax(fs))
    best = float(fine[ib])
    overlap = float(res[ib, 1])

    bins = []
    span = t_imu[-1] - t_imu[0]
    n_bins = max(1, int(drift_bins))
    for k in range(n_bins):
        t0 = t_imu[0] + span * k / n_bins
        t1 = t_imu[0] + span * (k + 1) / n_bins
        mask = (grid_f >= t0) & (grid_f <= t1)
        loc = best + np.arange(-drift_span_s, drift_span_s, fine_s)
        vals = np.array([_corr(grid_f, a_f, t_gt, om_gt, t, mask) for t in loc])
        peaks = float(vals[:, 0].max())
        covered = (t0 >= t_gt[0] - best) and (t1 <= t_gt[-1] - best)
        bins.append({"t0_s": float(t0 - t_imu[0]), "t1_s": float(t1 - t_imu[0]),
                     "offset_s": float(loc[int(np.argmax(vals[:, 0]))]),
                     "peak_corr": peaks, "fully_covered": bool(covered),
                     "usable": bool(covered and peaks > 0.5)})
    near = fine[fs > fs[ib] - 0.005]
    return {
        "offset_s": best,
        "peak_corr": float(fs[ib]),
        "overlap_s": overlap,
        "identifiable": bool(overlap >= min_overlap_s),
        "coarse_peak_corr": float(cs.max()),
        "search_range_s": [tau_lo, tau_hi],
        "resolution_s": float(near.max() - near.min()) if len(near) > 1 else fine_s,
        "drift_bins": bins,
        "drift_span_s": (float(bins[-1]["offset_s"] - bins[0]["offset_s"])
                         if sum(b["usable"] for b in bins) > 1 else None),
        "n_usable_bins": int(sum(b["usable"] for b in bins)),
        "curve_best_lags": fine[np.argsort(-fs)[:5]].tolist(),
    }
