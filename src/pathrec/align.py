"""Trajectory alignment and error metrics for the path-from-pixels track.

Two alignment families, because they answer different questions:

* **Sim(3)** (7 DoF: rotation, scale, translation) measures *shape*. It hides a
  scale error completely, which is why DASH-SLAM's authors explicitly warn that
  Sim(3) ATE alone "can exhibit errors more than an order of magnitude larger
  when evaluated under rigid SE(3) alignment".
* **SE(3)** (6 DoF, scale fixed at 1) measures a metric trajectory. A monocular
  extractor has no metric scale, so its SE(3) error is only meaningful after an
  external scale is supplied; we always report the recovered scale next to it.

A third, **yaw+scale in a gravity-aligned frame** (4 DoF), is what a
gravity-aware extractor can actually be aligned with: yaw is free, pitch and
roll come from gravity.

Nothing here uses ground truth except the evaluation functions themselves.
"""
from __future__ import annotations

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[np.ndarray, float, np.ndarray]:
    """Least-squares ``dst ~ s R src + t`` (Umeyama 1991).

    Parameters
    ----------
    src, dst : (N, 3) float arrays of matched points.
    with_scale : when False, ``s`` is fixed at 1 (rigid alignment).

    Returns
    -------
    (R, s, t) with ``R`` (3, 3) proper rotation.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"need matching (N,3) arrays, got {src.shape} and {dst.shape}")
    if len(src) < 3:
        raise ValueError(f"need at least 3 points, got {len(src)}")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    cov = b.T @ a / len(src)
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[2, 2] = -1.0
    rot = u @ s_mat @ vt
    if with_scale:
        var = (a ** 2).sum() / len(src)
        scale = float((d * np.diag(s_mat)).sum() / var) if var > 0 else 1.0
    else:
        scale = 1.0
    trans = mu_d - scale * rot @ mu_s
    return rot, scale, trans


def align_yaw_scale(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """4-DoF fit ``dst ~ s Rz(yaw) src + t`` for two already z-up frames.

    Used when gravity is known for both trajectories: pitch and roll are fixed by
    gravity, so only yaw, scale and translation remain free.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    c = float((a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]).sum())
    d = float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum())
    yaw = float(np.arctan2(d, c))
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw), np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0]])
    denom = float((a * a).sum())
    scale = float(((b @ rot) * a).sum() / denom) if denom > 0 else 1.0
    trans = mu_d - scale * rot @ mu_s
    return rot, scale, trans


def path_length(p: np.ndarray) -> float:
    """Total travelled distance of a (N, 3) track."""
    p = np.asarray(p, dtype=np.float64)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def trajectory_error(est: np.ndarray, true: np.ndarray, with_scale: bool = True) -> dict:
    """Alignment error of one recovered track against truth.

    Returns ATE RMSE (m), the aligned path length, the fitted scale (1.0 when the
    alignment is rigid), per-axis correlation and the path-length ratio. Both
    alignments are reported by :func:`both_errors`; use that in reports.
    """
    est = np.asarray(est, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    rot, scale, trans = umeyama(est, true, with_scale=with_scale)
    aligned = scale * est @ rot.T + trans
    err = np.linalg.norm(aligned - true, axis=1)
    lp_true = path_length(true)
    corr = []
    for i in range(3):
        if np.std(aligned[:, i]) < 1e-12 or np.std(true[:, i]) < 1e-12:
            corr.append(float("nan"))
        else:
            corr.append(float(np.corrcoef(aligned[:, i], true[:, i])[0, 1]))
    return {
        "alignment": "sim3" if with_scale else "se3",
        "scale": float(scale),
        "ate_rmse_m": float(np.sqrt((err ** 2).mean())),
        "ate_median_m": float(np.median(err)),
        "path_len_m": path_length(aligned),
        "path_len_true_m": lp_true,
        "path_ratio": path_length(aligned) / lp_true if lp_true > 0 else float("nan"),
        "axis_corr": corr,
    }


def both_errors(est: np.ndarray, true: np.ndarray) -> dict:
    """Sim(3) and SE(3) error side by side, so scale error cannot hide."""
    return {"sim3": trajectory_error(est, true, True), "se3": trajectory_error(est, true, False)}
