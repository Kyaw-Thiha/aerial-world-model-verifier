"""Score one recovered path against ground truth, the way the verifier needs it scored.

The verifier does not need a good trajectory; it needs the *horizontal acceleration* of the
trajectory, its direction and its magnitude, plus an honest coverage number. So every
clip is scored on:

* trajectory: Sim(3) and SE(3) ATE, the fitted scale, path-length ratio (DASH-SLAM's rule:
  never report Sim(3) alone);
* direction: per-step displacement direction error (scale-free);
* acceleration: horizontal direction error, magnitude ratio, correlation and R^2 after
  Savitzky-Golay differentiation with a declared window;
* required tilt: the tilt the recovered path implies versus the tilt the true path implies,
  drag-free and with ``k_d``;
* rotation: relative rotation error per frame, to confirm rotation stays good;
* coverage: the fraction of samples whose *true* path requires at least 10 deg of tilt,
  i.e. the fraction on which a verifier could say anything at all.

All metrics are computed after aligning the recovered path to truth (evaluation only). The
direction metrics use the alignment's rotation, not its scale, so they stay scale-free.
"""
from __future__ import annotations

import numpy as np

from src.pathrec.align import both_errors, path_length, umeyama
from src.pathrec.lean import (direction_error_deg, lean_correlation, lean_xy, magnitude_ratio,
                              required_tilt_deg)


def up_axis(pos: np.ndarray) -> dict:
    """Which world axis is up, from the *shape* of the flight (an indoor flight is flat).

    Returns the axis index, the per-axis standard deviations and whether the choice is
    unambiguous (chosen axis at most half the next largest). Evaluation must know this to
    split horizontal from vertical acceleration; when it is ambiguous the caller should
    treat tilt numbers as unreliable, and that is reported rather than hidden.
    """
    pos = np.asarray(pos, dtype=np.float64)
    std = pos.std(axis=0)
    order = np.argsort(std)
    axis = int(order[0])
    if std.max() <= 1e-9:                # a point: no extent, so no up direction
        return {"axis": axis, "std": std.tolist(), "margin": float("nan"),
                "unambiguous": False}
    # a perfectly flat flight has zero extent along up, which is the best evidence there is
    margin = float(std[order[1]] / std[order[0]]) if std[order[0]] > 1e-9 else float("inf")
    return {"axis": axis, "std": std.tolist(), "margin": margin,
            "unambiguous": bool(margin >= 2.0)}


def _rot_from_quat(q: np.ndarray) -> np.ndarray:
    """(N,4) xyzw -> (N,3,3) rotation matrices."""
    from scipy.spatial.transform import Rotation
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return Rotation.from_quat(q).as_matrix()


def relative_rotation_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> np.ndarray:
    """Angle between consecutive rotations of two sequences, per step, in degrees."""
    a = np.asarray(rot_a, dtype=np.float64)
    b = np.asarray(rot_b, dtype=np.float64)
    n = min(len(a), len(b))
    rel_a = np.einsum("nji,njk->nik", a[: n - 1], a[1:n])
    rel_b = np.einsum("nji,njk->nik", b[: n - 1], b[1:n])
    d = np.einsum("nij,nkj->nik", rel_a, rel_b)     # rel_a * rel_b^T
    cos = (np.trace(d, axis1=1, axis2=2) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def score_clip(est_pos: np.ndarray, true_pos: np.ndarray, dt: float, *,
               est_quat: np.ndarray | None = None, true_quat: np.ndarray | None = None,
               window_s: float = 0.5, polyorder: int = 3, k_d: float = 0.295,
               min_true_tilt_deg: float = 10.0) -> dict:
    """All metrics for one clip. ``dt`` is the sample interval actually used, in seconds."""
    est = np.asarray(est_pos, dtype=np.float64)
    true = np.asarray(true_pos, dtype=np.float64)
    n = min(len(est), len(true))
    est, true = est[:n], true[:n]
    out: dict = {"n": int(n), "dt_s": float(dt), "fps": float(1.0 / dt),
                 "window_s": float(window_s), "k_d": float(k_d)}

    errs = both_errors(est, true)
    out["ate_sim3_m"] = errs["sim3"]["ate_rmse_m"]
    out["scale_sim3"] = errs["sim3"]["scale"]
    out["ate_se3_m"] = errs["se3"]["ate_rmse_m"]
    out["path_len_est_m"] = errs["se3"]["path_len_m"]
    out["path_len_true_m"] = errs["se3"]["path_len_true_m"]
    out["path_ratio_se3"] = errs["se3"]["path_ratio"]

    rot, scale, trans = umeyama(est, true, with_scale=True)
    est_al = scale * est @ rot.T + trans

    # per-step direction, scale-free
    s_est, s_true = np.diff(est_al, axis=0), np.diff(true, axis=0)
    ne, nt = np.linalg.norm(s_est, axis=1), np.linalg.norm(s_true, axis=1)
    keep = (ne > 0) & (nt > 0.02 * np.median(nt))
    if keep.any():
        cos = np.clip((s_est[keep] * s_true[keep]).sum(1) / (ne[keep] * nt[keep]), -1, 1)
        out["step_dir_err_deg_median"] = float(np.degrees(np.median(np.arccos(cos))))
        out["step_dir_err_deg_p75"] = float(np.degrees(np.percentile(np.arccos(cos), 75)))
    else:
        out["step_dir_err_deg_median"] = float("nan")
        out["step_dir_err_deg_p75"] = float("nan")

    # acceleration, tilt and lean
    from src.pathrec.lean import differentiate
    t_e, v_e, a_e = differentiate(est_al, dt=dt, window_s=window_s, polyorder=polyorder)
    t_t, v_t, a_t = differentiate(true, dt=dt, window_s=window_s, polyorder=polyorder)
    m = min(len(a_e), len(a_t))
    a_e, a_t = a_e[:m], a_t[:m]
    v_e, v_t = v_e[:m], v_t[:m]

    up = up_axis(true)
    axis = up["axis"]
    h = [i for i in range(3) if i != axis]
    out["up_axis"] = axis
    out["up_margin"] = up["margin"]
    out["up_unambiguous"] = up["unambiguous"]

    # lean_xy is defined for a z-up world; permute so the detected up axis is last
    perm = [i for i in range(3) if i != axis] + [axis]
    a_e, a_t = a_e[:, perm], a_t[:, perm]
    v_e, v_t = v_e[:, perm], v_t[:, perm]
    l_est, l_true = lean_xy(a_e), lean_xy(a_t)
    dir_err = direction_error_deg(l_est, l_true)
    out["accel_dir_err_deg_median"] = float(np.median(dir_err)) if dir_err.size else float("nan")
    out["accel_dir_err_deg_p75"] = float(np.percentile(dir_err, 75)) if dir_err.size else float("nan")
    out["accel_mag_ratio"] = magnitude_ratio(l_est, l_true)
    fit = lean_correlation(l_est, l_true)
    out.update({"lean_corr_x": fit["corr_x"], "lean_corr_y": fit["corr_y"],
                "lean_r2": fit["r2"], "lean_slope": fit["slope"], "lean_n": fit["n"]})

    tilt_e = required_tilt_deg(a_e, None, k_d=0.0)
    tilt_t = required_tilt_deg(a_t, v_t, k_d=0.0)
    tilt_e_d = required_tilt_deg(a_e, v_e, k_d=k_d)
    tilt_t_d = required_tilt_deg(a_t, v_t, k_d=k_d)
    out["tilt_err_deg_median"] = float(np.median(np.abs(tilt_e - tilt_t)))
    out["tilt_err_deg_median_drag"] = float(np.median(np.abs(tilt_e_d - tilt_t_d)))
    out["tilt_true_median_deg"] = float(np.median(tilt_t))
    out["testable_fraction"] = float((tilt_t >= min_true_tilt_deg).mean())
    out["coverage_fraction"] = out["testable_fraction"]

    if est_quat is not None and true_quat is not None:
        r_e, r_t = _rot_from_quat(est_quat[:n]), _rot_from_quat(true_quat[:n])
        rr = relative_rotation_deg(r_e, r_t)
        out["rel_rot_err_deg_median"] = float(np.median(rr))
        out["rel_rot_err_deg_p90"] = float(np.percentile(rr, 90))
    else:
        out["rel_rot_err_deg_median"] = float("nan")
        out["rel_rot_err_deg_p90"] = float("nan")
    return out
