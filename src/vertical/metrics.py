"""The Track V vertical-motion statistic, computed from stored reconstruction output.

Statistic (``configs/v00_feasibility_prereg.yaml`` + amendment 1):

* per window ``[t1, t2]`` (3 s, stride 0.5 s):
  - ``ratio_dep = h(t2) / h(t1)`` with ``h`` the camera's signed distance to the window's ground
    plane (reconstruction units, scale-free);
  - ``ratio_cmd = z(t2)/z(t1)`` on the simulated clips (scene ground at z = 0, exact truth) and
    ``1 + (z(t2)-z(t1)) / (s h(t1))`` on real footage, with one similarity scale ``s`` per
    sequence;
  - ``e = log(ratio_dep) - log(ratio_cmd)``; ``|e|`` is the error.
* secondary, scale-free and exploratory:
  ``rho = dz_dep / L_dep`` against ``rho_cmd = dz_truth / L_truth`` and
  ``q = (rho_dep - rho_cmd) / (|rho_cmd| + 0.05)``.

Frames enter a window when their time is inside ``[t1, t2]``; heights are linearly
interpolated to the exact window ends from the two reconstructed frames that bracket them, and
only windows whose ends are both inside the reconstructed time span are used.
"""
from __future__ import annotations

import numpy as np

WINDOW_S = 3.0
WINDOW_STRIDE_S = 0.5
WINDOW_MIN_FRAMES = 6
RHO_FLOOR = 0.05
DEFECT_FAMILIES = ("removed", "scaled_0.5", "scaled_1.5", "reversed")


def windows(t: np.ndarray, length: float = WINDOW_S, stride: float = WINDOW_STRIDE_S,
            min_frames: int = WINDOW_MIN_FRAMES) -> list[dict]:
    """Sliding windows over the reconstructed frame times."""
    t = np.asarray(t, dtype=np.float64)
    if len(t) < 3:
        return []
    t0, t1 = float(t[0]), float(t[-1])
    if t1 - t0 < length + 1e-9:
        length = max(1.0, 0.5 * (t1 - t0))
    out = []
    start = t0
    while start + length <= t1 + 1e-9:
        a, b = start, start + length
        inside = np.flatnonzero((t >= a - 1e-9) & (t <= b + 1e-9))
        if len(inside) >= min_frames:
            out.append({"t1": a, "t2": b, "idx": inside})
        start += stride
    return out


def interp_series(t: np.ndarray, y: np.ndarray, tq: float) -> float:
    return float(np.interp(tq, t, y))


def _sign(x: float, digits: int = 12) -> float:
    return float(np.round(x, digits))


def window_statistics(t_rec: np.ndarray, c2w: np.ndarray, plane_n: np.ndarray, plane_d: float,
                      t_truth: np.ndarray, pos_truth: np.ndarray, *, mode: str,
                      scale_s: float | None = None) -> list[dict]:
    """Every pre-registered quantity for one unit, window by window.

    ``mode`` is ``sim`` (ratio_cmd from the truth altitude above z=0) or ``real`` (ratio_cmd
    with the sequence scale and the reconstruction's own initial height).
    """
    t_rec = np.asarray(t_rec, dtype=np.float64)
    c = np.asarray(c2w, dtype=np.float64)[:, :3, 3]
    n = np.asarray(plane_n, dtype=np.float64)
    h = c @ n - float(plane_d)
    t_truth = np.asarray(t_truth, dtype=np.float64)
    pos = np.asarray(pos_truth, dtype=np.float64)
    z = pos[:, 2]
    xy = pos[:, :2]

    rows = []
    for w in windows(t_rec):
        a, b = w["t1"], w["t2"]
        h1 = interp_series(t_rec, h, a)
        h2 = interp_series(t_rec, h, b)
        if h1 <= 0 or h2 <= 0:
            rows.append({"t1": a, "t2": b, "ok": False, "reason": "nonpositive_height",
                         "h1": h1, "h2": h2})
            continue
        z1 = interp_series(t_truth, z, a)
        z2 = interp_series(t_truth, z, b)
        x1 = interp_series(t_truth, xy[:, 0], a)
        y1 = interp_series(t_truth, xy[:, 1], a)
        x2 = interp_series(t_truth, xy[:, 0], b)
        y2 = interp_series(t_truth, xy[:, 1], b)
        L_truth = float(np.hypot(x2 - x1, y2 - y1))
        dz_truth = float(z2 - z1)
        ratio_dep = h2 / h1
        if mode == "sim":
            if z1 <= 0:
                rows.append({"t1": a, "t2": b, "ok": False, "reason": "nonpositive_z"})
                continue
            ratio_cmd = z2 / z1
        elif mode == "real":
            if scale_s is None or not np.isfinite(scale_s) or scale_s <= 0:
                raise ValueError("real mode needs a positive sequence scale")
            h1_m = scale_s * h1
            if h1_m <= 0:
                rows.append({"t1": a, "t2": b, "ok": False, "reason": "nonpositive_metric_h"})
                continue
            ratio_cmd = 1.0 + dz_truth / h1_m
        else:
            raise ValueError(mode)
        if ratio_cmd <= 0:
            rows.append({"t1": a, "t2": b, "ok": False, "reason": "nonpositive_ratio_cmd"})
            continue
        e = float(np.log(ratio_dep) - np.log(ratio_cmd))
        # secondary, scale-free
        idx = w["idx"]
        cs = c[idx]
        dperp = cs - np.outer(cs @ n, n)
        L_dep = float(np.sum(np.linalg.norm(np.diff(dperp, axis=0), axis=1)))
        dz_dep = float(n @ (c[idx[-1]] - c[idx[0]]))
        rho_dep = dz_dep / max(L_dep, 1e-9)
        rho_cmd = dz_truth / L_truth if L_truth > 1e-9 else float("nan")
        q = ((rho_dep - rho_cmd) / (abs(rho_cmd) + RHO_FLOOR)
             if np.isfinite(rho_cmd) else float("nan"))
        rows.append({"t1": a, "t2": b, "ok": True, "h1": h1, "h2": h2,
                     "ratio_dep": ratio_dep, "ratio_cmd": ratio_cmd, "e": e,
                     "abs_e": abs(e), "dz_truth": dz_truth, "L_truth": L_truth,
                     "dz_dep": dz_dep, "L_dep": L_dep, "rho_dep": rho_dep,
                     "rho_cmd": rho_cmd, "q": q, "n_frames": int(len(idx))})
    return rows


def defect_reference(ratio_cmd: float, family: str) -> float | None:
    """log(ratio_defect) - log(ratio_cmd) for one window and one defect family."""
    if family == "removed":
        rd = 1.0
    elif family == "scaled_0.5":
        rd = 1.0 + 0.5 * (ratio_cmd - 1.0)
    elif family == "scaled_1.5":
        rd = 1.0 + 1.5 * (ratio_cmd - 1.0)
    elif family == "reversed":
        rd = 1.0 - (ratio_cmd - 1.0)
    else:
        raise ValueError(family)
    if rd <= 1e-6:
        return None
    return float(np.log(rd) - np.log(ratio_cmd))


def defect_table(rows: list[dict]) -> dict:
    """Per-clip |defect log change| for every family, median over the clip's windows."""
    out = {}
    ok = [r for r in rows if r.get("ok")]
    for fam in DEFECT_FAMILIES:
        vals = [v for v in (defect_reference(r["ratio_cmd"], fam) for r in ok) if v is not None]
        out[fam] = {"median_abs": float(np.median(np.abs(vals))) if vals else None,
                    "n_windows": len(vals),
                    "per_window": [float(v) for v in vals]}
    return out


def clip_summary(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return {"n_windows": 0, "n_ok": 0}
    abs_e = np.array([r["abs_e"] for r in ok])
    e = np.array([r["e"] for r in ok])
    q = np.array([r["q"] for r in ok if np.isfinite(r["q"])])
    return {"n_windows": len(rows), "n_ok": len(ok),
            "median_abs_e": float(np.median(abs_e)), "p90_abs_e": float(np.percentile(abs_e, 90)),
            "median_e": float(np.median(e)),
            "median_abs_q": float(np.median(np.abs(q))) if len(q) else None,
            "median_q": float(np.median(q)) if len(q) else None,
            "median_ratio_cmd": float(np.median([r["ratio_cmd"] for r in ok])),
            "median_dz_truth": float(np.median([r["dz_truth"] for r in ok])),
            "window_ok_frac": float(len(ok) / len(rows))}


def sequence_scale(cams: np.ndarray, pos_truth: np.ndarray) -> float:
    """One similarity scale: ratio of RMS radii about the centroid (no rotation fitted).

    The reconstruction's orientation is arbitrary, so a full similarity fit would have to fit a
    rotation as well; the ratio of RMS distances from the centroid is the scale of a
    rotation-free alignment and is what the pre-registration names.
    """
    c = np.asarray(cams, dtype=np.float64)
    p = np.asarray(pos_truth, dtype=np.float64)
    c0 = c - c.mean(axis=0)
    p0 = p - p.mean(axis=0)
    num = float(np.sqrt(np.mean(np.sum(p0 ** 2, axis=1))))
    den = float(np.sqrt(np.mean(np.sum(c0 ** 2, axis=1))))
    return num / den if den > 0 else float("nan")


def pooled(rows_by_unit: dict[str, list[dict]]) -> dict:
    """Pooled |e| over every unit's windows."""
    vals = [r["abs_e"] for rows in rows_by_unit.values() for r in rows if r.get("ok")]
    if not vals:
        return {"n": 0}
    a = np.array(vals)
    return {"n": int(len(a)), "median_abs_e": float(np.median(a)),
            "p90_abs_e": float(np.percentile(a, 90))}


def evaluate_window(t_frames: np.ndarray, cams: np.ndarray, plane_n: np.ndarray,
                    plane_d: float, t1: float, t2: float, truth_t: np.ndarray,
                    truth_pos: np.ndarray, *, mode: str,
                    scale_s: float | None = None) -> dict:
    """One window's row, using the window's own plane and the frames inside it."""
    t_frames = np.asarray(t_frames, dtype=np.float64)
    cams = np.asarray(cams, dtype=np.float64)
    n = np.asarray(plane_n, dtype=np.float64)
    h = cams @ n - float(plane_d)
    if len(t_frames) < 2 or t1 < t_frames[0] - 1e-9 or t2 > t_frames[-1] + 1e-9:
        return {"t1": t1, "t2": t2, "ok": False, "reason": "window_not_covered"}
    h1 = float(np.interp(t1, t_frames, h))
    h2 = float(np.interp(t2, t_frames, h))
    if h1 <= 0 or h2 <= 0:
        return {"t1": t1, "t2": t2, "ok": False, "reason": "nonpositive_height",
                "h1": h1, "h2": h2}
    z = np.asarray(truth_pos, dtype=np.float64)[:, 2]
    xy = np.asarray(truth_pos, dtype=np.float64)[:, :2]
    z1 = float(np.interp(t1, truth_t, z))
    z2 = float(np.interp(t2, truth_t, z))
    L_truth = float(np.hypot(np.interp(t2, truth_t, xy[:, 0]) - np.interp(t1, truth_t, xy[:, 0]),
                             np.interp(t2, truth_t, xy[:, 1]) - np.interp(t1, truth_t, xy[:, 1])))
    dz_truth = z2 - z1
    ratio_dep = h2 / h1
    if mode == "sim":
        if z1 <= 0:
            return {"t1": t1, "t2": t2, "ok": False, "reason": "nonpositive_z"}
        ratio_cmd = z2 / z1
    elif mode == "real":
        if scale_s is None or not np.isfinite(scale_s) or scale_s <= 0:
            raise ValueError("real mode needs a positive scale")
        h1_m = float(scale_s) * h1
        ratio_cmd = 1.0 + dz_truth / h1_m
    else:
        raise ValueError(mode)
    if ratio_cmd <= 0:
        return {"t1": t1, "t2": t2, "ok": False, "reason": "nonpositive_ratio_cmd"}
    e = float(np.log(ratio_dep) - np.log(ratio_cmd))
    dperp = cams - np.outer(cams @ n, n)
    L_dep = float(np.sum(np.linalg.norm(np.diff(dperp, axis=0), axis=1)))
    dz_dep = float(n @ (cams[-1] - cams[0]))
    rho_dep = dz_dep / max(L_dep, 1e-9)
    rho_cmd = dz_truth / L_truth if L_truth > 1e-9 else float("nan")
    q = ((rho_dep - rho_cmd) / (abs(rho_cmd) + RHO_FLOOR)
         if np.isfinite(rho_cmd) else float("nan"))
    return {"t1": t1, "t2": t2, "ok": True, "h1": h1, "h2": h2,
            "ratio_dep": ratio_dep, "ratio_cmd": ratio_cmd, "e": e, "abs_e": abs(e),
            "dz_truth": dz_truth, "L_truth": L_truth, "dz_dep": dz_dep, "L_dep": L_dep,
            "rho_dep": rho_dep, "rho_cmd": rho_cmd, "q": q,
            "n_frames": int(len(t_frames))}
