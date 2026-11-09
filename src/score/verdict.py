"""The verdict: residual, per-clip nuisance fit, abstention, conformal thresholds, metrics.

Definitions follow ``configs/verdict_preregistration.yaml`` (do not change them here):
the **depicted** thrust axis is built from the pixel gravity estimate and the path's yaw,
the **implied** axis from the path's position, and the residual between them is scored by
its median over the frames that carry signal. Heading comes from the path because gravity
cannot observe it; the path is known in commanded-path mode, so this is not a fitted
nuisance but a stated restriction of the claim.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

E3 = np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------------ geometry
def rotation_between_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotations carrying each unit vector ``a[i]`` onto ``b[i]`` (N, 3, 3)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    v = np.cross(a, b)
    c = np.einsum("ij,ij->i", a, b)
    s = np.linalg.norm(v, axis=1)
    R = np.tile(np.eye(3), (len(a), 1, 1))
    ok = s > 1e-12
    K = np.zeros((len(a), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -v[:, 2], v[:, 1]
    K[:, 1, 0], K[:, 1, 2] = v[:, 2], -v[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -v[:, 1], v[:, 0]
    factor = np.where(ok, (1.0 - c) / np.where(ok, s * s, 1.0), 0.0)
    R[ok] = (np.eye(3)[None] + K[ok] + np.einsum("nij,njk->nik", K[ok], K[ok])
             * factor[ok][:, None, None])
    # anti-parallel: any half turn about an axis perpendicular to ``a``
    anti = (~ok) & (c < 0)
    if anti.any():
        axis = np.cross(a[anti], np.array([1.0, 0.0, 0.0]))
        bad = np.linalg.norm(axis, axis=1) < 1e-9
        axis[bad] = np.cross(a[anti][bad], np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        R[anti] = Rotation.from_rotvec(np.pi * axis).as_matrix()
    return R


def axis_angle_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle between unit vectors, robust near 0 and 180 deg."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    dot = np.clip(np.einsum("...i,...i->...", u, v), -1.0, 1.0)
    cross = np.linalg.norm(np.cross(u, v), axis=-1)
    return np.degrees(np.arctan2(cross, dot))


def tilt_deg_of(axis: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.asarray(axis)[..., 2], -1.0, 1.0)))


def depicted_axis_world(up_cam: np.ndarray, R_cb: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Body z axis in the world frame from the measured gravity and the path's yaw.

    ``up_cam`` is world up expressed in camera coordinates (GeoCalib's convention, the
    same one :mod:`src.perception.frames` uses) and ``R_cb`` maps camera->body, so
    ``u_b = R_cb @ up_cam`` is world up in body coordinates. For a ZYX attitude
    ``R_bw = Rz(yaw) Ry(pitch) Rx(roll)`` the body-frame up is

        u_b = (-sin p, sin r cos p, cos r cos p),

    so gravity fixes pitch and roll and the path's yaw closes the rotation. Then
    ``z_B = R_bw @ e3``. A level airframe gives ``u_b = e3`` and ``z_B = e3``.
    """
    up_cam = np.asarray(up_cam, dtype=np.float64).reshape(-1, 3)
    n = np.linalg.norm(up_cam, axis=1, keepdims=True)
    up_cam = up_cam / np.where(n > 0, n, 1.0)
    R_cb = np.broadcast_to(np.asarray(R_cb, dtype=np.float64), (len(up_cam), 3, 3))
    yaw = np.broadcast_to(np.asarray(yaw, dtype=np.float64).ravel(), (len(up_cam),))
    u_b = np.einsum("nij,nj->ni", R_cb, up_cam)
    pitch = np.arcsin(np.clip(-u_b[:, 0], -1.0, 1.0))
    roll = np.arctan2(u_b[:, 1], u_b[:, 2])
    N = len(yaw)
    rv = np.zeros((N, 3))
    rv[:, 2] = yaw
    Rz = Rotation.from_rotvec(rv).as_matrix()
    rv = np.zeros((N, 3))
    rv[:, 1] = pitch
    Ry = Rotation.from_rotvec(rv).as_matrix()
    rv = np.zeros((N, 3))
    rv[:, 0] = roll
    Rx = Rotation.from_rotvec(rv).as_matrix()
    R_bw = np.einsum("nij,njk,nkl->nil", Rz, Ry, Rx)
    return R_bw[:, :, 2]


# ------------------------------------------------------------------ residual
def residual_axis(dep_axis: np.ndarray, imp_axis: np.ndarray) -> np.ndarray:
    return axis_angle_deg(dep_axis, imp_axis)


def shift_series(values: np.ndarray, t: np.ndarray, lag_s: float) -> np.ndarray:
    """``values`` sampled at ``t + lag_s`` (linear, edge-clamped)."""
    if lag_s == 0.0:
        return np.asarray(values)
    values = np.asarray(values, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if values.ndim == 1:
        return np.interp(np.clip(t + lag_s, t[0], t[-1]), t, values)
    return np.column_stack([np.interp(np.clip(t + lag_s, t[0], t[-1]), t, values[:, k])
                            for k in range(values.shape[1])])


def clip_index(clip: dict) -> np.ndarray:
    """Frame indices of a clip: stored explicitly, never recomputed from time."""
    lo, hi = clip["index"]
    return np.arange(int(lo), int(hi))


# ------------------------------------------------------------------ nuisance fit
def fit_nuisances(up_cam: np.ndarray, imp_axis: np.ndarray, yaw: np.ndarray, t: np.ndarray,
                  uptilt_deg: float, mask: np.ndarray, prereg: dict,
                  lag_override: float | None = None,
                  uptilt_offset_override: float | None = None) -> dict:
    """Grid-search the two bounded nuisances (mount uptilt, time lag) on one clip.

    ``mask`` selects the testable frames (implied tilt above the signal floor). The loss
    is the median residual, the same statistic as the score, so the fit cannot buy a low
    tail by moving the body of the distribution.
    """
    from src.perception.frames import camera_to_body_pitched

    nu = prereg["nuisances"]
    u_lo, u_hi = nu["uptilt_offset_deg"]["bounds"]
    u_step = nu["uptilt_offset_deg"].get("step", 1.0)
    l_lo, l_hi = nu["lag_ms"]["bounds"]
    l_step = nu["lag_ms"].get("step", 25.0)
    offsets = [uptilt_offset_override] if uptilt_offset_override is not None else \
        np.arange(u_lo, u_hi + 1e-9, u_step)
    lags = [lag_override] if lag_override is not None else \
        np.arange(l_lo, l_hi + 1e-9, l_step) / 1000.0

    if not mask.any():
        return {"ok": False, "reason": "no testable frames"}
    best = {"loss": np.inf}
    for doff in offsets:
        R_cb = camera_to_body_pitched(np.deg2rad(uptilt_deg + float(doff)))
        dep = depicted_axis_world(up_cam, R_cb, yaw)
        for lag in lags:
            imp = shift_series(imp_axis, t, float(lag))
            r = residual_axis(dep, imp)
            loss = float(np.nanmedian(r[mask]))
            if loss < best["loss"]:
                best = {"loss": loss, "uptilt_offset_deg": float(doff),
                        "lag_ms": float(lag * 1000.0)}
    return {"ok": True, **best}


# ------------------------------------------------------------------ clip score
def score_clip(up_cam: np.ndarray, imp_axis: np.ndarray, yaw: np.ndarray, t: np.ndarray,
               uptilt_deg: float, implied_tilt_deg: np.ndarray, valid: np.ndarray,
               prereg: dict, nuisances: dict | None = None) -> dict:
    """Residual score for one clip under one condition, with the pre-registered rule."""
    up_cam = np.asarray(up_cam, dtype=np.float64)
    finite = np.isfinite(up_cam).all(axis=1) & np.isfinite(imp_axis).all(axis=1)
    signal = finite & (np.asarray(implied_tilt_deg) >= 10.0)
    valid = np.asarray(valid, dtype=bool) if valid is not None else np.ones(len(t), bool)
    from src.perception.frames import camera_to_body_pitched

    fit = nuisances or fit_nuisances(up_cam, imp_axis, yaw, t, uptilt_deg, signal, prereg)
    ab = prereg["abstention"]
    n_signal = int(signal.sum())
    frac_signal = n_signal / max(len(t), 1)
    frac_valid = int((finite & valid).sum()) / max(len(t), 1)
    reasons = []
    if n_signal < 12:
        reasons.append(f"signal_frames {n_signal} < 12")
    if frac_signal < 0.20:
        reasons.append(f"testable_fraction {frac_signal:.2f} < 0.20")
    if frac_valid < 0.40:
        reasons.append(f"usable_depicted {frac_valid:.2f} < 0.40")
    out = {"n_frames": int(len(t)), "n_signal": n_signal, "frac_signal": frac_signal,
           "frac_usable": frac_valid, "testable": not reasons, "abstain_reasons": reasons,
           "nuisances": fit}
    if not fit.get("ok"):
        out["testable"] = False
        out["abstain_reasons"] = out["abstain_reasons"] + ["nuisance fit failed"]
        return out
    R_cb = camera_to_body_pitched(np.deg2rad(uptilt_deg + fit["uptilt_offset_deg"]))
    dep = depicted_axis_world(up_cam, R_cb, yaw)
    imp = shift_series(imp_axis, t, fit["lag_ms"] / 1000.0)
    r_axis = residual_axis(dep, imp)
    r_tilt = np.abs(tilt_deg_of(dep) - tilt_deg_of(imp))
    use = signal & valid & np.isfinite(r_axis)
    if use.sum() == 0:
        out["testable"] = False
        out["abstain_reasons"] = out["abstain_reasons"] + ["no usable testable frames"]
        return out
    out.update({
        "score": float(np.median(r_axis[use])),
        "score_p75": float(np.percentile(r_axis[use], 75)),
        "score_median_tilt": float(np.median(r_tilt[use])),
        "n_scored": int(use.sum()),
        "tilt_dep_median": float(np.median(tilt_deg_of(dep)[use])),
        "tilt_imp_median": float(np.median(tilt_deg_of(imp)[use])),
        "nuisance_loss": fit["loss"],
    })
    return out


# ------------------------------------------------------------------ thresholds
def conformal_quantile(scores: np.ndarray, alpha: float = 0.05) -> float:
    """``ceil((n + 1)(1 - alpha))``-th smallest score, or +inf when n is too small."""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(s)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if n == 0:
        return float("nan")
    if k > n:
        return float("inf")
    return float(s[k - 1])


def speed_bin(speed: float, bins: list[list[float]]) -> str:
    if speed < bins[0][0]:          # clamped into the first bin, never silently the last
        return f"{bins[0][0]:g}-{bins[0][1]:g}"
    for lo, hi in bins:
        if lo <= speed < hi:
            return f"{lo:g}-{hi:g}"
    return f"{bins[-1][0]:g}-{bins[-1][1]:g}"


# ------------------------------------------------------------------ metrics
def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Rank AUROC; higher score = more likely to be rejected (positive = real)."""
    p = np.asarray(positive, dtype=np.float64)
    q = np.asarray(negative, dtype=np.float64)
    if len(p) == 0 or len(q) == 0:
        return float("nan")
    allv = np.concatenate([p, q])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    vals, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(vals))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    rp = ranks[: len(p)].sum()
    return float((rp - len(p) * (len(p) + 1) / 2.0) / (len(p) * len(q)))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return float("nan")

    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a))
        r[order] = np.arange(len(a))
        # average ties
        vals, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        if (counts > 1).any():
            sums = np.zeros(len(vals))
            np.add.at(sums, inv, r)
            r = (sums / counts)[inv]
        return r

    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den else float("nan")
