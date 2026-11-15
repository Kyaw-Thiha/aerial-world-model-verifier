"""Label-free confidence gate for the depicted-tilt estimator (pre-registration amendment 2).

Why this exists. Gate 3's operating point was set after excluding three flights on the
strength of their real-clip scores. That is selection by outcome: it inflates accepted
numbers and it cannot be repeated on generated video, where no mocap exists. Amendment 2
replaces it with a gate that decides *from the clip's own pixels and the estimator's own
outputs* whether the depicted tilt is trustworthy. A clip the gate drops is "not testable",
exactly like an excitation abstention. No flight is ever excluded by identity.

Everything here is pure numpy on arrays, so the gate can be tested without video and
applied unchanged to TII frames and to generated clips.

Label. The quantity the gate must predict is the depicted-gravity error against mocap,
``median_t angle(up_est(t), up_truth(t))``. The label is ``error > 10 deg``. The label is
computed here and is never an input to a signal.
"""
from __future__ import annotations

import numpy as np

from src.perception.frames import body_to_world, camera_to_body_pitched, up_in_camera

SIGNALS = ("S1_gravity_uncertainty", "S2_up_confidence", "S3_latitude_confidence",
           "S4_temporal_jitter", "S5_vo_agreement", "S6_uncertainty_instability")
SINGLE_FRAME = {"S1_gravity_uncertainty": True, "S2_up_confidence": True,
                "S3_latitude_confidence": True, "S4_temporal_jitter": False,
                "S5_vo_agreement": False, "S6_uncertainty_instability": True}
JITTER_WINDOW_S = 0.5
MIN_KEEP = 0.60


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise angle between two direction arrays, NaN-propagating."""
    a = _unit(np.asarray(a, dtype=np.float64))
    b = _unit(np.asarray(b, dtype=np.float64))
    c = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(c))


# ------------------------------------------------------------------ label (mocap side)
def up_truth_in_camera(quat: np.ndarray, uptilt_deg: float,
                       quat_order: str = "xyzw") -> np.ndarray:
    """World up expressed in camera coordinates, from mocap attitude and the recorded mount.

    ``quat_order`` is explicit because TII ships ``xyzw`` and a silent swap would look like
    a 180 deg error.
    """
    R_bw = body_to_world(quat, quat_order)
    R_cb = camera_to_body_pitched(np.deg2rad(uptilt_deg))
    return up_in_camera(R_bw, R_cb)


def clip_label(up_est: np.ndarray, up_truth: np.ndarray, threshold_deg: float = 10.0) -> dict:
    """Per-frame depicted-gravity error and the clip label (1 = estimator is wrong)."""
    err = angle_deg(up_est, up_truth)
    ok = np.isfinite(err)
    if not ok.any():
        return {"median_error_deg": float("nan"), "label": None, "n_frames": 0}
    med = float(np.median(err[ok]))
    return {"median_error_deg": med, "label": int(med > threshold_deg),
            "n_frames": int(ok.sum())}


# ------------------------------------------------------------------ signals (pixel side)
def slow_component(up: np.ndarray, window_samples: int) -> np.ndarray:
    """Running component-wise median of a unit-vector series, renormalised."""
    up = np.asarray(up, dtype=np.float64)
    n = len(up)
    w = max(1, int(window_samples))
    half = w // 2
    out = np.full_like(up, np.nan)
    for t in range(n):
        lo, hi = max(0, t - half), min(n, t + half + 1)
        seg = up[lo:hi]
        if not np.isfinite(seg).all():
            continue
        out[t] = np.median(seg, axis=0)
    return _unit(out)


def s_temporal_jitter(up: np.ndarray, dt_s: float, window_s: float = JITTER_WINDOW_S) -> float:
    """Robust spread of the gravity estimate after removing its slow component.

    ``dt_s`` is the spacing between consecutive kept frames. The window is defined in
    seconds, so the same statistic means the same thing at any sampling rate — which is
    also why it cannot be transferred to a 1 s step without becoming meaningless.
    """
    up = np.asarray(up, dtype=np.float64)
    ok = np.isfinite(up).all(axis=1)
    if ok.sum() < 3:
        return float("nan")
    w = max(1, int(round(window_s / dt_s)))
    if w < 3:
        # A window of one or two samples leaves nothing to smooth against, so the
        # statistic is not defined. Returning 0 here would read as "perfectly stable",
        # which is the opposite of what a coarse time base implies.
        return float("nan")
    slow = slow_component(np.where(ok[:, None], up, np.nan), w)
    d = angle_deg(up, slow)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else float("nan")


def s_vo_agreement(up: np.ndarray, R_rel: np.ndarray) -> float:
    """Angle between the measured gravity at t+1 and the frame-t gravity carried by VO.

    ``R_rel[t]`` carries camera frame t to t+1 (camera->camera). A pair without a VO row is
    skipped; a clip with no pairs is "not testable" for this signal (NaN).
    """
    up = np.asarray(up, dtype=np.float64)
    R_rel = np.asarray(R_rel, dtype=np.float64)
    if len(up) < 2 or len(R_rel) != len(up) - 1:
        return float("nan")
    pred = np.einsum("nij,nj->ni", R_rel, up[:-1])
    d = angle_deg(pred, up[1:])
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else float("nan")


def s_uncertainty_instability(unc: np.ndarray) -> float:
    """Fraction of frames whose uncertainty exceeds twice the clip's own median."""
    u = np.asarray(unc, dtype=np.float64)
    u = u[np.isfinite(u)]
    if u.size == 0:
        return float("nan")
    med = float(np.median(u))
    if med <= 0:
        return float("nan")
    return float(np.mean(u > 2.0 * med))


def clip_signal(name: str, rows: dict[str, np.ndarray], dt_s: float | None = None,
                R_rel: np.ndarray | None = None) -> float:
    """One clip-level signal value from per-frame arrays (larger = less trustworthy)."""

    def med(key: str) -> float:
        if key not in rows:
            return float("nan")
        v = np.asarray(rows[key], dtype=np.float64)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else float("nan")

    if name == "S1_gravity_uncertainty":
        return med("gravity_uncertainty_raw")
    if name == "S2_up_confidence":
        return -med("up_confidence_mean")
    if name == "S3_latitude_confidence":
        return -med("latitude_confidence_mean")
    if name == "S4_temporal_jitter":
        if dt_s is None:
            raise ValueError("S4 needs dt_s (the frame spacing in seconds)")
        return s_temporal_jitter(rows["up"], dt_s)
    if name == "S5_vo_agreement":
        if R_rel is None:
            raise ValueError("S5 needs the VO relative rotations")
        return s_vo_agreement(rows["up"], R_rel)
    if name == "S6_uncertainty_instability":
        return s_uncertainty_instability(rows.get("gravity_uncertainty_raw", np.array([])))
    raise ValueError(f"unknown gate signal {name!r}; known: {SIGNALS}")


# ------------------------------------------------------------------ threshold rule
def pick_tau(values: np.ndarray, labels: np.ndarray, min_keep: float = MIN_KEEP) -> float:
    """tau maximising (TPR - FPR) with the fit clips' kept fraction at least ``min_keep``.

    Larger signal = less trustworthy, so a clip is dropped when ``value > tau``. Ties go to
    the larger tau (keep more).
    """
    v = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    ok = np.isfinite(v) & np.isfinite(y)
    v, y = v[ok], y[ok]
    if v.size == 0 or y.sum() == 0 or (len(y) - y.sum()) == 0:
        return float("nan")
    cands = np.unique(np.concatenate([v, [np.inf]]))
    best, best_tau = -np.inf, float("nan")
    for tau in cands:
        drop = v > tau
        if (1.0 - drop.mean()) < min_keep:
            continue
        tpr = (drop & (y == 1)).sum() / max(1, (y == 1).sum())
        fpr = (drop & (y == 0)).sum() / max(1, (y == 0).sum())
        if (tpr - fpr) > best:
            best, best_tau = tpr - fpr, float(tau)
    return best_tau


def kept(value: float, tau: float) -> bool:
    """False when the gate says the clip's depicted tilt cannot be trusted."""
    if not np.isfinite(value) or not np.isfinite(tau):
        return False
    return not bool(value > tau)


def auroc_pred(values: np.ndarray, labels: np.ndarray) -> float:
    """P(signal of a label-1 clip > signal of a label-0 clip), ties counted as half."""
    v = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    ok = np.isfinite(v) & np.isfinite(y)
    v, y = v[ok], y[ok]
    pos, neg = v[y == 1], v[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def bootstrap_auroc_ci(values: np.ndarray, labels: np.ndarray, flights: np.ndarray,
                       n_boot: int = 2000, seed: int = 0) -> dict:
    """Flight-level bootstrap: resample whole flights, so correlated clips are not split."""
    v = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    f = np.asarray(flights)
    uniq = np.unique(f)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([np.flatnonzero(f == p) for p in pick])
        a = auroc_pred(v[idx], y[idx])
        if np.isfinite(a):
            out.append(a)
    if not out:
        return {"auroc": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_boot": 0, "frac_usable": 0.0}
    return {"auroc": auroc_pred(v, y), "lo": float(np.percentile(out, 2.5)),
            "hi": float(np.percentile(out, 97.5)), "n_boot": len(out),
            # a draw is usable only when it contains a label-1 clip. If positives live in
            # one flight, most draws are unusable and the interval is narrower than the
            # evidence; the fraction is reported so that can be said out loud.
            "frac_usable": float(len(out) / n_boot)}


def lofo_tau(values: np.ndarray, labels: np.ndarray, flights: np.ndarray,
             min_keep: float = MIN_KEEP) -> dict:
    """Refit tau on the other seven flights for each held-out flight."""
    v = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    f = np.asarray(flights)
    out = {}
    for u in np.unique(f):
        tr = f != u
        tau = pick_tau(v[tr], y[tr], min_keep)
        out[str(u)] = {"tau": tau, "n_fit": int(tr.sum()),
                       "n_fit_positive": int(np.nansum(y[tr]))}
    return out
