"""Body-rate consistency: the thrust axis must rotate at the rate the path demands.

Differential flatness fixes the thrust axis ``b3 = (a + g e3 + k_d v)/|.|``.  Its motion is
``d b3/dt = omega x b3`` (world frame), so the component of ``omega`` along ``b3`` -- the body
yaw rate -- drops out.  Both the required and the depicted rate are compared *without any start
attitude*:

* required: ``rb = d(b3_req)/dt`` computed from the commanded/true path alone;
* depicted: ``rd = R_ref (omega_body x e3)`` where ``omega_body = R_cb omega_cam`` comes from the
  video rotation and ``R_ref`` is *any* rotation with third column ``b3_req``.

``R_ref (omega_body x e3)`` equals ``omega_world x b3`` for the frame ``R_ref``; because
``omega_body x e3`` is perpendicular to ``e3`` the result does not depend on which ``R_ref`` is
chosen (the unknown yaw about ``b3`` cancels).  Substituting ``b3_req`` for the true depicted
thrust axis is the only approximation and it is quantified against ground truth in P0.

Rates are radians per second in the mathematics and degrees per second in the stored outputs.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from src.physics.flatness import G0

E3 = np.array([0.0, 0.0, 1.0])
K_D = 0.295            # the project's frozen world-frame drag, 1/s
FS_FILTER_HZ = 100.0   # uniform path grid
FILTER_WINDOW_S = 0.31  # Savitzky-Golay window in seconds (31 samples at 100 Hz)
FILTER_POLY = 3


# ----------------------------------------------------------------------------- path side
def uniform_path(t: np.ndarray, pos: np.ndarray, fs: float = FS_FILTER_HZ
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample a position track to ``fs`` and differentiate once with Savitzky-Golay.

    Returns ``(t_grid, vel, acc, interior)``.  ``interior`` is False within half a filter
    window of either end, where the polynomial fit is one-sided and its derivative is biased.
    """
    t = np.asarray(t, float)
    pos = np.asarray(pos, float)
    order = np.argsort(t)
    t, pos = t[order], pos[order]
    dt = 1.0 / float(fs)
    tg = np.arange(t[0], t[-1] + 1e-12, dt)
    pg = np.column_stack([np.interp(tg, t, pos[:, k]) for k in range(3)])
    w = int(round(FILTER_WINDOW_S * fs))
    if w % 2 == 0:
        w += 1
    w = min(w, len(tg) - 1 if (len(tg) - 1) % 2 else len(tg) - 2)
    if w < FILTER_POLY + 2:
        raise ValueError(f"track too short: {len(tg)} samples for window {w}")
    vel = np.column_stack([savgol_filter(pg[:, k], w, FILTER_POLY, deriv=1, delta=dt)
                           for k in range(3)])
    acc = np.column_stack([savgol_filter(pg[:, k], w, FILTER_POLY, deriv=2, delta=dt)
                           for k in range(3)])
    interior = np.zeros(len(tg), bool)
    interior[w // 2: len(tg) - w // 2] = True
    return tg, vel, acc, interior


def thrust_axis(acc: np.ndarray, vel: np.ndarray, k_d: float = K_D,
                g: float = G0) -> np.ndarray:
    """Unit thrust axis from acceleration and velocity; NaN where the norm is degenerate."""
    acc = np.asarray(acc, float).reshape(-1, 3)
    vel = np.asarray(vel, float).reshape(-1, 3)
    u = acc + np.array([0.0, 0.0, g]) + k_d * vel
    n = np.linalg.norm(u, axis=1, keepdims=True)
    out = np.full_like(u, np.nan)
    good = (n[:, 0] > 1e-6) & np.isfinite(n[:, 0])
    out[good] = u[good] / n[good]
    return out


def axis_rate(tg: np.ndarray, b3: np.ndarray,
              window_s: float = FILTER_WINDOW_S) -> np.ndarray:
    """``d b3/dt`` (rad/s) by Savitzky-Golay on the uniform grid; NaN at the edges."""
    b3 = np.asarray(b3, float)
    dt = float(np.median(np.diff(tg)))
    w = int(round(window_s / dt))
    if w % 2 == 0:
        w += 1
    w = min(w, len(b3) - 1 if (len(b3) - 1) % 2 else len(b3) - 2)
    if w < FILTER_POLY + 2:
        return np.full_like(b3, np.nan)
    r = np.column_stack([savgol_filter(b3[:, k], w, FILTER_POLY, deriv=1, delta=dt)
                         for k in range(3)])
    half = w // 2
    r[:half] = np.nan
    r[len(r) - half:] = np.nan
    return r


def filter_gain(window_s: float = FILTER_WINDOW_S, fs: float = FS_FILTER_HZ,
                poly: int = FILTER_POLY) -> dict:
    """-3 dB cutoff of the Savitzky-Golay first-derivative filter, measured numerically."""
    w = int(round(window_s * fs))
    if w % 2 == 0:
        w += 1
    dt = 1.0 / fs
    freqs = np.linspace(0.01, fs / 2.0, 4000)
    gains = []
    for f in freqs:
        x = np.sin(2 * np.pi * f * np.arange(w) * dt)
        d = savgol_filter(x, w, poly, deriv=1, delta=dt)
        ref = 2 * np.pi * f * np.cos(2 * np.pi * f * np.arange(w) * dt)
        mid = w // 2
        gains.append(abs(d[mid]) / max(abs(ref[mid]), 1e-12))
    gains = np.asarray(gains)
    below = np.flatnonzero(gains < 1.0 / np.sqrt(2.0))
    cut = float(freqs[below[0]]) if len(below) else float(freqs[-1])
    return {"window_s": window_s, "fs_hz": fs, "poly": poly, "cutoff_3db_hz": cut}


# ----------------------------------------------------------------------------- video side
def rotation_rates(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Angular velocity in the *camera* frame at every frame (rad/s).

    Central differences in SO(3): ``Log(R[k-1]^T R[k+1]) / (t[k+1] - t[k-1])``, one-sided at
    the two ends.  Rows whose neighbours are missing stay NaN.
    """
    R = np.asarray(R, float)
    t = np.asarray(t, float)
    n = len(R)
    out = np.full((n, 3), np.nan)
    if n < 2:
        return out
    lo = np.maximum(np.arange(n) - 1, 0)
    hi = np.minimum(np.arange(n) + 1, n - 1)
    M = np.einsum("nij,nik->njk", R[lo], R[hi])
    rv = Rotation.from_matrix(M).as_rotvec()
    dt = t[hi] - t[lo]
    ok = dt > 0
    out[ok] = rv[ok] / dt[ok, None]
    return out


def central_axis_rate(b3: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``d b3/dt`` by the same 3-point stencil as :func:`rotation_rates`."""
    b3 = np.asarray(b3, float)
    t = np.asarray(t, float)
    n = len(b3)
    out = np.full((n, 3), np.nan)
    if n < 2:
        return out
    lo = np.maximum(np.arange(n) - 1, 0)
    hi = np.minimum(np.arange(n) + 1, n - 1)
    dt = t[hi] - t[lo]
    ok = (dt > 0) & np.isfinite(b3[lo]).all(1) & np.isfinite(b3[hi]).all(1)
    out[ok] = (b3[hi][ok] - b3[lo][ok]) / dt[ok, None]
    return out


def kabsch_unit(a: np.ndarray, b: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Rotation ``G`` minimising ``sum_i w_i |G a_i - b_i|^2`` for unit vectors.

    This is the gauge fit: ``a`` is the thrust axis the video implies in its own (arbitrary,
    unknown) frame and ``b`` is the required thrust axis in the world frame.  A constant
    rotation is the *only* unknown -- no start attitude is supplied.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    good = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    a, b = a[good], b[good]
    if w is not None:
        w = np.asarray(w, float)[good]
    else:
        w = np.ones(len(a))
    if len(a) < 3:
        return np.full((3, 3), np.nan)
    H = (a * w[:, None]).T @ b
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def gauge_fit(R_est: np.ndarray, b3_req: np.ndarray, R_cb: np.ndarray) -> dict:
    """Fit the constant rotation that maps the video gauge to the world through the thrust axis.

    ``c_k = R_est[k] R_cb e3`` is the thrust axis in the video's own frame; the fit returns
    ``G`` with ``G c_k ~ b3_req[k]``, the rms fit angle, and the angular spread of the
    required axis (which decides whether the fit is determined at all).
    """
    R_est = np.asarray(R_est, float)
    b3_req = np.asarray(b3_req, float)
    R_cb = np.asarray(R_cb, float)
    c = np.einsum("nij,j->ni", R_est, np.asarray(R_cb, float).T @ E3)
    G = kabsch_unit(c, b3_req)
    good = np.isfinite(c).all(1) & np.isfinite(b3_req).all(1)
    fit_deg = float("nan")
    if np.isfinite(G).all() and good.sum():
        fit_deg = float(np.degrees(np.arccos(np.clip(
            np.sum((c[good] @ G.T) * b3_req[good], axis=1), -1, 1))).mean())
    bb = b3_req[good]
    spread = float("nan")
    if len(bb) >= 3:
        m = bb.mean(0)
        m /= max(np.linalg.norm(m), 1e-12)
        spread = float(np.degrees(np.arccos(np.clip(bb @ m, -1, 1)).max()))
    return {"G": G, "fit_deg": fit_deg, "spread_deg": spread, "n_fit": int(good.sum())}


def align_e3_to(b3: np.ndarray) -> np.ndarray:
    """Minimal rotations ``R_ref`` with ``R_ref e3 = b3`` (one per row)."""
    b3 = np.asarray(b3, float).reshape(-1, 3)
    out = np.empty((len(b3), 3, 3), float)
    for i, b in enumerate(b3):
        if not np.isfinite(b).all():
            out[i] = np.nan
            continue
        axis = np.cross(E3, b)
        s = np.linalg.norm(axis)
        c = float(np.clip(np.dot(E3, b), -1.0, 1.0))
        if s < 1e-9:
            out[i] = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            out[i] = Rotation.from_rotvec(axis / s * np.arctan2(s, c)).as_matrix()
    return out


def body_rate_from_camera(omega_cam: np.ndarray, R_cb: np.ndarray) -> np.ndarray:
    """Camera-frame angular velocity -> body frame."""
    return np.asarray(omega_cam, float) @ np.asarray(R_cb, float).T


def depicted_axis_rate(omega_cam: np.ndarray, R_est: np.ndarray, R_cb: np.ndarray,
                       G: np.ndarray) -> np.ndarray:
    """Depicted thrust-axis rate ``omega_world x b3`` from video rotation and gauge ``G``.

    ``R_hat_wb[k] = G R_est[k] R_cb`` is the body-to-world attitude the video implies; the
    unknown body yaw about ``b3`` never enters because ``omega_body x e3`` has no component
    along ``e3``.
    """
    w_body = body_rate_from_camera(omega_cam, R_cb)
    w = np.cross(w_body, E3)
    R_hat = np.einsum("ij,njk,kl->nil", np.asarray(G, float), np.asarray(R_est, float),
                      np.asarray(R_cb, float).T)
    return np.einsum("nij,nj->ni", R_hat, w)


def depicted_axis_rate_literal(omega_cam: np.ndarray, b3_req: np.ndarray,
                               R_cb: np.ndarray) -> np.ndarray:
    """The handoff-literal form ``(R_cb omega_cam) x b3_req`` with an arbitrary minimal frame.

    Kept only to quantify the error the arbitrary yaw about ``b3`` introduces (P0).
    """
    w_body = body_rate_from_camera(omega_cam, R_cb)
    w = np.cross(w_body, E3)
    R_ref = align_e3_to(b3_req)
    return np.einsum("nij,nj->ni", R_ref, w)


def world_axis_rate(omega_world: np.ndarray, b3: np.ndarray) -> np.ndarray:
    """Exact ``omega x b3`` in the world frame (for ground-truth checks)."""
    return np.cross(np.asarray(omega_world, float), np.asarray(b3, float))


def world_omega(R_bw: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World-frame angular velocity from body-to-world rotations (central differences)."""
    R = np.asarray(R_bw, float)
    w_cam = rotation_rates(R, t)                      # this is body-frame here
    return np.einsum("nij,nj->ni", R, w_cam)


# ----------------------------------------------------------------------------- statistics
def residuals(rb: np.ndarray, rd: np.ndarray) -> dict:
    """Per-frame ``|rb - rd|`` (deg/s), the ratio to ``|rb|``, and the cosine."""
    rb = np.asarray(rb, float)
    rd = np.asarray(rd, float)
    good = np.isfinite(rb).all(1) & np.isfinite(rd).all(1)
    e = np.full(len(rb), np.nan)
    cos = np.full(len(rb), np.nan)
    e[good] = np.degrees(np.linalg.norm(rb[good] - rd[good], axis=1))
    a = np.linalg.norm(rb[good], axis=1)
    b = np.linalg.norm(rd[good], axis=1)
    d = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    idx = np.flatnonzero(good)[d]
    if len(idx):
        cos[idx] = np.sum(rb[idx] * rd[idx], axis=1) / (a[d] * b[d])
    speed = np.full(len(rb), np.nan)
    speed[good] = np.degrees(np.linalg.norm(rb[good], axis=1))
    return {"err_deg_s": e, "cos": cos, "req_deg_s": speed, "ok": good}


def windows(n: int, seconds: float, fps: float, hop_fraction: float = 0.5) -> list[tuple[int, int]]:
    """Frame windows ``[lo, hi)`` of the given duration; hop is half a window by default."""
    k = max(1, int(round(seconds * fps)))
    hop = max(1, int(round(k * hop_fraction)))
    if n < k:
        return []
    return [(lo, lo + k) for lo in range(0, n - k + 1, hop)]


def window_scores(res: dict, seconds: float, hop_fraction: float = 0.5,
                  min_frac: float = 0.6, min_req_deg_s: float = 0.0) -> list[dict]:
    """One row per window: absolute, relative and cosine statistics plus its testability."""
    n = len(res["err_deg_s"])
    fps = 1.0 / np.median(np.diff(res["t"])) if "t" in res else 0.0
    rows = []
    for lo, hi in windows(n, seconds, fps, hop_fraction):
        e = res["err_deg_s"][lo:hi]
        q = res["req_deg_s"][lo:hi]
        c = res["cos"][lo:hi]
        good = np.isfinite(e) & np.isfinite(q)
        frac = float(good.mean())
        testable = frac >= min_frac
        med_req = float(np.median(q[good])) if good.sum() else float("nan")
        if testable and not (med_req >= min_req_deg_s):
            testable = False
        cm = c[np.isfinite(c) & (np.abs(q) > max(0.5 * min_req_deg_s, 1e-9))]
        rows.append({
            "lo": lo, "hi": hi,
            "abs_deg_s": float(np.median(e[good])) if good.sum() else float("nan"),
            "rel": (float(np.median(e[good]) / med_req)
                    if good.sum() and med_req > 0 else float("nan")),
            "cos": float(np.median(cm)) if len(cm) else float("nan"),
            "req_deg_s": med_req,
            "frac_ok": frac, "testable": bool(testable),
        })
    return rows


# ----------------------------------------------------------------------------- noise model
def step_errors(R_est: np.ndarray, R_true: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-step rotation error (deg) between two pose sequences, gauge free."""
    R_est = np.asarray(R_est, float)
    R_true = np.asarray(R_true, float)
    n = min(len(R_est), len(R_true))
    out = []
    for k in range(n - 1):
        de = R_est[k].T @ R_est[k + 1]
        dt_ = R_true[k].T @ R_true[k + 1]
        out.append(np.degrees(np.linalg.norm(
            Rotation.from_matrix(de.T @ dt_).as_rotvec())))
    return np.asarray(out)


def noisy_sequence(R: np.ndarray, sigma_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Add iid Gaussian rotation noise (per axis, camera frame) to a pose sequence."""
    R = np.asarray(R, float)
    s = np.deg2rad(sigma_deg)
    n = len(R)
    eta = Rotation.from_rotvec(rng.normal(0.0, s, (n - 1, 3))).as_matrix()
    steps = np.einsum("nij,njk->nik", np.einsum("nji,njk->nik", R[:-1], R[1:]), eta)
    out = np.empty_like(R)
    out[0] = R[0]
    cur = R[0]
    for k in range(n - 1):
        cur = cur @ steps[k]
        out[k + 1] = cur
    return out


def maxwell_median_deg(sigma_deg: float) -> float:
    """Median angle of a 3-D isotropic Gaussian rotation vector -- calibration constant."""
    rng = np.random.default_rng(12345)
    v = rng.normal(0.0, 1.0, size=(400000, 3))
    return float(np.median(np.linalg.norm(v, axis=1))) * sigma_deg


def match_sigma(step_err_deg: np.ndarray) -> float:
    """Gaussian per-axis sigma (deg) whose median step angle matches the measured one."""
    m = float(np.median(np.asarray(step_err_deg, float)))
    return m / maxwell_median_deg(1.0)
