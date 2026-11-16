"""Pixel-path verdict statistics that avoid the second derivative of the observed path.

Track F, stage 0. The question is C09's: a pixel-path verdict failed because the observed
path was differentiated twice (noise grows as frequency squared) and needed metric scale.
The variants here never differentiate the pixel path twice; three of them never
differentiate it at all.

Everything is expressed in a horizontal plane perpendicular to an *up* direction supplied by
the attitude source (truth quaternion, or the depicted lean through the published mount) and
a per-clip yaw offset that is fitted as a nuisance, because yaw is unconstrained in this
project. Positions, velocities and accelerations arrive with an unknown positive scale; each
fit carries that scale explicitly so the statistic is a ratio.

Definitions (``configs/f01_pixel_path_prereg.yaml``):

* ``shooting`` — integrate ``a_h = g Rz(psi + yaw) lean - k_d v_h`` twice from a free initial
  state and least-squares the observed path *undifferentiated*; statistic ``rho`` is the
  unexplained fraction of the path.
* ``preintegration`` — one Savitzky-Golay derivative of the path; integrate the same ODE once
  and fit scale and initial velocity; statistic the unexplained fraction of velocity.
* ``band_coherence`` — band-limited complex coherence between the depicted lean and the
  path's horizontal acceleration; invariant to a constant rotation, so yaw-free.
* ``velocity_drag`` — near-steady flight, ``lean ~ (k_d/g) v``; one derivative.
* ``physics_scale`` — the scale fitted by the shooting model, checked for identifiability and
  against a metric anchor.

No function here reads the truth path. ``src.pathrec`` scoring helpers are used only by the
runner, for provenance and the oracle comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter

G0 = 9.80665
KD_DEFAULT = 0.295
PSI_STEP_DEG = 1.0


# --------------------------------------------------------------------------- small helpers
def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.full_like(v, np.nan), where=n > 1e-12)


def horizontal_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane perpendicular to ``up``.

    The choice is arbitrary (a yaw offset is fitted anyway), but it must be deterministic:
    pick the world axis least aligned with ``up`` and orthogonalise the other two.
    """
    up = unit(np.asarray(up, dtype=np.float64).reshape(3))
    axis = np.abs(up)
    e = np.zeros(3)
    e[int(np.argmin(axis))] = 1.0
    e1 = unit(e - np.dot(e, up) * up)
    e2 = np.cross(up, e1)
    return e1, unit(e2)


def rot2(angle) -> np.ndarray:
    a = np.asarray(angle, dtype=np.float64)
    c, s = np.cos(a), np.sin(a)
    out = np.empty(a.shape + (2, 2))
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    return out


def apply_rot2(angle, vec: np.ndarray) -> np.ndarray:
    a = np.asarray(angle, dtype=np.float64)
    c, s = np.cos(a), np.sin(a)
    x, y = vec[..., 0], vec[..., 1]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1)


def odd_window(n: int, dt: float, window_s: float, polyorder: int = 3) -> int:
    """Odd Savitzky-Golay window, at least polyorder + 2 samples, at most the series length."""
    if window_s <= 0:
        raise ValueError("window must be positive")
    w = int(round(window_s / dt))
    if w % 2 == 0:
        w += 1
    w = max(w, polyorder + 2 if polyorder % 2 == 0 else polyorder + 2)
    if w % 2 == 0:
        w += 1
    top = n if n % 2 == 1 else n - 1
    if w > top:
        w = top
    return max(w, 3 if polyorder == 1 else 5)


def longest_finite_run(mask: np.ndarray) -> slice:
    """Longest contiguous run of True in a boolean mask (used to cut at NaN gaps)."""
    mask = np.asarray(mask, dtype=bool)
    best_start, best_len, start = 0, 0, None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start > best_len:
                best_start, best_len = start, i - start
            start = None
    if start is not None and len(mask) - start > best_len:
        best_start, best_len = start, len(mask)
    return slice(best_start, best_start + best_len)


def smooth_derivative(values: np.ndarray, dt: float, window: int, deriv: int,
                      polyorder: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window < polyorder + 2:
        raise ValueError(f"window {window} too short for polyorder {polyorder}")
    return savgol_filter(values, window, polyorder, deriv=deriv, delta=dt, axis=0,
                         mode="interp")


# --------------------------------------------------------------------------- linear response
@dataclass
class _Response:
    """Homogeneous and forced response of the drag ODE on a uniform grid."""

    H_v: np.ndarray   # (N, 2, 2)  velocity response to v0
    H_x: np.ndarray   # (N, 2, 2)  position response to v0
    F_v: np.ndarray   # (N, 2)     forced velocity
    F_x: np.ndarray   # (N, 2)     forced position


def drag_response(forcing: np.ndarray, dt: float, k_d: float) -> _Response:
    """Integrate ``v' = -k_d v + f`` and ``x' = v`` from zero state, first-order hold on ``f``.

    ``forcing`` is (N, 2); the value at sample n drives the step to n+1 (zero-order hold),
    which is exact for piecewise-constant forcing and second-order accurate otherwise. The
    response to the initial state is carried as matrices, so a least-squares fit never has
    to re-integrate.
    """
    f = np.asarray(forcing, dtype=np.float64)
    n = len(f)
    if dt <= 0:
        raise ValueError("dt must be positive")
    if k_d < 0:
        raise ValueError("k_d must be non-negative")
    if k_d > 0:
        decay = np.exp(-k_d * dt)
        a = (1.0 - decay) / k_d
        c = dt / k_d - a / k_d
    else:
        decay, a, c = 1.0, dt, 0.5 * dt * dt
    H_v = np.empty((n, 2, 2))
    H_x = np.empty((n, 2, 2))
    F_v = np.empty((n, 2))
    F_x = np.empty((n, 2))
    H_v[0] = np.eye(2)
    H_x[0] = 0.0
    F_v[0] = 0.0
    F_x[0] = 0.0
    for i in range(n - 1):
        H_v[i + 1] = decay * H_v[i]
        H_x[i + 1] = H_x[i] + a * H_v[i]
        F_v[i + 1] = decay * F_v[i] + a * f[i]
        F_x[i + 1] = F_x[i] + a * F_v[i] + c * f[i]
    return _Response(H_v=H_v, H_x=H_x, F_v=F_v, F_x=F_x)


# --------------------------------------------------------------------------- fits
@dataclass
class FitResult:
    statistic: float
    psi_deg: float
    scale: float
    scale_rel_se: float
    sse: float
    sst: float
    r2: float
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(np.isfinite(self.statistic))


def _fit_linear(A: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    sse = float(resid @ resid)
    dof = max(1, A.shape[0] - A.shape[1])
    sigma2 = sse / dof
    try:
        cov = sigma2 * np.linalg.pinv(A.T @ A)
    except np.linalg.LinAlgError:      # pragma: no cover - pinv does not raise
        cov = np.full((A.shape[1], A.shape[1]), np.nan)
    return beta, sse, cov


def _psi_grid() -> np.ndarray:
    return np.deg2rad(np.arange(-180.0, 180.0, PSI_STEP_DEG))


def _project_out(matrix: np.ndarray, vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal basis of ``matrix`` (2-D columns) and ``vec`` projected off its span."""
    q, _ = np.linalg.qr(matrix)
    return q, vec - q @ (q.T @ vec)


def _scaled_fit(model_free: np.ndarray, observed: np.ndarray, offset: np.ndarray
                ) -> tuple[float, float, float, np.ndarray] | None:
    """Fit ``s * observed = offset + B beta`` with ``s > 0`` eliminated in closed form.

    ``model_free`` holds the nuisance columns (each a flattened (N, 2) response), ``offset``
    the forced response. Returns ``(s, sse, rel_se, beta)`` or None when the observed series
    has no component outside the nuisance span (the scale is then unidentifiable).
    """
    q, _ = np.linalg.qr(model_free)
    u = observed - q @ (q.T @ observed)
    w = offset - q @ (q.T @ offset)
    uu = float(u @ u)
    if uu <= 1e-18:
        return None
    s = float((u @ w) / uu)
    if not np.isfinite(s) or s <= 0:
        return None
    resid = s * u - w
    sse = float(resid @ resid)
    dof = max(1, len(u) - model_free.shape[1] - 1)
    rel_se = float(np.sqrt(sse / dof / uu) / s)
    beta, *_ = np.linalg.lstsq(model_free, s * observed - offset, rcond=None)
    return s, sse, rel_se, beta


def shooting_fit(X_h: np.ndarray, yaw: np.ndarray, lean: np.ndarray, t: np.ndarray,
                 k_d: float = KD_DEFAULT, g: float = G0,
                 psi_grid: np.ndarray | None = None) -> FitResult:
    """V2. Integrate the depicted lean twice; fit scale, yaw, initial velocity and position.

    ``X_h`` (N, 2) is the observed horizontal path, ``lean`` (N, 2) the depicted/attitude lean
    in the yaw-aligned frame, ``yaw`` (N,) the heading used to rotate it into the path frame.
    The returned statistic is ``sqrt(SSE / SST)`` with ``SST`` about the observed path mean
    scaled by the fitted scale: 0 means the attitude explains the path shape exactly.

    The scale is constrained positive and eliminated in closed form (see ``_scaled_fit``):
    an unconstrained least squares would otherwise prefer ``s = 0`` whenever the attitude
    implies *no* acceleration, which is exactly the tilt-removed case this test must detect.
    """
    X_h = np.asarray(X_h, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64).ravel()
    lean = np.asarray(lean, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel()
    n = len(t)
    if len(X_h) != n or len(yaw) != n or len(lean) != n:
        raise ValueError("length mismatch")
    dt = float(np.median(np.diff(t)))
    grid = _psi_grid() if psi_grid is None else np.asarray(psi_grid, dtype=np.float64)
    Xc = X_h - X_h.mean(axis=0)
    eye_x = np.tile([1.0, 0.0], (n, 1)).ravel()
    eye_y = np.tile([0.0, 1.0], (n, 1)).ravel()
    best: FitResult | None = None
    for psi in grid:
        forcing = g * apply_rot2(psi + yaw, lean)
        resp = drag_response(forcing, dt, k_d)
        basis = np.column_stack([resp.H_x[:, 0, :].ravel(), resp.H_x[:, 1, :].ravel(),
                                 eye_x, eye_y])
        out = _scaled_fit(basis, Xc.ravel(), resp.F_x.ravel())
        if out is None:
            continue
        s, sse, rel_se, beta = out
        sst = float(s * s * np.sum(np.sum(Xc ** 2, axis=1)))
        if sst <= 0 or not np.isfinite(sst):
            continue
        rho = float(np.sqrt(sse / sst))
        cand = FitResult(statistic=rho, psi_deg=float(np.degrees(psi)), scale=s,
                         scale_rel_se=rel_se, sse=sse, sst=sst, r2=1.0 - sse / sst,
                         extra={"v0": [float(beta[0]), float(beta[1])],
                                "x0": [float(beta[2]) + float(X_h.mean(axis=0)[0]),
                                       float(beta[3]) + float(X_h.mean(axis=0)[1])]})
        if best is None or cand.statistic < best.statistic:
            best = cand
    if best is None:
        return FitResult(float("nan"), float("nan"), float("nan"), float("nan"),
                         float("nan"), float("nan"), float("nan"))
    return best


def preintegration_fit(v_obs: np.ndarray, yaw: np.ndarray, lean: np.ndarray, t: np.ndarray,
                       k_d: float = KD_DEFAULT, g: float = G0,
                       psi_grid: np.ndarray | None = None) -> FitResult:
    """V1. Integrate the ODE once and fit scale and initial velocity to the observed velocity.

    ``v_obs`` is one derivative of the pixel path, so the statistic is exposed to a single
    differentiation. ``extra`` carries the per-axis correlation and sign agreement at the
    chosen yaw offset, which is what the pre-registration says to report beside it.
    """
    v_obs = np.asarray(v_obs, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64).ravel()
    lean = np.asarray(lean, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel()
    n = len(t)
    dt = float(np.median(np.diff(t)))
    grid = _psi_grid() if psi_grid is None else np.asarray(psi_grid, dtype=np.float64)
    best: FitResult | None = None
    for psi in grid:
        forcing = g * apply_rot2(psi + yaw, lean)
        resp = drag_response(forcing, dt, k_d)
        basis = np.column_stack([resp.H_v[:, 0, :].ravel(), resp.H_v[:, 1, :].ravel()])
        out = _scaled_fit(basis, v_obs.ravel(), resp.F_v.ravel())
        if out is None:
            continue
        s, sse, rel_se, beta = out
        sst = float(s * s * np.sum(v_obs ** 2))
        if sst <= 0:
            continue
        rho = float(np.sqrt(sse / sst))
        v_model = (resp.H_v[:, 0, :] * beta[0] + resp.H_v[:, 1, :] * beta[1] + resp.F_v)
        cand = FitResult(statistic=rho, psi_deg=float(np.degrees(psi)), scale=s,
                         scale_rel_se=rel_se, sse=sse, sst=sst, r2=1.0 - sse / sst,
                         extra={"v_model": v_model, "v_obs": v_obs})
        if best is None or cand.statistic < best.statistic:
            best = cand
    if best is None:
        return FitResult(float("nan"), float("nan"), float("nan"), float("nan"),
                         float("nan"), float("nan"), float("nan"))
    vm = best.extra["v_model"]
    vo = best.extra["v_obs"]
    r = [_pearson(vo[:, i], vm[:, i]) for i in range(2)]
    sg = [_sign_agree(vo[:, i], vm[:, i]) for i in range(2)]
    best.extra = {"r_x": r[0], "r_y": r[1], "s_x": sg[0], "s_y": sg[1],
                  "r_complex": complex_correlation(vo, vm)}
    return best


def velocity_drag_fit(v_obs: np.ndarray, yaw: np.ndarray, lean: np.ndarray,
                      k_d: float = KD_DEFAULT, g: float = G0,
                      psi_grid: np.ndarray | None = None) -> FitResult:
    """V4. Near-steady flight: ``v_h ~ (g / k_d) Rz(psi + yaw) lean``, scale fitted."""
    v_obs = np.asarray(v_obs, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64).ravel()
    lean = np.asarray(lean, dtype=np.float64)
    grid = _psi_grid() if psi_grid is None else np.asarray(psi_grid, dtype=np.float64)
    best: FitResult | None = None
    for psi in grid:
        model = apply_rot2(psi + yaw, lean)
        denom = float(np.sum(model ** 2))
        if denom <= 1e-18:
            continue
        s = float(np.sum(model * v_obs) / denom)
        if s <= 0:
            continue
        resid = v_obs - s * model
        sse = float(np.sum(resid ** 2))
        sst = float(np.sum(v_obs ** 2))
        cand = FitResult(statistic=float(np.sqrt(sse / sst)), psi_deg=float(np.degrees(psi)),
                         scale=s, scale_rel_se=float("nan"), sse=sse, sst=sst,
                         r2=1.0 - sse / sst)
        if best is None or cand.statistic < best.statistic:
            best = cand
    if best is None:
        return FitResult(float("nan"), float("nan"), float("nan"), float("nan"),
                         float("nan"), float("nan"), float("nan"))
    return best


# --------------------------------------------------------------------------- coherence
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3:
        return float("nan")
    if np.var(a) <= 1e-18 or np.var(b) <= 1e-18:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _sign_agree(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = (np.abs(a) > eps) & (np.abs(b) > eps)
    if not keep.any():
        return float("nan")
    return float(np.mean(np.sign(a[keep]) == np.sign(b[keep])))


def complex_correlation(z1: np.ndarray, z2: np.ndarray) -> float:
    """``|sum conj(z1) z2| / sqrt(sum|z1|^2 sum|z2|^2)`` on 2-D real series.

    This is the magnitude of the complex correlation: a positive rotation of either series
    (a yaw offset) cannot change it, and a positive scale cannot change it. 1 means the two
    series are identical up to yaw and scale, 0 means orthogonal.
    """
    a = np.asarray(z1, dtype=np.float64)
    b = np.asarray(z2, dtype=np.float64)
    ca = a[:, 0] + 1j * a[:, 1]
    cb = b[:, 0] + 1j * b[:, 1]
    na = float(np.sum(np.abs(ca) ** 2))
    nb = float(np.sum(np.abs(cb) ** 2))
    if na <= 1e-18 or nb <= 1e-18:
        # an all-zero series carries no phase to agree with, so there is no coherence
        return 0.0
    return float(abs(np.sum(np.conj(ca) * cb)) / np.sqrt(na * nb))


def bandpass(values: np.ndarray, dt: float, f_lo: float, f_hi: float,
             order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth band-pass; returns the input unchanged if the band is empty."""
    from scipy.signal import butter, filtfilt

    values = np.asarray(values, dtype=np.float64)
    fs = 1.0 / dt
    nyq = fs / 2.0
    lo, hi = max(f_lo, 0.0) / nyq, min(f_hi, nyq * 0.999) / nyq
    if not (0 < lo < hi < 1):
        return values - values.mean(axis=0)
    if len(values) < 12:
        return values - values.mean(axis=0)
    b, a = butter(order, [lo, hi], btype="band")
    pad = min(len(values) - 1, 3 * max(len(a), len(b)))
    return filtfilt(b, a, values, axis=0, padlen=pad)


def band_coherence(a_obs: np.ndarray, yaw: np.ndarray, lean: np.ndarray, dt: float,
                   f_lo: float, f_hi: float) -> dict:
    """V3. Band-limited complex coherence between the depicted lean and path acceleration.

    Both series are band-passed to ``[f_lo, f_hi]``; the statistic is ``1 - |rho_complex|``.
    The Welch magnitude-squared coherence over the same band is returned beside it as a
    cross-check with a different estimator.
    """
    f_l = apply_rot2(yaw, lean)
    bl = bandpass(f_l, dt, f_lo, f_hi)
    ba = bandpass(a_obs, dt, f_lo, f_hi)
    rho = complex_correlation(bl, ba)
    return {"statistic": float(1.0 - rho) if np.isfinite(rho) else float("nan"),
            "rho_complex": rho,
            "welch_max": welch_max_coherence(bl, ba, dt, f_lo, f_hi)}


def welch_max_coherence(z1: np.ndarray, z2: np.ndarray, dt: float,
                        f_lo: float, f_hi: float) -> float:
    """Max magnitude-squared coherence over the band, from a Welch CSD (Hann, 50 % overlap).

    Removed from the decision statistic because a 4 s clip has only a few independent
    segments in a 0.05-1.5 Hz band; reported as a cross-check.
    """
    from scipy.signal import csd, welch

    a = np.asarray(z1, dtype=np.float64)
    b = np.asarray(z2, dtype=np.float64)
    ca = a[:, 0] + 1j * a[:, 1]
    cb = b[:, 0] + 1j * b[:, 1]
    nperseg = min(len(ca), 32)
    if nperseg < 8:
        return float("nan")
    f, s_ab = csd(ca, cb, fs=1.0 / dt, nperseg=nperseg, noverlap=nperseg // 2,
                  detrend="linear")
    _, s_aa = welch(ca, fs=1.0 / dt, nperseg=nperseg, noverlap=nperseg // 2,
                    detrend="linear")
    _, s_bb = welch(cb, fs=1.0 / dt, nperseg=nperseg, noverlap=nperseg // 2,
                    detrend="linear")
    band = (f >= f_lo) & (f <= f_hi) & (f > 0)
    if not band.any():
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = np.abs(s_ab) ** 2 / (s_aa * s_bb)
    coh = coh[band]
    coh = coh[np.isfinite(coh)]
    return float(np.max(coh)) if coh.size else float("nan")


# --------------------------------------------------------------------------- scale check
def physics_scale_check(fit: FitResult, path_steps_m: np.ndarray, depth_m: np.ndarray,
                        rel_se_max: float = 1.0,
                        ratio_bounds: tuple[float, float] = (1e-4, 0.5)) -> dict:
    """V5. Is the scale the shooting fit recovers identifiable and physically plausible?

    ``path_steps_m`` is the per-frame displacement of the *pixel* path (arbitrary units) and
    ``depth_m`` the per-frame metric scene depth. The plausibility band compares the metric
    step the fitted scale implies against the scene depth: a frame-to-frame step of more than
    half the scene depth, or less than 1e-4 of it, is implausible for this footage. The band
    is deliberately loose; it is a consistency check, not a calibrated test.
    """
    rel_se = float(getattr(fit, "scale_rel_se", np.nan))
    s = float(getattr(fit, "scale", np.nan))
    steps = np.asarray(path_steps_m, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    finite = np.isfinite(steps) & np.isfinite(depth) & (depth > 0)
    ratio = float("nan")
    if finite.any() and np.isfinite(s) and s > 0:
        ratio = float(np.median(s * steps[finite] / depth[finite]))
    ident = bool(np.isfinite(s) and s > 0 and np.isfinite(rel_se) and rel_se <= rel_se_max)
    lo, hi = ratio_bounds
    plausible = bool(np.isfinite(ratio) and lo < ratio < hi)
    flag = (not ident) or (not plausible)
    return {"identifiable": ident, "rel_se": rel_se, "scale": s,
            "depth_ratio": ratio, "plausible": plausible, "flag": bool(flag)}
