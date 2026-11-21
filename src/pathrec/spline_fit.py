"""Spline-smoothed joint fit of the flatness model to a pixel path (Track F2, part 1).

The round-1 statistics either integrated the depicted lean (F01 V2) or differentiated the
observed path (F01 V1/V3). This one fits a cubic B-spline to the observed horizontal path with
knots spaced by the E05 window rule (``0.5 / f95``), takes velocity and acceleration
analytically from the spline, and then fits the flatness model

    v_m' = -k_d v_m + g Rz(psi + yaw) lean,     v_m(t_0) = v0
    a_m  = g Rz(psi + yaw) lean - k_d v_m

to the spline's acceleration by least squares in the positive path scale ``s`` and the initial
velocity ``v0``, over a grid of yaw offsets ``psi``. The statistic is the fraction of observed
horizontal acceleration energy the prediction does not explain:

    rho = sqrt(SSE / sum |s a_spline|^2).

Pre-registration: ``configs/f04_round2_prereg.yaml``. Everything here is scale-free except the
fitted ``s`` it reports for diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import make_lsq_spline

G0 = 9.80665
KD_DEFAULT = 0.295
PSI_STEP_DEG = 1.0
BOUNDARY_SAMPLES = 3


def horizontal_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane perpendicular to ``up``.

    Same convention as round 1's ``src/pathrec/pathverdict.py``: the choice is arbitrary (a
    yaw offset is fitted) but deterministic, and the world axis least aligned with ``up`` is
    dropped.
    """
    up = np.asarray(up, dtype=np.float64).reshape(3)
    up = up / np.linalg.norm(up)
    e = np.zeros(3)
    e[int(np.argmin(np.abs(up)))] = 1.0
    e1 = e - np.dot(e, up) * up
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return e1, e2 / np.linalg.norm(e2)


# --------------------------------------------------------------------------- spline
@dataclass
class SplinePath:
    """Spline-evaluated path on its interior, with analytic derivatives."""

    t: np.ndarray            # (M,) seconds, relative
    X: np.ndarray            # (M, 2) path units
    v: np.ndarray            # (M, 2) path units / s
    a: np.ndarray            # (M, 2) path units / s^2
    n_intervals: int         # K, the pre-registered knot count
    degree: int
    knot_u: np.ndarray


def knot_intervals(n: int, dt: float, window_s: float) -> int:
    """Number of spline intervals under the pre-registered rule.

    ``K = max(4, round(T / window_s))``, capped so every interval carries at least three
    samples. With the 60-frame UZH clips and the E05 windows this is 4-12.
    """
    if window_s <= 0 or dt <= 0 or n < 4:
        raise ValueError("window_s, dt and n must be positive and n >= 4")
    span = (n - 1) * dt
    k = int(round(span / float(window_s)))
    k = max(4, k)
    cap = max(4, (n - 1) // 3)
    return min(k, cap)


def fit_spline_path(t: np.ndarray, X: np.ndarray, window_s: float, degree: int = 3,
                    n_intervals: int | None = None,
                    boundary: int = BOUNDARY_SAMPLES) -> SplinePath:
    """Fit a cubic B-spline to a (N, 2) track and return interior derivatives.

    ``n_intervals`` overrides the pre-registered knot count (used only by the declared
    knot-density sensitivity). A clamped cubic spline's second derivative is unreliable at the
    ends, so the first and last ``boundary`` samples are dropped from the returned arrays.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError(f"X must be (N, >=2), got {X.shape}")
    n = len(t)
    if len(X) != n or n < 8:
        raise ValueError("t and X must have the same length, at least 8")
    if not np.all(np.diff(t) > 0):
        raise ValueError("t must be strictly increasing")
    dt = float(np.median(np.diff(t)))
    span = float(t[-1] - t[0])
    k = int(n_intervals) if n_intervals is not None else knot_intervals(n, dt, window_s)
    k = max(4, min(k, (n - 1) // 3))

    u = (t - t[0]) / span
    inner = np.linspace(0.0, 1.0, k + 1)[1:-1]
    knot_u = np.concatenate([np.zeros(degree + 1), inner, np.ones(degree + 1)])
    spl = make_lsq_spline(u, X, knot_u, degree)

    # derivatives with respect to u, then chain rule to seconds
    d1 = spl.derivative(1)
    d2 = spl.derivative(2)
    sl = slice(boundary, n - boundary)
    ue = u[sl]
    t_e = t[sl] - t[0]
    Xe = spl(ue)
    v = d1(ue) / span
    a = d2(ue) / (span * span)
    return SplinePath(t=t_e, X=Xe, v=v, a=a, n_intervals=k, degree=degree, knot_u=knot_u)


# --------------------------------------------------------------------------- drag model
def drag_velocity(t: np.ndarray, forcing: np.ndarray, k_d: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Velocity response to a free initial velocity and a forcing, zero-order hold.

    Returns ``(H, F)`` with ``v_m = H[:, None] * v0 + F``: ``H`` is the scalar homogeneous
    decay (the drag is isotropic) and ``F`` the forced response, on the sample grid ``t``.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    f = np.asarray(forcing, dtype=np.float64)
    n = len(t)
    if len(f) != n:
        raise ValueError("forcing and t must have the same length")
    if k_d < 0:
        raise ValueError("k_d must be non-negative")
    dt = float(np.median(np.diff(t)))
    decay = np.exp(-k_d * dt) if k_d > 0 else 1.0
    a_step = (1.0 - decay) / k_d if k_d > 0 else dt
    H = decay ** np.arange(n)
    F = np.zeros((n, 2))
    if n > 1:
        # F[n] = a * sum_{i<n} decay^(n-1-i) f[i]: one IIR step, shifted by one sample
        from scipy.signal import lfilter
        F[1:] = lfilter([a_step], [1.0, -decay], f[:-1], axis=0)
    return H, F


# --------------------------------------------------------------------------- fit
@dataclass
class DynFit:
    statistic: float
    statistic_model: float
    statistic_demeaned: float
    psi_deg: float
    scale: float
    v0: tuple[float, float]
    sse: float
    sst: float
    n: int
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(np.isfinite(self.statistic))


def _psi_grid(step_deg: float = PSI_STEP_DEG) -> np.ndarray:
    return np.deg2rad(np.arange(-180.0, 180.0, step_deg))


def _rot2(angle, vec):
    c, s = np.cos(angle), np.sin(angle)
    x, y = vec[..., 0], vec[..., 1]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1)


def _denominators(a_obs_scaled: np.ndarray, a_model: np.ndarray) -> tuple[float, float, float]:
    obs = float(np.sum(a_obs_scaled ** 2))
    centred = a_obs_scaled - a_obs_scaled.mean(axis=0)
    obs_c = float(np.sum(centred ** 2))
    mod = float(np.sum(a_model ** 2))
    return obs, obs_c, mod


def _rho(sse: float, denom: float) -> float:
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.sqrt(max(sse, 0.0) / denom))


def dyn_fit(t: np.ndarray, a_spline: np.ndarray, yaw: np.ndarray, lean: np.ndarray,
            k_d: float = KD_DEFAULT, g: float = G0, psi_grid: np.ndarray | None = None,
            scale: float | None = None, psi: float | None = None,
            v0: np.ndarray | None = None) -> DynFit:
    """Fit the flatness model to the spline acceleration.

    With ``scale=None`` the scale is fitted (positive, eliminated in closed form per psi);
    with ``scale`` given the ruler is fixed, which is how an edit is scored after the real
    condition set it (pre-registration amendment 1's pairing). ``psi`` and ``v0`` likewise
    override the fitted values. Residuals are computed on the full 2-D acceleration.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    a_spline = np.asarray(a_spline, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64).ravel()
    lean = np.asarray(lean, dtype=np.float64)
    n = len(t)
    if not (len(a_spline) == len(yaw) == len(lean) == n):
        raise ValueError("length mismatch")
    grid = _psi_grid() if psi_grid is None else np.asarray(psi_grid, dtype=np.float64)
    if psi is not None:
        grid = np.asarray([psi], dtype=np.float64)
    fixed = scale is not None and psi is not None and v0 is not None

    if fixed:
        return _score_fixed(t, a_spline, yaw, lean, float(scale), float(psi),
                            np.asarray(v0, dtype=np.float64), k_d, g)

    best: DynFit | None = None
    for psi_v in grid:
        forcing = g * _rot2(psi_v + yaw, lean)
        H, F = drag_velocity(t, forcing, k_d)
        # s a_spline + k_d H v0 = forcing - k_d F, flattened component-wise
        kdH = k_d * np.repeat(H, 2)
        parity = np.arange(2 * n) % 2
        basis = np.column_stack([np.where(parity == 0, kdH, 0.0),
                                 np.where(parity == 1, kdH, 0.0)])
        y = (forcing - k_d * F).ravel()
        obs = a_spline.ravel()
        q, _ = np.linalg.qr(basis)
        u = obs - q @ (q.T @ obs)
        w = y - q @ (q.T @ y)
        uu = float(u @ u)
        if uu <= 1e-18:
            continue
        s = float((u @ w) / uu)
        if not np.isfinite(s) or s <= 0:
            continue
        resid = s * u - w
        sse = float(resid @ resid)
        # basis . v0 = y - s * obs  (the residual is s*obs - y + basis.v0)
        v0_fit, *_ = np.linalg.lstsq(basis, y - s * obs, rcond=None)
        a_model = forcing - k_d * (H[:, None] * v0_fit + F)
        denom, denom_c, denom_m = _denominators(s * a_spline, a_model)
        cand = DynFit(statistic=_rho(sse, denom), statistic_model=_rho(sse, denom_m),
                      statistic_demeaned=_rho(sse, denom_c), psi_deg=float(np.degrees(psi_v)),
                      scale=s, v0=(float(v0_fit[0]), float(v0_fit[1])), sse=sse,
                      sst=denom, n=n, extra={"k_d": k_d})
        if best is None or cand.statistic < best.statistic:
            best = cand
    if best is None:
        return DynFit(float("nan"), float("nan"), float("nan"), float("nan"),
                      float("nan"), (float("nan"), float("nan")), float("nan"),
                      float("nan"), n)
    return best


def _score_fixed(t: np.ndarray, a_spline: np.ndarray, yaw: np.ndarray, lean: np.ndarray,
                 scale: float, psi: float, v0: np.ndarray, k_d: float, g: float) -> DynFit:
    forcing = g * _rot2(psi + yaw, lean)
    H, F = drag_velocity(t, forcing, k_d)
    v_model = H[:, None] * v0 + F
    a_model = forcing - k_d * v_model
    resid = scale * a_spline - a_model
    sse = float(np.sum(resid ** 2))
    denom, denom_c, denom_m = _denominators(scale * a_spline, a_model)
    return DynFit(statistic=_rho(sse, denom), statistic_model=_rho(sse, denom_m),
                  statistic_demeaned=_rho(sse, denom_c), psi_deg=float(np.degrees(psi)),
                  scale=float(scale), v0=(float(v0[0]), float(v0[1])), sse=sse,
                  sst=denom, n=len(t), extra={"k_d": k_d, "fixed": True})


def dyn_score_psi(t: np.ndarray, a_spline: np.ndarray, yaw: np.ndarray, lean: np.ndarray,
                  scale: float, v0: np.ndarray, k_d: float = KD_DEFAULT,
                  g: float = G0, psi_grid: np.ndarray | None = None) -> DynFit:
    """Score with the scale and initial velocity fixed but the yaw offset re-searched.

    This is the pre-registered psi-refit sensitivity, not a gate path: the headline holds psi
    at the real-condition value so an edit cannot rotate itself into agreement.
    """
    grid = _psi_grid() if psi_grid is None else np.asarray(psi_grid, dtype=np.float64)
    best: DynFit | None = None
    for psi in grid:
        cand = _score_fixed(t, a_spline, yaw, lean, float(scale), float(psi),
                            np.asarray(v0, dtype=np.float64), k_d, g)
        if best is None or cand.statistic < best.statistic:
            best = cand
    return best


def dyn_fit_nov0(t: np.ndarray, a_spline: np.ndarray, v_spline: np.ndarray, yaw: np.ndarray,
                 lean: np.ndarray, k_d: float = KD_DEFAULT, g: float = G0,
                 psi_grid: np.ndarray | None = None, scale: float | None = None,
                 psi: float | None = None) -> DynFit:
    """EXPLORATORY (declared after round-2 results): the same model without a free v0.

    The registered fit lets v0 absorb a constant part of the forcing and drives s -> 0
    (``decisions.md`` [F2]); here the drag term uses the *observed* spline velocity, so the
    only free parameter is the positive scale:

        s * (a_spline + k_d v_spline) = g Rz(psi + yaw) lean.

    One scalar least squares per psi, s = f.x / |x|^2, s > 0 required. Statistics use the same
    normalisation as the registered fit (observed acceleration energy), so the two are
    comparable; the pairing rule (scale and psi from the real condition) is unchanged.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    a_spline = np.asarray(a_spline, dtype=np.float64)
    v_spline = np.asarray(v_spline, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64).ravel()
    lean = np.asarray(lean, dtype=np.float64)
    n = len(t)
    x_all = (a_spline + k_d * v_spline).ravel()
    grid = _psi_grid() if psi_grid is None else np.asarray(psi_grid, dtype=np.float64)
    if psi is not None:
        grid = np.asarray([psi], dtype=np.float64)
    best: DynFit | None = None
    for psi_v in grid:
        f = g * _rot2(psi_v + yaw, lean)
        ff = f.ravel()
        if scale is not None:
            s = float(scale)
        else:
            denom = float(x_all @ x_all)
            if denom <= 1e-18:
                continue
            s = float((ff @ x_all) / denom)
            if not np.isfinite(s) or s <= 0:
                continue
        resid = s * x_all - ff
        sse = float(resid @ resid)
        denom_obs = float(np.sum((s * a_spline) ** 2))
        denom_c = float(np.sum((s * a_spline - (s * a_spline).mean(axis=0)) ** 2))
        denom_m = float(np.sum(ff ** 2))
        cand = DynFit(statistic=_rho(sse, denom_obs), statistic_model=_rho(sse, denom_m),
                      statistic_demeaned=_rho(sse, denom_c), psi_deg=float(np.degrees(psi_v)),
                      scale=s, v0=(0.0, 0.0), sse=sse, sst=denom_obs, n=n,
                      extra={"k_d": k_d, "no_v0": True})
        if best is None or cand.statistic < best.statistic:
            best = cand
    if best is None:
        return DynFit(float("nan"), float("nan"), float("nan"), float("nan"),
                      float("nan"), (float("nan"), float("nan")), float("nan"),
                      float("nan"), n)
    return best
