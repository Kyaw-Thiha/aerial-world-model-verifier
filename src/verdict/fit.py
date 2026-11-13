"""Per-clip regression of depicted lean on required lean (the "gain test").

Model (pre-registered in ``configs/gain_prereg.yaml``)::

    l_dep = G1 * l_req0 + G2 * (Rz(-yaw) v / g) + b

``G1`` is the acceleration gain the verdict tests, ``G2`` is a drag coefficient in 1/s
(fitted nuisance unless fixed), and ``b`` is a free 2-D intercept that absorbs a constant
mount offset.  A real flight gives a positive ``G1`` calibrated on real clips; tilt removal
gives 0, a 50 % scale gives 0.5, and a mirrored roll flips the sign of the roll-axis gain
``gy`` while leaving ``gx`` positive.

Everything here is fitted from the shared clip schema (``t, pos, yaw, depicted_lean,
depicted_conf``); no mocap quantity other than ``pos`` and ``yaw`` is read.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .lean import depicted_lean_from_gravity, principal_spread_deg, required_lean

DRAG_FIXED_DEFAULT = 0.295


@dataclass
class FitOptions:
    drag: str = "fitted"            # "fitted" | "fixed" | "free"
    fixed_drag: float = DRAG_FIXED_DEFAULT
    intercept: bool = True
    lag_search: bool = True
    lag_bounds_ms: tuple[float, float] = (-150.0, 150.0)
    lag_step_ms: float = 25.0
    estimator: str = "ols"          # "ols" | "huber"
    smoothing_s: float = 0.0
    confidence_weights: bool = False
    sg_window_s: float = 0.37
    sg_polyorder: int = 3
    bootstrap: bool = True
    n_boot: int = 400
    seed: int = 20261113
    min_frames: int = 24

    def validate(self) -> None:
        if self.drag not in ("fitted", "fixed", "free"):
            raise ValueError(f"unknown drag mode {self.drag!r}")
        if self.estimator not in ("ols", "huber"):
            raise ValueError(f"unknown estimator {self.estimator!r}")
        if self.drag == "fixed" and not np.isfinite(self.fixed_drag):
            raise ValueError("fixed drag needs a finite fixed_drag")
        if self.n_boot < 20:
            raise ValueError("n_boot must be at least 20")


@dataclass
class GainResult:
    """A per-clip gain fit, with everything the graded output (T5) reports."""

    gain: float
    gain_ci: tuple[float, float]
    axis_gains: tuple[float, float]
    axis_ci: tuple[tuple[float, float], tuple[float, float]]
    drag: float
    drag_ci: tuple[float, float]
    intercept: tuple[float, float]
    r2: float
    cond_number: float
    lag_ms: float
    n_frames: int
    n_used: int
    used_fraction: float
    finite_fraction: float
    required_tilt_deg: float
    depicted_tilt_deg: float
    excitation_deg: float
    n_signal_frames: int
    method: dict = field(default_factory=dict)
    bootstrap_ok: bool = False

    @property
    def gain_half_width(self) -> float:
        lo, hi = self.gain_ci
        return float("nan") if not np.isfinite([lo, hi]).all() else 0.5 * (hi - lo)

    @property
    def axis_half_widths(self) -> tuple[float, float]:
        out = []
        for lo, hi in self.axis_ci:
            out.append(float("nan") if not np.isfinite([lo, hi]).all() else 0.5 * (hi - lo))
        return tuple(out)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["gain_half_width"] = self.gain_half_width
        d["axis_half_widths"] = list(self.axis_half_widths)
        return d


# ------------------------------------------------------------------ small helpers
def _finite_rows(l_dep: np.ndarray, l_req: np.ndarray, w: np.ndarray,
                 mask: np.ndarray) -> np.ndarray:
    return (mask & np.isfinite(l_dep).all(axis=1)
            & np.isfinite(l_req).all(axis=1) & np.isfinite(w).all(axis=1))


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware centered moving average along axis 0."""
    if window <= 1:
        return values.copy()
    values = np.asarray(values, dtype=np.float64)
    half = window // 2
    out = np.full_like(values, np.nan)
    n = len(values)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = values[lo:hi]
        ok = np.isfinite(seg)
        count = ok.sum(axis=0)
        with np.errstate(invalid="ignore"):
            out[i] = np.where(count > 0, np.nansum(seg, axis=0) / np.maximum(count, 1),
                              np.nan)
    return out


def _weights(conf: np.ndarray) -> np.ndarray | None:
    conf = np.asarray(conf, dtype=np.float64)
    ok = np.isfinite(conf)
    if not ok.any():
        return None
    med = float(np.median(conf[ok]))
    if not np.isfinite(med) or med == 0:
        return None
    ratio = np.where(ok, conf / med, 1.0)
    return np.clip(ratio, 0.1, 10.0)


def _shift_series(values: np.ndarray, t: np.ndarray, lag_s: float) -> np.ndarray:
    """``values`` sampled at ``t + lag_s`` (linear, edge clamped), per column."""
    values = np.asarray(values, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    target = np.clip(t + lag_s, t[0], t[-1])
    if values.ndim == 1:
        return np.interp(target, t, values)
    return np.column_stack([np.interp(target, t, values[:, k])
                            for k in range(values.shape[1])])


def _design(l_req: np.ndarray, w: np.ndarray, drag: str, intercept: bool,
            fixed_drag: float = DRAG_FIXED_DEFAULT
            ) -> tuple[np.ndarray, np.ndarray, float]:
    """Stacked scalar design and response offset for the fitted drag mode."""
    l_req = np.asarray(l_req, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n = len(l_req)
    cols: list[np.ndarray] = []
    names: list[str] = ["G1"]
    cols.append(np.column_stack([l_req[:, 0], l_req[:, 1]]).ravel())
    offset = np.zeros(2 * n)
    if drag == "fitted":
        cols.append(np.column_stack([w[:, 0], w[:, 1]]).ravel())
        names.append("G2")
    elif drag == "fixed":
        offset = np.column_stack([fixed_drag * w[:, 0],
                                  fixed_drag * w[:, 1]]).ravel()
    if intercept:
        bx = np.zeros(2 * n)
        by = np.zeros(2 * n)
        bx[0::2] = 1.0
        by[1::2] = 1.0
        cols.extend([bx, by])
        names.extend(["b_x", "b_y"])
    return np.column_stack(cols), offset, names


def _axis_design(l_req: np.ndarray, w: np.ndarray, axis: int, drag: str,
                 intercept: bool, fixed_drag: float = DRAG_FIXED_DEFAULT
                 ) -> tuple[np.ndarray, np.ndarray]:
    x = l_req[:, axis]
    ww = w[:, axis]
    cols = [x]
    offset = np.zeros(len(x))
    if drag == "fitted":
        cols.append(ww)
    elif drag == "fixed":
        offset = fixed_drag * ww
    if intercept:
        cols.append(np.ones(len(x)))
    return np.column_stack(cols), offset


def _solve_linear(A: np.ndarray, y: np.ndarray, weights: np.ndarray | None,
                  estimator: str, f_scale: float) -> np.ndarray:
    if A.shape[1] == 0:
        return np.zeros(0)
    if weights is not None:
        sw = np.sqrt(weights)
        Aw, yw = A * sw[:, None], y * sw
    else:
        Aw, yw = A, y
    beta, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    if estimator == "huber":
        res = least_squares(lambda b: (Aw @ b - yw), beta, loss="huber",
                            f_scale=f_scale)
        beta = res.x
    return beta


@dataclass
class _RawFit:
    gain: float
    drag: float
    intercept: np.ndarray
    axis_gains: np.ndarray
    axis_drags: np.ndarray
    axis_intercepts: np.ndarray
    r2: float
    cond: float
    sse: float


def _fit_once(l_req: np.ndarray, w: np.ndarray, l_dep: np.ndarray, opts: FitOptions,
              weights: np.ndarray | None = None) -> _RawFit:
    A, offset, _ = _design(l_req, w, opts.drag, opts.intercept, opts.fixed_drag)
    y = np.column_stack([l_dep[:, 0], l_dep[:, 1]]).ravel() - offset
    Aw = np.repeat(weights, 2) if weights is not None else None
    beta = _solve_linear(A, y, Aw, opts.estimator, 0.05)
    resid = y - A @ beta
    sse = float(resid @ resid)
    yc = np.column_stack([l_dep[:, 0] - l_dep[:, 0].mean(),
                          l_dep[:, 1] - l_dep[:, 1].mean()]).ravel()
    sst = float(yc @ yc)
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    gain = float(beta[0])
    pos = 1
    drag = float(beta[pos]) if opts.drag == "fitted" else (
        opts.fixed_drag if opts.drag == "fixed" else 0.0)
    if opts.drag == "fitted":
        pos += 1
    if opts.intercept:
        intercept = np.array([beta[pos], beta[pos + 1]])
    else:
        intercept = np.zeros(2)
    axis_gains, axis_drags, axis_ints = [], [], []
    for axis in range(2):
        Ax, offx = _axis_design(l_req, w, axis, opts.drag, opts.intercept,
                                opts.fixed_drag)
        wx = weights if weights is not None else None
        bx = _solve_linear(Ax, l_dep[:, axis] - offx, wx, opts.estimator, 0.05)
        axis_gains.append(float(bx[0]))
        k = 1
        axis_drags.append(float(bx[k]) if opts.drag == "fitted" else drag)
        if opts.drag == "fitted":
            k += 1
        axis_ints.append(float(bx[k]) if opts.intercept else 0.0)
    try:
        cond = float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        cond = float("inf")
    return _RawFit(gain=gain, drag=drag, intercept=intercept,
                   axis_gains=np.array(axis_gains), axis_drags=np.array(axis_drags),
                   axis_intercepts=np.array(axis_ints), r2=r2, cond=cond, sse=sse)


def _percentile_ci(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 20:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def fit_gain(clip, opts: FitOptions | None = None) -> GainResult:
    """Fit the gain statistic to one clip (see the module docstring)."""
    from .schema import Clip

    if not isinstance(clip, Clip):
        raise TypeError(f"expected a Clip, got {type(clip)!r}")
    opts = opts or FitOptions()
    opts.validate()

    req = required_lean(clip.pos, clip.yaw, clip.t, window_s=opts.sg_window_s,
                        polyorder=opts.sg_polyorder)
    l_dep = np.asarray(clip.depicted_lean, dtype=np.float64).copy()
    if opts.smoothing_s > 0:
        window = max(1, int(round(opts.smoothing_s / req.dt_s)))
        l_dep = _smooth_series(l_dep, window)
    weights = _weights(clip.depicted_conf) if opts.confidence_weights else None

    base_mask = req.mask & _finite_rows(l_dep, req.l_req0, req.velocity_reg, req.mask)
    n_frames = len(clip.t)
    finite_frac = float(np.mean(np.isfinite(l_dep).all(axis=1)))

    # lag search: shift the required side onto the depicted side
    lag_s = 0.0
    if opts.lag_search:
        lags = np.arange(opts.lag_bounds_ms[0], opts.lag_bounds_ms[1] + 1e-9,
                         opts.lag_step_ms) / 1000.0
        best = (np.inf, 0.0)
        for candidate in lags:
            lr = _shift_series(req.l_req0, clip.t, float(candidate))
            vv = _shift_series(req.velocity_reg, clip.t, float(candidate))
            ok = base_mask & np.isfinite(lr).all(axis=1) & np.isfinite(vv).all(axis=1)
            if ok.sum() < opts.min_frames:
                continue
            wsub = weights[ok] if weights is not None else None
            raw = _fit_once(lr[ok], vv[ok], l_dep[ok], opts, wsub)
            if raw.sse < best[0]:
                best = (raw.sse, float(candidate))
        if np.isfinite(best[0]):
            lag_s = best[1]

    lr = _shift_series(req.l_req0, clip.t, lag_s)
    vv = _shift_series(req.velocity_reg, clip.t, lag_s)
    ok = base_mask & np.isfinite(lr).all(axis=1) & np.isfinite(vv).all(axis=1)
    if ok.sum() < opts.min_frames:
        return _nan_result(clip, req, l_dep, ok, lag_s, opts,
                           reason="too few usable frames")

    wsub = weights[ok] if weights is not None else None
    raw = _fit_once(lr[ok], vv[ok], l_dep[ok], opts, wsub)

    gain_ci = (float("nan"), float("nan"))
    axis_ci = ((float("nan"), float("nan")), (float("nan"), float("nan")))
    drag_ci = (float("nan"), float("nan"))
    boot_ok = False
    if opts.bootstrap:
        idx = np.flatnonzero(ok)
        dt_s = req.dt_s
        block = max(1, int(round(0.5 / dt_s)))
        starts = np.arange(0, len(idx), block)
        if len(starts) >= 5:
            rng = np.random.default_rng(opts.seed)
            boot = []
            for _ in range(opts.n_boot):
                chosen = []
                while len(chosen) < len(idx):
                    k = int(rng.integers(len(starts)))
                    chosen.extend(idx[starts[k]:starts[k] + block].tolist())
                chosen = np.array(chosen[:len(idx)])
                wb = weights[chosen] if weights is not None else None
                boot.append(_fit_once(lr[chosen], vv[chosen], l_dep[chosen], opts, wb))
            gains = np.array([b.gain for b in boot])
            gain_ci = _percentile_ci(gains)
            axis_ci = (_percentile_ci(np.array([b.axis_gains[0] for b in boot])),
                       _percentile_ci(np.array([b.axis_gains[1] for b in boot])))
            drag_ci = _percentile_ci(np.array([b.drag for b in boot]))
            boot_ok = bool(np.isfinite(gain_ci).all())

    req_tilt = req.tilt_deg[ok]
    dep_norm = np.linalg.norm(l_dep[ok], axis=1)
    dep_tilt = np.degrees(np.arcsin(np.clip(dep_norm, 0.0, 1.0)))
    return GainResult(
        gain=raw.gain, gain_ci=gain_ci,
        axis_gains=(raw.axis_gains[0], raw.axis_gains[1]), axis_ci=axis_ci,
        drag=raw.drag, drag_ci=drag_ci,
        intercept=(float(raw.intercept[0]), float(raw.intercept[1])),
        r2=raw.r2, cond_number=raw.cond, lag_ms=float(lag_s * 1000.0),
        n_frames=n_frames, n_used=int(ok.sum()),
        used_fraction=float(ok.sum() / max(n_frames, 1)), finite_fraction=finite_frac,
        required_tilt_deg=float(np.nanmedian(req_tilt)),
        depicted_tilt_deg=float(np.nanmedian(dep_tilt)),
        excitation_deg=principal_spread_deg(lr[ok]),
        n_signal_frames=int(np.sum(req_tilt >= 10.0)),
        method={"drag": opts.drag, "intercept": opts.intercept,
                "estimator": opts.estimator, "lag_search": opts.lag_search,
                "smoothing_s": opts.smoothing_s,
                "confidence_weights": opts.confidence_weights,
                "sg_window_s": opts.sg_window_s, "sg_polyorder": opts.sg_polyorder},
        bootstrap_ok=boot_ok,
    )


def _nan_result(clip, req, l_dep, ok, lag_s, opts, reason: str = "") -> GainResult:
    return GainResult(
        gain=float("nan"), gain_ci=(float("nan"), float("nan")),
        axis_gains=(float("nan"), float("nan")),
        axis_ci=((float("nan"), float("nan")), (float("nan"), float("nan"))),
        drag=float("nan"), drag_ci=(float("nan"), float("nan")),
        intercept=(float("nan"), float("nan")), r2=float("nan"),
        cond_number=float("nan"), lag_ms=float(lag_s * 1000.0),
        n_frames=len(clip.t), n_used=int(ok.sum()),
        used_fraction=float(ok.sum() / max(len(clip.t), 1)),
        finite_fraction=float(np.mean(np.isfinite(l_dep).all(axis=1))),
        required_tilt_deg=float("nan"), depicted_tilt_deg=float("nan"),
        excitation_deg=float("nan"), n_signal_frames=0,
        method={"drag": opts.drag, "intercept": opts.intercept,
                "estimator": opts.estimator, "lag_search": opts.lag_search,
                "smoothing_s": opts.smoothing_s,
                "confidence_weights": opts.confidence_weights,
                "sg_window_s": opts.sg_window_s, "sg_polyorder": opts.sg_polyorder,
                "reason": reason}, bootstrap_ok=False)


def lean_from_gravity(clip, uptilt_deg: float, mount_rotation: np.ndarray | None = None
                      ) -> np.ndarray:
    """Convenience wrapper: depicted lean from a gravity array stored in ``meta``.

    The builder uses this for TII; the shared clip file already carries the lean, so
    scoring never needs camera extrinsics.
    """
    from src.perception.frames import camera_to_body_pitched

    up = np.asarray(clip.meta.get("up_cam", []), dtype=np.float64)
    R_cb = (np.asarray(mount_rotation, dtype=np.float64) if mount_rotation is not None
            else camera_to_body_pitched(np.deg2rad(uptilt_deg)))
    return depicted_lean_from_gravity(up, R_cb)
