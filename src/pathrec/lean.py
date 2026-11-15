"""Acceleration, lean and required tilt from a recovered path.

The verifier compares the lean depicted in pixels with the lean the path
requires. To first order the horizontal lean vector is

    l = a_h / (g + a_z)                       (dimensionless, tan of the tilt)

so its *direction* is the direction of horizontal acceleration and is free of
any scale factor, while its *magnitude* needs metric scale and a time base. The
same quantity falls out of the flatness model: the horizontal part of
``(a + g e3 + k_d v) / ||a + g e3 + k_d v||``.

Acceleration is a double derivative, so every function here takes ``dt`` or
``fps`` explicitly and refuses silent defaults. A window may be given in
seconds (video) or in steps (fixed-step generated clips).
"""
from __future__ import annotations

import numpy as np

from src.physics.diff import edge_slice, savgol_derivatives
from src.physics.flatness import implied_thrust_axis

G0 = 9.80665


def odd_window(n: int, polyorder: int) -> int:
    """Largest odd window <= n that still leaves interior samples."""
    w = n if n % 2 == 1 else n - 1
    if w < polyorder + 2:
        raise ValueError(f"window {w} too short for polyorder {polyorder} (n={n})")
    return w


def differentiate(pos: np.ndarray, dt: float, window_s: float | None = None,
                  window_steps: int | None = None, polyorder: int = 3
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Velocity and acceleration on the interior of a (N, 3) track.

    Exactly one of ``window_s`` (seconds, for video) and ``window_steps``
    (samples, for fixed-step clips) must be given.

    Returns
    -------
    (t, v, a) arrays trimmed to the interior, so all three have one sample per
    usable timestamp. ``t = arange(N) * dt``.
    """
    pos = np.asarray(pos, dtype=np.float64)
    if (window_s is None) == (window_steps is None):
        raise ValueError("give exactly one of window_s / window_steps")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    n = len(pos)
    if window_steps is not None:
        window = int(window_steps)
        if window % 2 == 0:
            window += 1
    else:
        window = odd_window(max(int(round(float(window_s) / dt)), polyorder + 2), polyorder)
    window = min(window, odd_window(n, polyorder))
    v_full, a_full = savgol_derivatives(pos, dt, window, polyorder)
    sl = edge_slice(n, window)
    t = np.arange(n, dtype=np.float64)[sl] * dt
    return t, v_full[sl], a_full[sl]


def lean_xy(a: np.ndarray, g: float = G0) -> np.ndarray:
    """Horizontal lean vector ``a_h / (g + a_z)`` from (N, 3) acceleration.

    Samples where ``g + a_z`` is non-positive (sustained free fall, or a sign
    error in the acceleration) are returned as NaN rather than as a flipped
    lean: the thrust axis there points below the horizon, which the flatness
    model cannot represent.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    den = g + a[:, 2]
    out = np.full((len(a), 2), np.nan)
    ok = den > 1e-6
    out[ok, 0] = a[ok, 0] / den[ok]
    out[ok, 1] = a[ok, 1] / den[ok]
    return out


def tilt_deg(a: np.ndarray, v: np.ndarray | None = None, k_d: float = 0.0,
             g: float = G0) -> np.ndarray:
    """Angle between the implied thrust axis and world up, in degrees."""
    z = implied_thrust_axis(a, v, k_d=k_d, g=g)
    return np.degrees(np.arccos(np.clip(z[:, 2], -1.0, 1.0)))


def direction_error_deg(l_est: np.ndarray, l_true: np.ndarray,
                        remove_constant_yaw: bool = False,
                        min_magnitude: float = 0.0) -> np.ndarray:
    """Angle between estimated and true lean directions, in degrees.

    Samples with a NaN on either side, or with ``|l_true| <= min_magnitude``, are
    dropped. With ``remove_constant_yaw`` one constant in-plane rotation is taken
    out first (the verifier's yaw is free, and a global frame offset is not a
    failure of the extractor).
    """
    l_est = np.asarray(l_est, dtype=np.float64).reshape(-1, 2)
    l_true = np.asarray(l_true, dtype=np.float64).reshape(-1, 2)
    if l_est.shape != l_true.shape:
        raise ValueError(f"shape mismatch {l_est.shape} vs {l_true.shape}")
    ok = (np.isfinite(l_est).all(1) & np.isfinite(l_true).all(1)
          & (np.linalg.norm(l_true, axis=1) > min_magnitude)
          & (np.linalg.norm(l_est, axis=1) > 0))
    if not ok.any():
        return np.array([])
    e, t = l_est[ok], l_true[ok]
    te = np.arctan2(e[:, 1], e[:, 0])
    tt = np.arctan2(t[:, 1], t[:, 0])
    diff = te - tt
    if remove_constant_yaw:
        mean = np.angle(np.mean(np.exp(1j * diff)))
        diff = diff - mean
    diff = np.angle(np.exp(1j * diff))  # wrap to (-pi, pi]
    return np.degrees(np.abs(diff))


def magnitude_ratio(l_est: np.ndarray, l_true: np.ndarray,
                    min_magnitude: float = 0.0) -> float:
    """Median ``|l_est| / |l_true|`` over testable samples (NaN if none)."""
    l_est = np.asarray(l_est, dtype=np.float64).reshape(-1, 2)
    l_true = np.asarray(l_true, dtype=np.float64).reshape(-1, 2)
    ok = (np.isfinite(l_est).all(1) & np.isfinite(l_true).all(1)
          & (np.linalg.norm(l_true, axis=1) > min_magnitude)
          & (np.linalg.norm(l_est, axis=1) > 0))
    if not ok.any():
        return float("nan")
    return float(np.median(np.linalg.norm(l_est[ok], axis=1) / np.linalg.norm(l_true[ok], axis=1)))


def lean_correlation(l_est: np.ndarray, l_true: np.ndarray,
                     min_magnitude: float = 0.0) -> dict:
    """Linear agreement between estimated and true lean vectors.

    Returns the two per-axis correlations, the component-wise R^2 of a
    through-origin linear fit ``l_true ~ beta l_est``, and the slope ``beta``.
    With a correct extractor ``beta = 1``; a global scale error multiplies it,
    and a time-base error with the true ``dt`` divides it.
    """
    l_est = np.asarray(l_est, dtype=np.float64).reshape(-1, 2)
    l_true = np.asarray(l_true, dtype=np.float64).reshape(-1, 2)
    ok = (np.isfinite(l_est).all(1) & np.isfinite(l_true).all(1)
          & (np.linalg.norm(l_true, axis=1) > min_magnitude))
    if ok.sum() < 3:
        return {"n": int(ok.sum()), "corr_x": float("nan"), "corr_y": float("nan"),
                "r2": float("nan"), "slope": float("nan")}
    e, t = l_est[ok], l_true[ok]
    corrx = float(np.corrcoef(e[:, 0], t[:, 0])[0, 1]) if e[:, 0].std() > 0 else float("nan")
    corry = float(np.corrcoef(e[:, 1], t[:, 1])[0, 1]) if e[:, 1].std() > 0 else float("nan")
    ef, tf = e.ravel(), t.ravel()
    denom = float(ef @ ef)
    slope = float(ef @ tf / denom) if denom > 0 else float("nan")
    resid = tf - slope * ef
    r2 = 1.0 - float(resid @ resid) / float(((tf - tf.mean()) ** 2).sum()) if len(tf) > 2 else float("nan")
    return {"n": int(ok.sum()), "corr_x": corrx, "corr_y": corry, "r2": r2, "slope": slope}


def required_tilt_deg(a: np.ndarray, v: np.ndarray | None = None, k_d: float = 0.0,
                      g: float = G0) -> np.ndarray:
    """Flatness tilt of a recovered path (drag-free by default, ``k_d`` in 1/s)."""
    return tilt_deg(a, v, k_d=k_d, g=g)
