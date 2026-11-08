"""Savitzky-Golay derivatives of a position track.

Separated from the experiment so the same differentiator is used by the mocap-only
floor, the ANWM excitation audit and the pixel stages. Window length and polynomial order are
explicit parameters: they trade noise against trajectory curvature, and the
choice must be recorded per run.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def check_window(n: int, window: int, polyorder: int) -> None:
    """Validate a Savitzky-Golay configuration against the available samples."""
    if polyorder < 1:
        raise ValueError(f"polyorder must be >= 1, got {polyorder}")
    if window % 2 != 1:
        raise ValueError(f"window must be odd, got {window}")
    if window < polyorder + 2:
        raise ValueError(f"window ({window}) must be >= polyorder + 2 ({polyorder + 2})")
    if window > n:
        raise ValueError(f"window ({window}) exceeds samples ({n})")


def edge_slice(n: int, window: int) -> slice:
    """Index range whose derivatives are not dominated by one-sided stencils."""
    half = window // 2
    if 2 * half + 1 > n:
        raise ValueError(f"window ({window}) leaves no interior samples for n={n}")
    return slice(half, n - half)


def savgol_derivatives(
    pos: np.ndarray, dt: float, window: int, polyorder: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (velocity, acceleration) in m/s and m/s^2 for an (N, 3) position track.

    ``dt`` is the sample interval in seconds. Passing the wrong ``dt`` scales
    velocity by 1/dt and acceleration by 1/dt^2, which is exactly the failure
    mode the sample-rate test guards against.
    """
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos must be (N, 3), got {pos.shape}")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt must be positive, got {dt!r}")
    check_window(len(pos), window, polyorder)
    vel = savgol_filter(pos, window, polyorder, deriv=1, delta=dt, axis=0, mode="interp")
    acc = savgol_filter(pos, window, polyorder, deriv=2, delta=dt, axis=0, mode="interp")
    return vel, acc
