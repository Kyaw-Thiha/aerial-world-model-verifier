"""Rigorous peak bounds from sampled positions (Track A2).

For any *continuous* trajectory p(t) through the sampled positions, with step dt and

    v_i = (p_{i+1} - p_i) / dt          the mean velocity of step i,

write A for the largest acceleration magnitude over the two-step window [t_i, t_{i+2}].
Since v_i is the mean of v(s) over step i,

    v_{i+1} - v_i = (1/dt) * integral_{t_i}^{t_{i+1}} [v(s + dt) - v(s)] ds
                  = (1/dt) * integral_{t_i}^{t_{i+1}} integral_s^{s+dt} a(tau) dtau ds,

so |v_{i+1} - v_i| <= dt * A and therefore

    A >= |v_{i+1} - v_i| / dt .

The version with 2 dt in the denominator, |v_{i+1} - v_i| / (2 dt), is therefore also valid
and is a factor two more conservative. Both are exposed here; the sharper one is primary.

This is a bound on *acceleration*, and on thrust through the triangle inequality. It is NOT
a bound on tilt on its own: a large acceleration pointing straight up produces little tilt,
so a tilt lower bound needs an explicit cap on the vertical acceleration an airframe can
produce (``tilt_lower_bound_deg`` takes that cap as an argument rather than fitting it).

Everything assumes a continuous realization. The commanded ANWM paths are teleports: under a
piecewise-constant velocity reading the acceleration is impulsive and unbounded, so these
bounds describe the *smoothest continuous flight* through the sampled points, which is the
only reading under which "required acceleration" is meaningful.
"""
from __future__ import annotations

import numpy as np

from src.physics.flatness import G0


def step_velocities(pos: np.ndarray, dt: float) -> np.ndarray:
    """Mean velocity of each step, (n-1, 3)."""
    p = np.asarray(pos, dtype=np.float64)
    return np.diff(p, axis=0) / float(dt)


def velocity_changes(pos: np.ndarray, dt: float) -> np.ndarray:
    """``v_{i+1} - v_i`` for each interior pair, (n-2, 3)."""
    v = step_velocities(pos, dt)
    return np.diff(v, axis=0)


def peak_accel_lower_bound(pos: np.ndarray, dt: float) -> np.ndarray:
    """Rigorous lower bound on the peak |acceleration| in each two-step window."""
    return np.linalg.norm(velocity_changes(pos, dt), axis=1) / float(dt)


def peak_accel_lower_bound_conservative(pos: np.ndarray, dt: float) -> np.ndarray:
    """The same bound with the factor two stated by hand: |dv| / (2 dt). Always weaker."""
    return np.linalg.norm(velocity_changes(pos, dt), axis=1) / (2.0 * float(dt))


def peak_horizontal_accel_lower_bound(pos: np.ndarray, dt: float) -> np.ndarray:
    """Same argument applied to the horizontal components: a bound on peak |a_h|."""
    d = velocity_changes(pos, dt)[:, :2]
    return np.linalg.norm(d, axis=1) / float(dt)


def tilt_lower_bound_deg(peak_a_h: np.ndarray, max_vertical_accel: float,
                         g: float = G0) -> np.ndarray:
    """Tilt required at that instant if the vertical acceleration is at most ``max_vertical_accel``.

    ``tilt = atan2(|u_h|, u_z)`` with ``u = a + g e3``, so ``u_z <= max_vertical_accel + g``
    and ``u_h >= peak_a_h`` (drag-free). If the vertical acceleration were unbounded the
    bound would be zero, which is why the cap is an explicit argument and not a fitted value.
    """
    denom = float(max_vertical_accel) + g
    return np.degrees(np.arctan2(np.asarray(peak_a_h, dtype=np.float64), denom))


def thrust_ratio_lower_bound(pos: np.ndarray, dt: float, k_d: float = 0.0) -> np.ndarray:
    """Rigorous lower bound on |u|/g, i.e. thrust-to-weight, in each two-step window.

    ``u = a + g e3 + k_d v`` and ``u_h = a_h + k_d v_h``, so the horizontal part alone
    already gives ``|u| >= |u_h| >= |a_h| - k_d |v_h| >= |dv_h| / dt - k_d |v_h|``. Gravity
    drops out because this bounds the horizontal component (a conservative reading: the
    vertical component can only add to |u|).
    """
    v = step_velocities(pos, dt)
    d = velocity_changes(pos, dt)
    a_h = np.linalg.norm(d[:, :2], axis=1) / float(dt)
    v_h = np.linalg.norm(0.5 * (v[:-1, :2] + v[1:, :2]), axis=1)
    return np.maximum(a_h - float(k_d) * v_h, 0.0) / G0


def summarise_bounds(pos: np.ndarray, dt: float, k_d: float = 0.0,
                     max_vertical_accel: float = 2.0 * G0) -> dict:
    """Peak bounds for one path, as used by the ANWM analysis."""
    peak = peak_accel_lower_bound(pos, dt)
    horiz = peak_horizontal_accel_lower_bound(pos, dt)
    tilt = tilt_lower_bound_deg(horiz, max_vertical_accel)
    twr = thrust_ratio_lower_bound(pos, dt, k_d)
    return {
        "n_pairs": int(len(peak)),
        "peak_accel_lower_median": float(np.median(peak)) if len(peak) else None,
        "peak_accel_lower_max": float(np.max(peak)) if len(peak) else None,
        "peak_horizontal_accel_lower_max": float(np.max(horiz)) if len(horiz) else None,
        "tilt_lower_bound_median_deg": float(np.median(tilt)) if len(tilt) else None,
        "tilt_lower_bound_max_deg": float(np.max(tilt)) if len(tilt) else None,
        "thrust_ratio_lower_max": float(np.max(twr)) if len(twr) else None,
        "max_vertical_accel_assumed": float(max_vertical_accel),
    }
