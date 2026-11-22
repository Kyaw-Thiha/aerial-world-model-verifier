"""Candidate airframe models M0-M5 (Track K, Part 2).

Each model returns the unit thrust axis implied by the path under that model.  The
parameter names and bounds are the pre-registered ones (``configs/k02_model_prereg.yaml``).
Nothing here uses the depicted side, the camera, a mount or an edit.
"""
from __future__ import annotations

import numpy as np

from src.depicted.clips import up_body_from_lean
from src.depicted.stat_a2 import depicted_axis_from_up_body
from src.physics.flatness import G0
from src.verdict.lean import world_to_yaw_frame

K_D_SCHEMA = 0.295
MODELS = ("M0", "M1", "M2", "M3", "M4", "M5")


def unit(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, float)
    n = np.linalg.norm(u, axis=1, keepdims=True)
    bad = np.isfinite(n[:, 0]) & (n[:, 0] < 1e-9)
    if bad.any():
        raise ValueError("degenerate required axis")
    n = np.where(np.isfinite(n) & (n > 1e-9), n, 1.0)
    return u / n


def heading_components(vec: np.ndarray, yaw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World horizontal (x, y) -> heading (forward, left)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return c * vec[:, 0] + s * vec[:, 1], -s * vec[:, 0] + c * vec[:, 1]


def from_heading_components(f: np.ndarray, l: np.ndarray,
                            yaw: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.column_stack([c * f - s * l, s * f + c * l])


def lean_of_axis(axis: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    return world_to_yaw_frame(axis[:, :2], yaw)


def axis_of_lean(lean: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    return depicted_axis_from_up_body(up_body_from_lean(lean), yaw)


# --------------------------------------------------------------------------- M0/M1/M3
def axis_M0(k, p: dict | None = None) -> np.ndarray:
    return unit(k.a + np.array([0.0, 0.0, G0]) + K_D_SCHEMA * k.v)


def _linear_axis(k, k_forward: float, k_lateral: float,
                 k_quadratic: float = 0.0) -> np.ndarray:
    vf, vl = heading_components(k.v, k.yaw)
    speed = np.linalg.norm(k.v[:, :2], axis=1)
    kf = k_forward + k_quadratic * speed
    kl = k_lateral + k_quadratic * speed
    d = from_heading_components(kf * vf, kl * vl, k.yaw)
    u = k.a + np.array([0.0, 0.0, G0])
    u = u + np.column_stack([d[:, 0], d[:, 1], K_D_SCHEMA * k.v[:, 2]])
    return unit(u)


def axis_M1(k, p: dict) -> np.ndarray:
    return _linear_axis(k, p["k_forward"], p["k_lateral"])


def axis_M3(k, p: dict) -> np.ndarray:
    return _linear_axis(k, p["k_forward"], p["k_lateral"], p["k_quadratic"])


# --------------------------------------------------------------------------- M2
def lowpass_axis(u: np.ndarray, tau: float, dt: float) -> np.ndarray:
    """First-order lag on the unit axis, renormalised each step.

    ``tau > 0`` is the causal lag ``tau du/dt + u = u_in``; ``tau < 0`` is the same
    filter run backwards, i.e. a lead, which is what lets the data reject the lag
    direction instead of being forced into it.  ``|dt/tau| < 0.04`` on this grid.
    """
    if abs(tau) < 1e-9:
        return u
    from scipy.signal import lfilter
    alpha = dt / abs(tau)
    b, a = [alpha], [1.0, alpha - 1.0]
    out = np.full_like(u, np.nan)
    finite = np.isfinite(u).all(axis=1)
    # filter each contiguous finite run on its own, so a masked edge cannot poison the
    # whole series through the recursion
    edges = np.flatnonzero(np.diff(np.r_[0, finite.astype(int), 0]))
    for lo, hi in zip(edges[::2], edges[1::2]):
        seg = u[lo:hi]
        if tau > 0:
            out[lo:hi] = lfilter(b, a, seg, axis=0)
        else:
            out[lo:hi] = lfilter(b, a, seg[::-1], axis=0)[::-1]
    return unit(out)


def axis_M2(k, p: dict, dt: float | None = None) -> np.ndarray:
    base = _linear_axis(k, p["k_forward"], p["k_lateral"])
    return lowpass_axis(base, float(p["tau"]), k.dt if dt is None else dt)


# --------------------------------------------------------------------------- M4
def _R_from_yaw_axis(yaw: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Body->world rotation with the given heading and body z axis (zero twist in heading)."""
    c, s = np.cos(yaw), np.sin(yaw)
    ux = c * axis[:, 0] + s * axis[:, 1]
    uy = -s * axis[:, 0] + c * axis[:, 1]
    uz = axis[:, 2]
    # Rodrigues rotation from e3 to (ux, uy, uz), then Rz(yaw).
    vx = np.cross(np.tile([0.0, 0.0, 1.0], (len(axis), 1)),
                  np.column_stack([ux, uy, uz]))
    cs = uz
    vv = np.sum(vx * vx, axis=1)
    K = np.zeros((len(axis), 3, 3))
    K[:, 0, 1] = -vx[:, 2]
    K[:, 0, 2] = vx[:, 1]
    K[:, 1, 0] = vx[:, 2]
    K[:, 1, 2] = -vx[:, 0]
    K[:, 2, 0] = -vx[:, 1]
    K[:, 2, 1] = vx[:, 0]
    fac = np.where(vv > 1e-12, 1.0 / (1.0 + cs), 0.0)
    Rt = np.eye(3)[None] + K + fac[:, None, None] * (K @ K)
    Rz = np.zeros((len(axis), 3, 3))
    Rz[:, 0, 0] = c
    Rz[:, 0, 1] = -s
    Rz[:, 1, 0] = s
    Rz[:, 1, 1] = c
    Rz[:, 2, 2] = 1.0
    return Rz @ Rt


def axis_M4(k, p: dict, iters: int = 12) -> np.ndarray:
    """Body-frame rotor drag, implicit in the attitude (Faessler et al., RA-L 2018)."""
    u = axis_M0(k)
    d = np.array([p["d_x"], p["d_y"], 0.0])
    for _ in range(iters):
        R = _R_from_yaw_axis(k.yaw, u)
        v_body = np.einsum("nji,nj->ni", R, k.v)
        drag = np.einsum("nij,nj->ni", R, d * v_body)
        u = unit(k.a + np.array([0.0, 0.0, G0]) + drag)
    return u


# --------------------------------------------------------------------------- M5
M5_FEATURES = ["1", "v_f", "v_l", "a_f", "a_l", "j_f", "j_l", "|v|^2",
               "v_f|v|", "v_l|v|", "a.v"]


def m5_features(k) -> np.ndarray:
    vf, vl = heading_components(k.v, k.yaw)
    af, al = heading_components(k.a, k.yaw)
    jf, jl = heading_components(k.j, k.yaw)
    vf, vl, af, al, jf, jl = (x / G0 for x in (vf, vl, af, al, jf, jl))
    jf, jl = 0.1 * jf, 0.1 * jl
    speed = np.hypot(vf, vl)
    return np.column_stack([np.ones(len(vf)), vf, vl, af, al, jf, jl,
                            vf ** 2 + vl ** 2, vf * speed, vl * speed,
                            af * vf + al * vl])


def std_features(k, p: dict) -> np.ndarray:
    return (m5_features(k) - p["feat_mean"]) / p["feat_sd"]


def m5_residual(k, p: dict) -> np.ndarray:
    return std_features(k, p) @ p["W"]


def axis_M5(k, p: dict) -> np.ndarray:
    base = _linear_axis(k, p["k_forward"], p["k_lateral"])
    lean = lean_of_axis(base, k.yaw) + m5_residual(k, p)
    return axis_of_lean(lean, k.yaw)


# --------------------------------------------------------------------------- dispatch
def axis_for(model: str, params: dict, k) -> np.ndarray:
    if model == "M0":
        return axis_M0(k)
    if model == "M1":
        return axis_M1(k, params)
    if model == "M2":
        return axis_M2(k, params)
    if model == "M3":
        return axis_M3(k, params)
    if model == "M4":
        return axis_M4(k, params)
    if model == "M5":
        return axis_M5(k, params)
    raise ValueError(model)


def parameter_names(model: str) -> tuple[str, ...]:
    return {
        "M0": (),
        "M1": ("k_forward", "k_lateral"),
        "M2": ("k_forward", "k_lateral", "tau"),
        "M3": ("k_forward", "k_lateral", "k_quadratic"),
        "M4": ("d_x", "d_y"),
        "M5": ("k_forward", "k_lateral"),
    }[model]
