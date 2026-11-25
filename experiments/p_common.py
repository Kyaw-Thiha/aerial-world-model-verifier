"""Shared rate-consistency analysis for Track P (P0, P1, P2)."""
from __future__ import annotations

import numpy as np

from experiments.p_units import attitude_from_axis_and_yaw
from src.verdict.rate import (K_D, axis_rate, central_axis_rate, depicted_axis_rate,
                              depicted_axis_rate_literal, gauge_fit, residuals,
                              rotation_rates, thrust_axis, uniform_path, window_scores)

E3 = np.array([0.0, 0.0, 1.0])
WINDOWS = (0.25, 0.5, 1.0)


def path_side(unit: dict, filter_window_s: float | None = None) -> dict:
    """100 Hz path kinematics, required thrust axis, and its Savitzky-Golay rate."""
    tg, vel, acc, interior = uniform_path(unit["t"], unit["pos"])
    b3 = thrust_axis(acc, vel)
    if filter_window_s is None:
        rb = axis_rate(tg, b3)
    else:
        rb = axis_rate(tg, b3, window_s=filter_window_s)
    return {"tg": tg, "vel": vel, "acc": acc, "b3": b3, "rb": rb, "interior": interior}


def sample_at(tg: np.ndarray, arr: np.ndarray, tq: np.ndarray) -> np.ndarray:
    tq = np.asarray(tq, float)
    out = np.column_stack([np.interp(tq, tg, arr[:, k]) for k in range(arr.shape[1])])
    return out


def video_side(unit: dict, path: dict, attitude: np.ndarray | None = None,
               t_override: np.ndarray | None = None) -> dict:
    """Frame-grid required axis, and the depicted rate for the given attitude sequence."""
    tq = np.asarray(unit["t_video"] if t_override is None else t_override, float)
    inside = (tq >= path["tg"][0]) & (tq <= path["tg"][-1])
    b3 = sample_at(path["tg"], path["b3"], tq)
    n = np.linalg.norm(b3, axis=1, keepdims=True)
    b3 = np.where(n > 1e-9, b3 / np.maximum(n, 1e-12), np.nan)
    R_wc = np.asarray(unit["R_est"] if attitude is None else attitude, float)[: len(tq)]
    g = gauge_fit(R_wc, b3, unit["R_cb"])
    omega = rotation_rates(R_wc, tq)
    rd = depicted_axis_rate(omega, R_wc, unit["R_cb"], g["G"])
    rb = central_axis_rate(b3, tq)
    res = residuals(rb, rd)
    res["t"] = tq
    res["inside"] = inside
    res["err_deg_s"][~inside] = np.nan
    res["cos"][~inside] = np.nan
    res["req_deg_s"][~inside] = np.nan
    return {"res": res, "gauge": g, "b3": b3, "R_wc": R_wc}


def literal_side(unit: dict, path: dict, attitude: np.ndarray) -> dict:
    """The handoff-literal form, only used to quantify its yaw-gauge error."""
    tq = np.asarray(unit["t_video"], float)
    b3 = sample_at(path["tg"], path["b3"], tq)
    n = np.linalg.norm(b3, axis=1, keepdims=True)
    b3 = b3 / np.maximum(n, 1e-12)
    R_wc = np.asarray(attitude, float)[: len(tq)]
    omega = rotation_rates(R_wc, tq)
    rd = depicted_axis_rate_literal(omega, b3, unit["R_cb"])
    rb = central_axis_rate(b3, tq)
    res = residuals(rb, rd)
    res["t"] = tq
    return res


def null_attitude(unit: dict, path: dict) -> np.ndarray:
    """Feasible camera attitude through the mount: flatness body attitude at the frame times."""
    tq = np.asarray(unit["t_video"], float)
    b3 = sample_at(path["tg"], path["b3"], tq)
    b3 = b3 / np.linalg.norm(b3, axis=1, keepdims=True)
    if unit["yaw"] is None:
        yaw = np.zeros(len(tq))
    else:
        yaw = np.interp(tq, unit["t"], unit["yaw"])
    R_wb = attitude_from_axis_and_yaw(b3, yaw)
    return np.einsum("nij,jk->nik", R_wb, unit["R_cb"])


def window_stats(res: dict, seconds: float, min_req_deg_s: float) -> list[dict]:
    return [w for w in window_scores(res, seconds, min_req_deg_s=min_req_deg_s) if w["testable"]]


def clip_score(rows: list[dict], key: str = "abs_deg_s") -> float:
    if not rows:
        return float("nan")
    v = np.asarray([r[key] for r in rows], float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else float("nan")


def quantile(v, q):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, q)) if len(v) else float("nan")
