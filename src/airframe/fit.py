"""Fitting and evaluation for the Track K airframe models.

Discipline implemented here:

* the lag nuisance is a pure time shift, searched on the pre-registered grid and fitted
  per unit (flight / sequence) on that unit's own truth;
* model parameters see only path kinematics and truth attitude, never a depicted lean,
  an edit, a threshold or a verdict outcome;
* the leave-one-flight-out wrapper fits every parameter without the held-out unit.

Residuals are reported both as an angle between thrust axes (the verdict's statistic)
and as a yaw-aligned lean-vector difference (the fitting objective).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import minimize

from src.physics.flatness import angle_between_deg
from src.verdict.lean import world_to_yaw_frame

from .kinematics import Kinematics
from .models import (axis_for, lean_of_axis, m5_features, parameter_names,
                     std_features)

LAG_GRID = np.arange(-0.150, 0.150 + 1e-9, 0.025)
STRIDE_FIT = 4


# --------------------------------------------------------------------------- utilities
def shift_kinematics(k: Kinematics, lag: float) -> Kinematics:
    """Path inputs sampled at ``t + lag``; the truth attitude stays at ``t``."""
    if abs(lag) < 1e-12:
        return k
    tq = k.t + lag
    ok = (tq >= k.t[0]) & (tq <= k.t[-1])

    def interp(x):
        out = np.column_stack([np.interp(tq, k.t, x[:, i], left=np.nan, right=np.nan)
                               for i in range(x.shape[1])])
        return out

    yaw_u = np.unwrap(k.yaw)
    yaw = np.interp(tq, k.t, yaw_u, left=np.nan, right=np.nan)
    return replace(k, v=interp(k.v), a=interp(k.a), j=interp(k.j), yaw=yaw,
                   valid=k.valid & ok)


def residual(k: Kinematics, model: str, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """(angle deg, lean-vector difference) of the model against the truth attitude."""
    u = axis_for(model, params, k)
    ang = angle_between_deg(u, k.z)
    lean = lean_of_axis(u, k.yaw_truth) - k.lean
    mask = k.valid & np.isfinite(ang) & np.isfinite(lean).all(axis=1)
    return np.where(mask, ang, np.nan), np.where(mask[:, None], lean, np.nan)


def fit_lag(k: Kinematics, model: str, params: dict) -> tuple[float, float]:
    """Best lag on the pre-registered grid and its median angle residual."""
    best = (np.inf, 0.0)
    for lag in LAG_GRID:
        ang, _ = residual(shift_kinematics(k, lag), model, params)
        m = np.isfinite(ang)
        if m.sum() < 12:
            continue
        loss = float(np.median(ang[m]))
        if loss < best[0]:
            best = (loss, float(lag))
    return best[1], best[0]


# --------------------------------------------------------------------------- fit
def _objective(units: list[Kinematics], model: str, params: dict, sub: list[np.ndarray]) -> float:
    tot = 0.0
    n = 0
    for k, idx in zip(units, sub):
        _, lean = residual(k, model, params)
        r = lean[idx]
        r = r[np.isfinite(r).all(axis=1)]
        tot += float(np.sum(r ** 2))
        n += len(r)
    return tot / max(n, 1)


def _bounds(model: str) -> list[tuple[float, float]]:
    base = [(0.0, 1.0), (0.0, 1.0)]
    if model == "M0":
        return []
    if model in ("M1", "M5"):
        return base
    if model == "M2":
        return base + [(-0.10, 0.20)]
    if model == "M3":
        return base + [(0.0, 0.10)]
    if model == "M4":
        return [(0.0, 1.0), (0.0, 1.0)]
    raise ValueError(model)


def _pack(model: str, p: dict) -> np.ndarray:
    return np.array([p[n] for n in parameter_names(model)], float)


def _unpack(model: str, x: np.ndarray) -> dict:
    return {n: float(v) for n, v in zip(parameter_names(model), x)}


def fit_parameters(units: list[Kinematics], model: str, lags: list[float],
                   init: dict | None = None, maxiter: int = 400) -> dict:
    """Fit one model's parameters on ``units`` with the given per-unit lags fixed."""
    if model == "M0":
        return {}
    shifted = [shift_kinematics(k, lag) for k, lag in zip(units, lags)]
    sub = [np.arange(0, len(k), STRIDE_FIT) for k in shifted]
    init = init or {}
    defaults = {"k_forward": 0.295, "k_lateral": 0.295, "tau": 0.0,
                "k_quadratic": 0.005, "d_x": 0.295, "d_y": 0.295}
    x0 = [init.get(n, defaults[n]) for n in parameter_names(model)]
    bounds = _bounds(model)
    res = minimize(lambda x: _objective(shifted, model, _unpack(model, x), sub),
                   np.asarray(x0, float), method="Powell", bounds=bounds,
                   options={"maxiter": maxiter, "xtol": 1e-4, "ftol": 1e-8})
    return _unpack(model, np.asarray(res.x, float))


def fit_model(units: list[Kinematics], model: str, init: dict | None = None,
              passes: int = 2) -> tuple[dict, list[float], dict]:
    """Alternate lag refits and parameter fits (``passes`` times)."""
    params = {} if model == "M0" else (init or fit_parameters(units, model, [0.0] * len(units)))
    lags = []
    for _ in range(passes):
        lags = [fit_lag(k, model, params)[0] for k in units]
        if model != "M0":
            params = fit_parameters(units, model, lags, init=params)
    lags = [fit_lag(k, model, params)[0] for k in units]
    info = {"lags_s": lags, "n_units": len(units), "model": model}
    return params, lags, info


# --------------------------------------------------------------------------- M5
def _ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    XtX = X.T @ X
    XtX[np.diag_indices_from(XtX)] += alpha * len(X)
    return np.linalg.solve(XtX, X.T @ Y)


def fit_m5(units: list[Kinematics], lags: list[float], base_params: dict,
           alphas=(0.01, 0.1, 1.0, 10.0, 100.0)) -> tuple[dict, dict]:
    """Ridge residual on the M1 model; alpha by inner leave-one-unit-out."""
    shifted = [shift_kinematics(k, lag) for k, lag in zip(units, lags)]
    X = [m5_features(k) for k in shifted]
    mu = np.vstack(X).mean(axis=0)
    sd = np.vstack(X).std(axis=0)
    sd[sd < 1e-9] = 1.0
    Xs = [(x - mu) / sd for x in X]
    Y = []
    for orig, k, x in zip(units, shifted, Xs):
        # target: truth lean minus M1's required lean, both in the TRUTH frame at t
        from .models import axis_M1
        u = axis_M1(k, base_params)
        Y.append(orig.lean - lean_of_axis(u, orig.yaw_truth))
    scores = {}
    for alpha in alphas:
        errs = []
        for i in range(len(units)):
            tr = [j for j in range(len(units)) if j != i]
            Xt = np.vstack([Xs[j] for j in tr])
            Yt = np.vstack([Y[j] for j in tr])
            ok = np.isfinite(Yt).all(axis=1)
            W = _ridge(Xt[ok], Yt[ok], alpha)
            r = Y[i] - Xs[i] @ W
            r = r[np.isfinite(r).all(axis=1)]
            errs.append(float(np.mean(np.degrees(r) ** 2)))
        scores[alpha] = float(np.mean(errs))
    alpha = min(scores, key=scores.get)
    Xt = np.vstack(Xs)
    Yt = np.vstack(Y)
    ok = np.isfinite(Yt).all(axis=1)
    W = _ridge(Xt[ok], Yt[ok], alpha)
    params = {**base_params, "W": W, "feat_mean": mu, "feat_sd": sd, "alpha": alpha,
              "alpha_scores": scores}
    return params, {"alpha": alpha, "alpha_scores": scores, "n_features": Xt.shape[1]}


# --------------------------------------------------------------------------- LOFO
def lofo(units: list[Kinematics], model: str, init: dict | None = None) -> dict:
    """Leave-one-unit-out fit; returns per-unit predictions and pooled metrics."""
    out = {"model": model, "units": []}
    for i, k in enumerate(units):
        train = [u for j, u in enumerate(units) if j != i]
        if model == "M5":
            base, lags, _ = fit_model(train, "M1", init=None)
            params, m5info = fit_m5(train, lags, base)
            params["m5_alpha"] = m5info["alpha"]
        else:
            params, lags, _ = fit_model(train, model, init=init)
        lag, _ = fit_lag(k, model, params)
        ang, lean = residual(shift_kinematics(k, lag), model, params)
        m = np.isfinite(ang)
        out["units"].append({
            "name": k.name, "n": int(m.sum()),
            "rms_deg": float(np.sqrt(np.nanmean(ang[m] ** 2))),
            "median_deg": float(np.nanmedian(ang[m])),
            "lean_rms_deg": float(np.degrees(np.sqrt(np.nanmean(
                (lean[m] ** 2).sum(axis=1))))),
            "lag_s": float(lag),
            "params": {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                       for kk, vv in params.items() if kk not in ("feat_mean", "feat_sd")},
        })
    angs = np.concatenate([np.array([u["rms_deg"]]) for u in out["units"]])
    out["pooled_rms_mean"] = float(np.mean(angs))
    out["n_units"] = len(units)
    weights = np.array([u["n"] for u in out["units"]], float)
    sq = np.array([u["rms_deg"] ** 2 for u in out["units"]])
    out["pooled_rms_frame_weighted"] = float(np.sqrt(np.sum(weights * sq) / np.sum(weights)))
    return out


def evaluate_fixed(units: list[Kinematics], model: str, params: dict) -> dict:
    """Evaluate fixed parameters on units, fitting only each unit's lag nuisance."""
    rows = []
    for k in units:
        lag, _ = fit_lag(k, model, params)
        ang, lean = residual(shift_kinematics(k, lag), model, params)
        m = np.isfinite(ang)
        rows.append({"name": k.name, "n": int(m.sum()),
                     "rms_deg": float(np.sqrt(np.nanmean(ang[m] ** 2))),
                     "median_deg": float(np.nanmedian(ang[m])),
                     "lean_rms_deg": float(np.degrees(np.sqrt(np.nanmean(
                         (lean[m] ** 2).sum(axis=1))))),
                     "lag_s": float(lag)})
    w = np.array([r["n"] for r in rows], float)
    sq = np.array([r["rms_deg"] ** 2 for r in rows])
    return {"model": model, "units": rows,
            "pooled_rms_mean": float(np.mean(sq) ** 0.5),
            "pooled_rms_frame_weighted": float(np.sqrt(np.sum(w * sq) / np.sum(w)))}
