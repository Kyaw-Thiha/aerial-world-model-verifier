"""Track J, Stage 1c: learned reader confidence and its effect on the window statistic.

Pre-registered in ``configs/j04_learned_confidence_prereg.yaml``, written before any fit.
CPU only, stored reader outputs, zero spend.

Usage
-----
PYTHONPATH=. python experiments/j04_learned_confidence.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.depicted.anatomy import axis_errors, contiguous_runs, load_flight
from src.verdict.schema import load_clip
from src.verdict.segments import frame_residual_deg

TII = [
    "flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
    "flight-07p-lemniscate", "flight-13a-trackRATM", "flight-13p-trackRATM", "flight-16p-trackRATM",
]
DEV = ["flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse", "flight-07p-lemniscate",
       "flight-13a-trackRATM", "flight-16p-trackRATM"]
HELDOUT = ["flight-13p-trackRATM", "flight-07a-lemniscate"]
OUT = Path("results/j04_confidence")
WINDOW_S = 0.5
WIN, HOP = 8, 4


# --------------------------------------------------------------------------- features
RAW = ["gravity_uncertainty_raw", "roll_uncertainty_raw", "pitch_uncertainty_raw",
       "up_confidence_mean", "up_confidence_p10", "latitude_confidence_mean",
       "latitude_confidence_p10", "valid_frac"]


def load_features(flight: str) -> dict:
    tab = load_flight(flight)
    rows = [json.loads(l) for l in
            (Path("results/gate_a2/tii") / f"{flight}__identity.jsonl").read_text().splitlines()
            if l.strip()]
    by_name = {r["frame"]: r for r in rows}
    raw = np.array([[float(by_name[n].get(k, np.nan)) for k in RAW] for n in tab.names])
    err = axis_errors(tab, "frozen")
    t = tab.t
    # 0.5 s centred aggregates, computed inside contiguous runs only
    n = len(t)
    agg = np.zeros((n, len(RAW) * 5))
    for a, b in contiguous_runs(t):
        for i in range(a, b):
            lo = np.searchsorted(t, t[i] - WINDOW_S / 2, "left")
            hi = np.searchsorted(t, t[i] + WINDOW_S / 2, "right")
            blk = raw[lo:hi]
            for k in range(len(RAW)):
                v = blk[:, k]
                v = v[np.isfinite(v)]
                if len(v) == 0:
                    continue
                agg[i, k * 5:(k + 1) * 5] = [np.median(v), np.max(v), np.min(v), np.std(v),
                                            raw[i, k] - np.median(v)]
    geo = np.column_stack([tab.tilt_deg, tab.speed, np.cos(tab.yaw), np.sin(tab.yaw),
                           np.abs(err["pitch"]), np.abs(err["roll"])])  # last two never used
    X = np.column_stack([raw, agg, geo[:, :4]])
    names = (RAW + [f"{k}_{a}" for k in RAW for a in
                    ("med", "max", "min", "std", "dev")] +
             ["tilt_deg", "speed_mps", "cos_yaw", "sin_yaw"])
    return {"flight": flight, "X": X, "names": names, "theta": err["theta"], "t": t,
            "s3": raw[:, RAW.index("latitude_confidence_mean")], "clip_index": tab.clip_index}


# --------------------------------------------------------------------------- model
def auc(score: np.ndarray, label: np.ndarray) -> float:
    ok = np.isfinite(score) & np.isfinite(label)
    s, y = score[ok], label[ok]
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def fit_logistic(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 50) -> np.ndarray:
    """L2 logistic regression by Newton / IRLS (numpy only). Converges in ~10 steps."""
    n, k = X.shape
    w = np.zeros(k)
    for _ in range(iters):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = X.T @ (p - y) / n + lam * w
        W = p * (1 - p)
        H = (X * W[:, None]).T @ X / n + lam * np.eye(k)
        step = np.linalg.solve(H + 1e-9 * np.eye(k), g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    k = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(k), X.T @ y)


def standardise(Xfit: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    mu = np.nanmean(Xfit, axis=0)
    sd = np.nanstd(Xfit, axis=0)
    sd[sd < 1e-9] = 1.0
    out = [(Xfit - mu) / sd]
    out += [(X - mu) / sd for X in others]
    return [np.nan_to_num(X, nan=0.0) for X in out]


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    q[-1] += 1e-9
    out = []
    for i in range(bins):
        m = (p >= q[i]) & (p < q[i + 1])
        if m.sum() == 0:
            continue
        out.append({"p_mean": float(p[m].mean()), "obs": float(y[m].mean()), "n": int(m.sum())})
    return out


# --------------------------------------------------------------------------- downstream
def wmedian(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return float(np.median(x))
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def windows_of(resid: np.ndarray, length: int, hop: int) -> list[tuple[int, int]]:
    n = len(resid)
    if n <= length:
        return [(0, n)]
    starts = list(range(0, n - length + 1, hop))
    return [(s, s + length) for s in starts]


def clip_stats(lean_dep: np.ndarray, lean_req: np.ndarray, yaw: np.ndarray,
               w: np.ndarray | None, mask: np.ndarray | None) -> float:
    """Max over complete windows of the (optionally weighted/masked) window-median residual."""
    resid = frame_residual_deg(lean_dep, lean_req, yaw)
    vals = []
    for lo, hi in windows_of(resid, WIN, HOP):
        r = resid[lo:hi]
        ww = None if w is None else w[lo:hi]
        keep = np.isfinite(r)
        if mask is not None:
            keep &= mask[lo:hi]
        if keep.sum() < 6:
            continue
        vals.append(wmedian(r[keep], ww[keep]) if ww is not None else float(np.median(r[keep])))
    return float(max(vals)) if vals else float("nan")


def compose_partial(real: np.ndarray, edit: np.ndarray, frac: float) -> np.ndarray:
    n = len(real)
    m = max(1, int(round(frac * n)))
    a = (n - m) // 2
    out = real.copy()
    out[a:a + m] = edit[a:a + m]
    return out


def load_clip_set(flights: list[str]) -> list[dict]:
    out = []
    for f in flights:
        for d in sorted((Path("data/processed/track_d/clips") / f).glob("*")):
            cond = d.name.split("__")[-1]
            c = load_clip(d / "verdict_clip.npz")
            out.append({"flight": f, "clip": d.name, "cond": cond,
                        "dep": np.asarray(c.depicted_lean, float),
                        "req": np.asarray(c.lean_req, float),
                        "yaw": np.asarray(c.yaw, float), "t": np.asarray(c.t, float)})
    return out


# --------------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {f: load_features(f) for f in TII}
    flights = np.concatenate([[f] * len(data[f]["theta"]) for f in TII])

    # ---------- per-frame prediction, leave-one-flight-out
    folds = []
    pred = np.full(len(flights), np.nan)
    pred_s3 = np.full(len(flights), np.nan)
    pred_geo = np.full(len(flights), np.nan)
    offset = 0
    starts = {}
    for f in TII:
        starts[f] = offset
        offset += len(data[f]["theta"])
    y_all = np.concatenate([(data[f]["theta"] > 10.0).astype(float) for f in TII])
    for f in TII:
        fit = np.array([u for u in TII if u != f])
        Xf = np.concatenate([data[u]["X"] for u in fit])
        yf = np.concatenate([(data[u]["theta"] > 10.0).astype(float) for u in fit])
        Xh = data[f]["X"]
        Xf_s, Xh_s = standardise(Xf, Xh)
        # lambda by fit-fold log-loss over the same grid on held-in data (no leakage)
        best, best_lam = np.inf, None
        for lam in (1e-3, 1e-2, 1e-1):
            w = fit_logistic(Xf_s, yf, lam)
            z = Xf_s @ w
            p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
            ll = -np.mean(yf * np.log(p + 1e-12) + (1 - yf) * np.log(1 - p + 1e-12))
            if ll < best:
                best, best_lam = ll, lam
        w = fit_logistic(Xf_s, yf, best_lam)
        sl = starts[f]
        pred[sl:sl + len(Xh)] = 1 / (1 + np.exp(-np.clip(Xh_s @ w, -30, 30)))
        # physics-only baseline: tilt + speed
        idx = [data[f]["names"].index(k) for k in ("tilt_deg", "speed_mps")]
        Xg = Xf[:, idx]
        Xg_s, Xh_g = standardise(Xg, Xh[:, idx])
        wg = fit_logistic(Xg_s, yf, 1e-2)
        pred_geo[sl:sl + len(Xh)] = 1 / (1 + np.exp(-np.clip(Xh_g @ wg, -30, 30)))
        # S3 baseline: a one-feature logistic on the frozen signal, fitted on the fit flights
        # (the frozen signal is -median(latitude_conf), so lower confidence = more suspicious)
        s3_f = np.column_stack([-np.concatenate([data[u]["s3"] for u in fit])])
        s3_h = np.column_stack([-data[f]["s3"]])
        s3_f_s, s3_h_s = standardise(s3_f, s3_h)
        ws3 = fit_logistic(s3_f_s, yf, 1e-3)
        pred_s3[sl:sl + len(Xh)] = 1 / (1 + np.exp(-np.clip(s3_h_s @ ws3, -30, 30)))
        yh = (data[f]["theta"] > 10.0).astype(float)
        folds.append({"held_out": f, "n": int(len(Xh)), "n_bad_held_out": int(yh.sum()),
                      "lambda": best_lam,
                      "auroc": auc(pred[sl:sl + len(Xh)], y_all[sl:sl + len(Xh)]),
                      "auroc_s3": auc(pred_s3[sl:sl + len(Xh)], y_all[sl:sl + len(Xh)]),
                      "auroc_geo": auc(pred_geo[sl:sl + len(Xh)], y_all[sl:sl + len(Xh)])})

    ok = np.isfinite(pred)
    p = np.clip(pred[ok], 1e-6, 1 - 1e-6)
    y = y_all[ok]
    # recalibration: fit logit(y) ~ a + b * logit(p_held_out) with logistic regression, no
    # penalty. For a calibrated model b = 1 and a = 0. (An OLS slope of the binary outcome on
    # the logit would instead estimate E[p(1-p)] ~ 0.11 and is NOT the calibration slope.)
    logit = np.log(p / (1 - p))
    A = np.column_stack([np.ones_like(logit), logit])
    coef = fit_logistic(A, y, 1e-9)
    # flight-clustered bootstrap of the pooled AUROC
    g = flights[ok]
    uniq = np.unique(g)
    rng = np.random.default_rng(1)
    draws = []
    for _ in range(5000):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(g == q) for q in pick])
        draws.append(auc(pred[ok][idx], y[idx]))
    draws = np.array([d for d in draws if np.isfinite(d)])
    pooled = {
        "auroc_learned": auc(pred[ok], y),
        "auroc_ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "n_boot": int(len(draws)), "frac_usable": float(len(draws) / 5000),
        "auroc_s3_framewise": auc(pred_s3[ok], y),
        "auroc_physics_only": auc(pred_geo[ok], y),
        "brier_learned": float(np.mean((p - y) ** 2)),
        "brier_s3_learned": float(np.mean((np.clip(pred_s3[ok], 1e-6, 1 - 1e-6) - y) ** 2)),
        "calibration_slope": float(coef[1]), "calibration_intercept": float(coef[0]),
        "prevalence": float(y.mean()), "n_frames": int(ok.sum()),
        "reliability": reliability(p, y),
    }
    # clip-level AUROC of the clip-mean probability, the unit of the published 0.946
    clip_rows = []
    for f in TII:
        idx = np.arange(starts[f], starts[f] + len(data[f]["theta"]))
        for c in np.unique(data[f]["clip_index"]):
            m = idx[data[f]["clip_index"] == c]
            if len(m) == 0:
                continue
            clip_rows.append({"flight": f, "clip": str(c),
                              "p_mean": float(np.nanmean(pred[m])),
                              "err_median": float(np.nanmedian(data[f]["theta"][
                                  data[f]["clip_index"] == c]))})
    clip_label = np.array([r["err_median"] > 10.0 for r in clip_rows], float)
    clip_score = np.array([r["p_mean"] for r in clip_rows])
    pooled["clip_level"] = {"n": len(clip_rows), "n_positive": int(clip_label.sum()),
                            "auroc": auc(clip_score, clip_label)}

    (OUT / "per_frame.json").write_text(json.dumps(
        {"folds": folds, "pooled": pooled, "features": data[TII[0]]["names"]}, indent=1))

    # ---------- downstream: the weighted window statistic on the dev block
    clips = load_clip_set(TII)
    real = [c for c in clips if c["cond"] == "real"]
    by_flight = {}
    for c in clips:
        by_flight.setdefault((c["flight"], c["clip"].split("__")[0]), {})[c["cond"]] = c

    # per-clip frame weights from the learned probability, aligned by time
    pmap = {}
    for f in TII:
        t = data[f]["t"]
        pmap[f] = (t, pred[starts[f]:starts[f] + len(t)])

    rows = []
    for key, conds in by_flight.items():
        c0 = conds.get("real")
        if c0 is None:
            continue
        f = c0["flight"]
        t_ref, p_ref = pmap[f]
        idx = np.searchsorted(t_ref, c0["t"]).clip(0, len(t_ref) - 1)
        w = np.clip(1.0 - p_ref[idx], 0.1, 1.0)
        mask = p_ref[idx] <= np.nanpercentile(p_ref, 90)
        for cond in ("real", "tilt_removal", "mirror_roll", "yaw_control", "null_edit"):
            if cond not in conds:
                continue
            ce = conds[cond]
            fracs = (1.0,) if cond == "real" else (1.0, 0.5, 0.25)
            for frac in fracs:
                dep = compose_partial(c0["dep"], ce["dep"], frac) if frac < 1 else ce["dep"]
                lab = "full" if frac == 1 else f"partial_{int(frac * 100)}"
                for name, (ww, mm) in {"plain": (None, None), "weighted": (w, None),
                                       "masked": (None, mask),
                                       "weighted_masked": (w, mask)}.items():
                    s = clip_stats(dep, c0["req"], c0["yaw"], ww, mm)
                    rows.append({"clip": c0["clip"], "flight": f, "cond": cond,
                                 "coverage": lab, "arm": name, "stat": s})

    # LOFO thresholds on the real clips, per arm
    out = {}
    real_rows = [r for r in rows if r["cond"] == "real"]
    for arm in ("plain", "weighted", "masked", "weighted_masked"):
        det = {}
        for cov in ("partial_25", "partial_50", "full"):
            for cond in ("tilt_removal", "mirror_roll", "yaw_control", "null_edit"):
                hits = 0
                tot = 0
                for r in rows:
                    if r["arm"] != arm or r["coverage"] != cov or r["cond"] != cond:
                        continue
                    if r["flight"] in HELDOUT:
                        continue
                    fit = np.array([x["stat"] for x in real_rows
                                    if x["arm"] == arm and x["flight"] != r["flight"]
                                    and x["flight"] not in HELDOUT])
                    fit = fit[np.isfinite(fit)]
                    tau = float(np.quantile(fit, 0.95))
                    hits += float(r["stat"] > tau)
                    tot += 1
                det[f"{cond}/{cov}"] = f"{int(hits)}/{tot}"
        real_frr = 0
        real_n = 0
        for r in real_rows:
            if r["arm"] != arm or r["flight"] in HELDOUT:
                continue
            fit = np.array([x["stat"] for x in real_rows if x["arm"] == arm
                            and x["flight"] != r["flight"] and x["flight"] not in HELDOUT])
            fit = fit[np.isfinite(fit)]
            real_frr += float(r["stat"] > float(np.quantile(fit, 0.95)))
            real_n += 1
        out[arm] = {"real_false_reject": f"{int(real_frr)}/{real_n}", "detection": det}
    (OUT / "downstream.json").write_text(json.dumps(out, indent=1))
    (OUT / "rows.csv").write_text(
        "clip,flight,cond,coverage,arm,stat\n" +
        "\n".join(f"{r['clip']},{r['flight']},{r['cond']},{r['coverage']},{r['arm']},"
                  f"{r['stat']:.4f}" for r in rows) + "\n")

    print(json.dumps({k: v for k, v in pooled.items() if k != "reliability"}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
