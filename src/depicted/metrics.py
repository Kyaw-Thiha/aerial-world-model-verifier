"""Error summaries used by every Track E table.

Two conventions matter and the project has been inconsistent about one of them: TII's
published "trimmed RMSE 4.80 deg" drops the worst **10%** of frames
(`experiments/phase1_depicted.py`: `np.sort(err)[:int(0.9 * len(err))]`), while the handoff
describes it as "drop the worst 5%". Both are computed here and every table says which.
"""
from __future__ import annotations

import numpy as np

SPEED_BINS = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0), (20.0, 1e9)]


def trimmed_rmse(err: np.ndarray, drop: float) -> float:
    """RMSE after dropping the worst ``drop`` fraction of the largest errors."""
    e = np.sort(np.asarray(err, dtype=np.float64))
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return float("nan")
    k = int(round(drop * len(e)))
    keep = e[:len(e) - k] if k > 0 else e
    return float(np.sqrt(np.mean(keep ** 2)))


def summarize(err: np.ndarray) -> dict:
    """Central, tail and bios of one error sample; NaN-safe; n reported always."""
    e = np.asarray(err, dtype=np.float64)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return {"n": 0, "median_deg": None, "trimmed_rmse_5pct_deg": None,
                "trimmed_rmse_10pct_deg": None, "rmse_deg": None, "p90_deg": None,
                "p99_deg": None, "frac_over_10_deg": None, "frac_over_20_deg": None}
    return {
        "n": int(len(e)),
        "median_deg": float(np.median(e)),
        "mean_deg": float(np.mean(e)),
        "trimmed_rmse_5pct_deg": trimmed_rmse(e, 0.05),
        "trimmed_rmse_10pct_deg": trimmed_rmse(e, 0.10),
        "rmse_deg": float(np.sqrt(np.mean(e ** 2))),
        "p75_deg": float(np.percentile(e, 75)),
        "p90_deg": float(np.percentile(e, 90)),
        "p99_deg": float(np.percentile(e, 99)),
        "frac_over_10_deg": float(np.mean(e > 10.0)),
        "frac_over_20_deg": float(np.mean(e > 20.0)),
    }


def angle_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle between unit vectors (rows), degrees; NaN-safe."""
    u = np.asarray(u, dtype=np.float64).reshape(-1, 3)
    v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
    nu = np.linalg.norm(u, axis=1)
    nv = np.linalg.norm(v, axis=1)
    ok = (nu > 1e-9) & (nv > 1e-9) & np.isfinite(nu) & np.isfinite(nv)
    c = np.full(len(u), np.nan)
    c[ok] = np.einsum("ij,ij->i", u[ok] / nu[ok, None], v[ok] / nv[ok, None])
    out = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
    out[~ok] = np.nan
    return out


def running_median(vectors: np.ndarray, t: np.ndarray, window_s: float) -> np.ndarray:
    """Component-wise running median of a vector time series, renormalised.

    Windows are centred and measured in seconds (project convention), so the result is
    comparable across frame rates. Frames with no finite input stay NaN.
    """
    vectors = np.asarray(vectors, dtype=np.float64).reshape(len(t), -1)
    n = len(t)
    out = np.full_like(vectors, np.nan)
    half = window_s / 2.0
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    for i in range(n):
        blk = vectors[lo[i]:hi[i]]
        if np.size(blk) == 0 or not np.isfinite(blk).any():
            continue
        med = np.nanmedian(blk, axis=0)
        nrm = np.linalg.norm(med)
        out[i] = med / nrm if nrm > 1e-12 else np.nan
    return out


def speed_bin_label(speed: float) -> str:
    for lo, hi in SPEED_BINS:
        if lo <= speed < hi:
            return f"{int(lo)}-{int(hi)}" if hi < 1e8 else f"{int(lo)}+"
    return "unknown"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; (nan, nan) when n == 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    s = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - s) / d), float((c + s) / d)


def auroc(positive: np.ndarray, negative: np.ndarray) -> tuple[float, int, int]:
    """Rank-based AUROC with ties handled by average rank; nan if a class is empty."""
    p = np.asarray(positive, dtype=np.float64)
    q = np.asarray(negative, dtype=np.float64)
    p = p[np.isfinite(p)]
    q = q[np.isfinite(q)]
    if len(p) == 0 or len(q) == 0:
        return float("nan"), len(p), len(q)
    allv = np.concatenate([p, q])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    vals, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(vals))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    rp = ranks[:len(p)].sum()
    return float((rp - len(p) * (len(p) + 1) / 2.0) / (len(p) * len(q))), len(p), len(q)


def bootstrap_auroc(positive: np.ndarray, negative: np.ndarray, groups: np.ndarray,
                    n_boot: int = 2000, seed: int = 0) -> dict:
    """AUROC with a group-level (sequence-level) bootstrap interval."""
    rng = np.random.default_rng(seed)
    p = np.asarray(positive, dtype=np.float64)
    q = np.asarray(negative, dtype=np.float64)
    g = np.asarray(groups)
    pg, qg = g[:len(p)], g[len(p):]
    ug = np.unique(g)
    point = auroc(p, q)[0]
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        pi = np.concatenate([np.flatnonzero(pg == u) for u in pick]) if len(p) else np.array([], int)
        qi = np.concatenate([np.flatnonzero(qg == u) for u in pick]) if len(q) else np.array([], int)
        if len(pi) == 0 or len(qi) == 0:
            continue
        v = auroc(p[pi], q[qi])[0]
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    out = {"auroc": point, "n_positive": int(len(p)), "n_negative": int(len(q)),
           "n_boot_used": int(len(vals))}
    if len(vals) >= 20:
        out["lo"] = float(np.percentile(vals, 2.5))
        out["hi"] = float(np.percentile(vals, 97.5))
    else:
        out["lo"] = out["hi"] = None
    return out
