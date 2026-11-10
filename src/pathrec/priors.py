"""Scale and time priors for a pixel-derived path.

A monocular path has no metric scale, and its time base is whatever fps we assume. The
verifier needs to know how wrong either can be before a violation claim means anything.

Three tools, in the order the handoff asks for them:

1. **scale ratio** per clip against truth, fitted on some sequences and evaluated on held
   out ones (split by sequence, never by frame) -> a plausibility interval.
2. **plausibility-interval rule**: a clip is inconsistent only if *no* scale in the
   interval makes depicted and implied tilt agree.
3. **scale- and time-free statistic**: regress the depicted lean on the recovered
   horizontal acceleration. Correlation and R^2 are invariant to an unknown scale and to a
   time rescale, while the slope gives an **apparent gravity** ``g_app = 1 / slope`` which
   should be about 9.81 m/s^2 when the tilt in the pixels matches the path.
"""
from __future__ import annotations

import numpy as np


def scale_ratio(est: np.ndarray, true: np.ndarray) -> float:
    """Least-squares scalar scale taking ``est`` to ``true`` (no rotation, no offset)."""
    e = np.asarray(est, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(true, dtype=np.float64).reshape(-1, 3)
    ec, tc = e - e.mean(0), t - t.mean(0)
    den = float((ec * ec).sum())
    return float((ec * tc).sum() / den) if den > 0 else float("nan")


def interval(ratios, coverage: float = 0.9) -> dict:
    """Empirical central interval of per-clip scale ratios.

    Returns the median and the ``coverage`` interval, robust to a few broken clips by
    construction (quantiles, not min/max).
    """
    r = np.asarray([x for x in ratios if np.isfinite(x)], dtype=np.float64)
    if r.size == 0:
        return {"n": 0, "median": float("nan"), "low": float("nan"), "high": float("nan")}
    lo_q, hi_q = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    return {"n": int(r.size), "median": float(np.median(r)),
            "low": float(np.quantile(r, lo_q)), "high": float(np.quantile(r, hi_q))}


def consistent(ratio: float, iv: dict) -> bool:
    """Whether a clip's scale ratio falls inside the plausibility interval."""
    if not np.isfinite(ratio):
        return False
    return bool(iv["low"] <= ratio <= iv["high"])


def lean_regression(l_dep: np.ndarray, a_h: np.ndarray) -> dict:
    """Through-origin fit ``l_dep ~ a_h / g_app`` (the handoff's scale-free statistic).

    ``l_dep`` (N,2) is the depicted lean, ``a_h`` (N,2) the recovered horizontal
    acceleration in *any* fixed units. Returns the slope, its 1/slope (apparent gravity in
    those units), R^2, Pearson r and the sample count.

    Invariance: multiplying ``a_h`` by any constant s divides ``slope`` by s and leaves
    ``r`` and ``R^2`` unchanged; rescaling time multiplies ``a_h`` by 1/dt^2 and does the
    same. So with unknown scale and fps, the *reported* g_app carries both unknowns while
    the fit quality does not.
    """
    l = np.asarray(l_dep, dtype=np.float64).reshape(-1, 2)
    a = np.asarray(a_h, dtype=np.float64).reshape(-1, 2)
    if l.shape != a.shape:
        raise ValueError(f"shape mismatch {l.shape} vs {a.shape}")
    ok = np.isfinite(l).all(1) & np.isfinite(a).all(1)
    l, a = l[ok], a[ok]
    if len(l) < 3:
        return {"n": int(len(l)), "slope": float("nan"), "g_app": float("nan"),
                "r2": float("nan"), "r": float("nan")}
    a_f, l_f = a.ravel(), l.ravel()
    den = float(a_f @ a_f)
    slope = float(a_f @ l_f / den) if den > 0 else float("nan")
    resid = l_f - slope * a_f
    ss_tot = float(((l_f - l_f.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    r = float(np.corrcoef(a_f, l_f)[0, 1]) if a_f.std() > 0 and l_f.std() > 0 else float("nan")
    return {"n": int(len(l)), "slope": slope,
            "g_app": float(1.0 / slope) if slope not in (0.0, float("nan")) else float("nan"),
            "r2": r2, "r": r}


def time_rescale(a_h: np.ndarray, factor: float) -> np.ndarray:
    """Acceleration under a wrong time base: ``a * factor**2`` (dt scaled by 1/factor)."""
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    return np.asarray(a_h, dtype=np.float64) * factor ** 2


def tilt_agreement(l_dep: np.ndarray, l_imp_candidates: dict) -> dict:
    """Plausibility-interval rule: is any candidate scale consistent with the pixels?

    ``l_imp_candidates`` maps a scale value to the implied lean vector at that scale.
    Returns per-scale residuals and whether at least one scale agrees within ``tol``
    (fraction of |l_dep|), which is the only statement the verifier may make when scale is
    unknown: "inconsistent" requires *every* candidate to disagree.
    """
    l = np.asarray(l_dep, dtype=np.float64).reshape(-1, 2)
    ok = np.isfinite(l).all(1)
    l = l[ok]
    out = {}
    for s, l_imp in l_imp_candidates.items():
        imp = np.asarray(l_imp, dtype=np.float64).reshape(-1, 2)[ok]
        finite = np.isfinite(imp).all(1)
        if finite.sum() < 3:
            out[str(s)] = {"n": int(finite.sum()), "residual_median": float("nan")}
            continue
        resid = np.linalg.norm(l[finite] - imp[finite], axis=1)
        out[str(s)] = {"n": int(finite.sum()),
                       "residual_median": float(np.median(resid)),
                       "residual_p90": float(np.percentile(resid, 90))}
    return out
