"""Selective prediction: risk-coverage curves, conformal risk control, cluster intervals.

Track M treats every abstention gate as **selective prediction**: the gate accepts or
rejects a clip, and the quantity that matters is the risk *among accepted clips* at the
coverage it achieves.  Three tools are used, all of them distribution-free:

* :func:`risk_coverage` — for a grid of thresholds, the coverage and the risk (a mean of
  0/1 indicators, or a continuous error) among accepted clips.
* :func:`crc_threshold` — conformal risk control (Angelopoulos et al., 2023) on the
  accept-above rule ``accept iff score >= lam``.  The reported bound is the exact
  binomial-style upper bound ``(k + 1) / (n + 1)`` on the accepted risk, which holds with
  probability ``>= 1 - alpha`` for exchangeable clips; the smallest ``lam`` whose bound is
  at most ``alpha`` is selected.  If no ``lam`` reaches ``alpha`` the function says so
  instead of returning the best of a bad set.
* :func:`cluster_interval` — a flight/sequence-clustered bootstrap.  With 6-8 clusters the
  interval is **descriptive**: the report says so wherever one is printed.

Nothing here reads a dataset.  The caller supplies arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOOT_DRAWS = 5000
BOOT_SEED = 1


# ------------------------------------------------------------------ basic statistics
def auroc(scores, labels) -> float:
    """Rank AUROC with ties counted as half.  ``labels`` are 0/1; 1 is the positive class."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    ok = np.isfinite(s) & np.isfinite(y)
    s, y = s[ok], y[ok]
    n1 = int(np.sum(y == 1))
    n0 = int(np.sum(y == 0))
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def conformal_quantile(values, alpha: float = 0.05) -> float:
    """Upper conformal quantile: the ``ceil((n+1)(1-alpha))/n`` empirical quantile.

    This is the finite-sample-corrected version of ``np.quantile(v, 1-alpha)``; it is
    slightly conservative at small ``n`` and equals the plain quantile in the limit.
    """
    v = np.asarray([x for x in np.ravel(values) if np.isfinite(x)], dtype=np.float64)
    n = len(v)
    if n < 8:
        return float("nan")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(v)[k - 1])


# ------------------------------------------------------------------ risk-coverage
def risk_coverage(score, risk, thresholds) -> dict:
    """Coverage and accepted risk at each threshold, for the rule ``accept iff score >= t``.

    ``risk`` is a 0/1 indicator (for a rate) or a continuous per-clip error.  ``nan``
    entries in ``score`` are never accepted (a clip with no score is not judged).
    Returns dict of arrays ``threshold, coverage, risk, n_accept, n_total``.
    """
    s = np.asarray(score, dtype=np.float64).ravel()
    r = np.asarray(risk, dtype=np.float64).ravel()
    if s.shape != r.shape:
        raise ValueError("score and risk must have the same length")
    ok = np.isfinite(s)
    ts = np.asarray(list(thresholds), dtype=np.float64)
    cov = np.zeros(len(ts))
    rk = np.zeros(len(ts))
    nacc = np.zeros(len(ts), dtype=int)
    for i, t in enumerate(ts):
        acc = ok & (s >= t)
        nacc[i] = int(acc.sum())
        cov[i] = nacc[i] / max(int(ok.sum()), 1)
        rr = r[acc & np.isfinite(r)]
        rk[i] = float(np.mean(rr)) if len(rr) else np.nan
    return {"threshold": ts, "coverage": cov, "risk": rk, "n_accept": nacc,
            "n_total": int(ok.sum())}


@dataclass
class CRCResult:
    lam: float
    achieved_coverage: float
    n_accept: int
    n_total: int
    bound: float
    alpha: float
    reached: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def crc_threshold(score, risk, alpha: float = 0.05,
                  min_accept: int = 10) -> CRCResult:
    """Smallest threshold whose accepted risk has an exact-conservative bound <= alpha.

    The rule is ``accept iff score >= lam`` with higher score = more trustworthy.  For a
    0/1 risk indicator the accepted risk is estimated by ``(k + 1) / (n + 1)``; for a
    continuous risk the same correction is applied to the mean by treating it as
    ``k = n * mean``.  Candidate thresholds are the observed scores.  If no threshold
    reaches the target, ``reached`` is False and the reason is stated.
    """
    s = np.asarray(score, dtype=np.float64).ravel()
    r = np.asarray(risk, dtype=np.float64).ravel()
    ok = np.isfinite(s) & np.isfinite(r)
    if ok.sum() == 0:
        return CRCResult(float("nan"), 0.0, 0, 0, float("nan"), alpha, False,
                         "no finite scores")
    s, r = s[ok], r[ok]
    n_total = len(s)
    cands = np.unique(s)
    best = None
    for lam in cands:
        acc = s >= lam
        n = int(acc.sum())
        if n < min_accept:
            continue
        k = float(np.sum(r[acc]))
        bound = (k + 1.0) / (n + 1.0)
        if bound <= alpha:
            best = (float(lam), n, bound)
            break
    if best is None:
        n = n_total
        k = float(np.sum(r))
        return CRCResult(float("nan"), 0.0, 0, n_total, (k + 1.0) / (n + 1.0), alpha,
                         False,
                         f"no threshold reaches alpha={alpha:g} with >= {min_accept} "
                         f"accepted; all-accepted bound is {(k + 1.0) / (n + 1.0):.3f}")
    lam, n, bound = best
    return CRCResult(lam, n / n_total, n, n_total, bound, alpha, True)


# ------------------------------------------------------------------ cluster bootstrap
def _cluster_indices(clusters) -> list[np.ndarray]:
    c = np.asarray(clusters).astype(str)
    return [np.flatnonzero(c == u) for u in np.unique(c)]


def cluster_interval(values, clusters, statistic=None, alpha: float = 0.05,
                     draws: int = BOOT_DRAWS, seed: int = BOOT_SEED) -> dict:
    """Clustered bootstrap interval for a statistic of per-clip ``values``.

    ``statistic`` takes a 1-D array and returns a float; it defaults to the mean.  Flights
    are resampled with replacement.  ``n_clusters < 10`` makes the interval descriptive:
    the result says so in ``note`` and the report must repeat it.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if statistic is None:
        def statistic(a):
            a = np.asarray(a, dtype=np.float64)
            a = a[np.isfinite(a)]
            return float(np.mean(a)) if len(a) else float("nan")
    groups = _cluster_indices(clusters)
    point = statistic(v)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        out.append(statistic(v[idx]))
    out = np.asarray(out, dtype=np.float64)
    out = out[np.isfinite(out)]
    if len(out) < 100:
        lo = hi = float("nan")
    else:
        lo, hi = np.quantile(out, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "n_clusters": len(groups), "draws": int(draws), "seed": int(seed),
            "note": ("descriptive only: fewer than 10 clusters"
                     if len(groups) < 10 else "clustered bootstrap")}


def cluster_auroc(scores, labels, clusters, alpha: float = 0.05,
                  draws: int = BOOT_DRAWS, seed: int = BOOT_SEED) -> dict:
    """AUROC with a cluster-resampled interval."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    point = auroc(s, y)
    groups = _cluster_indices(clusters)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        a = auroc(s[idx], y[idx])
        if np.isfinite(a):
            out.append(a)
    out = np.asarray(out)
    if len(out) < 100:
        lo = hi = float("nan")
    else:
        lo, hi = np.quantile(out, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "n_clusters": len(groups), "draws": int(draws), "seed": int(seed),
            "note": ("descriptive only: fewer than 10 clusters"
                     if len(groups) < 10 else "clustered bootstrap")}


def rate_interval(flags, clusters, alpha: float = 0.05,
                  draws: int = BOOT_DRAWS, seed: int = BOOT_SEED) -> dict:
    """A 0/1 rate with a cluster-resampled interval and its denominator."""
    f = np.asarray(flags, dtype=np.float64).ravel()
    f = np.where(np.isfinite(f), f, 0.0)
    return cluster_interval(f, clusters, alpha=alpha, draws=draws, seed=seed)


# ------------------------------------------------------------------ CRC from counts
def crc_from_counts(lam, n_accept, k_risk, alpha: float = 0.05) -> CRCResult:
    """Conformal risk control from precomputed (threshold, accepted, failures) counts.

    The gate is ``accept iff score >= lam``.  Candidate thresholds are scanned in
    increasing order (decreasing coverage); the first whose exact-conservative bound
    ``(k + 1) / (n + 1) <= alpha`` is returned.  ``k_risk`` may be fractional when the risk
    is a mean of continuous per-clip errors.
    """
    lam = np.asarray(lam, dtype=np.float64)
    n = np.asarray(n_accept, dtype=np.int64)
    k = np.asarray(k_risk, dtype=np.float64)
    order = np.argsort(lam)
    n_total = int(n[order][-1]) if len(order) else 0
    for i in order:
        if n[i] < 1:
            continue
        bound = (k[i] + 1.0) / (n[i] + 1.0)
        if bound <= alpha:
            return CRCResult(float(lam[i]), n[i] / max(n_total, 1), int(n[i]), n_total,
                             float(bound), alpha, True)
    if not len(order):
        return CRCResult(float("nan"), 0.0, 0, 0, float("nan"), alpha, False, "empty grid")
    i = order[0]
    return CRCResult(float("nan"), 0.0, 0, int(n[i]), float((k[i] + 1) / (n[i] + 1)), alpha,
                     False,
                     f"no threshold reaches alpha={alpha:g}; the full-coverage bound is "
                     f"{(k[i] + 1.0) / (n[i] + 1.0):.3f}")


def pooled_rate_interval(hits, ns, clusters, alpha: float = 0.05,
                         draws: int = BOOT_DRAWS, seed: int = BOOT_SEED) -> dict:
    """A pooled rate ``sum(hits)/sum(ns)`` with a cluster-resampled interval.

    The point estimate is the pooled rate (not the mean of per-cluster rates, which weights a
    one-clip flight like a twenty-six-clip flight).  The interval resamples clusters and pools
    the resampled hits and denominators.
    """
    h = np.asarray(hits, dtype=np.float64).ravel()
    n = np.asarray(ns, dtype=np.float64).ravel()
    c = np.asarray(clusters).astype(str).ravel()
    if not (len(h) == len(n) == len(c)):
        raise ValueError("hits, ns and clusters must have the same length")
    ok = np.isfinite(n) & (n > 0)
    h, n, c = h[ok], n[ok], c[ok]
    if len(h) == 0 or n.sum() == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_clusters": 0, "note": "no data"}
    point = float(h.sum() / n.sum())
    groups = _cluster_indices(c)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        tot = n[idx].sum()
        if tot > 0:
            out.append(h[idx].sum() / tot)
    out = np.asarray(out)
    lo = float(np.quantile(out, alpha / 2.0)) if len(out) > 100 else float("nan")
    hi = float(np.quantile(out, 1.0 - alpha / 2.0)) if len(out) > 100 else float("nan")
    return {"point": point, "lo": lo, "hi": hi, "n_clusters": len(groups),
            "draws": int(draws), "seed": int(seed),
            "note": ("descriptive only: fewer than 10 clusters"
                     if len(groups) < 10 else "clustered bootstrap")}
