"""Conformal bands and the three pre-registered decision rules.

The bands are fitted on **real** clips of the fit flights only.  A held-out flight is
never used to set its own bar.  Two conventions are used:

* ``ScalarBand``: a two-sided conformal interval, ``alpha`` split over both tails.
* ``JointBand``: standardise each gain by its fit-flight median and robust scale
  (``(q97.5 - q2.5) / 3.92``), take the largest absolute z-score, and put a one-sided
  conformal 95th percentile on that maximum.  This gives one joint false-reject budget
  for two or three gains instead of one budget per gain.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fit import GainResult

RULES = ("R1_scalar_g1", "R2_axis_gains", "R3_scalar_or_axis")
SIMPLICITY = {"R1_scalar_g1": 0, "R2_axis_gains": 1, "R3_scalar_or_axis": 2}


def _finite(values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).ravel()
    return v[np.isfinite(v)]


@dataclass
class ScalarBand:
    lo: float
    hi: float
    alpha: float
    n: int

    def flag(self, value: float) -> bool:
        return bool(np.isfinite(value) and (value < self.lo or value > self.hi))

    def as_dict(self) -> dict:
        return {"lo": self.lo, "hi": self.hi, "alpha": self.alpha, "n": self.n}


@dataclass
class JointBand:
    keys: tuple[str, ...]
    center: dict
    scale: dict
    threshold: float
    alpha: float
    n: int

    def statistic(self, result: GainResult) -> float:
        vals = []
        for key in self.keys:
            v = _gain_value(result, key)
            if not np.isfinite(v):
                return float("nan")
            vals.append(abs((v - self.center[key]) / self.scale[key]))
        return float(max(vals)) if vals else float("nan")

    def flag(self, result: GainResult) -> bool:
        stat = self.statistic(result)
        return bool(np.isfinite(stat) and stat > self.threshold)

    def as_dict(self) -> dict:
        return {"keys": list(self.keys), "center": self.center, "scale": self.scale,
                "threshold": self.threshold, "alpha": self.alpha, "n": self.n}


def _gain_value(result: GainResult, key: str) -> float:
    if key == "G1":
        return float(result.gain)
    if key == "gx":
        return float(result.axis_gains[0])
    if key == "gy":
        return float(result.axis_gains[1])
    raise KeyError(key)


def conformal_band(values, alpha: float = 0.05) -> ScalarBand:
    """Two-sided conformal interval from the fit-clip values."""
    v = _finite(values)
    if len(v) < 8:
        return ScalarBand(float("nan"), float("nan"), alpha, len(v))
    lo, hi = np.quantile(v, [alpha / 2.0, 1.0 - alpha / 2.0])
    return ScalarBand(float(lo), float(hi), alpha, len(v))


def fit_joint_band(results: list[GainResult], keys: tuple[str, ...],
                   alpha: float = 0.05) -> JointBand:
    values = {key: _finite([_gain_value(r, key) for r in results]) for key in keys}
    n = min((len(v) for v in values.values()), default=0)
    if n < 8:
        return JointBand(keys, {k: float("nan") for k in keys},
                         {k: float("nan") for k in keys}, float("nan"), alpha, n)
    center, scale = {}, {}
    for key, v in values.items():
        center[key] = float(np.median(v))
        q_lo, q_hi = np.quantile(v, [0.025, 0.975])
        scale[key] = float(max((q_hi - q_lo) / 3.92, 1e-6))
    stats = []
    for r in results:
        vals = [_gain_value(r, k) for k in keys]
        if not np.isfinite(vals).all():
            continue
        stats.append(max(abs((v - center[k]) / scale[k]) for v, k in zip(vals, keys)))
    if len(stats) < 8:
        return JointBand(keys, center, scale, float("nan"), alpha, len(stats))
    threshold = float(np.quantile(stats, 1.0 - alpha))
    return JointBand(keys, center, scale, threshold, alpha, len(stats))


@dataclass
class Bands:
    scalar: ScalarBand | None = None
    axis: JointBand | None = None
    joint: JointBand | None = None


def fit_bands(results: list[GainResult], rule: str, alpha: float = 0.05) -> Bands:
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}")
    out = Bands()
    if rule in ("R1_scalar_g1", "R3_scalar_or_axis"):
        out.scalar = conformal_band([r.gain for r in results], alpha)
    if rule in ("R2_axis_gains", "R3_scalar_or_axis"):
        out.axis = fit_joint_band(results, ("gx", "gy"), alpha)
    if rule == "R3_scalar_or_axis":
        out.joint = fit_joint_band(results, ("G1", "gx", "gy"), alpha)
    return out


def flag(result: GainResult, rule: str, bands: Bands) -> bool:
    """Apply one decision rule.  A missing/undefined band abstains rather than flags."""
    if rule == "R1_scalar_g1":
        return False if bands.scalar is None else bands.scalar.flag(result.gain)
    if rule == "R2_axis_gains":
        return False if bands.axis is None else bands.axis.flag(result)
    if rule == "R3_scalar_or_axis":
        if bands.scalar is not None and bands.scalar.flag(result.gain):
            return True
        return False if bands.joint is None else bands.joint.flag(result)
    raise ValueError(f"unknown rule {rule!r}")
