"""Scale-free coherence between depicted and required lean.

The gain test asks whether the *magnitude* of the depicted lean tracks the required lean.
This tier asks whether the *shape and sign* do, with no scale at all:

* per-axis Pearson correlation over usable frames (pitch/forward and roll/lateral);
* per-axis sign agreement, the fraction of frames where the two series have the same sign.

A positive scale edit (x0.5, x1.5) leaves Pearson correlation unchanged; a tilt removal
destroys it; a mirrored roll flips the sign on the roll axis while leaving pitch positive.
That complementarity is the point, and it is why the scaling families are expected to sit
at chance and are reported as a negative result rather than tuned.

Definition and thresholds: ``configs/coherence_prereg.yaml``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .fit import _shift_series

EPS = 1e-4
VAR_FLOOR = 1e-12


@dataclass
class CoherenceResult:
    r_x: float
    r_y: float
    s_x: float
    s_y: float
    n_used: int
    lag_ms: float

    @property
    def r_min(self) -> float:
        return float(min(self.r_x, self.r_y))

    @property
    def s_min(self) -> float:
        return float(min(self.s_x, self.s_y))

    @property
    def correlation_score(self) -> float:
        """Higher is more suspicious (the preregistered primary)."""
        return float(1.0 - self.r_min)

    @property
    def sign_score(self) -> float:
        return float(1.0 - self.s_min)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update({"r_min": self.r_min, "s_min": self.s_min,
                  "correlation_score": self.correlation_score,
                  "sign_score": self.sign_score})
        return d


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3:
        return 0.0
    va, vb = float(np.var(a)), float(np.var(b))
    if va < VAR_FLOOR or vb < VAR_FLOOR:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = (np.abs(a) > EPS) & (np.abs(b) > EPS)
    if not keep.any():
        return 0.5
    return float(np.mean(np.sign(a[keep]) == np.sign(b[keep])))


def coherence(clip, lag_ms: float = 0.0, mask: np.ndarray | None = None
              ) -> CoherenceResult:
    """Per-axis correlation and sign agreement between depicted and required lean."""
    l_dep = np.asarray(clip.depicted_lean, dtype=np.float64)
    if clip.lean_req is None:
        raise ValueError("coherence needs schema v2 lean_req; re-export the clip")
    l_req = _shift_series(np.asarray(clip.lean_req, dtype=np.float64),
                          np.asarray(clip.t, dtype=np.float64), lag_ms / 1000.0)
    usable = np.isfinite(l_dep).all(axis=1) & np.isfinite(l_req).all(axis=1)
    if mask is not None:
        usable &= np.asarray(mask, dtype=bool)
    if usable.sum() < 3:
        return CoherenceResult(float("nan"), float("nan"), float("nan"), float("nan"),
                               int(usable.sum()), lag_ms)
    rs = [_pearson(l_dep[usable, i], l_req[usable, i]) for i in range(2)]
    ss = [_sign_agreement(l_dep[usable, i], l_req[usable, i]) for i in range(2)]
    return CoherenceResult(rs[0], rs[1], ss[0], ss[1], int(usable.sum()), lag_ms)
