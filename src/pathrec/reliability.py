"""Reliability signals that need no ground truth.

The verifier must be able to abstain, so every extractor run has to produce a signal that
predicts its own error *before* truth is available. Four families, all computable from the
recovered poses (and depth where the extractor provides it):

* **parallax** — median baseline per frame against median scene depth. Low parallax is the
  documented failure mode for learned VO (VGGT paper), so this is the primary signal.
* **tracker health** — whatever the extractor reports (patch count, confidence, valid
  depth fraction). Passed through, not interpreted here.
* **smoothness** — the ratio of per-step displacement noise to the overall motion: a track
  that is mostly jitter cannot support an acceleration.
* **cross-extractor agreement** — whether two independent extractors agree on the
  *direction* of motion. Scale-free, so it can be computed without truth.

Nothing in this module reads ground truth; the bake-off measures how well these predict the
error. That evaluation is where truth enters.
"""
from __future__ import annotations

import numpy as np


def steps(pos: np.ndarray) -> np.ndarray:
    """Per-step displacement vectors of a (N, 3) track."""
    pos = np.asarray(pos, dtype=np.float64)
    return np.diff(pos, axis=0)


def parallax_to_depth(pos: np.ndarray, depth: np.ndarray | float) -> float:
    """Median per-frame baseline divided by median scene depth (dimensionless).

    ``depth`` is a per-frame scalar list (median depth of each frame) or one number. A
    value near zero means the camera barely translates relative to what it sees, which is
    where monocular translation estimates are known to fail.
    """
    d = np.asarray(depth, dtype=np.float64).ravel()
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return float("nan")
    s = np.linalg.norm(steps(pos), axis=1)
    if s.size == 0:
        return float("nan")
    return float(np.median(s) / np.median(d))


def jitter_ratio(pos: np.ndarray) -> float:
    """Step-to-step change relative to the step size: ``median|d2p| / median|dp|``.

    A smooth flight, however tightly curved, has a small value (the second difference is
    curvature times dt^2); a noisy or jittery track is dominated by it. Deliberately not
    "path length over net displacement", which flags any curved flight as noise.
    """
    pos = np.asarray(pos, dtype=np.float64)
    if len(pos) < 3:
        return float("nan")
    d1 = np.diff(pos, axis=0)
    d2 = np.diff(pos, axis=0, n=2)
    s1 = float(np.median(np.linalg.norm(d1, axis=1)))
    s2 = float(np.median(np.linalg.norm(d2, axis=1)))
    if s1 <= 0:
        return float("nan")
    return s2 / s1


def rotation_rate_deg(rot_mats: np.ndarray) -> float:
    """Median rotation between consecutive camera orientations, in degrees."""
    r = np.asarray(rot_mats, dtype=np.float64)
    if len(r) < 2:
        return float("nan")
    rel = np.einsum("nji,njk->nik", r[:-1], r[1:])
    cos = (np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0
    return float(np.degrees(np.median(np.arccos(np.clip(cos, -1.0, 1.0)))))


def direction_agreement(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """Median angle (deg) between the per-step motion directions of two tracks.

    Scale-free and translation-free by construction: only unit step vectors are compared,
    and steps shorter than 1 % of the median step are ignored as numerically meaningless.
    """
    a = steps(pos_a)
    b = steps(pos_b)
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    a, b = a[:n], b[:n]
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    keep = (na > 0.01 * np.median(na)) & (nb > 0.01 * np.median(nb))
    if not keep.any():
        return float("nan")
    ua, ub = a[keep] / na[keep, None], b[keep] / nb[keep, None]
    cos = np.clip((ua * ub).sum(1), -1.0, 1.0)
    return float(np.median(np.degrees(np.arccos(cos))))


def flatten(features: dict) -> dict:
    """Drop non-finite entries so a feature table stays JSON-safe."""
    out = {}
    for k, v in features.items():
        if isinstance(v, (int, float, np.floating)) and not np.isfinite(v):
            out[k] = None
        else:
            out[k] = v.item() if isinstance(v, np.generic) else v
    return out
