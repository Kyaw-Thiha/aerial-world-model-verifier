"""Ground-plane extraction from a multi-view point cloud (Track V).

Frozen in ``configs/v00_feasibility_prereg.yaml`` (+ amendment 1) before any output existed.
The rule, restated so the code and the config cannot drift:

* pool the points of the frames inside the window, at most 20,000 per frame, at most 200,000
  per window, finite and above the confidence floor;
* scale every length by ``scene_scale`` = median distance of the pooled points to the nearest
  camera centre;
* RANSAC, 2000 trials, inlier band 1 % of scene scale, seed 1, three mutually distinct
  candidates extracted greedily (each round's inliers removed before the next);
* orient each normal so the mean camera centre has positive signed distance;
* a candidate is admissible when >= 95 % of camera centres sit above it, the mean camera height
  is in [0.01, 1.0] scene scale, and the normal is within 70 deg of the mean camera-down axis;
* the winner is the admissible candidate with the most inliers (ties: smaller facing angle);
* a window is reconstructable when the winner has >= 10 % of the pooled points as inliers.

Nothing here uses truth. It is the estimator, and it is what a verifier would have.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONF_FLOOR = 0.5
POINTS_PER_FRAME = 20_000
POINTS_PER_WINDOW = 200_000
TRIALS = 2000
INLIER_FRAC = 0.01
N_CANDIDATES = 3
CAMERAS_ABOVE_FRAC = 0.95
HEIGHT_BAND = (0.01, 1.0)
FACING_DEG = 70.0
COVERAGE_INLIER_FRAC = 0.10
RANSAC_POINTS = 20_000


@dataclass
class Plane:
    n: np.ndarray              # unit normal, oriented so the cameras are above
    d: float                   # offset: signed distance of the plane is n . p - d
    inlier_frac: float
    facing_deg: float
    admissible: bool
    n_inliers: int

    def height(self, points: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return p @ self.n - self.d

    def as_dict(self) -> dict:
        return {"n": [float(x) for x in self.n], "d": float(self.d),
                "inlier_frac": float(self.inlier_frac), "facing_deg": float(self.facing_deg),
                "admissible": bool(self.admissible), "n_inliers": int(self.n_inliers)}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def scene_scale(points: np.ndarray, cams: np.ndarray) -> float:
    """Median distance of the pooled points to the nearest camera centre."""
    p = np.asarray(points, dtype=np.float64)
    c = np.asarray(cams, dtype=np.float64)
    if len(p) == 0 or len(c) == 0:
        return float("nan")
    # nearest camera, chunked so a 200k x 64 matrix never materialises
    best = np.full(len(p), np.inf)
    for i in range(0, len(c), 16):
        cc = c[i:i + 16]
        d = np.linalg.norm(p[:, None, :] - cc[None, :, :], axis=2).min(axis=1)
        best = np.minimum(best, d)
    return float(np.median(best))


def subsample(points: np.ndarray, cap: int) -> np.ndarray:
    if len(points) <= cap:
        return points
    idx = np.linspace(0, len(points) - 1, cap).round().astype(int)
    return points[idx]


def _fit(pts: np.ndarray, cams_mean: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Least-squares plane through ``pts``, oriented so ``cams_mean`` is on the positive side."""
    if len(pts) < 3:
        return None
    centre = pts.mean(axis=0)
    u, s, vt = np.linalg.svd(pts - centre, full_matrices=False)
    if s[1] <= 0:
        return None
    n = _unit(vt[-1])
    if float(n @ (cams_mean - centre)) < 0:
        n = -n
    return n, float(n @ centre)


def _ransac(pts: np.ndarray, thr: float, rng: np.random.Generator,
            trials: int = TRIALS) -> tuple[np.ndarray, float, np.ndarray] | None:
    n_pts = len(pts)
    if n_pts < 3:
        return None
    best = None
    best_count = -1
    for _ in range(trials):
        idx = rng.choice(n_pts, 3, replace=False)
        a, b, c = pts[idx]
        n = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(n))
        if norm < 1e-12:
            continue
        n = n / norm
        d = float(n @ a)
        count = int(np.count_nonzero(np.abs(pts @ n - d) < thr))
        if count > best_count:
            best_count = count
            best = (n, d)
    if best is None:
        return None
    n, d = best
    inl = np.abs(pts @ n - d) < thr
    refit = _fit(pts[inl], pts.mean(axis=0))
    if refit is not None:
        n, d = refit
        inl = np.abs(pts @ n - d) < thr
    return n, d, inl


def fit_ground_plane(points: np.ndarray, cams: np.ndarray, cams_down: np.ndarray,
                     rng: np.random.Generator | None = None) -> dict:
    """Fit the ground plane of one window. Returns the winner plus every candidate."""
    rng = rng or np.random.default_rng(1)
    p = np.asarray(points, dtype=np.float64)
    c = np.asarray(cams, dtype=np.float64)
    if len(p) < 100 or len(c) < 2:
        return {"ok": False, "reason": "too_few_points", "scale": float("nan")}
    scale = scene_scale(p, c)
    if not np.isfinite(scale) or scale <= 0:
        return {"ok": False, "reason": "bad_scale", "scale": float("nan")}
    thr = INLIER_FRAC * scale
    sub = subsample(p, RANSAC_POINTS)
    cams_mean = c.mean(axis=0)
    down = _unit(np.asarray(cams_down, dtype=np.float64).mean(axis=0))
    heights_band = (HEIGHT_BAND[0] * scale, HEIGHT_BAND[1] * scale)

    candidates: list[Plane] = []
    remaining = sub
    for _ in range(N_CANDIDATES):
        got = _ransac(remaining, thr, rng)
        if got is None:
            break
        n, d, _ = got
        dist = np.abs(sub @ n - d)
        cand_in = sub[dist < thr]
        if len(cand_in) < 50:
            break
        refit = _fit(cand_in, cams_mean)
        if refit is not None:
            n, d = refit
        dist = sub @ n - d
        inl = np.abs(dist) < thr
        inlier_frac = float(np.count_nonzero(inl)) / float(len(sub))
        # orientation against the cameras
        cdist = c @ n - d
        if float(cdist.mean()) < 0:
            n, d = -n, -d
            cdist = -cdist
        above = float(np.mean(cdist > 0.005 * scale))
        h_mean = float(cdist.mean())
        facing = float(np.degrees(np.arccos(np.clip(abs(float(n @ down)), -1.0, 1.0))))
        admissible = bool(above >= CAMERAS_ABOVE_FRAC
                          and heights_band[0] <= h_mean <= heights_band[1]
                          and facing < FACING_DEG
                          and inlier_frac >= COVERAGE_INLIER_FRAC)
        candidates.append(Plane(n=n, d=d, inlier_frac=inlier_frac, facing_deg=facing,
                                admissible=admissible,
                                n_inliers=int(np.count_nonzero(inl))))
        remaining = sub[~inl]
        if len(remaining) < 100:
            break

    admissible = [c for c in candidates if c.admissible]
    pool = admissible or candidates
    if not pool:
        return {"ok": False, "reason": "no_candidate", "scale": scale}
    winner = max(pool, key=lambda c: (c.n_inliers, -c.facing_deg))
    return {"ok": True, "scale": float(scale), "plane": winner,
            "winning_admissible": bool(admissible),
            "n_candidates": len(candidates),
            "candidates": [c.as_dict() for c in candidates],
            "coverage_ok": bool(winner.inlier_frac >= COVERAGE_INLIER_FRAC
                                and winner.admissible)}
