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
  exceeds 0.01 scene scale, and -- when the window's cameras travel at least one scene scale --
  the plane is not parallel to the flight (|n . v_hat| <= 0.6). Amendment 3 removed the former
  camera-axis "facing" test: Pi3X's world frame is an arbitrary gauge, not gravity-aligned;
* the winner is the admissible candidate with the most inliers (ties: lower mean camera height);
* a window is reconstructable when the winner has >= 10 % of the pooled points as inliers.

Nothing here uses truth. It is the estimator, and it is what a verifier would have.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONF_FLOOR = -1.0        # Pi3X conf is a log confidence (amendment 3)
POINTS_PER_FRAME = 20_000
POINTS_PER_WINDOW = 200_000
TRIALS = 2000
INLIER_FRAC = 0.01
N_CANDIDATES = 3
CAMERAS_ABOVE_FRAC = 0.95
HEIGHT_MIN_FRAC = 0.01
MOTION_PERP_MAX = 0.6        # |n . v_hat| bound when the window travels >= 1 scene scale
FACING_DEG = 70.0
COVERAGE_INLIER_FRAC = 0.10
RANSAC_POINTS = 20_000


@dataclass
class Plane:
    n: np.ndarray              # unit normal, oriented so the cameras are above
    d: float                   # offset: signed distance of the plane is n . p - d
    inlier_frac: float
    facing_deg: float          # |n . v_hat|; nan when the window barely moves (amendment 3)
    admissible: bool
    n_inliers: int
    height_ref: float = 0.0    # mean camera height above this plane, scene-scale units

    def height(self, points: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return p @ self.n - self.d

    def as_dict(self) -> dict:
        return {"n": [float(x) for x in self.n], "d": float(self.d),
                "motion_abs_dot": float(self.facing_deg),
                "inlier_frac": float(self.inlier_frac), "facing_deg": float(self.facing_deg),
                "admissible": bool(self.admissible), "n_inliers": int(self.n_inliers),
                "height_scene_units": float(self.height_ref)}


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
    """Vectorised RANSAC: the same rule as the loop (trials, band), batched over trials."""
    n_pts = len(pts)
    if n_pts < 3:
        return None
    idx = rng.choice(n_pts, size=(trials, 3), replace=True)
    a, b, c = pts[idx[:, 0]], pts[idx[:, 1]], pts[idx[:, 2]]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=1)
    good = norm > 1e-12
    if not np.any(good):
        return None
    n[good] /= norm[good, None]
    d = np.einsum("ij,ij->i", n[good], a[good])
    ng = n[good]
    best_count, best_n, best_d = -1, None, None
    p32 = pts.astype(np.float32)
    for i in range(0, len(ng), 256):
        nb, db = ng[i:i + 256], d[i:i + 256]
        dist = np.abs(p32 @ nb.T.astype(np.float32) - db.astype(np.float32))
        counts = np.count_nonzero(dist < thr, axis=0)
        k = int(np.argmax(counts))
        if counts[k] > best_count:
            best_count = int(counts[k])
            best_n, best_d = nb[k].astype(np.float64), float(db[k])
    if best_n is None:
        return None
    inl = np.abs(pts @ best_n - best_d) < thr
    refit = _fit(pts[inl], pts.mean(axis=0))
    if refit is not None:
        best_n, best_d = refit
        inl = np.abs(pts @ best_n - best_d) < thr
    return best_n, best_d, inl


def fit_ground_plane(points: np.ndarray, cams: np.ndarray, rng: np.random.Generator | None = None,
                     travel: np.ndarray | None = None) -> dict:
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
    # gauge-invariant motion condition (amendment 3, A6): the mean camera displacement
    v_hat = None
    if travel is not None and np.linalg.norm(travel) >= scale:
        v_hat = _unit(np.asarray(travel, dtype=np.float64))

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
        motion = float(abs(n @ v_hat)) if v_hat is not None else float("nan")
        admissible = bool(above >= CAMERAS_ABOVE_FRAC
                          and h_mean > HEIGHT_MIN_FRAC * scale
                          and (v_hat is None or motion <= MOTION_PERP_MAX)
                          and inlier_frac >= COVERAGE_INLIER_FRAC)
        candidates.append(Plane(n=n, d=d, inlier_frac=inlier_frac, facing_deg=motion,
                                admissible=admissible,
                                n_inliers=int(np.count_nonzero(inl)),
                                height_ref=float(np.mean(cdist))))
        remaining = sub[~inl]
        if len(remaining) < 100:
            break

    admissible = [c for c in candidates if c.admissible]
    pool = admissible or candidates
    if not pool:
        return {"ok": False, "reason": "no_candidate", "scale": scale}
    winner = max(pool, key=lambda c: (c.n_inliers, -float(c.height_ref)))
    return {"ok": True, "scale": float(scale), "plane": winner,
            "winning_admissible": bool(admissible),
            "n_candidates": len(candidates),
            "candidates": [c.as_dict() for c in candidates],
            "coverage_ok": bool(winner.inlier_frac >= COVERAGE_INLIER_FRAC
                                and winner.admissible)}
