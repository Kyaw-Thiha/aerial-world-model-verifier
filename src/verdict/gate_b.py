"""Gate B: do two independent tilt readers disagree where the verdict reader is wrong?

The verdict reader (GeoCalib on the raw fisheye) is the only reader the verdict has.  Gate B
runs a second and a third estimator on the *same* pixels and abstains when they disagree,
on the argument that an estimator whose reading is not reproducible across independent models
is not reporting the scene's gravity.

Readers (see ``remote/m_readers.py``)
-------------------------------------
``R_frozen``  GeoCalib simple_divisional on the raw fisheye -- stored in the schema clips,
              never re-run here.
``R_pf``      Perspective Fields, Paramnet-360Cities-edina-centered.
``R_gc224``   GeoCalib pinhole with a scalar focal prior, on the same 224 px centre crop.

The disagreement statistic is computed on the matched feed (``R_pf`` vs ``R_gc224``), because
a lean comparison across different camera models and fields is not a like-for-like quantity.
Each reader's gravity direction in the camera frame is turned into the project's lean vector
through the clip's own mount by :func:`src.verdict.lean.depicted_lean_from_gravity`, so the
difference is in the same units the verdict uses.

Nothing here reads a dataset; the caller supplies per-frame rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.verdict.lean import depicted_lean_from_gravity

VAR_FLOOR = 1e-12


def gravity_pf(row: dict, roll_sign: float = -1.0, pitch_sign: float = 1.0) -> np.ndarray:
    """PF's world-up direction in camera coordinates.

    Built from PF's own predicted roll and pitch.  PF's angle signs are not documented in a
    way that matches GeoCalib's, so both are fixed once from data and held fixed:

    * ``roll_sign = -1``: on the rotation probe a +20 deg image rotation moves PF's
      ``pred_roll`` by +18.7 deg and GeoCalib's roll by -19.3 deg, so PF's roll is negated
      (H03's convention).  With it, PF's roll agrees with ``R_gc224`` to a median 1.41 deg
      (p90 5.83) over the simulated frames.
    * ``pitch_sign = +1``: the probe cannot resolve the pitch sign (its frames are near
      level, so the two hypotheses differ by 0.5 deg), so it is fixed on the simulated
      development clips against the render's own truth, where +1 gives 2.4-6.5 deg of lean
      error against -1's 3-20 deg.

    Camera convention: x right, y down, z forward, with ``roll = atan2(u_x, -u_y)``.
    """
    r = np.radians(roll_sign * float(row["pf_roll_deg"]))
    p = np.radians(pitch_sign * float(row["pf_pitch_deg"]))
    return np.array([np.sin(r) * np.cos(p), -np.cos(r) * np.cos(p), np.sin(p)])


def fit_pf_signs(rows) -> dict:
    """The four sign hypotheses, scored by median angular disagreement with R_gc224."""
    best = None
    table = {}
    for rs in (1.0, -1.0):
        for ps in (1.0, -1.0):
            ang = []
            for row in rows:
                u_pf = gravity_pf(row, rs, ps)
                u_gc = gravity_gc(row)
                nu = np.linalg.norm(u_pf) * np.linalg.norm(u_gc)
                if nu > 0:
                    ang.append(float(np.degrees(np.arccos(
                        np.clip(np.dot(u_pf, u_gc) / nu, -1.0, 1.0)))))
            med = float(np.median(ang)) if ang else float("nan")
            table[f"{rs:+.0f}/{ps:+.0f}"] = med
            if best is None or med < best[2]:
                best = (rs, ps, med)
    return {"median_deg_by_sign": table, "roll_sign": best[0], "pitch_sign": best[1],
            "median_deg": best[2]}


def gravity_gc(row: dict) -> np.ndarray:
    return np.asarray(row["gc_gravity"], dtype=np.float64)


def lean_from_gravity(up_cam: np.ndarray, R_cb: np.ndarray) -> np.ndarray:
    """The project's lean vector from a camera-frame up direction and the mount."""
    up = np.atleast_2d(np.asarray(up_cam, dtype=np.float64))
    return depicted_lean_from_gravity(up, np.asarray(R_cb, dtype=np.float64))


def lean_difference_deg(l_a: np.ndarray, l_b: np.ndarray) -> np.ndarray:
    """Per-frame angle between two lean vectors, in degrees (NaN passes through)."""
    a = np.asarray(l_a, dtype=np.float64)
    b = np.asarray(l_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    d = np.linalg.norm(a - b, axis=1)
    out = np.degrees(np.arcsin(np.clip(d, 0.0, 1.0)))
    out[~np.isfinite(d)] = np.nan
    return out


def residual_deg(l_a: np.ndarray, l_b: np.ndarray) -> np.ndarray:
    """Alias of :func:`lean_difference_deg`, used for reader-versus-truth error."""
    return lean_difference_deg(l_a, l_b)


@dataclass
class Disagreement:
    median: float
    p90: float
    window_max: float
    n_frames: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _windows(n: int, length: int = 8, hop: int = 4) -> list[tuple[int, int]]:
    if n <= length:
        return [(0, n)]
    return [(s, s + length) for s in range(0, n - length + 1, hop)]


def disagreement(d_l: np.ndarray, window: int = 8, hop: int = 4,
                 min_frames: int = 6) -> Disagreement:
    """Clip median, p90 and window-max of a per-frame lean difference."""
    d = np.asarray(d_l, dtype=np.float64)
    ok = np.isfinite(d)
    if ok.sum() < min_frames:
        nan = float("nan")
        return Disagreement(nan, nan, nan, int(ok.sum()))
    vals = d[ok]
    wins = []
    for lo, hi in _windows(len(d), window, hop):
        w = d[lo:hi]
        w = w[np.isfinite(w)]
        if len(w) >= min_frames:
            wins.append(float(np.median(w)))
    return Disagreement(float(np.median(vals)), float(np.percentile(vals, 90)),
                        float(max(wins)) if wins else float("nan"), int(ok.sum()))


def gate_b(value: float, tau: float) -> dict:
    """Abstain when the clip's disagreement exceeds ``tau``."""
    if not np.isfinite(value):
        return {"testable": False, "reasons": ["readers_disagree"],
                "value": value, "note": "no usable disagreement"}
    if np.isfinite(tau) and value > tau:
        return {"testable": False, "reasons": ["readers_disagree"], "value": value,
                "tau": tau}
    return {"testable": True, "reasons": [], "value": value, "tau": tau}


def ensemble_lean(l_a: np.ndarray, l_b: np.ndarray) -> np.ndarray:
    """Mean of two lean series; a NaN in either reader gives NaN for that frame."""
    a = np.asarray(l_a, dtype=np.float64)
    b = np.asarray(l_b, dtype=np.float64)
    out = 0.5 * (a + b)
    bad = ~(np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1))
    out[bad] = np.nan
    return out


def roll_of(gravity: np.ndarray) -> np.ndarray:
    """Image-plane roll of an up direction: atan2(u_x, -u_y), degrees."""
    g = np.atleast_2d(np.asarray(gravity, dtype=np.float64))
    return np.degrees(np.arctan2(g[:, 0], -g[:, 1]))
