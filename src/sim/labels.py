"""Infeasible flights for the simulated set: the defect is a camera attitude, not a render edit.

Everything here is a function of the clip's frozen row and its flight. A defect replaces the
attitude the camera is rendered on, over an interval of the flight:

* ``tilt_removed``  -- camera level (zero roll and pitch, the path's yaw kept);
* ``mirrored_roll`` -- roll sign flipped, pitch and yaw untouched;
* ``tilt_scaled``   -- roll and pitch multiplied by ``s`` (0.5 under-tilts, 1.5 over-tilts).

The path, the speed and the position are identical to the feasible clip's, so the required
lean is identical and the only thing that changed is what the pixels would show. The interval
is exact (``interval_s``) and the fraction is recorded, so the same labels serve a
segment-level detector (Track G2) and a clip-level verdict.

The thrust-limit family is different in kind and lives in ``dataset.py``: there the *path* is
what is infeasible, the camera is honest, and nothing here applies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.generation.attitude import body_rotation
from src.verdict.lean import depicted_lean_from_gravity

DEFECT_FAMILIES = ("tilt_removed", "mirrored_roll", "tilt_scaled")
E3 = np.array([0.0, 0.0, 1.0])


@dataclass
class Defect:
    kind: str                      # one of DEFECT_FAMILIES
    magnitude: float               # 1.0 | -1.0 | 0.5 | 1.5
    interval_s: tuple[float, float] | None   # None = the whole flight
    fraction: float                # 1.0 for a whole-flight defect

    def active(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        if self.interval_s is None:
            return np.ones_like(t, dtype=bool)
        t0, t1 = self.interval_s
        return (t >= t0 - 1e-12) & (t <= t1 + 1e-12)


def interval_for(duration_s: float, fraction: float, seed: int) -> tuple[float, float]:
    """Place a partial-defect interval of length ``fraction * duration`` inside the flight.

    The start is drawn uniformly in ``[0.05, 0.95 - fraction]`` of the flight, rounded to the
    video frame grid, so the interval never touches the first or last 5 % (a reader that needs
    context at the clip boundary is not being tested by an interval starting at t = 0).
    """
    rng = np.random.default_rng(int(seed))
    f = float(fraction)
    if not 0.0 < f < 1.0:
        raise ValueError(f"partial fraction must be in (0, 1), got {f}")
    lo, hi = 0.05 * duration_s, (0.95 - f) * duration_s
    start = float(rng.uniform(lo, max(lo, hi)))
    start = round(start * 30.0) / 30.0                  # snap to the 30 fps frame grid
    start = max(0.0, min(start, duration_s * (1.0 - f)))
    return (start, start + f * duration_s)


def defect_from_row(row: dict, duration_s: float) -> Defect | None:
    """Resolve a frozen row's defect (None for feasible and thrust-limit clips)."""
    fam = row.get("defect_family")
    if fam in (None, "feasible", "thrust_limit"):
        return None
    kind = row.get("violation") if fam == "partial" else fam
    if kind not in DEFECT_FAMILIES:
        raise KeyError(f"unknown defect kind {kind!r}")
    if fam == "partial":
        frac = float(row["fraction"])
        return Defect(kind=kind, magnitude=float(row["magnitude"]),
                      interval_s=interval_for(duration_s, frac, row["interval_seed"]),
                      fraction=frac)
    return Defect(kind=kind, magnitude=float(row["magnitude"]), interval_s=None, fraction=1.0)


def camera_attitude(flight, defect: Defect | None) -> np.ndarray:
    """(N,3,3) body-to-world rotation the *camera* is rendered on."""
    base = flight.R_bw
    if defect is None:
        return base
    mask = defect.active(flight.t)
    out = base.copy()
    yaw, pitch, roll = flight.yaw, flight.pitch, flight.roll
    if defect.kind == "tilt_removed":
        out[mask] = body_rotation(yaw[mask], np.zeros(mask.sum()), np.zeros(mask.sum()))
    elif defect.kind == "mirrored_roll":
        out[mask] = body_rotation(yaw[mask], pitch[mask], -roll[mask])
    elif defect.kind == "tilt_scaled":
        out[mask] = body_rotation(yaw[mask], defect.magnitude * pitch[mask],
                                  defect.magnitude * roll[mask])
    else:
        raise KeyError(defect.kind)
    return out


def depicted_gt(R_bw_cam: np.ndarray, R_cb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Depicted lean and tilt from the rendered camera pose and the mount (exact truth)."""
    up_cam = np.einsum("nji,j->ni", np.asarray(R_bw_cam) @ R_cb, E3)   # (R_bw R_cb)^T e3
    lean = depicted_lean_from_gravity(up_cam, R_cb)
    up_body = np.einsum("ij,nj->ni", R_cb, up_cam)      # R_cb @ up_cam, row by row
    tilt = np.degrees(np.arccos(np.clip(up_body @ E3, -1.0, 1.0)))
    return lean, tilt


def required_gt(flight) -> tuple[np.ndarray, np.ndarray]:
    """Required lean and tilt from the commanded attitude (the path), exact."""
    from src.verdict.lean import roll_pitch_to_lean

    lean = roll_pitch_to_lean(flight.roll, flight.pitch)
    tilt = np.degrees(np.arccos(np.clip(flight.thrust_axis @ E3, -1.0, 1.0)))
    return lean, tilt


def realised_magnitude(defect: Defect | None, tilt_req: np.ndarray, mask: np.ndarray) -> dict:
    """The defect's size in degrees, measured on the clip's own required tilt."""
    if defect is None:
        return {"magnitude_deg": 0.0, "magnitude_unit": "none"}
    active = tilt_req[mask]
    if active.size == 0:
        return {"magnitude_deg": 0.0, "magnitude_unit": "none"}
    med = float(np.median(active))
    if defect.kind == "tilt_removed":
        return {"magnitude_deg": med, "magnitude_unit": "tilt removed (deg)"}
    if defect.kind == "mirrored_roll":
        return {"magnitude_deg": 2.0 * med, "magnitude_unit": "roll inversion (deg of swing)"}
    return {"magnitude_deg": abs(1.0 - defect.magnitude) * med,
            "magnitude_unit": f"tilt scaled x{defect.magnitude:g} (deg of change)"}
