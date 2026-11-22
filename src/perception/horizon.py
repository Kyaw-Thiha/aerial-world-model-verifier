"""A non-learned horizon/vanishing-point roll estimator (H03's classical reader).

Purpose
-------
H03 asks whether the learned depicted-attitude reader (GeoCalib) is valid on *generated*
frames.  This module is the independent, non-learned reading used as one check.  It returns
the camera roll -- the angle of the projected world-up direction in the image plane -- which
is the one part of the gravity estimate that is invariant to focal length for a pinhole
camera and to any centred radial fisheye model.

Definition
----------
Camera coordinates are x right, y down, z forward; image coordinates are the same.  For a
unit world-up vector ``u`` expressed in camera coordinates,

    roll = degrees(atan2(u_x, -u_y))

so a level camera (u = (0, -1, 0)) has roll 0.  This is also the image tilt of the horizon
line (a line perpendicular to the projected up), measured with y down, so a positive roll is
a horizon that falls to the right.

Two classical readings, both focal-free for roll
------------------------------------------------
``sky_line``
    Segment the sky (blue-dominant or bright/low-saturation top region), take the lowest sky
    pixel per column, and fit a line with RANSAC.  Its slope is the roll.  This is the
    long-line/horizon reading, and it abstains when no sky is visible (nadir aerial views,
    forest canopies).
``line_family``
    LSD line segments, length >= 40 px.  Their orientations are folded modulo 90 deg (a
    rectilinear scene's horizontal and vertical families both encode the same roll), and the
    circular median of the folded orientations is the roll when the family is concentrated
    enough (>= 12 lines, concentration >= 0.6).  This is the vanishing-point reading without
    an explicit VP solve: for a rectilinear scene the vertical family *is* the projected up
    and the horizontal family is the horizon direction.

The reading with the larger normalised support wins; if neither reaches its floor the frame
abstains.  Both readings are returned so a report can show which one was used.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SKY_MIN_INLIERS = 25
SKY_MIN_SPAN_FRAC = 0.60
SKY_RANSAC_PX = 2.0
LINE_MIN_LEN_PX = 40.0
LINE_MIN_COUNT = 12
LINE_MIN_CONCENTRATION = 0.60
LINE_MAX_IQR_DEG = 8.0


@dataclass
class Reading:
    roll_deg: float
    support: float
    method: str
    n_inliers: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return bool(np.isfinite(self.roll_deg))


def roll_from_up(u: np.ndarray) -> np.ndarray:
    """Roll in degrees from a unit up vector in camera coordinates, (N, 3) or (3,)."""
    u = np.asarray(u, dtype=np.float64)
    return np.degrees(np.arctan2(u[..., 0], -u[..., 1]))


def up_from_roll(roll_deg: float, z: float = 0.0) -> np.ndarray:
    """One up vector with the given roll and z-component, for synthetic checks."""
    r = np.deg2rad(roll_deg)
    xy = np.array([np.sin(r), -np.cos(r)]) * np.sqrt(max(0.0, 1.0 - z * z))
    return np.array([xy[0], xy[1], z])


def _sky_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Blue-dominant or bright low-saturation pixels, in the upper part of the frame."""
    import cv2

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = (h >= 90) & (h <= 140) & (s >= 25) & (v >= 55)
    bright = (v >= 195) & (s <= 45)
    mask = (blue | bright).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    # keep only the sky component that reaches the top of the frame: a blue patch on the
    # ground or a clipped bright band must not invent a horizontal boundary
    n, labels = cv2.connectedComponents(mask)
    keep = np.zeros_like(mask)
    for lab in range(1, n):
        comp = labels == lab
        if comp[:4].any():
            keep |= comp.astype(np.uint8)
    return keep


def sky_line_reading(img_bgr: np.ndarray) -> Reading:
    import cv2

    mask = _sky_mask(img_bgr)
    h, w = mask.shape
    cols, rows = [], []
    for c in range(w):
        idx = np.flatnonzero(mask[:, c])
        if idx.size:
            # the sky-ground boundary is the lowest sky pixel of the column
            cols.append(c)
            rows.append(float(idx[-1]))
    if len(cols) < SKY_MIN_INLIERS:
        return Reading(float("nan"), 0.0, "sky_line", len(cols))
    x = np.asarray(cols, dtype=np.float64)
    y = np.asarray(rows, dtype=np.float64)
    span = (x.max() - x.min()) / max(w - 1, 1)
    if span < SKY_MIN_SPAN_FRAC:
        return Reading(float("nan"), 0.0, "sky_line", len(cols), {"span": span})
    # deterministic RANSAC: try the least-squares line and iteratively reweighted refits
    best = None
    a, b = np.polyfit(x, y, 1)
    for _ in range(5):
        res = np.abs(y - (a * x + b))
        inl = res <= max(SKY_RANSAC_PX, np.percentile(res, 50.0))
        if inl.sum() < 2:
            break
        a, b = np.polyfit(x[inl], y[inl], 1)
        best = (a, b, int(inl.sum()))
    if best is None or best[2] < SKY_MIN_INLIERS:
        return Reading(float("nan"), 0.0, "sky_line", 0 if best is None else best[2])
    slope = float(best[0])
    roll = float(np.degrees(np.arctan(slope)))
    support = best[2] / max(w, 1)
    return Reading(roll, support, "sky_line", best[2],
                   {"slope": slope, "span": float(span), "intercept": float(best[1])})


def line_family_reading(img_bgr: np.ndarray) -> Reading:
    import cv2

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    if lines is None:
        return Reading(float("nan"), 0.0, "line_family", 0)
    seg = lines.reshape(-1, 4)
    d = seg[:, 2:4] - seg[:, 0:2]
    length = np.linalg.norm(d, axis=1)
    keep = length >= LINE_MIN_LEN_PX
    seg, d, length = seg[keep], d[keep], length[keep]
    if len(seg) < LINE_MIN_COUNT:
        return Reading(float("nan"), 0.0, "line_family", int(len(seg)))
    theta = np.degrees(np.arctan2(d[:, 1], d[:, 0]))      # y down
    folded = ((theta + 45.0) % 90.0) - 45.0               # orientation mod 90 deg
    z4 = np.exp(4j * np.deg2rad(folded))
    concentration = float(np.abs(np.mean(z4)))
    mean_angle = float(np.degrees(np.angle(np.mean(z4))) / 4.0)
    if concentration < LINE_MIN_CONCENTRATION:
        return Reading(float("nan"), 0.0, "line_family", int(len(seg)),
                       {"concentration": concentration})
    dev = ((folded - mean_angle + 45.0) % 90.0) - 45.0
    spread = float(np.percentile(dev, 75) - np.percentile(dev, 25))
    if spread > LINE_MAX_IQR_DEG:
        # a rectilinear scene gives one tight family; scattered orientations (diagonal roads,
        # roof edges, foliage) do not encode the horizon and must not be averaged into it
        return Reading(float("nan"), 0.0, "line_family", int(len(seg)),
                       {"concentration": concentration, "iqr_deg": spread})
    roll = float(mean_angle)
    support = concentration * min(1.0, len(seg) / 30.0)
    return Reading(roll, support, "line_family", int(len(seg)),
                   {"concentration": concentration, "n_lines": int(len(seg))})


def estimate_roll(img_bgr: np.ndarray) -> tuple[Reading, dict]:
    """Best classical roll reading plus both candidates for the record."""
    sky = sky_line_reading(img_bgr)
    lines = line_family_reading(img_bgr)
    cands = [r for r in (sky, lines) if r.usable]
    if not cands:
        return Reading(float("nan"), 0.0, "none"), {"sky": sky.__dict__, "lines": lines.__dict__}
    best = max(cands, key=lambda r: r.support)
    return best, {"sky": sky.__dict__, "lines": lines.__dict__}
