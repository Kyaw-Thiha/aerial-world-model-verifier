"""One clip schema for real, rendered and generated flights.

All arrays are (N, ...) and time-ordered. World frame is z-up, SI units.
Nothing here has an implicit default that changes physics: the sample rate and
the quaternion order are required arguments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

QUAT_ORDERS = ("xyzw", "wxyz")


@dataclass
class Clip:
    """A single flight or generated video segment.

    Attributes
    ----------
    name : str
        Identifier used in results (e.g. ``flight-01a-ellipse``).
    t : (N,) float
        Timestamps in seconds, increasing, uniform.
    pos : (N, 3) float
        Position in metres, world frame, z-up.
    quat : (N, 4) float
        Body-to-world attitude as a unit quaternion, laid out per ``quat_order``.
    quat_order : str
        ``"xyzw"`` or ``"wxyz"``. Required: the two are silently different.
    fps : float
        Sample rate in hertz, i.e. 1/median(dt). Acceleration scales with fps^2,
        so this is never inferred at use sites.
    vel : (N, 3) float or None
        Optional velocity in m/s (world frame). Kept when the source provides a
        cleaner estimate than differentiating position.
    cam_uptilt_deg : float or None
        Camera mount pitch offset relative to the body, in degrees. Recorded for
        every TII flight; unused by the mocap-only floor.
    frame_paths : list[str] or None
        Frame filenames in time order, when the clip has pixels.
    intrinsics : dict or None
        Camera intrinsics; None until the pixel stage needs them.
    meta : dict
        Free-form provenance (source file, licence, notes).
    """

    name: str
    t: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    quat_order: str
    fps: float
    vel: np.ndarray | None = None
    cam_uptilt_deg: float | None = None
    frame_paths: list[str] | None = None
    intrinsics: dict | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=np.float64).ravel()
        self.pos = np.asarray(self.pos, dtype=np.float64).reshape(-1, 3)
        self.quat = np.asarray(self.quat, dtype=np.float64).reshape(-1, 4)
        if self.vel is not None:
            self.vel = np.asarray(self.vel, dtype=np.float64).reshape(-1, 3)
        if self.quat_order not in QUAT_ORDERS:
            raise ValueError(f"quat_order must be one of {QUAT_ORDERS}, got {self.quat_order!r}")
        n = len(self.t)
        if n < 3:
            raise ValueError(f"clip {self.name}: need at least 3 samples, got {n}")
        if len(self.pos) != n or len(self.quat) != n:
            raise ValueError(
                f"clip {self.name}: length mismatch t={n} pos={len(self.pos)} quat={len(self.quat)}"
            )
        if self.vel is not None and len(self.vel) != n:
            raise ValueError(f"clip {self.name}: vel length {len(self.vel)} != {n}")
        if not np.isfinite(self.t).all():
            raise ValueError(f"clip {self.name}: non-finite timestamps")
        if not (np.diff(self.t) > 0).all():
            raise ValueError(f"clip {self.name}: timestamps are not strictly increasing")
        if not np.isfinite(self.pos).all() or not np.isfinite(self.quat).all():
            raise ValueError(f"clip {self.name}: NaNs in pos/quat; drop rows before building a Clip")
        if not self.fps or not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"clip {self.name}: fps must be a positive finite number, got {self.fps!r}")

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def dt(self) -> float:
        """Nominal sample interval in seconds."""
        return 1.0 / self.fps

    def __repr__(self) -> str:  # keep logs short
        return (
            f"Clip(name={self.name!r}, n={self.n}, fps={self.fps:.2f}, "
            f"dur={self.t[-1] - self.t[0]:.2f}s, quat_order={self.quat_order!r}, "
            f"uptilt={self.cam_uptilt_deg})"
        )
