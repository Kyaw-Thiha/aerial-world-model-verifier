"""Analytic flights: exact position, velocity, acceleration and attitude at a high rate.

Why analytic: the ground truth of this track has to be *exact*, so the path is written in
closed form and differentiated in closed form. No Savitzky-Golay, no renderer-side
integration, no dependence on a physics engine's substepping. The renderer only consumes
``(t, R_wc, p_wc, K)``; everything else is the verifier's truth.

Conventions are the project's, stated once in ``src/perception/frames.py`` and
``docs/decisions.md``:

* world z up, SI, angles in degrees only at the reporting boundary;
* body FLU (x forward, y left, z up), body z is the thrust axis;
* camera OpenCV RDF (x right, y down, z optical axis); ``R_cb`` maps camera->body, so the
  camera-to-world rotation is ``R_bw @ R_cb``;
* attitude from flatness with the project's single drag coefficient,
  ``z_B = unit(a + g e3 + k_d v)``, and heading is the yaw of the path.

A ``Flight`` carries both the commanded (physical) attitude and, separately, the attitude
the *camera* is rendered with. That split is the whole point of the track: a defect is a
mismatch between the two, and in a simulator it can be applied on purpose.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.generation.attitude import body_rotation, roll_pitch_from_thrust
from src.perception.frames import camera_to_body_pitched
from src.physics.flatness import G0, implied_thrust_axis

E3 = np.array([0.0, 0.0, 1.0])
K_D = 0.295          # 1/s, the project's world-frame drag coefficient
GT_HZ = 250.0        # ground-truth rate; the handoff asks for >= 200 Hz
VIDEO_FPS = 30.0


# --------------------------------------------------------------------------- flight
@dataclass
class Flight:
    """One flight: analytic truth at ``gt_hz`` plus the render scheduling.

    ``camera_R_bw`` is what the *camera* is physically attached to; for a feasible flight it
    equals ``R_bw`` and ``camera_R_cb`` is the fixed mount. A defective flight replaces
    ``camera_R_bw`` (e.g. an identity rotation for tilt removal) while ``R_bw`` stays the
    truth the verifier should have inferred from the path.
    """

    name: str
    t: np.ndarray                 # (N,) seconds, uniform, starts at 0
    pos: np.ndarray               # (N,3) world metres
    vel: np.ndarray               # (N,3)
    acc: np.ndarray               # (N,3)
    yaw: np.ndarray               # (N,) radians, heading of the path
    gt_hz: float = GT_HZ
    video_fps: float = VIDEO_FPS
    k_d: float = K_D
    meta: dict = field(default_factory=dict)

    # filled by __post_init__ / camera_poses
    thrust_axis: np.ndarray | None = None      # (N,3) commanded body z in world
    roll: np.ndarray | None = None             # (N,) radians
    pitch: np.ndarray | None = None
    R_bw: np.ndarray | None = None             # (N,3,3) commanded body->world
    camera_uptilt_deg: float | None = None     # fixed mount pitch, degrees
    camera_R_cb: np.ndarray | None = None      # (3,3) camera->body
    camera_R_bw: np.ndarray | None = None      # (N,3,3) attitude the camera is rendered on
    frame_t: np.ndarray | None = None          # (M,) render timestamps
    frame_idx: np.ndarray | None = None         # (M,) indices into the GT arrays

    def __post_init__(self) -> None:
        for k in ("pos", "vel", "acc"):
            setattr(self, k, np.asarray(getattr(self, k), dtype=np.float64).reshape(-1, 3))
        self.t = np.asarray(self.t, dtype=np.float64).ravel()
        self.yaw = np.asarray(self.yaw, dtype=np.float64).ravel()
        n = len(self.t)
        for k in ("pos", "vel", "acc", "yaw"):
            if len(getattr(self, k)) != n:
                raise ValueError(f"{k} has {len(getattr(self, k))} rows, t has {n}")
        if n < 2:
            raise ValueError("a flight needs at least two samples")
        dt = np.diff(self.t)
        if not np.allclose(dt, dt[0], rtol=0, atol=1e-9):
            raise ValueError("flight timestamps must be uniform; resample before constructing")
        self._attitude_from_flatness()

    # -- commanded side -----------------------------------------------------------
    def _attitude_from_flatness(self) -> None:
        self.thrust_axis = implied_thrust_axis(self.acc, self.vel, self.k_d, G0)
        self.roll, self.pitch = roll_pitch_from_thrust(self.thrust_axis, self.yaw)
        self.R_bw = body_rotation(self.yaw, self.pitch, self.roll)
        # The construction is exact, but the whole track rests on it, so it is checked
        # against the definition rather than trusted (a silent 90 deg error here would
        # look like a verifier failure later).
        got = self.R_bw[:, :, 2]
        if not np.allclose(got, self.thrust_axis, atol=1e-9):
            worst = float(np.degrees(np.arccos(np.clip((got * self.thrust_axis).sum(1), -1, 1))).max())
            raise ValueError(f"attitude/ thrust-axis mismatch of {worst:.3g} deg")

    # -- camera side ---------------------------------------------------------------
    def camera_poses(self, uptilt_deg: float = 0.0, level_camera: bool = False,
                     mount_yaw_deg: float = 0.0, lever_m: float = 0.0,
                     mirror_roll: bool = False) -> dict:
        """Camera-to-world poses for rendering.

        Parameters
        ----------
        uptilt_deg : fixed mount pitch (camera optical axis above the body x axis).
        level_camera : render the camera level no matter what the body does (this is the
            tilt-removal defect: the path still demands tilt, the pixels do not show it).
        mount_yaw_deg : fixed mount yaw about body z (0 = camera looking along body x).
        lever_m : camera displacement along body z from the body origin, metres. 0 puts the
            camera at the centre of mass, which is what the flatness model assumes.
        mirror_roll : render the camera with the sign of the body roll flipped and everything
            else (yaw, pitch, position, speed) unchanged -- the mirrored-roll defect. The
            required lean is unchanged, so the depicted lean's roll component flips sign.
        """
        n = len(self.t)
        self.camera_uptilt_deg = float(uptilt_deg)
        R_cb = camera_to_body_pitched(np.radians(uptilt_deg))
        if abs(mount_yaw_deg) > 1e-12:
            c, s = np.cos(np.radians(mount_yaw_deg)), np.sin(np.radians(mount_yaw_deg))
            Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            R_cb = Rz @ R_cb
        self.camera_R_cb = R_cb
        if level_camera:
            # a level camera still points somewhere: keep the path's heading
            R_bw = body_rotation(self.yaw, np.zeros(n), np.zeros(n))
        elif mirror_roll:
            R_bw = body_rotation(self.yaw, self.pitch, -self.roll)
        else:
            R_bw = self.R_bw
        self.camera_R_bw = R_bw
        R_wc = R_bw @ R_cb
        offset = np.zeros((n, 3)) if lever_m == 0 else R_bw[:, :, 2] * float(lever_m)
        p_wc = self.pos + offset
        poses = np.tile(np.eye(4), (n, 1, 1))
        poses[:, :3, :3] = R_wc
        poses[:, :3, 3] = p_wc
        return self.schedule(video_fps=self.video_fps, poses=poses)

    def schedule(self, video_fps: float | None = None, poses: np.ndarray | None = None) -> dict:
        """Pick the render frames on a uniform ``video_fps`` grid and pack the render job."""
        fps = float(self.video_fps if video_fps is None else video_fps)
        duration = self.t[-1] - self.t[0]
        m = int(np.floor(duration * fps + 1e-9)) + 1
        ft = np.arange(m) / fps
        idx = np.round(ft * self.gt_hz).astype(int).clip(0, len(self.t) - 1)
        self.frame_t = ft
        self.frame_idx = idx
        return {"frame_t": ft, "frame_idx": idx, "poses": None if poses is None else poses[idx]}

    # -- reporting -----------------------------------------------------------------
    def tilt_deg(self, mask: np.ndarray | None = None) -> np.ndarray:
        """Commanded body tilt = angle between thrust axis and world up, degrees."""
        cos = np.clip(self.thrust_axis @ E3, -1.0, 1.0)
        out = np.degrees(np.arccos(cos))
        return out if mask is None else out[mask]

    def summary(self) -> dict:
        tilt = self.tilt_deg()
        speed = np.linalg.norm(self.vel, axis=1)
        acc = np.linalg.norm(self.acc, axis=1)
        return {
            "name": self.name,
            "duration_s": float(self.t[-1] - self.t[0]),
            "gt_hz": float(self.gt_hz),
            "n_gt": int(len(self.t)),
            "video_fps": float(self.video_fps),
            "n_frames": int(len(self.frame_t)) if self.frame_t is not None else None,
            "k_d": float(self.k_d),
            "speed_max": float(speed.max()),
            "accel_max": float(acc.max()),
            "tilt_med_deg": float(np.median(tilt)),
            "tilt_p95_deg": float(np.percentile(tilt, 95)),
            "tilt_max_deg": float(tilt.max()),
            "frac_tilt_gt10_deg": float((tilt > 10).mean()),
            "camera_uptilt_deg": self.camera_uptilt_deg,
            **self.meta,
        }


# ------------------------------------------------------------------------ maneuvers
def _rollout(name: str, t: np.ndarray, pos, vel, acc, yaw, **meta) -> Flight:
    n = len(t)
    return Flight(name=name, t=t, pos=pos, vel=vel, acc=acc,
                  yaw=np.broadcast_to(np.asarray(yaw, dtype=np.float64), (n,)).copy(), meta=meta)


def hover(duration: float = 4.0, pos=(0.0, 0.0, 10.0), yaw: float = 0.0,
          gt_hz: float = GT_HZ) -> Flight:
    """Level hover: zero acceleration, so the thrust axis is world up and tilt is 0 deg."""
    t = np.arange(0.0, duration + 1e-9, 1.0 / gt_hz)
    p = np.tile(np.asarray(pos, dtype=np.float64), (len(t), 1))
    z = np.zeros_like(p)
    return _rollout("hover", t, p, z, z, yaw, maneuver="hover")


def ellipse(radius: float = 10.0, speed: float = 8.0, height: float = 12.0,
            duration: float | None = None, clockwise: bool = False,
            gt_hz: float = GT_HZ) -> Flight:
    """Horizontal circle at constant speed; peak tilt = atan(w^2 r / g) (plus drag).

    ``speed`` is the tangential speed: w = v / r.
    """
    w = speed / radius * (-1.0 if clockwise else 1.0)
    duration = (2 * np.pi / abs(w)) if duration is None else duration
    t = np.arange(0.0, duration + 1e-9, 1.0 / gt_hz)
    u = w * t
    pos = np.column_stack([radius * np.cos(u), radius * np.sin(u), np.full_like(u, height)])
    vel = np.column_stack([-radius * w * np.sin(u), radius * w * np.cos(u), np.zeros_like(u)])
    acc = np.column_stack([-radius * w * w * np.cos(u), -radius * w * w * np.sin(u), np.zeros_like(u)])
    yaw = np.arctan2(vel[:, 1], vel[:, 0])          # heading along the path
    return _rollout("ellipse", t, pos, vel, acc, yaw, maneuver="ellipse",
                    radius=radius, speed=speed, height=height)


def lemniscate(scale: float = 8.0, speed: float = 8.0, height: float = 12.0,
               duration: float | None = None, gt_hz: float = GT_HZ) -> Flight:
    """Gerono lemniscate (figure of eight) parameterised by arc-rate ``w``.

    x = a sin(u), y = a sin(u) cos(u), u = w t. Derivatives are closed form; ``speed`` sets
    ``w`` by the small-angle relation ``w = speed / a`` and is therefore only approximate,
    which is reported in the summary.
    """
    a = scale
    w = speed / a
    duration = (2 * np.pi / w) if duration is None else duration
    t = np.arange(0.0, duration + 1e-9, 1.0 / gt_hz)
    u = w * t
    su, cu = np.sin(u), np.cos(u)
    pos = np.column_stack([a * su, a * su * cu, np.full_like(u, height)])
    vx = a * w * cu
    vy = a * w * (cu * cu - su * su)
    ax = -a * w * w * su
    ay = -4.0 * a * w * w * su * cu
    vel = np.column_stack([vx, vy, np.zeros_like(u)])
    acc = np.column_stack([ax, ay, np.zeros_like(u)])
    yaw = np.arctan2(vy, vx)
    return _rollout("lemniscate", t, pos, vel, acc, yaw, maneuver="lemniscate",
                    scale=scale, speed_param=speed, speed_peak_approx=speed)


def slalom(speed: float = 10.0, amplitude: float = 4.0, wavelength: float = 16.0,
           duration: float = 8.0, height: float = 12.0, gt_hz: float = GT_HZ) -> Flight:
    """x advances at ``speed``, y oscillates sinusoidally in x, z constant."""
    t = np.arange(0.0, duration + 1e-9, 1.0 / gt_hz)
    k = 2 * np.pi / wavelength
    x = speed * t
    y = amplitude * np.sin(k * x)
    vx = np.full_like(t, speed)
    vy = amplitude * k * speed * np.cos(k * x)
    ax = np.zeros_like(t)
    ay = -amplitude * k * k * speed * speed * np.sin(k * x)
    pos = np.column_stack([x, y, np.full_like(t, height)])
    vel = np.column_stack([vx, vy, np.zeros_like(t)])
    acc = np.column_stack([ax, ay, np.zeros_like(t)])
    yaw = np.arctan2(vy, vx)
    return _rollout("slalom", t, pos, vel, acc, yaw, maneuver="slalom",
                    speed=speed, amplitude=amplitude, wavelength=wavelength)


def turn(speed: float = 12.0, radius: float = 6.0, arc_deg: float = 180.0,
         height: float = 12.0, gt_hz: float = GT_HZ) -> Flight:
    """Constant-speed circular arc: sustained lateral acceleration v^2/r."""
    w = speed / radius
    duration = np.radians(arc_deg) / w
    t = np.arange(0.0, duration + 1e-9, 1.0 / gt_hz)
    u = w * t
    pos = np.column_stack([radius * np.sin(u), radius * (1 - np.cos(u)), np.full_like(u, height)])
    vel = np.column_stack([speed * np.cos(u), speed * np.sin(u), np.zeros_like(u)])
    acc = np.column_stack([-speed * w * np.sin(u), speed * w * np.cos(u), np.zeros_like(u)])
    yaw = u
    return _rollout("turn", t, pos, vel, acc, yaw, maneuver="turn",
                    speed=speed, radius=radius, arc_deg=arc_deg)


MANEUVERS = {"hover": hover, "ellipse": ellipse, "lemniscate": lemniscate,
             "slalom": slalom, "turn": turn}


def build(maneuver: str, **kwargs) -> Flight:
    if maneuver not in MANEUVERS:
        raise KeyError(f"unknown maneuver {maneuver!r}; have {sorted(MANEUVERS)}")
    return MANEUVERS[maneuver](**kwargs)


# --------------------------------------------------------------------- projection
def pinhole_K(width: int = 640, height: int = 360, hfov_deg: float = 90.0) -> np.ndarray:
    """Square-pixel pinhole intrinsics with the principal point at the image centre."""
    f = 0.5 * width / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array([[f, 0.0, (width - 1) / 2.0],
                     [0.0, f, (height - 1) / 2.0],
                     [0.0, 0.0, 1.0]])


def project_world_points(poses: np.ndarray, K: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Project world points into the cameras of ``poses`` (camera-to-world), (N,M,2)."""
    poses = np.asarray(poses, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out = np.full((len(poses), len(points), 2), np.nan)
    for i, T in enumerate(poses):
        R_wc, p_wc = T[:3, :3], T[:3, 3]
        pc = (points - p_wc) @ R_wc          # R_cw (points - p)
        z = pc[:, 2]
        ok = z > 1e-6
        pix = (pc[ok] @ K.T)[:, :2] / z[ok, None]
        out[i, ok] = pix
    return out
