"""Path library for ANWM inputs: three input regimes in one schema.

A `PathSpec` is a geometric path (time, world position z-up, camera yaw) plus the metadata the
verifier needs. Regimes:
  A  native       ANWM's own released test trajectories (teleport polylines).
  B  real-flight  TII mocap, resampled and rescaled into ANWM's trained speed/yaw envelope.
  C  design       factorial smooth maneuvers with a controlled required tilt.

dt is an *analysis* parameter, never a generation parameter: ANWM consumes relative
displacements and yaw changes, so each path is generated once and `audit()` evaluates the
required tilt at 0.25 / 0.667 / 1.0 s afterwards with `src/physics`. Actions are built by
ANWM's own `_compute_actions` / `normalize_data` / `get_delta_np` (see `anwm_env`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.generation.audit import per_step_tilt, pooled_stats, smoothed_tilt
from src.physics.flatness import G0

ROOT = Path(__file__).resolve().parents[2]

# ANWM's trained envelope (AerialVLN action set; third_party/anwm/config/data_config.yaml).
CAP_H_SPEED = 5.0        # m/s, MOVE_FORWARD_STEP_SIZE / 1 s
CAP_V_SPEED = 2.0        # m/s, GO_UP/DOWN_STEP_SIZE / 1 s
CAP_YAW_RATE = 15.0      # deg/s, TURN_ANGLE / 1 s (Regime B rescaling constraint)
CAP_TRAVEL_TURN = 45.0   # deg per step of *travel direction* change (AerialVLN MOVE_LEFT is 45 deg
                         # at 5 m forward / 5 m lateral); above this the polyline aliases
SPACING = 3.30           # metric_waypoint_spacing (normaliser, not a duration)
K_D = 0.295              # world-frame drag, 1/s (Phase 0 fit; holds for both control modes)
ANALYSIS_DTS = (0.25, 2.0 / 3.0, 1.0)


def spacing_envelope() -> dict:
    """Per-step local displacement allowed by `action_stats` after the 3.30 m normaliser."""
    from src.generation.anwm_env import data_config
    st = data_config()["action_stats"]
    return {"x": (st["min"][0] * SPACING, st["max"][0] * SPACING),
            "y": (st["min"][1] * SPACING, st["max"][1] * SPACING),
            "z": (st["min"][2] * SPACING, st["max"][2] * SPACING)}


@dataclass
class PathSpec:
    name: str
    regime: str                      # "A" | "B" | "C"
    pos: np.ndarray                  # (N, 3) world metres, z-up
    yaw: np.ndarray                  # (N,) radians, heading
    nominal_dt: float = 1.0
    t: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=np.float64).reshape(-1, 3)
        self.yaw = np.asarray(self.yaw, dtype=np.float64).ravel()
        if len(self.yaw) != len(self.pos):
            raise ValueError(f"{self.name}: pos/yaw length mismatch")
        if self.t is None:
            self.t = np.arange(len(self.pos)) * self.nominal_dt
        self.t = np.asarray(self.t, dtype=np.float64)

    # ---- derived quantities -------------------------------------------------
    def speeds(self) -> dict:
        v = np.diff(self.pos, axis=0) / self.nominal_dt
        h = np.linalg.norm(v[:, :2], axis=1)
        yaw_rate = np.degrees(np.diff(self.yaw)) / self.nominal_dt
        step = np.linalg.norm(np.diff(self.pos, axis=0), axis=1)
        return {"peak_h_ms": float(h.max()), "peak_v_ms": float(np.abs(v[:, 2]).max()),
                "peak_speed_ms": float(np.linalg.norm(v, axis=1).max()),
                "peak_yaw_rate_deg_s": float(np.abs(yaw_rate).max()),
                "max_step_m": float(step.max()), "mean_step_m": float(step.mean()),
                "n_steps": int(len(v))}

    def travel_turn_deg(self) -> float:
        """Largest per-step change of horizontal travel direction (deg)."""
        d = np.diff(self.pos[:, :2], axis=0)
        keep = np.linalg.norm(d, axis=1) > 0.5
        ang = np.degrees(np.arctan2(d[keep, 1], d[keep, 0]))
        return float(np.abs((np.diff(ang) + 180) % 360 - 180).max()) if len(ang) > 1 else 0.0

    def caps_violations(self, strict_yaw: bool = False) -> list[str]:
        """`strict_yaw` adds the 15 deg/s yaw-rate cap (Regime B requirement).

        Regime C paths are checked against the travel-direction cap instead, because ANWM's own
        data contains lateral strafes: 45 deg of travel-direction change at 0 deg of yaw change.
        """
        s = self.speeds()
        out = []
        if s["peak_h_ms"] > CAP_H_SPEED + 1e-9:
            out.append(f"horizontal {s['peak_h_ms']:.2f} > {CAP_H_SPEED} m/s")
        if s["peak_v_ms"] > CAP_V_SPEED + 1e-9:
            out.append(f"vertical {s['peak_v_ms']:.2f} > {CAP_V_SPEED} m/s")
        if strict_yaw and s["peak_yaw_rate_deg_s"] > CAP_YAW_RATE + 1e-9:
            out.append(f"yaw {s['peak_yaw_rate_deg_s']:.1f} > {CAP_YAW_RATE} deg/s")
        if self.regime == "C" and self.travel_turn_deg() > CAP_TRAVEL_TURN + 1e-6:
            out.append(f"travel turn {self.travel_turn_deg():.1f} > {CAP_TRAVEL_TURN} deg/step")
        env = spacing_envelope()
        if s["max_step_m"] > 3 * min(-env["x"][0], env["x"][1], env["y"][1], env["z"][1]):
            out.append(f"step {s['max_step_m']:.1f} m outside action_stats envelope")
        return out

    def tilt(self, dt: float | None = None, k_d: float = 0.0, estimator: str = "per_step"):
        dt = self.nominal_dt if dt is None else float(dt)
        if estimator == "per_step":
            return per_step_tilt(self.pos, dt, k_d)
        if estimator == "smoothed":
            return smoothed_tilt(self.pos, dt, k_d)
        raise ValueError(estimator)

    def peak_tilt(self, dt=None, k_d=0.0, estimator="per_step") -> float:
        a = self.tilt(dt, k_d, estimator)
        a = a[np.isfinite(a)]
        return float(a.max()) if a.size else float("nan")

    def audit(self, dts=ANALYSIS_DTS, k_d=K_D) -> dict:
        """Required tilt vs dt, both estimators, each cell one pooled population."""
        out = {}
        for dt in dts:
            cell = {}
            for est in ("per_step", "smoothed"):
                for kname, k in (("drag_free", 0.0), ("k_d", k_d)):
                    st = pooled_stats([self.tilt(dt, k, est)])
                    cell[f"{est}_{kname}"] = st
            out[f"{dt:.3f}"] = cell
        return out

    def summary(self, dts=ANALYSIS_DTS) -> dict:
        return {"name": self.name, "regime": self.regime, "n": int(len(self.pos)),
                "nominal_dt": self.nominal_dt, **self.speeds(),
                "caps_violations": self.caps_violations(),
                "designed_tilt_deg": self.meta.get("target_tilt_deg"),
                "peak_tilt_per_step_drag_free_deg": self.peak_tilt(None, 0.0),
                "peak_tilt_per_step_k_d_deg": self.peak_tilt(None, K_D),
                "peak_tilt_smoothed_drag_free_deg": self.peak_tilt(None, 0.0, "smoothed"),
                "tilt_vs_dt": {f"{dt:.3f}": {
                    "per_step_drag_free": self.peak_tilt(dt, 0.0),
                    "per_step_k_d": self.peak_tilt(dt, K_D),
                    "smoothed_drag_free": self.peak_tilt(dt, 0.0, "smoothed")} for dt in dts},
                "meta": self.meta}

    # ---- ANWM action format -------------------------------------------------
    def anwm_actions(self, start: int = 0, len_traj_pred: int = 16):
        """(actions_normalised, delta) exactly as ANWM's EvalDataset builds them."""
        from src.generation.anwm_env import anwm_modules
        TrainingDataset, get_delta_np, normalize_data, cfg = anwm_modules()
        end = start + len_traj_pred + 1
        if end > len(self.pos):
            raise ValueError(f"{self.name}: need {len_traj_pred + 1} future poses from {start}, "
                             f"have {len(self.pos) - start}")
        traj = {"point": self.pos, "yaw": self.yaw.reshape(-1, 1)}

        class _Shim:  # the only attributes `_compute_actions` touches
            len_traj_pred = 16
            normalize = True
            data_config = cfg["airvln_16"]

        shim = _Shim()
        shim.len_traj_pred = len_traj_pred
        # goal_time must be a sequence: `_compute_actions` indexes with it and then slices 2-D
        actions, _ = TrainingDataset._compute_actions(shim, traj, start, np.array([end - 1]))
        actions[:, :3] = normalize_data(actions[:, :3], {
            k: np.expand_dims(np.asarray(v, dtype=np.float32), 0)
            for k, v in cfg["action_stats"].items()})
        return actions, get_delta_np(actions)


# --------------------------------------------------------------------------- regime A
def from_native(npz_path: str | Path, start: int = 0, n: int = 17, name: str | None = None,
                nominal_dt: float = 1.0) -> PathSpec:
    """Slice of ANWM's own released trajectory (pose keys only, as fetched by scripts/)."""
    z = np.load(npz_path, allow_pickle=False)
    pts = (np.asarray(z["point"], np.float64) if "point" in z.files else
           np.column_stack([np.asarray(z["position"], np.float64),
                            np.zeros(len(z["position"]))]))
    yaw = np.asarray(z["yaw"], np.float64).ravel()
    if start + n > len(pts):
        raise ValueError(f"{npz_path}: only {len(pts)} poses")
    return PathSpec(name=name or f"A:{Path(npz_path).stem[:8]}:{start}", regime="A",
                    pos=pts[start:start + n], yaw=yaw[start:start + n], nominal_dt=nominal_dt,
                    meta={"source": str(npz_path), "start_index": int(start),
                          "n_frames": int(n),
                          "note": "released teleport polyline; step length is a command, not a duration"})


# --------------------------------------------------------------------------- regime B
def from_mocap(pos: np.ndarray, yaw: np.ndarray, t: np.ndarray, t0: float, duration: float,
               nominal_dt: float = 1.0, name: str = "B", v_h_max: float = CAP_H_SPEED,
               v_z_max: float = CAP_V_SPEED, yaw_rate_max: float = CAP_YAW_RATE) -> PathSpec:
    """TII mocap window, rescaled into ANWM's envelope.

    Space scale `lam` and time scale `tau`: v -> (lam/tau) v, a -> (lam/tau^2) a,
    yaw rate -> yawrate/tau. `tau` is fixed by the yaw-rate cap, then `lam` by the speed caps,
    never upscaling space (a slow flight is not inflated to 5 m/s).
    """
    from scipy.interpolate import CubicSpline

    m = (t >= t0) & (t <= t0 + duration)
    if m.sum() < 8:
        raise ValueError("window too short")
    tp, p, y = t[m] - t0, np.asarray(pos, float)[m], np.asarray(yaw, float)[m]
    from scipy.signal import savgol_filter

    v = np.gradient(p, tp, axis=0)            # (N,3)
    # yaw rate from raw finite differences is mocap noise (1000s of deg/s); use the same
    # Savitzky-Golay + high-percentile convention as the Phase 0 floor so the cap reflects
    # real rotation, not jitter
    win = min(101, (len(tp) // 2) * 2 - 1)
    raw_rate = np.abs(savgol_filter(np.unwrap(y), win, 3, deriv=1,
                                    delta=float(np.median(np.diff(tp)))))
    yaw_rate = raw_rate
    hmax = float(np.linalg.norm(v[:, :2], axis=1).max())
    vmax = float(np.abs(v[:, 2]).max())
    tau = max(1.0, float(np.degrees(np.percentile(yaw_rate, 99.5))) / yaw_rate_max)
    lam = tau                                  # speeds unchanged by default
    if hmax > 0:
        lam = min(lam, v_h_max * tau / hmax)
    if vmax > 1e-9:
        lam = min(lam, v_z_max * tau / vmax)
    # stretched time t' = tau * tp, so the new path runs tau times longer. Evaluating the
    # original path at tp = t' / tau gives dp'/dt' = (lam/tau) dp/dt: speeds scale by lam/tau,
    # accelerations by lam/tau^2, yaw rates by 1/tau, as recorded in `meta`.
    grid = np.arange(0.0, tau * tp[-1] + 1e-9, nominal_dt)
    ps = CubicSpline(tp, lam * (p - p[0]), axis=0)(grid / tau)
    ys = CubicSpline(tp, np.unwrap(y))(grid / tau)
    return PathSpec(name=name, regime="B", pos=ps, yaw=ys, nominal_dt=nominal_dt,
                    t=grid, meta={
                        "source": "TII mocap", "window_s": [float(t0), float(t0 + duration)],
                        "length_scale_lam": lam, "time_scale_tau": tau,
                        "accel_scale_lam_over_tau2": lam / tau ** 2,
                        "raw_peak_h_ms": hmax, "raw_peak_v_ms": vmax,
                        "raw_peak_yaw_rate_deg_s": float(np.degrees(yaw_rate.max())),
                        "raw_p99p5_yaw_rate_deg_s": float(np.degrees(np.percentile(yaw_rate, 99.5))),
                        "duration_m": float(np.linalg.norm(ps[-1] - ps[0]))})


# --------------------------------------------------------------------------- regime C
def _yaw_from_course(p: np.ndarray) -> np.ndarray:
    d = np.diff(p[:, :2], axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    return np.concatenate([[ang[0]], ang])


def _cruise(v: float, n: int, dt: float, yaw0: float = 0.0, z0: float = 0.0):
    s = np.arange(n) * v * dt
    p = np.column_stack([s * np.cos(yaw0), s * np.sin(yaw0), np.full(n, z0)])
    return p, np.full(n, yaw0)


def _turn(v: float, kappa: float, n: int, dt: float, yaw0: float = 0.0, z0: float = 0.0):
    tt = np.arange(n) * dt
    psi = yaw0 + v * kappa * tt
    p = np.column_stack([np.sin(psi) / kappa, (1 - np.cos(psi)) / kappa, np.full(n, z0)])
    return p, psi


def _turn_yaw_fixed(v: float, kappa: float, n: int, dt: float, yaw0: float = 0.0):
    p, _ = _turn(v, kappa, n, dt, yaw0)
    return p, np.full(n, yaw0)


def _slalom(v: float, amp: float, wavelength: float, n: int, dt: float,
            yaw_policy: str = "course"):
    tt = np.arange(n) * dt
    k = 2 * np.pi / wavelength
    # constant x-speed chosen so the path speed peaks at v
    vx = v / np.sqrt(1 + (amp * k) ** 2)
    x = vx * tt
    p = np.column_stack([x, amp * np.sin(k * x), np.zeros(n)])
    return p, (_yaw_from_course(p) if yaw_policy == "course" else np.zeros(n))


def _quintic(u: np.ndarray) -> np.ndarray:
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def _accel_brake(v0: float, v_peak: float, ramp_s: float, n: int, dt: float, yaw0: float = 0.0):
    """Quintic speed ramp v0 -> v_peak over `ramp_s`, hold, then mirror back to v0.

    Peak longitudinal acceleration is 1.875 (v_peak - v0) / ramp_s, so a short ramp reaches
    high required pitch inside the 5 m/s speed cap.
    """
    T = (n - 1) * dt
    ramp_s = min(ramp_s, T / 2)
    tt = np.arange(n) * dt
    u = np.clip(tt / ramp_s, 0, 1)
    up = _quintic(u)
    v = np.where(tt <= ramp_s,
                 v0 + (v_peak - v0) * up,
                 np.where(tt <= T - ramp_s, v_peak,
                          v0 + (v_peak - v0) * _quintic(np.clip((T - tt) / ramp_s, 0, 1))))
    s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]))]) * dt
    p = np.column_stack([s, np.zeros(n), np.zeros(n)])
    return p, np.full(n, yaw0)


def _climb_turn(v: float, kappa: float, vz: float, n: int, dt: float, yaw0: float = 0.0,
                vz_ramp_s: float = 2.0):
    """Helix: horizontal arc plus a quintic vertical-speed ramp to `vz`."""
    p, psi = _turn(v, kappa, n, dt, yaw0)
    tt = np.arange(n) * dt
    u = np.clip(tt / vz_ramp_s, 0, 1)
    p[:, 2] = np.cumsum(np.concatenate([[0.0], 0.5 * (vz * _quintic(u[1:]) +
                                                   vz * _quintic(u[:-1]))])) * dt
    return p, psi


def _accel_brake_steps(v0: float, a: float, n: int, dt: float, yaw0: float = 0.0):
    """Accelerate then brake *in step-velocity space*: v_k changes by +-a per step.

    The path is the cumulative sum of the step-average velocities, so its per-step required
    tilt is exactly atan(a / g) drag-free (an upward speed change means the thrust vector
    tilts back, a downward one tilts forward). Working in step space matters because a 1 s
    sample cannot resolve a ramp shorter than a step, and the per-step estimator is the
    honest reading of what the commanded polyline asks for.
    """
    v = np.empty(n)
    v[0] = v0
    half = n // 2
    for k in range(1, half + 1):
        v[k] = v[k - 1] + a * dt
    for k in range(half + 1, n):
        v[k] = max(v[k - 1] - a * dt, 0.0)
    v = np.clip(v, -CAP_H_SPEED, CAP_H_SPEED)
    s = np.concatenate([[0.0], np.cumsum(v[:-1])]) * dt
    p = np.column_stack([s, np.zeros(n), np.zeros(n)])
    return p, np.full(n, yaw0)


def _amplitudes(lo: float, hi: float, n: int = 120):
    return np.linspace(lo, hi, n)


def _design_one(maneuver: str, target: float, speed: float, n: int, dt: float,
                yaw_policy: str, tol: float = 0.5):
    """Pick the amplitude whose drag-free peak per-step tilt is closest to `target`."""
    def build(amp):
        if maneuver == "turn":
            return _turn(speed, amp, n, dt) if yaw_policy == "course" else \
                _turn_yaw_fixed(speed, amp, n, dt)
        if maneuver == "slalom":
            return _slalom(speed, amp, 40.0, n, dt, yaw_policy)
        if maneuver == "climb_turn":
            p, y = _climb_turn(speed, amp, 1.5, n, dt)
            return (p, y) if yaw_policy == "course" else (p, np.zeros(n))
        raise ValueError(maneuver)

    best = None
    # turn / climb_turn amplitude is curvature; cap it below the 45 deg/step aliasing bound
    kappa_max = min(1.0, np.radians(CAP_TRAVEL_TURN) / (speed * dt))
    hi = {"turn": kappa_max, "slalom": 20.0, "climb_turn": kappa_max}[maneuver]
    for amp in _amplitudes(1e-4, hi):
        p, y = build(amp)
        ps = PathSpec(name="cand", regime="C", pos=p, yaw=y, nominal_dt=dt)
        viol = ps.caps_violations(strict_yaw=(yaw_policy == "course"))
        if viol:
            continue
        t_peak = ps.peak_tilt(dt, 0.0)
        score = abs(t_peak - target)
        if best is None or score < best[0]:
            best = (score, amp, p, y, t_peak, viol)
    if best is None:
        raise RuntimeError(f"{maneuver}: no amplitude satisfied the caps")
    score, amp, p, y, t_peak, viol = best
    return p, y, {"amplitude": float(amp), "achieved_tilt_deg": float(t_peak),
                  "reachable": bool(score <= tol)}


def design(maneuver: str, target_tilt_deg: float | None, speed: float = 5.0, n: int = 17,
           dt: float = 1.0, yaw_policy: str = "course", name: str | None = None) -> PathSpec:
    """One Regime C maneuver at one required-tilt target.

    `target_tilt_deg = None` gives the straight-cruise control (no shaping).
    """
    if maneuver == "cruise" or target_tilt_deg is None:
        p, y = _cruise(speed, n, dt)
        meta = {"achieved_tilt_deg": 0.0, "reachable": True, "note": "drag-free cruise"}
        
    elif maneuver == "accel_brake":
        best = None
        for v0 in np.linspace(0.0, speed, 21):
            for a in np.linspace(0.05, G0 * np.tan(np.radians(35.0)), 120):
                p, y = _accel_brake_steps(v0, a, n, dt)
                ps = PathSpec(name="cand", regime="C", pos=p, yaw=y, nominal_dt=dt)
                if ps.caps_violations():
                    continue
                t_peak = ps.peak_tilt(dt, 0.0)
                sc = abs(t_peak - target_tilt_deg)
                if best is None or sc < best[0]:
                    best = (sc, v0, a, p, y, t_peak)
        if best is None:
            raise RuntimeError("accel_brake: no feasible profile")
        sc, v0, a, p, y, t_peak = best
        meta = {"v_start_ms": float(v0), "accel_ms2": float(a),
                "achieved_tilt_deg": float(t_peak), "reachable": bool(sc <= 0.5)}
    else:
        p, y, extra = _design_one(maneuver, target_tilt_deg, speed, n, dt, yaw_policy)
        meta = {**extra}
    spec = PathSpec(name=name or f"C:{maneuver}:{target_tilt_deg}:{speed}:{yaw_policy}",
                    regime="C", pos=p, yaw=y, nominal_dt=dt,
                    meta={"maneuver": maneuver, "speed_ms": speed, "yaw_policy": yaw_policy,
                          "target_tilt_deg": target_tilt_deg, "nominal_dt": dt,
                          "reachable_deg_target": meta["reachable"], **meta})
    if spec.caps_violations():
        raise RuntimeError(f"{spec.name}: caps violated {spec.caps_violations()}")
    return spec
