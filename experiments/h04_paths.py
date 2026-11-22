"""H04: the continuous-attitude commanded paths.

A path is a closed-form position ``p(t)`` on a 1 kHz grid over the 5.0625 s clip (81 frames
at 16 fps).  The attitude-active window is the middle 3.0 s with a 0.5 s raised-cosine
ramp-in/out.

The design is written in terms of the commanded **bank profile** ``phi(t)`` because that is
the quantity the campaign asks the generator to follow:

    turn rate   omega(t) = -(g / v0) * tan(phi(t))
    heading     psi(t)   = integral omega dt
    position    p(t)     = integral v0 (cos psi, sin psi) dt

so a smooth sinusoid in ``phi`` gives a smooth, continuously changing commanded attitude at a
known peak.  Velocity and acceleration are recovered from the 1 kHz positions with the same
Savitzky-Golay differentiation the frozen statistic uses, and the commanded attitude is the
flatness attitude ``z_B = unit(a + g e3 + k_d v)`` with ``k_d = 0.295``.  The reported peak is
the peak of that flatness attitude, which includes the 4.8 deg drag pitch at 2.8 m/s; the
requested peak is the peak *bank* (|roll|) of the commanded attitude.

"Peak" therefore means the requested peak bank in degrees; the achieved peak bank, the peak
tilt and the maximum per-frame attitude rate are all recorded and checked.

    pixi run -e gpu python experiments/h04_paths.py --out results/h04/paths
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from src.generation.attitude import roll_pitch_from_thrust
from src.physics.flatness import G0, implied_thrust_axis

ROOT = Path(__file__).resolve().parents[1]
RATE_HZ = 1000.0
DT = 1.0 / RATE_HZ
CLIP_S = 81.0 / 16.0
T0, T1 = 1.0, 4.0
RAMP_S = 0.5
V0 = 2.8
K_D = 0.295

# peak per-frame commanded attitude change allowed by the design check (deg/frame at 16 fps)
MAX_RATE_DEG_PER_FRAME = 4.6


def _smoothstep(s: np.ndarray) -> np.ndarray:
    """Quintic smoothstep on [0, 1]: value, first and second derivatives vanish at both ends."""
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def envelope(t: np.ndarray) -> np.ndarray:
    """C2 ramp-in/out over the active window (a second derivative must exist: the required
    lean is a second derivative of position)."""
    e = np.zeros_like(t)
    act = (t >= T0) & (t <= T1)
    x = t[act] - T0
    span = T1 - T0
    e[act] = np.where(x < RAMP_S, _smoothstep(x / RAMP_S),
                      np.where(x > span - RAMP_S, _smoothstep((span - x) / RAMP_S), 1.0))
    return e


def x_of(t: np.ndarray) -> np.ndarray:
    return np.clip(t - T0, 0.0, T1 - T0)


def roll_profile(family: str, peak_deg: float, t: np.ndarray) -> np.ndarray:
    """Commanded bank angle (rad) over the clip."""
    e = envelope(t)
    x = x_of(t)
    span = T1 - T0
    p = np.deg2rad(peak_deg)
    if family in ("arc", "climb_turn"):
        return p * np.sin(np.pi * x / span) * e
    if family == "s":
        return p * np.sin(2 * np.pi * x / span) * e
    if family == "slalom":
        # 1.5 cycles over the active window (3 lobes); see amendment 1 for the rate ceiling
        return p * np.sin(3 * np.pi * x / span) * e
    if family == "fig8":
        # figure-eight-like asymmetric S: one heading cycle with 0.6 : 1 lobes
        prof = np.sin(2 * np.pi * x / span) * (0.6 + 0.4 * np.cos(2 * np.pi * x / span))
        return p * prof * e / max(np.max(np.abs(prof)), 1e-9)
    raise ValueError(family)


def pitch_profile(family: str, peak_deg: float, t: np.ndarray) -> np.ndarray:
    """Commanded extra pitch (rad) for the pitch-only and climb variants."""
    e = envelope(t)
    x = x_of(t)
    span = T1 - T0
    p = np.deg2rad(peak_deg)
    if family == "climb_turn":
        return 0.5 * p * np.sin(np.pi * x / span) * e
    return np.zeros_like(t)


def positions(family: str, peak_deg: float, t: np.ndarray,
              aux: float | None = None) -> np.ndarray:
    """Positions for one path; ``aux`` is the vertical/climb or speed amplitude."""
    span = T1 - T0
    x = x_of(t)
    e = envelope(t)
    if family == "accel_brake":
        m = _default_aux("accel_brake", peak_deg) if aux is None else aux
        v = V0 * (1.0 + m * (np.sin(np.pi * x / span) ** 2 - 0.5) * e)
        pos = np.zeros((len(t), 3))
        pos[:, 0] = np.cumsum(v) * DT
        return pos
    phi = roll_profile(family, peak_deg, t)
    omega = -(G0 / V0) * np.tan(phi)
    psi = np.cumsum(omega) * DT
    pos = np.zeros((len(t), 3))
    pos[:, 0] = np.cumsum(V0 * np.cos(psi)) * DT
    pos[:, 1] = np.cumsum(V0 * np.sin(psi)) * DT
    if family == "climb_turn":
        z = _climb_amp(peak_deg) if aux is None else aux
        pos[:, 2] = z * np.sin(np.pi * x / span) ** 2 * e
    return pos


def _default_aux(family: str, peak_deg: float) -> float:
    if family == "accel_brake":
        return np.tan(np.deg2rad(peak_deg)) * G0 / (V0 * 2 * np.pi / (T1 - T0))
    return 0.0


def _climb_amp(peak_deg: float) -> float:
    """Vertical amplitude whose peak vertical acceleration equals 0.5 g tan(peak/2)."""
    span = T1 - T0
    a_z = 0.5 * G0 * np.tan(np.deg2rad(peak_deg) / 2.0)
    return a_z / (2.0 * (np.pi / span) ** 2)


def _attitude_series(family: str, scale: float, t: np.ndarray) -> dict:
    pos = positions(family, scale, t)
    return attitude(pos, derivatives(pos))


def peak_metric(family: str, scale: float, t: np.ndarray) -> float:
    """Achieved peak of the design metric: |roll| (bank), or |pitch| for accel_brake."""
    a = _attitude_series(family, scale, t)
    m = (t >= T0) & (t <= T1)
    key = "pitch" if family == "accel_brake" else "roll"
    return float(np.max(np.abs(np.degrees(a[key])[m])))


def solve_scale(family: str, target_deg: float, t: np.ndarray) -> float:
    lo, hi = 1e-4, max(4.0 * target_deg, 1.0)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if peak_metric(family, mid, t) < target_deg:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def derivatives(pos: np.ndarray, window_s: float = 0.37) -> dict:
    w = int(round(window_s / DT))
    w += (w + 1) % 2
    vel = savgol_filter(pos, w, 3, deriv=1, delta=DT, axis=0, mode="interp")
    acc = savgol_filter(pos, w, 3, deriv=2, delta=DT, axis=0, mode="interp")
    return {"vel": vel, "acc": acc, "window": w}


def attitude(pos: np.ndarray, deriv: dict) -> dict:
    z = implied_thrust_axis(deriv["acc"], deriv["vel"], K_D, G0)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    yaw = np.unwrap(np.arctan2(deriv["vel"][:, 1], deriv["vel"][:, 0]))
    phi, theta = roll_pitch_from_thrust(z, yaw)
    tilt = np.degrees(np.arccos(np.clip(z[:, 2], -1.0, 1.0)))
    return {"z_flat": z, "yaw": yaw, "roll": phi, "pitch": theta, "tilt_deg": tilt}


PATHS = [
    ("arc_5", "arc", 5.0), ("arc_10", "arc", 10.0), ("arc_20", "arc", 20.0),
    ("s_5", "s", 5.0), ("s_10", "s", 10.0), ("s_20", "s", 20.0),
    ("slalom_5", "slalom", 5.0), ("slalom_10", "slalom", 10.0),
    ("slalom_20", "slalom", 20.0),
    ("fig8_5", "fig8", 5.0), ("fig8_10", "fig8", 10.0), ("fig8_20", "fig8", 20.0),
    ("arc_8", "arc", 8.0), ("s_8", "s", 8.0), ("slalom_8", "slalom", 8.0),
    ("fig8_8", "fig8", 8.0),
    ("climb_turn_10", "climb_turn", 10.0), ("climb_turn_20", "climb_turn", 20.0),
    ("accel_brake_10", "accel_brake", 10.0), ("accel_brake_12", "accel_brake", 12.0),
    ("s_12", "s", 12.0), ("slalom_12", "slalom", 12.0),
]
# first frame per path index, fixed by the pre-registration: F(1 + i mod 6)
FRAMES = ["F1", "F2", "F3", "F4", "F5", "F6"]


def check(name: str, family: str, target: float, t: np.ndarray) -> dict:
    scale = solve_scale(family, target, t)
    pos = positions(family, scale, t)
    d = derivatives(pos)
    a = attitude(pos, d)
    act = (t >= T0) & (t <= T1)
    idx = np.round(np.linspace(0, len(t) - 1, 81)).astype(int)
    roll16 = np.degrees(a["roll"][idx])
    pitch16 = np.degrees(a["pitch"][idx])
    thrust_ratio = np.linalg.norm(d["acc"] + G0 * np.array([0, 0, 1.0]), axis=1) / G0
    rel = pos - pos[0]
    m16 = (t[idx] >= T0) & (t[idx] <= T1)
    tilt16 = a["tilt_deg"][idx]
    return {"path": name, "family": family, "requested_peak_bank_deg": target,
            "design_metric": "pitch" if family == "accel_brake" else "bank",
            "solved_amplitude": float(scale),
            "achieved_peak_bank_deg": float(np.max(np.abs(roll16[m16]))),
            "achieved_peak_pitch_deg": float(np.max(np.abs(pitch16[m16]))),
            "peak_tilt_deg": float(np.max(tilt16[m16])),
            "median_tilt_deg": float(np.median(tilt16[m16])),
            "roll_minmax_deg": [float(roll16[m16].min()), float(roll16[m16].max())],
            "max_roll_rate_deg_per_frame": float(np.max(np.abs(np.diff(roll16)))),
            "max_pitch_rate_deg_per_frame": float(np.max(np.abs(np.diff(pitch16)))),
            "peak_thrust_ratio": float(np.max(thrust_ratio)),
            "max_rel_translation_m": float(np.linalg.norm(rel, axis=1).max()),
            "n_roll_sign_changes": int(np.sum(np.diff(np.sign(np.round(roll16, 1))) != 0)),
            "first_frame": FRAMES[PATHS.index(next(p for p in PATHS if p[0] == name)) % 6]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/h04/paths"))
    args = ap.parse_args()
    t = np.arange(0.0, CLIP_S, DT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    idx81 = np.round(np.linspace(0, len(t) - 1, 81)).astype(int)
    arrays, summary = {}, []
    for name, family, target in PATHS:
        info = check(name, family, target, t)
        summary.append(info)
        pos = positions(family, info["solved_amplitude"], t)
        d = derivatives(pos)
        att = attitude(pos, d)
        arrays[f"{name}__t"] = t[idx81]
        arrays[f"{name}__pos"] = pos[idx81]
        arrays[f"{name}__vel"] = d["vel"][idx81]
        arrays[f"{name}__acc"] = d["acc"][idx81]
        for k in ("yaw", "roll", "pitch", "tilt_deg"):
            arrays[f"{name}__{k}"] = att[k][idx81]
        arrays[f"{name}__t_1khz"] = t[::10]
        arrays[f"{name}__pos_1khz"] = pos[::10]
        print(f"{name:16s} bank {info['achieved_peak_bank_deg']:5.2f}/{target:4.0f} "
              f"tilt {info['peak_tilt_deg']:5.2f} thrust {info['peak_thrust_ratio']:4.2f} "
              f"rel {info['max_rel_translation_m']:5.1f} "
              f"rate {max(info['max_roll_rate_deg_per_frame'], info['max_pitch_rate_deg_per_frame']):4.2f}",
              flush=True)
    np.savez_compressed(out / "paths.npz", **arrays)
    (out / "paths_summary.json").write_text(json.dumps(summary, indent=1))
    bad = [s["path"] for s in summary
           if abs((s["achieved_peak_pitch_deg"] if s["design_metric"] == "pitch"
                   else s["achieved_peak_bank_deg"]) - s["requested_peak_bank_deg"]) > 0.2
           or s["peak_thrust_ratio"] >= 2.0 or s["max_rel_translation_m"] > 16.0
           or max(s["max_roll_rate_deg_per_frame"], s["max_pitch_rate_deg_per_frame"])
           > MAX_RATE_DEG_PER_FRAME]
    print(json.dumps({"paths": len(summary), "failed_checks": bad}, indent=1))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
