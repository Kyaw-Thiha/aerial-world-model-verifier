"""X01: the common path set for four path-conditioned aerial generators.

Builds the 26 paths of ``configs/x01_common_paths_prereg.yaml`` (24 design/real paths plus two
scene-swap copies) as commanded trajectories on each model's own native frame grid, checks every
instantiation against the design rule, and writes the per-model inputs the generation apps mount.

A path is a FAMILY and a requested peak; each model gets the same family at the same peak on its
own time base (the three camera models at 16 fps over 5 s, ANWM at its 1 s action steps over
17 s).  The requested peak is solved by bisection on the achieved peak of the commanded attitude,
as H04 did.

Estimators: the camera models are differentiated on the 1 kHz design grid with the frozen 0.37 s
Savitzky-Golay window and then sampled at their frame times (H04's route); ANWM is evaluated with
the per-step second difference its action space actually commands (``src/generation/audit.py``).

    pixi run python experiments/x01_common_paths.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.attitude import body_rotation, roll_pitch_from_thrust  # noqa: E402
from src.perception.frames import camera_to_body_pitched  # noqa: E402
from src.physics.flatness import G0, implied_thrust_axis  # noqa: E402
from src.verdict.lean import world_to_yaw_frame  # noqa: E402

RATE_HZ = 1000.0
DT = 1.0 / RATE_HZ
K_D = 0.295
V0 = 2.8
S_AMP = 0.0
WINDOW_S = 0.37

PROMPT = ("Aerial drone footage over a sunlit city with a river, tall buildings and a "
          "hazy horizon, photorealistic, camera moving over the city.")

CAMERA_MODELS = {
    "solarwm": {"frames": 81, "fps": 16.0, "size": (864, 480), "f_px": 240.0},
    "sana": {"frames": 81, "fps": 16.0, "size": (1280, 704), "f_px": 352.0},
    "lingbot": {"frames": 81, "fps": 16.0, "size": (832, 480), "f_px": 240.0},
}
ANWM = {"frames": 17, "dt": 1.0}

PATHS = [
    ("hover_00", "hover", 0.0),
    ("cruise_00", "cruise", 0.0),
    ("s_5", "s", 5.0),
    ("slalom_5", "slalom", 5.0),
    ("fig8_5", "fig8", 5.0),
    ("s_10", "s", 10.0),
    ("slalom_10", "slalom", 10.0),
    ("arc_10", "arc", 10.0),
    ("arc_15", "arc", 15.0),
    ("arc_20", "arc", 20.0),
    ("arc_25", "arc", 25.0),
    ("climb_02", "climb", 0.2),      # requested peak vertical acceleration, in g
    ("climb_04", "climb", 0.4),
    ("descend_02", "descend", 0.2),
    ("descend_04", "descend", 0.4),
    ("climb_turn_10", "climb_turn", 10.0),
    ("climb_turn_20", "climb_turn", 20.0),
    ("accel_brake_10", "accel_brake", 10.0),
    ("accel_brake_20", "accel_brake", 20.0),
    ("arc_35", "arc", 35.0),
    ("slalom_25_fast", "slalom_fast", 25.0),
    ("climb_turn_30", "climb_turn", 30.0),
    ("real_uzh_of1", "real", None),
    ("real_uzh_of3", "real", None),
    ("real_uzh_of5", "real", None),
]
SCENE_SWAPS = [("arc_20__S2", "arc_20"), ("slalom_10__S2", "slalom_10")]

SCENES = {
    "S1": "302OLP89E75X7MFPRR58QG7CJPZCA7_processed",
    "S2": "33J5JKFMLEW1H2QNNURTNDINLEQ3QX_processed",
}
ANWM_START_TIME = 39
# ANWM takes context frames from its own released trajectories, which are the only five whose
# ``traj_data.pkl`` is on the native volume.  38RHUL is the one trajectory that is also in the
# downloaded frame set, but H01 verified 3KA7 at start_time 39, so 3KA7 is S1 here and 30EMX is
# S2; the scene is therefore not shared with the camera models and that is reported, not hidden.
ANWM_TRAJ = {"S1": "3KA7IJSNXE3LKLTL2ARA5RQP4ACBP0_processed",
             "S2": "30EMX9PEWSHJQGD7048XZZEQ9SBSK3_processed"}
REAL_FLIGHTS = {"real_uzh_of1": "outdoor_forward_1", "real_uzh_of3": "outdoor_forward_3",
                "real_uzh_of5": "outdoor_forward_5"}

ENVELOPE = {"peak_tilt_max_deg": 30.0, "thrust_ratio_max": 1.8,
            "tilt_rate_max_deg_s": 200.0}
LONGITUDINAL = ("climb", "descend", "accel_brake")
VERTICAL_G = ("climb", "descend")


# ------------------------------------------------------------------ profiles
def smoothstep(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def envelope(t: np.ndarray, t0: float, t1: float, ramp: float) -> np.ndarray:
    e = np.zeros_like(t)
    act = (t >= t0) & (t <= t1)
    x = t[act] - t0
    span = t1 - t0
    e[act] = np.where(x < ramp, smoothstep(x / ramp),
                      np.where(x > span - ramp, smoothstep((span - x) / ramp), 1.0))
    return e


def x_of(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return np.clip(t - t0, 0.0, t1 - t0)


def roll_profile(family: str, peak_deg: float, t: np.ndarray, t0: float, t1: float,
                 ramp: float) -> np.ndarray:
    e = envelope(t, t0, t1, ramp)
    x = x_of(t, t0, t1)
    span = t1 - t0
    p = np.deg2rad(peak_deg)
    if family in ("hover", "cruise", "accel_brake", "climb", "descend"):
        return np.zeros_like(t)
    if family in ("arc", "climb_turn"):
        return p * np.sin(np.pi * x / span) * e
    if family == "s":
        return p * np.sin(2 * np.pi * x / span) * e
    if family == "slalom":
        return p * np.sin(3 * np.pi * x / span) * e
    if family == "slalom_fast":
        return p * np.sin(2 * np.pi * x / span) * e
    if family == "fig8":
        prof = np.sin(2 * np.pi * x / span) * (0.6 + 0.4 * np.cos(2 * np.pi * x / span))
        return p * prof * e / max(np.max(np.abs(prof)), 1e-9)
    raise ValueError(family)


def vertical_profile(family: str, peak_deg: float, t: np.ndarray, t0: float, t1: float,
                     ramp: float, aux: float | None = None) -> np.ndarray:
    if family not in ("climb", "descend", "climb_turn"):
        return np.zeros_like(t)
    e = envelope(t, t0, t1, ramp)
    x = x_of(t, t0, t1)
    span = t1 - t0
    amp = (0.0 if aux is None else aux) / (2.0 * (np.pi / span) ** 2)
    sign = -1.0 if family == "descend" else 1.0
    return sign * amp * np.sin(np.pi * x / span) ** 2 * e


def positions(family: str, peak_deg: float, t: np.ndarray, t0: float, t1: float,
              ramp: float, aux: float | None = None) -> np.ndarray:
    span = t1 - t0
    x = x_of(t, t0, t1)
    e = envelope(t, t0, t1, ramp)
    pos = np.zeros((len(t), 3))
    if family == "hover":
        return pos
    if family == "cruise":
        pos[:, 0] = V0 * t
        return pos
    if family == "accel_brake":
        m = aux if aux is not None else _accel_aux(peak_deg, span)
        v = V0 * (1.0 + m * (np.sin(np.pi * x / span) ** 2 - 0.5) * e)
        pos[:, 0] = np.cumsum(v) * DT
        return pos
    phi = roll_profile(family, peak_deg, t, t0, t1, ramp)
    speed = V0 * (1.0 + S_AMP * np.sin(2 * np.pi * x / span) * e)
    omega = -(G0 / np.maximum(speed, 0.5)) * np.tan(phi)
    psi = np.cumsum(omega) * DT
    pos[:, 0] = np.cumsum(speed * np.cos(psi)) * DT
    pos[:, 1] = np.cumsum(speed * np.sin(psi)) * DT
    pos[:, 2] = vertical_profile(family, peak_deg, t, t0, t1, ramp, aux)
    return pos


def _accel_aux(peak_deg: float, span: float) -> float:
    return np.tan(np.deg2rad(peak_deg)) * G0 / (V0 * 2 * np.pi / span)


# ------------------------------------------------------------------ estimators
def dense_derivatives(pos: np.ndarray, dt: float, window_s: float = WINDOW_S) -> dict:
    w = int(round(window_s / dt))
    w += (w + 1) % 2
    w = max(5, min(w, len(pos) - 1 if (len(pos) - 1) % 2 else len(pos) - 2))
    vel = savgol_filter(pos, w, 3, deriv=1, delta=dt, axis=0, mode="interp")
    acc = savgol_filter(pos, w, 3, deriv=2, delta=dt, axis=0, mode="interp")
    return {"vel": vel, "acc": acc, "window": w}


def attitude(pos: np.ndarray, vel: np.ndarray, acc: np.ndarray) -> dict:
    z = implied_thrust_axis(acc, vel, K_D, G0)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    yaw = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    phi, theta = roll_pitch_from_thrust(z, yaw)
    tilt = np.degrees(np.arccos(np.clip(z[:, 2], -1.0, 1.0)))
    thr = np.linalg.norm(acc + G0 * np.array([0.0, 0.0, 1.0]) + K_D * vel,
                         axis=1) / G0
    return {"z": z, "yaw": yaw, "roll": phi, "pitch": theta, "tilt_deg": tilt,
            "thrust_ratio": thr}


def perstep(pos: np.ndarray, dt: float) -> dict:
    """The per-step attitude ANWM's piecewise-constant-velocity commands imply."""
    v = np.diff(pos, axis=0) / dt                 # (n-1, 3)
    a = np.diff(v, axis=0) / dt                   # (n-2, 3) at interior frames
    vm = 0.5 * (v[1:] + v[:-1])
    z = implied_thrust_axis(a, vm, K_D, G0)
    yaw = np.unwrap(np.arctan2(vm[:, 1], vm[:, 0]))
    phi, theta = roll_pitch_from_thrust(z, yaw)
    tilt = np.degrees(np.arccos(np.clip(z[:, 2], -1.0, 1.0)))
    thr = np.linalg.norm(a + G0 * np.array([0.0, 0.0, 1.0]) + K_D * vm, axis=1) / G0
    n = len(pos)
    out = {k: np.full(n, np.nan) for k in ("yaw", "roll", "pitch", "tilt_deg",
                                           "thrust_ratio")}
    out["yaw"][1:-1], out["roll"][1:-1], out["pitch"][1:-1] = yaw, phi, theta
    out["tilt_deg"][1:-1], out["thrust_ratio"][1:-1] = tilt, thr
    zf = np.full((n, 3), np.nan)
    zf[1:-1] = z
    out["z"] = zf
    out["vel"] = np.vstack([v[:1], vm, v[-1:]])
    out["acc"] = np.vstack([a[:1], a, a[-1:]])
    return out


def lean_req_dense(pos: np.ndarray, vel: np.ndarray, acc: np.ndarray,
                   window: int) -> dict:
    q = acc + G0 * np.array([0.0, 0.0, 1.0]) + K_D * vel
    n = np.linalg.norm(q, axis=1)
    u0 = q / np.maximum(n, 1e-9)[:, None]
    yaw = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    l_req = world_to_yaw_frame(u0[:, :2], yaw)
    tilt = np.degrees(np.arccos(np.clip(u0[:, 2], -1.0, 1.0)))
    half = max(window // 2, 1)
    l_req[:half] = np.nan
    l_req[len(l_req) - half:] = np.nan
    return {"lean_req": l_req, "tilt_req_deg": tilt}


# ------------------------------------------------------------------ instantiation
def instantiate(family: str, peak: float, frames: int, dt: float, t0: float, t1: float,
                ramp: float) -> dict:
    """Build one (family, request) instantiation on a model's frame grid.

    Turning families are solved on the bank amplitude, longitudinal families (climb, descend,
    accel_brake) on the vertical / speed-modulation amplitude, so that in every case the
    requested number is the achieved peak of the family's own design metric (|roll| for turns,
    |pitch| for the longitudinal families).
    """
    span_total = (frames - 1) * dt
    t = np.arange(0.0, span_total + 2 * dt, DT)
    tf = np.arange(frames) * dt
    metric = "pitch" if family in LONGITUDINAL else "roll"
    scale = 0.0
    if peak > 0.0:
        lo, hi = 1e-4, 2.0 * peak
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            got = _achieved(family, mid, frames, dt, t0, t1, ramp, metric,
                            solve_aux=(family in LONGITUDINAL))
            if got < peak:
                lo = mid
            else:
                hi = mid
        scale = 0.5 * (lo + hi)
    return _finish(family, peak, frames, dt, t0, t1, ramp, metric, scale)


def _finish(family: str, peak: float, frames: int, dt: float, t0: float, t1: float,
            ramp: float, metric: str, scale: float) -> dict:
    span_total = (frames - 1) * dt
    t = np.arange(0.0, span_total + 2 * dt, DT)
    tf = np.arange(frames) * dt
    if family in LONGITUDINAL:
        pos_d = positions(family, 0.0, t, t0, t1, ramp, aux=scale)
    elif family == "climb_turn":
        a_z = 0.5 * G0 * np.tan(np.deg2rad(abs(scale)) / 2.0)
        pos_d = positions(family, scale, t, t0, t1, ramp, aux=a_z)
    else:
        pos_d = positions(family, scale, t, t0, t1, ramp)
    if dt >= 0.5:                       # ANWM: per-step estimator at its 1 s action steps
        posf = _sample(pos_d, t, tf)
        att = perstep(posf, dt)
        w = 1
        dense = None
        lr = lean_req_dense(posf, att["vel"], att["acc"], w)
        rate = float(np.nanmax(np.abs(np.diff(np.degrees(att["roll"]))))) / dt
    else:                               # camera models: dense differentiation, then sample
        d = dense_derivatives(pos_d, DT)
        att_d = attitude(pos_d, d["vel"], d["acc"])
        idx = np.round(tf / DT).astype(int)
        posf = pos_d[idx]
        att = {k: att_d[k][idx] for k in ("z", "yaw", "roll", "pitch", "tilt_deg",
                                          "thrust_ratio")}
        att["vel"], att["acc"] = d["vel"][idx], d["acc"][idx]
        dense = {"t": t, "pos": pos_d, "vel": d["vel"], "acc": d["acc"], **att_d}
        w = d["window"]
        lr = lean_req_dense(pos_d, d["vel"], d["acc"], w)
        lr = {k: v[idx] for k, v in lr.items()}
        m = (t >= t0) & (t <= t1)
        rate = max(float(np.max(np.abs(np.diff(np.degrees(att_d["roll"])[m])))),
                   float(np.max(np.abs(np.diff(np.degrees(att_d["pitch"])[m]))))) / DT
    return {"pos": posf, "t": tf, "window": w, "design_rate_deg_s": rate,
            "lean_req": lr["lean_req"], "tilt_req": lr["tilt_req_deg"], "aux": scale,
            "metric": metric, **att}


def _achieved(family: str, x: float, frames: int, dt: float, t0: float, t1: float,
              ramp: float, metric: str, solve_aux: bool) -> float:
    r = _finish(family, 0.0, frames, dt, t0, t1, ramp, metric, x)
    m = (r["t"] >= t0) & (r["t"] <= t1) & np.isfinite(r[metric])
    return float(np.max(np.abs(np.degrees(r[metric])[m]))) if m.any() else 0.0


def _sample(pos: np.ndarray, t: np.ndarray, tf: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(tf, t, pos[:, k]) for k in range(3)])


def summarise(name: str, family: str, request, r: dict, t0: float, t1: float) -> dict:
    m = (r["t"] >= t0) & (r["t"] <= t1)
    roll = np.degrees(r["roll"])[m]
    pitch = np.degrees(r["pitch"])[m]
    tilt = r["tilt_deg"][m]
    thr = r["thrust_ratio"][m]
    dtf = float(np.median(np.diff(r["t"])))
    rate = float(r["design_rate_deg_s"])
    peak_tilt = float(np.nanmax(tilt))
    peak_thr = float(np.nanmax(thr))
    out = {"path": name, "family": family, "requested_peak_deg": request,
           "design_metric": r["metric"],
           "achieved_peak_bank_deg": float(np.nanmax(np.abs(roll))),
           "achieved_peak_pitch_deg": float(np.nanmax(np.abs(pitch))),
           "peak_tilt_deg": peak_tilt,
           "median_tilt_deg": float(np.nanmedian(tilt)),
           "peak_thrust_ratio": peak_thr, "peak_tilt_rate_deg_s": rate,
           "max_rel_translation_m": float(np.linalg.norm(r["pos"] - r["pos"][0],
                                                         axis=1).max()),
           "n_frames": int(len(r["t"])), "frame_dt_s": dtf,
           "window_samples": int(r["window"])}
    out["in_envelope"] = bool(peak_tilt <= ENVELOPE["peak_tilt_max_deg"]
                              and peak_thr <= ENVELOPE["thrust_ratio_max"]
                              and rate <= ENVELOPE["tilt_rate_max_deg_s"])
    out["envelope_failures"] = [
        k for k, ok in (("tilt", peak_tilt <= ENVELOPE["peak_tilt_max_deg"]),
                        ("thrust", peak_thr <= ENVELOPE["thrust_ratio_max"]),
                        ("rate", rate <= ENVELOPE["tilt_rate_max_deg_s"])) if not ok]
    if request is None:
        out["check_peak_ok"] = True
    else:
        got = (out["achieved_peak_pitch_deg"] if r["metric"] == "pitch"
               else out["achieved_peak_bank_deg"])
        out["check_peak_ok"] = bool(abs(got - request) <= 0.25)
    return out


# ------------------------------------------------------------------ real paths
def _real_window(flight: str, frames: int, dt: float, endpoint_max_m: float,
                 tilt_range: tuple[float, float]) -> dict:
    """The admissible real window with the largest smoothed peak required tilt."""
    z = np.load(ROOT / f"data/uzh/{flight}/truth.npz")
    p = np.asarray(z["pos"], float)
    t = np.asarray(z["t"], float)
    fs = 1.0 / np.median(np.diff(t))
    n = int(round((frames - 1) * dt * fs)) + 1
    if n >= len(p):
        raise RuntimeError(f"{flight}: only {len(p)} samples, need {n}")
    w_sg = int(round(1.0 * fs))            # 1 s smoothing: UZH truth differentiated twice
    w_sg += (w_sg + 1) % 2
    best = None
    for s in range(0, len(p) - n + 1, 2):
        w = p[s:s + n]
        if float(np.linalg.norm(w[-1] - w[0])) > endpoint_max_m:
            continue
        d = dense_derivatives(w, 1.0 / fs, window_s=1.0)
        a = attitude(w, d["vel"], d["acc"])
        peak = float(np.nanmax(a["tilt_deg"]))
        if not (tilt_range[0] <= peak <= tilt_range[1]):
            continue
        if best is None or peak > best["peak_tilt_deg"]:
            best = {"start": s, "peak_tilt_deg": peak, "pos": w,
                    "t": t[s:s + n] - t[s], "fs": fs,
                    "vel": d["vel"], "acc": d["acc"], "att": a}
    if best is None:
        raise RuntimeError(f"{flight}: no admissible window ({(frames-1)*dt:.2f} s, "
                           f"<={endpoint_max_m} m, tilt {tilt_range})")
    return best


def real_path(flight: str, frames: int, dt: float, endpoint_max_m: float,
              tilt_range: tuple[float, float], max_step_m: float | None = None) -> dict:
    w = _real_window(flight, frames, dt, endpoint_max_m, tilt_range)
    tf = np.arange(frames) * dt
    posf = _sample(w["pos"], w["t"], tf)
    scale = 1.0
    if max_step_m is not None:
        step = float(np.max(np.linalg.norm(np.diff(posf, axis=0), axis=1)))
        scale = min(1.0, max_step_m / max(step, 1e-9))
        posf = posf * scale
    if dt >= 0.5:                       # ANWM: per-step estimator on its 1 s steps
        att = perstep(posf, dt)
        lr = lean_req_dense(posf, att["vel"], att["acc"], 1)
        rate = float(np.nanmax(np.abs(np.diff(np.degrees(att["roll"]))))) / dt
        ww = 1
    else:                               # camera: sample the native 1 s-smoothed attitude
        idx = np.clip(np.round(tf * w["fs"]).astype(int), 0, len(w["pos"]) - 1)
        att = {k: w["att"][k][idx] * (1.0 if k in ("yaw", "roll", "pitch", "tilt_deg",
                                                   "thrust_ratio") else 1.0)
               for k in ("z", "yaw", "roll", "pitch", "tilt_deg", "thrust_ratio")}
        att["vel"], att["acc"] = w["vel"][idx] * scale, w["acc"][idx] * scale
        lr = lean_req_dense(w["pos"] * scale, w["vel"] * scale, w["acc"] * scale,
                            int(round(0.37 * w["fs"])) | 1)
        lr = {k: v[idx] for k, v in lr.items()}
        rate = float(np.nanmax(np.abs(np.diff(np.degrees(w["att"]["roll"]))))) * w["fs"]
        ww = int(round(0.37 * w["fs"])) | 1
    return {"pos": posf, "t": tf, "window": ww, "design_rate_deg_s": rate,
            "lean_req": lr["lean_req"], "tilt_req": lr["tilt_req_deg"], "aux": scale,
            "metric": "none",
            "source": {"flight": flight, "start_s": float(w["t"][0]),
                       "peak_tilt_native_deg": w["peak_tilt_deg"],
                       "position_scale": scale}, **att}

# ------------------------------------------------------------------ images and poses
def _first_frame(scene: str) -> Path:
    d = ROOT / "data/raw/anwm_frames" / SCENES[scene]
    return sorted(d.glob("*.jpg"), key=lambda p: int(p.stem))[0]


def prepare_image(src: Path, dst: Path, size: tuple[int, int]) -> None:
    """Scale to the target height keeping aspect, then edge-pad to the target width."""
    W, H = size
    im = Image.open(src).convert("RGB")
    w, h = im.size
    nw = max(1, int(round(w * H / h)))
    im = im.resize((nw, H), Image.BICUBIC)
    if nw > W:
        im = im.resize((W, H), Image.BICUBIC)
        nw = W
    left = (W - nw) // 2
    right = W - nw - left
    canvas = Image.new("RGB", (W, H))
    canvas.paste(im, (left, 0))
    if left:
        canvas.paste(im.crop((0, 0, 1, H)).resize((left, H)), (0, 0))
    if right:
        canvas.paste(im.crop((nw - 1, 0, nw, H)).resize((right, H)), (W - right, 0))
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)


def c2w_positions(pos: np.ndarray, yaw: np.ndarray, roll: np.ndarray,
                  pitch: np.ndarray) -> np.ndarray:
    R_bc = camera_to_body_pitched(0.0)
    ps = np.tile(np.eye(4), (len(pos), 1, 1))
    ps[:, :3, :3] = body_rotation(yaw, pitch, roll) @ R_bc[None]
    ps[:, :3, 3] = pos
    return ps


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/x01"))
    ap.add_argument("--data", default=str(ROOT / "data/x01"))
    args = ap.parse_args()
    out, data = Path(args.out), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    bases = {m: {"frames": c["frames"], "dt": 1.0 / c["fps"], "t0": 1.0, "t1": 4.0,
                 "ramp": 0.5} for m, c in CAMERA_MODELS.items()}
    bases["anwm"] = {"frames": ANWM["frames"], "dt": ANWM["dt"], "t0": 4.0, "t1": 12.0,
                     "ramp": 2.0}
    fast_window = {"solarwm": (2.2, 3.8), "sana": (2.2, 3.8), "lingbot": (2.2, 3.8),
                   "anwm": (6.0, 10.0)}

    arrays: dict[str, np.ndarray] = {}
    summary: list[dict] = []
    for model, base in bases.items():
        frames, dt = base["frames"], base["dt"]
        for name, family, peak in PATHS:
            t0, t1 = (fast_window[model] if family == "slalom_fast"
                      else (base["t0"], base["t1"]))
            try:
                if family == "real":
                    if model == "anwm":
                        r = real_path(REAL_FLIGHTS[name], frames, dt, 120.0, (10.0, 60.0),
                                      max_step_m=4.5)
                    else:
                        r = real_path(REAL_FLIGHTS[name], frames, dt, 15.0, (10.0, 45.0))
                    request = None
                elif family in VERTICAL_G:
                    r = _finish(family, 0.0, frames, dt, t0, t1, base["ramp"], "pitch",
                                float(peak) * G0)
                    request = None
                else:
                    r = instantiate(family, float(peak or 0.0), frames, dt, t0, t1,
                                    base["ramp"])
                    request = peak
            except RuntimeError as exc:
                print(f"{model:8s} {name:16s} SKIP {exc}", flush=True)
                continue
            info = summarise(name, family, request, r, t0, t1)
            info["model"] = model
            info["scene"] = "S1"
            if family == "real":
                info["real_source"] = r["source"]
            info["median_tilt_req_deg"] = float(np.nanmedian(r["tilt_req"]))
            if family in VERTICAL_G:
                info["vertical_peak_g"] = peak
            info["expressible"] = info["check_peak_ok"] if request is not None else True
            summary.append(info)
            key = f"{model}__{name}"
            for k in ("pos", "vel", "acc", "yaw", "roll", "pitch", "tilt_deg",
                      "thrust_ratio", "t", "lean_req", "tilt_req", "z"):
                arrays[f"{key}__{k}"] = np.asarray(r[k])
            flag = "IN " if info["in_envelope"] else "OUT"
            print(f"{model:8s} {name:16s} bank {info['achieved_peak_bank_deg']:5.2f} "
                  f"pitch {info['achieved_peak_pitch_deg']:5.2f} "
                  f"tilt {info['peak_tilt_deg']:5.2f} thr {info['peak_thrust_ratio']:4.2f} "
                  f"rate {info['peak_tilt_rate_deg_s']:6.1f} "
                  f"rel {info['max_rel_translation_m']:5.1f} {flag}"
                  f"{'' if info['check_peak_ok'] else ' PEAK-MISS'}"
                  f"{','.join(info['envelope_failures'])}", flush=True)

    for swap_name, base_name in SCENE_SWAPS:
        for model in bases:
            key = f"{model}__{base_name}"
            if f"{key}__pos" not in arrays:
                continue
            for k in ("pos", "vel", "acc", "yaw", "roll", "pitch", "tilt_deg",
                      "thrust_ratio", "t", "lean_req", "tilt_req", "z"):
                arrays[f"{model}__{swap_name}__{k}"] = arrays[f"{key}__{k}"]
            src = next(s for s in summary if s["model"] == model and s["path"] == base_name)
            summary.append({**src, "path": swap_name, "scene": "S2"})

    np.savez_compressed(out / "paths.npz", **arrays)
    (out / "paths_summary.json").write_text(json.dumps(summary, indent=1))

    # ------------------------------------------------------------- per-model inputs
    cache: dict = {}
    for scene in SCENES:
        for model, c in CAMERA_MODELS.items():
            dst = data / model / "frames" / f"{scene}.png"
            prepare_image(_first_frame(scene), dst, c["size"])
            cache[(scene, model)] = dst

    def write_camera(model: str, sid: str, name: str, scene: str) -> None:
        c = CAMERA_MODELS[model]
        W, H = c["size"]
        key = f"{model}__{name}"
        d = data / model / sid
        d.mkdir(parents=True, exist_ok=True)
        Image.open(cache[(scene, model)]).save(d / "image.png")
        poses = c2w_positions(arrays[f"{key}__pos"], arrays[f"{key}__yaw"],
                              arrays[f"{key}__roll"], arrays[f"{key}__pitch"])
        np.save(d / "poses.npy", poses.astype(np.float32))
        if model == "sana":
            np.save(d / "camera.npy", poses.astype(np.float32))
        if model == "sana":
            np.save(d / "intrinsics.npy", np.array([c["f_px"], c["f_px"], W / 2.0, H / 2.0],
                                                   np.float32))
        if model == "lingbot":
            # LingBot's get_Ks_transformed does ``Ks.chunk(4, dim=-1)`` on a (F, 4) tensor of
            # [fx, fy, cx, cy]; a 1-D vector fails the [:, 0:1] index and a 3x3 fails the chunk.
            K = np.array([[c["f_px"], c["f_px"], W / 2.0, H / 2.0]], np.float32)
            np.save(d / "intrinsics.npy", K)
        (d / "prompt.txt").write_text(PROMPT)
        meta = next(s for s in summary if s["model"] == model and s["path"] == name
                    and s.get("scene") == scene)
        (d / "meta.json").write_text(json.dumps(meta, indent=1))

    for model in CAMERA_MODELS:
        for name, family, _ in PATHS:
            if f"{model}__{name}__pos" in arrays:
                write_camera(model, f"{name}__S1", name, "S1")
        for swap_name, _ in SCENE_SWAPS:
            if f"{model}__{swap_name}__pos" in arrays:
                write_camera(model, swap_name, swap_name, "S2")

    specs = []
    for name, family, _ in PATHS + [(s, "", None) for s, _ in SCENE_SWAPS]:
        key = f"anwm__{name}"
        if f"{key}__pos" not in arrays:
            continue
        scene = "S2" if name.endswith("__S2") else "S1"
        info = next(s for s in summary if s["model"] == "anwm" and s["path"] == name)
        specs.append({
            "id": f"x01_{name}_{scene}",
            "regime": "C",
            "positions": np.asarray(arrays[f"{key}__pos"]).tolist(),
            "yaw": np.asarray(arrays[f"{key}__yaw"]).tolist(),
            "start_traj": ANWM_TRAJ[scene],
            "start_time": ANWM_START_TIME,
            "seed": 0,
            "designed_tilt_deg": info["peak_tilt_deg"],
            "meta": {"source": "x01_common_paths", "path": name, "family": family,
                     "scene": scene, "in_envelope": info["in_envelope"],
                     "envelope_failures": info["envelope_failures"]},
        })
    (data / "anwm").mkdir(parents=True, exist_ok=True)
    (data / "anwm" / "specs.json").write_text(json.dumps(specs, indent=1))

    (data / "manifest.json").write_text(json.dumps({
        "n_paths": len(PATHS), "n_swaps": len(SCENE_SWAPS),
        "models": {**{m: {**c, "n_samples": len(list((data / m).glob("*__S*")))}
                      for m, c in CAMERA_MODELS.items()},
                   "anwm": {"frames": ANWM["frames"], "dt": ANWM["dt"],
                            "n_samples": len(specs)}},
        "envelope": ENVELOPE,
    }, indent=1))
    n_miss = sum(1 for s in summary if not s["check_peak_ok"])
    print(f"\nwrote {out/'paths.npz'} ({len(arrays)} arrays), {len(summary)} cells, "
          f"{n_miss} peak misses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
