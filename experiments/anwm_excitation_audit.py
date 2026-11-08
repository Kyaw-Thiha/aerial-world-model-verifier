"""Excitation audit for ANWM's action space, using the floor physics code.

ANWM releases no action sequences or trajectories, so this is a *distribution*
audit of the action set the paper describes, not a replay of their data:

* 48 actions per segment, one recorded frame per action.
* Action set: forward/backward, left/right, up/down, yaw left/right.
* Speeds: 5 m/s horizontal, 2 m/s vertical, 15 deg/s yaw.
* Time step per action is not stated in the paper. It is inferred from two
  independent statements: evaluation horizons of 4/8/16/32 s over 48 frames give
  0.667 s, and 48 steps at the action speeds give a path length close to the
  reported 80.7 m. The audit therefore runs 1/3, 2/3 and 1 s and shows the
  sensitivity, because implied tilt scales with 1/dt.

Two readings of the same data are reported, because they answer different
questions:

* ``step``: acceleration as the exact velocity change between consecutive
  actions (velocity is piecewise constant). This is what the commanded
  trajectory implies.
* ``sg``: the same positions put through the verifier's own differentiator at
  the frame rate, which is what a pixel-based verifier would effectively see.

Outputs ``results/anwm_excitation/audit.json`` plus a printed table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.physics.diff import savgol_derivatives
from src.physics.flatness import G0, angle_between_deg, implied_thrust_axis, tilt_deg

ROOT = Path(__file__).resolve().parents[1]

# Paper-stated action set (Sec. 3.2): body-frame velocity commands.
ACTIONS = {
    "forward": (5.0, 0.0, 0.0, 0.0),
    "backward": (-5.0, 0.0, 0.0, 0.0),
    "left": (0.0, 5.0, 0.0, 0.0),
    "right": (0.0, -5.0, 0.0, 0.0),
    "up": (0.0, 0.0, 2.0, 0.0),
    "down": (0.0, 0.0, -2.0, 0.0),
    "yaw_left": (0.0, 0.0, 0.0, np.radians(15.0)),
    "yaw_right": (0.0, 0.0, 0.0, -np.radians(15.0)),
}
STEPS_PER_SEGMENT = 48
N_SEGMENTS = 4000
K_D_FITTED = 0.297  # from the mocap floor run
TILT_SIGNAL_DEG = 10.0
SUSTAIN_S = 0.2  # a frame counts as testable only if the tilt holds this long


def _rot_z(yaw: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.zeros((len(yaw), 3, 3))
    R[:, 0, 0], R[:, 0, 1] = c, -s
    R[:, 1, 0], R[:, 1, 1] = s, c
    R[:, 2, 2] = 1.0
    return R


def simulate(rng: np.random.Generator, dt: float, n_seg: int) -> dict:
    """Random 48-action segments; returns post-action positions and velocities."""
    names = list(ACTIONS)
    acts = rng.integers(0, len(names), size=(n_seg, STEPS_PER_SEGMENT))
    params = np.array([ACTIONS[n] for n in names])  # (8, 4)

    yaw = np.zeros(n_seg)
    pos = np.zeros((n_seg, 3))
    pos_hist = np.zeros((n_seg, STEPS_PER_SEGMENT, 3))
    vel_hist = np.zeros((n_seg, STEPS_PER_SEGMENT, 3))  # world velocity during each step
    for k in range(STEPS_PER_SEGMENT):
        p = params[acts[:, k]]
        yaw_mid = yaw + 0.5 * p[:, 3] * dt
        v_world = _rot_z(yaw_mid)[:, :, :] @ np.stack(
            [p[:, 0], p[:, 1], p[:, 2]], axis=1)[:, :, None]
        v_world = v_world[:, :, 0]
        pos = pos + v_world * dt
        yaw = yaw + p[:, 3] * dt
        pos_hist[:, k] = pos
        vel_hist[:, k] = v_world
    return {"pos": pos_hist, "vel": vel_hist, "actions": acts, "names": names}


def step_acceleration(sim: dict, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact acceleration at each boundary, and the mid-step velocity."""
    v = sim["vel"]
    dv = np.diff(v, axis=1) / dt
    v_mid = 0.5 * (v[:, 1:] + v[:, :-1])
    return dv, v_mid


def sustained_fraction(tilt: np.ndarray, dt: float, threshold: float, span_s: float) -> float:
    """Fraction of frames whose tilt stays above threshold for at least span_s."""
    n_need = max(1, int(round(span_s / dt)))
    above = tilt >= threshold
    if n_need == 1:
        return float(above.mean())
    kernel = np.ones(n_need)
    run = np.apply_along_axis(lambda r: np.convolve(r.astype(float), kernel, "valid"), -1, above)
    return float(np.mean(run >= n_need - 1) if run.size else 0.0)


def audit_dt(dt: float, rng: np.random.Generator) -> dict:
    sim = simulate(rng, dt, N_SEGMENTS)
    dv, v_mid = step_acceleration(sim, dt)

    out = {"dt_s": dt, "frames_per_second": 1.0 / dt}

    # (a) exact commanded acceleration between actions
    for label, k_d in (("drag_free", 0.0), ("fitted_drag", K_D_FITTED)):
        a = dv.reshape(-1, 3)
        v = v_mid.reshape(-1, 3)
        tilt = tilt_deg(implied_thrust_axis(a, v, k_d))
        out[f"step_{label}"] = {
            "median_tilt_deg": float(np.median(tilt)),
            "p90_tilt_deg": float(np.percentile(tilt, 90)),
            "p99_tilt_deg": float(np.percentile(tilt, 99)),
            "frac_frames_gt_10deg": float(np.mean(tilt > TILT_SIGNAL_DEG)),
            "frac_sustained_gt_10deg_0.2s": sustained_fraction(
                tilt.reshape(N_SEGMENTS, -1), dt, TILT_SIGNAL_DEG, SUSTAIN_S),
        }

    # (b) the verifier's own differentiator on the sampled positions
    window, poly = 5, 3
    n_need = max(window, 4 * window + 1)
    clips_pos = sim["pos"][: max(1, n_need // STEPS_PER_SEGMENT) + 1]
    frames = clips_pos.reshape(-1, 3)
    t = np.arange(len(frames)) * dt
    vel, acc = savgol_derivatives(frames, dt, window, poly)
    half = window // 2
    sl = slice(half, len(frames) - half)
    for label, k_d in (("drag_free", 0.0), ("fitted_drag", K_D_FITTED)):
        a, v = acc[sl], vel[sl]
        ok = np.linalg.norm(a + np.array([0.0, 0.0, G0]), axis=1) > 1.0
        tilt = np.full(len(a), np.nan)
        tilt[ok] = tilt_deg(implied_thrust_axis(a[ok], v[ok], k_d))
        out[f"sg_{label}"] = {
            "window": window, "polyorder": poly,
            "median_tilt_deg": float(np.nanmedian(tilt)),
            "p90_tilt_deg": float(np.nanpercentile(tilt, 90)),
            "frac_frames_gt_10deg": float(np.nanmean(tilt > TILT_SIGNAL_DEG)),
        }

    # cruise tilt: what a level camera would need to be at 5 m/s and 2 m/s
    out["cruise_tilt_deg"] = {
        f"k_d={K_D_FITTED}": {
            "horizontal_5ms": float(np.degrees(np.arctan(K_D_FITTED * 5.0 / G0))),
            "vertical_2ms": float(np.degrees(np.arctan(K_D_FITTED * 2.0 / G0))),
        }
    }
    sp = np.linalg.norm(sim["vel"].reshape(-1, 3), axis=1)
    out["speed_stats_ms"] = {"mean": float(sp.mean()), "max": float(sp.max()),
                             "frac_frames_moving_gt_10ms": float(np.mean(sp > 10.0))}
    out["mean_path_length_m"] = float(
        np.linalg.norm(np.diff(sim["pos"], axis=1), axis=2).sum(axis=1).mean())
    return out


def main() -> int:
    rng = np.random.default_rng(0)
    results = {f"dt_{dt:.3f}": audit_dt(dt, rng) for dt in (1 / 3, 2 / 3, 1.0)}
    out_dir = ROOT / "results" / "anwm_excitation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit.json").write_text(json.dumps(
        {"actions": {k: list(map(float, v)) for k, v in ACTIONS.items()},
         "steps_per_segment": STEPS_PER_SEGMENT, "n_segments": N_SEGMENTS,
         "tilt_signal_deg": TILT_SIGNAL_DEG, "k_d_fitted": K_D_FITTED,
         "results": results}, indent=2))

    print(f"{'dt (s)':>7} {'fps':>6} {'step p50':>9} {'step p90':>9} {'step >10deg':>11} "
          f"{'sustained':>10} {'sg p50':>8} {'sg >10deg':>10} {'path (m)':>9}")
    for name, r in results.items():
        s, g = r["step_fitted_drag"], r["sg_fitted_drag"]
        print(f"{r['dt_s']:7.3f} {r['frames_per_second']:6.2f} {s['median_tilt_deg']:9.1f} "
              f"{s['p90_tilt_deg']:9.1f} {s['frac_frames_gt_10deg']:11.3f} "
              f"{s['frac_sustained_gt_10deg_0.2s']:10.3f} {g['median_tilt_deg']:8.1f} "
              f"{g['frac_frames_gt_10deg']:10.3f} {r['mean_path_length_m']:9.1f}")
    print("\ncruise tilt with fitted drag:", results["dt_0.667"]["cruise_tilt_deg"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
