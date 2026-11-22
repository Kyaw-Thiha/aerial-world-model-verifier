"""H04 amendment 3: the required side for H02's held-bank clips.

H02's 15 held-bank clips are the `flatness` arm of the three turn paths on H02's five first
frames.  Their required side must be rebuilt because H04 scores against the *commanded* path:
the ANWM designed path (17 waypoints at 1 s), a cubic spline through it, the flatness
expression `unit(a + g e3)` differentiated on a dense grid with the same `0.5/f95` window as
the 5B campaign, sampled at the 81 frame indices.

The construction is checked against H02's stored point required tilt (`turn5` 9.8, `turn10`
13.0, `turn20` 21.2 deg) before any 14B clip is read.

    PYTHONPATH=. pixi run -e gpu python experiments/h04_heldbank_required.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

from src.physics.flatness import G0
from src.verdict.lean import world_to_yaw_frame

ROOT = Path(__file__).resolve().parents[1]
SPECS = Path("/home/kevin/Documents/Projects/aerial-model-verifier/results/anwm_clips/specs.json")
SPECS = Path("/home/kevin/Documents/Projects/aerial-model-verifier") / "results/anwm_clips/specs.json"
# the H02/H01 worktree is the authoritative copy of the ANWM clip specs
SPECS = Path("/home/kevin/Documents/Projects/aerial-model-verifier/data/anwm_clips/specs.json")
SPECS_FALLBACK = [Path("/home/kevin/Documents/Projects/awmv-h2/results/anwm_clips/specs.json"),
                  Path("/home/kevin/Documents/Projects/aerial-world-model-verifier/results/anwm_clips/specs.json")]
RATE = 1000.0
DT = 1.0 / RATE
DT_PATH = 1.0
N_OUT = 81
C_WINDOW = 0.5
SPECTRUM_WINDOW_S = 0.10

PATHS = {"turn5": "C_turn_5_fixed_0", "turn10": "C_turn_10_fixed_0",
         "turn20": "C_turn_20_fixed_0"}


def _specs() -> dict:
    for p in [SPECS] + SPECS_FALLBACK:
        if p.exists():
            return {s["id"]: s for s in json.loads(p.read_text())}
    raise SystemExit("ANWM clip specs not found")


def _resample(positions: np.ndarray) -> np.ndarray:
    """H02's construction: cubic spline through the waypoints, 81 samples, rebased."""
    p = np.asarray(positions, float)
    t = np.arange(len(p)) * DT_PATH
    cs = CubicSpline(t, p, bc_type="natural", axis=0)
    tt = np.linspace(t[0], t[-1], N_OUT)
    pos = cs(tt)
    return pos - pos[0]


def f95_and_window(pos: np.ndarray, dt: float) -> dict:
    n = len(pos)
    w = int(round(SPECTRUM_WINDOW_S / dt))
    w += (w + 1) % 2
    w = max(5, min(w, n if n % 2 else n - 1))
    acc = savgol_filter(pos, w, 3, deriv=2, delta=dt, axis=0, mode="interp")
    acc = acc - acc.mean(axis=0, keepdims=True)
    spec = np.fft.rfft(acc, axis=0)
    energy = (np.abs(spec) ** 2).sum(axis=1)
    freqs = np.fft.rfftfreq(n, d=dt)
    keep = freqs > 0.0
    e, f = energy[keep], freqs[keep]
    if e.sum() <= 0:
        return {"f95_hz": float("nan"), "window_s": float("nan")}
    cum = np.cumsum(e) / e.sum()
    f95 = float(f[min(int(np.searchsorted(cum, 0.95)), len(f) - 1)])
    ws = C_WINDOW / f95 if f95 > 0 else float("nan")
    return {"f95_hz": f95, "window_s": min(ws, (n * dt) / 2.0)}


def main(window_override: float | None = None, tag: str = "") -> int:
    specs = _specs()
    out, summary = {}, []
    for name, cid in PATHS.items():
        pos81 = _resample(specs[cid]["positions"])
        # dense path at 1 kHz over the same 16 s, then differentiate and sample at 81 frames
        t_nodes = np.arange(len(specs[cid]["positions"])) * DT_PATH
        cs = CubicSpline(t_nodes, np.asarray(specs[cid]["positions"], float),
                         bc_type="natural", axis=0)
        t = np.arange(t_nodes[0], t_nodes[-1], DT)
        pos = cs(t)
        pos = pos - pos[0]
        win = f95_and_window(pos, DT)
        if window_override is not None:
            win = {**win, "window_s": window_override, "override": True}
        w = int(round(win["window_s"] / DT))
        w += (w + 1) % 2
        vel = savgol_filter(pos, w, 3, deriv=1, delta=DT, axis=0, mode="interp")
        acc = savgol_filter(pos, w, 3, deriv=2, delta=DT, axis=0, mode="interp")
        q = acc + G0 * np.array([0.0, 0.0, 1.0])
        u0 = q / np.maximum(np.linalg.norm(q, axis=1), 1e-9)[:, None]
        yaw = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
        idx = np.round(np.linspace(0, len(t) - 1, N_OUT)).astype(int)
        lean = world_to_yaw_frame(u0[idx, :2], yaw[idx])
        vreq = world_to_yaw_frame(vel[idx, :2] / G0, yaw[idx])
        tilt = np.degrees(np.arccos(np.clip(u0[idx, 2], -1.0, 1.0)))
        # frames whose stencil is not interior are NaN on the required side
        half = max(w // 2, 1)
        t_frame = t[idx]
        bad = (t_frame < t[0] + half * DT) | (t_frame > t[-1] - half * DT)
        lean[bad] = np.nan
        vreq[bad] = np.nan
        out[f"{name}__lean_req"] = lean
        out[f"{name}__vel_req"] = vreq
        out[f"{name}__tilt_req"] = tilt
        out[f"{name}__t"] = idx * DT
        info = {"path": name, "source_clip": cid, "f95_hz": win["f95_hz"],
                "window_s": win["window_s"], "window_samples": w,
                "n_masked": int(bad.sum()),
                "peak_tilt_deg": float(np.nanmax(tilt)),
                "median_tilt_deg": float(np.nanmedian(tilt)),
                "pos81_range_m": float(np.linalg.norm(pos81 - pos81[0], axis=1).max())}
        summary.append(info)
        print(info)
    (ROOT / "results/h04").mkdir(parents=True, exist_ok=True)
    suffix = tag or ""
    np.savez_compressed(ROOT / f"results/h04/heldbank_required{suffix}.npz", **out)
    (ROOT / f"results/h04/heldbank_required{suffix}.json").write_text(
        json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    import sys
    ov = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "rule" else None
    raise SystemExit(main(ov, "" if ov is None else f"_w{ov:g}"))
