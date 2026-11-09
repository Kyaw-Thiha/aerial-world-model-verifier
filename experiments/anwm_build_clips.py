"""Build the clip specs for the first generation pass (Task 5) without touching a GPU.

Inputs:
  results/anwm_starts/starts.json   from remote/b_anwm.py::list_starts (native start states)
  data/raw/tii/*/mocap_*.csv        real flights for Regime B
Output: results/anwm_clips/specs.json  (positions/yaw per clip, 17 poses each)
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from src.datasets.tii import load_tii
from src.generation.paths import K_D, design, from_mocap

ROOT = Path(__file__).resolve().parents[1]
STARTS = ROOT / "results/anwm_starts/starts.json"
COVERAGE = ROOT / "results/anwm_starts/starts_coverage.json"
OUT = ROOT / "results/anwm_clips/specs.json"
MANEUVERS = ("cruise", "turn", "slalom", "accel_brake", "climb_turn")
TILTS = (5.0, 10.0, 20.0)
SEEDS = (0, 1, 2)


def usable_starts() -> list[dict]:
    """Starts whose projected prior is measured non-empty (coverage), not depth-heuristic ones."""
    if not STARTS.exists():
        raise SystemExit(f"{STARTS} missing; run remote/b_anwm.py::list_starts first")
    data = json.loads(STARTS.read_text())
    cov = {}
    if COVERAGE.exists():
        for traj, v in json.loads(COVERAGE.read_text()).items():
            for r in v["candidates"]:
                cov[(traj, r["ct"])] = r
    out = []
    for traj, v in data.items():
        for r in v["candidates"]:
            c = cov.get((traj, r["ct"]))
            if c is not None and not c["usable"]:
                continue
            if r["usable"] and r["positions"]:
                out.append({"traj": traj, "ct": r["ct"], "positions": r["positions"],
                            "yaw": r["yaw"],
                            "prior_coverage_min": (c or {}).get("coverage_min"),
                            "depth_p50_m": (c or {}).get("depth_p50_m")})
    return out


def place(design_pos: np.ndarray, design_yaw: np.ndarray, start: dict) -> tuple[np.ndarray, np.ndarray]:
    """Put a designed path into the world frame at a native start pose (same heading convention
    as the released `point`/`yaw` keys, which is what ANWM's action builder consumes)."""
    p0 = np.asarray(start["positions"], float)[0]
    y0 = float(np.asarray(start["yaw"], float)[0])
    c, s = np.cos(y0), np.sin(y0)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return p0 + design_pos @ Rz.T, np.asarray(design_yaw, float) + y0


def spec(cid, regime, pos, yaw, start, seed=0, tilt=None, meta=None):
    return {"id": cid, "regime": regime, "positions": np.asarray(pos).tolist(),
            "yaw": np.asarray(yaw).tolist(), "start_traj": start["traj"],
            "start_time": int(start["ct"]), "seed": seed, "designed_tilt_deg": tilt,
            "meta": meta or {}}


def main() -> int:
    starts = usable_starts()
    print(f"{len(starts)} usable native starts")
    chosen = starts[:20]
    specs = []

    # Regime A: native recorded paths from native starts (16 frames)
    for i, s in enumerate(chosen):
        specs.append(spec(f"A{i:02d}", "A", s["positions"], s["yaw"], s,
                          meta={"source": "released trajectory, recorded commands",
                               "prior_coverage_min": s.get("prior_coverage_min")}))

    # Regime C: factorial designs, three starts each
    for m in MANEUVERS:
        levels = (None,) if m == "cruise" else TILTS
        for lvl in levels:
            for pol in (("course", "fixed") if m in ("turn", "slalom", "climb_turn")
                        else ("course",)):
                try:
                    d = design(m, lvl, 5.0, 17, 1.0, pol)
                except Exception as e:
                    print("skip design", m, lvl, pol, e)
                    continue
                if lvl is not None and not d.meta.get("reachable_deg_target"):
                    print("skip unreachable", m, lvl, pol, round(d.peak_tilt(1.0, 0.0), 1))
                    continue
                for k, s in enumerate(chosen[:3]):
                    pos, yaw = place(d.pos, d.yaw, s)
                    specs.append(spec(f"C_{m}_{'ctl' if lvl is None else int(lvl)}_{pol}_{k}",
                                      "C", pos, yaw, s, tilt=lvl,
                                      meta={"maneuver": m, "policy": pol,
                                            "reachable": d.meta.get("reachable_deg_target"),
                                            "achieved_1s_drag_free": d.peak_tilt(1.0, 0.0),
                                            "achieved_1s_k_d": d.peak_tilt(1.0, K_D)}))

    # Regime B: two real flights (one autonomous, one piloted), four starts each
    for flight in ("flight-01a-ellipse", "flight-01p-ellipse"):
        hits = glob.glob(str(ROOT / f"data/raw/tii/*/mocap_{flight}.csv"))
        if not hits:
            print("missing mocap", flight)
            continue
        c = load_tii(hits[0])
        from scipy.spatial.transform import Rotation
        R = Rotation.from_quat(c.quat).as_matrix()
        yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
        t0 = float(c.t[0] + 0.15 * (c.t[-1] - c.t[0]))
        for k, s in enumerate(chosen[:4]):
            try:
                b = from_mocap(c.pos, yaw, c.t, t0, 40.0, name=f"B_{flight}_{k}")
            except Exception as e:
                print("skip B", flight, k, e)
                continue
            if len(b.pos) < 17:
                continue
            pos, yw = place(b.pos[:17], b.yaw[:17], s)
            specs.append(spec(f"B_{flight}_{k}", "B", pos, yw, s,
                              meta={"flight": flight, **{kk: vv for kk, vv in b.meta.items()
                                                         if isinstance(vv, (int, float, str))}}))

    # Variance: five paths x three seeds (seed 0 already exists)
    for base in ([x for x in specs if x["regime"] == "A"][:3] +
                 [x for x in specs if x["regime"] == "C"][:2]):
        for seed in SEEDS[1:]:
            v = dict(base)
            v["id"] = f"{base['id']}_s{seed}"
            v["seed"] = seed
            specs.append(v)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(specs))
    by_regime = {}
    for s in specs:
        by_regime[s["regime"]] = by_regime.get(s["regime"], 0) + 1
    print("specs", len(specs), by_regime, "->", OUT)
    # measured on A10G: 17.57 s per generated frame at 250 sampling steps (remote/b_anwm.py::smoke)
    per_clip_s = 17.57 * 16
    hours = len(specs) * per_clip_s / 3600
    print(f"projected: {len(specs)} clips x 16 frames x 17.57 s = {hours:.1f} GPU-hours "
          f"= US${hours * 1.10:.2f} at the A10G rate (unverified per region)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
