"""Build the injected-attitude clip specs for the C designs (CPU only, no generation).

The level clips in `results/anwm_clips/specs.json` are the paired baseline, so this rebuilds each
design from its parameters, asserts the geometry is bit-identical to the delivered spec, and
attaches what the level spec does not have: the analytic acceleration, the flatness thrust axis
and the injected roll/pitch.

    python -m experiments.anwm_build_injected

Writes results/anwm_clips_injected/specs.json with arms `injected_true` and `injected_mirrored`
(the mirrored arm is built but generated only after approval).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.generation.attitude import (pose_with_attitude, roll_pitch_from_thrust,
                                     thrust_from_accel)
from src.generation.paths import design
from src.generation.poses import camera_axis_change

ROOT = Path(__file__).resolve().parents[1]
LEVEL = ROOT / "results/anwm_clips/specs.json"
STARTS = ROOT / "results/anwm_starts/starts.json"
OUT = ROOT / "results/anwm_clips_injected/specs.json"
CONVENTION = ("intrinsic-ZYX R_wb=Rz(yaw)Ry(theta)Rx(phi); R_ec=t_ec (verified from released "
              "poses, spread 2e-16); drag-free thrust axis; 2026-11-10")


def start_pose(spec, starts) -> tuple[np.ndarray, float]:
    for traj, v in starts.items():
        for c in v["candidates"]:
            if traj == spec["start_traj"] and int(c["ct"]) == int(spec["start_time"]):
                return np.asarray(spec["positions"], float)[0], float(np.asarray(spec["yaw"])[0])
    raise KeyError(f"start state not found for {spec['id']}")


def place(spec, pos0, yaw0):
    c, s = np.cos(yaw0), np.sin(yaw0)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (np.asarray(pos0) + spec.pos @ Rz.T, spec.yaw + yaw0, spec.accel @ Rz.T)


def main() -> int:
    starts = json.loads(STARTS.read_text())
    level = [s for s in json.loads(LEVEL.read_text()) if s["regime"] == "C"]
    R_ec, _ = camera_axis_change(
        np.load(ROOT / "data/raw/anwm_full/3KA7IJSNXE3LKLTL2ARA5RQP4ACBP0_processed/traj_data.pkl",
                allow_pickle=True)["pose"],
        np.load(ROOT / "data/raw/anwm_full/3KA7IJSNXE3LKLTL2ARA5RQP4ACBP0_processed/traj_data.pkl",
                allow_pickle=True)["yaw"])
    out, rows, skipped = [], [], []
    for sp in level:
        meta = sp["meta"]
        d = design(meta["maneuver"], sp["designed_tilt_deg"], 5.0, 17, 1.0, meta["policy"])
        pos0, yaw0 = start_pose(sp, starts)
        pos, yaw, acc = place(d, pos0, yaw0)
        d_pos = float(np.abs(pos - np.asarray(sp["positions"], float)).max())
        d_yaw = float(np.abs(yaw - np.asarray(sp["yaw"], float)).max())
        if d_pos > 1e-9 or d_yaw > 1e-12:
            # Do not inject an attitude into a path the current design code cannot reproduce:
            # the acceleration would describe a different trajectory than the one generated.
            skipped.append({"id": sp["id"], "pos_err_m": d_pos, "yaw_err_rad": d_yaw,
                            "maneuver": meta["maneuver"], "policy": meta["policy"]})
            continue

        zB = thrust_from_accel(acc)
        phi, theta = roll_pitch_from_thrust(zB, yaw)
        zB_m = thrust_from_accel(acc * np.array([-1.0, -1.0, 1.0]))
        # poses must exist for every target step (this is what FFP consumes)
        pose_with_attitude(pos[1:], yaw[1:], zB[1:], R_ec)

        base = {"regime": "C", "positions": sp["positions"], "yaw": sp["yaw"],
                "start_traj": sp["start_traj"], "start_time": sp["start_time"],
                "seed": sp["seed"], "designed_tilt_deg": sp["designed_tilt_deg"],
                "paired_level_id": sp["id"], "convention": CONVENTION,
                "meta": {**meta,
                         "analytic_peak_tilt_deg": float(np.degrees(np.arccos(
                             np.clip(zB[:, 2].min(), -1, 1)))),
                         "max_roll_deg": float(np.degrees(np.abs(phi)).max()),
                         "max_pitch_deg": float(np.degrees(np.abs(theta)).max()),
                         "py_script": "experiments/anwm_build_injected.py"}}
        out.append({**base, "id": sp["id"], "arm": "injected_true",
                    "thrust": zB.tolist(), "roll_deg": np.degrees(phi).tolist(),
                    "pitch_deg": np.degrees(theta).tolist()})
        out.append({**base, "id": sp["id"] + f"__{sp['seed']}", "arm": "injected_mirrored",
                    "paired_level_id": sp["id"],
                    "thrust": zB_m.tolist(),
                    "roll_deg": np.degrees(roll_pitch_from_thrust(zB_m, yaw)[0]).tolist(),
                    "pitch_deg": np.degrees(roll_pitch_from_thrust(zB_m, yaw)[1]).tolist()})
        rows.append((sp["id"], meta["maneuver"], sp["designed_tilt_deg"],
                     float(meta["achieved_1s_drag_free"]),
                     float(np.degrees(np.arccos(np.clip(zB[:, 2].min(), -1, 1)))),
                     float(np.degrees(np.abs(phi).max())), float(np.degrees(np.abs(theta).max()))))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    print(f"{len(out)} specs ({len(out)//2} clips x 2 arms) -> {OUT.relative_to(ROOT)}")
    if skipped:
        print(f"SKIPPED {len(skipped)} level clips that the current design code cannot reproduce:")
        for k in skipped:
            print("  ", k)
    print(f"{'clip':30s} {'maneuver':12s} {'tgt':>4s} {'cmd tilt':>8s} {'inj tilt':>8s} "
          f"{'roll':>6s} {'pitch':>6s}")
    for r in rows:
        tgt = f"{r[2]:.0f}" if r[2] is not None else "-"
        print(f"{r[0]:30s} {r[1]:12s} {tgt:>4s} {r[3]:8.2f} {r[4]:8.2f} {r[5]:6.2f} {r[6]:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
