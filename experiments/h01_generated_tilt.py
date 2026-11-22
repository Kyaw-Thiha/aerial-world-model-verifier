"""H01 Parts 2-3: depicted tilt (and lean) on LingBot-World and Wan clips.

Reads the reader rows written by ``remote/h01_gate.py`` (GeoCalib 224 px, same row schema as
the ANWM scoring track) and the injected thrust axes saved by
``experiments/h01_lingbot_inputs.py``. Nothing is recomputed from pixels here.

Statistic: frame-level depicted tilt = angle between the body z-axis implied by the
estimated gravity (through the recorded forward mount) and world up; clip value = median.
The depicted lean vector is the horizontal part of that body z-axis in the world frame,
using the commanded yaw recovered from the injected pose; yaw itself is unobservable.

    pixi run python experiments/h01_generated_tilt.py --depicted results/h01_lingbot_depicted/depicted.jsonl \\
        --inputs data/h01_lingbot_inputs --out results/h01_lingbot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.a2_anwm_score import frame_arrays
from src.perception.frames import camera_to_body_pitched
from src.physics.flatness import tilt_deg
from src.score.verdict import depicted_axis_world

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def yaw_from_poses(poses: np.ndarray, R_bc: np.ndarray) -> np.ndarray:
    R_wb = poses[:, :3, :3] @ R_bc.T
    return np.arctan2(R_wb[:, 1, 0], R_wb[:, 0, 0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depicted", required=True)
    ap.add_argument("--inputs", default="data/h01_lingbot_inputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--uptilt-deg", type=float, default=0.0)
    args = ap.parse_args()

    rows = load_jsonl(ROOT / args.depicted)
    man = json.loads((ROOT / args.inputs / "manifest.json").read_text())
    grid = man["grid"]
    by_grid = {f"{g['traj']}_{g['frame']:03d}__{g['path']}__{g['arm']}": g for g in grid}
    probes = man.get("convention_probe", {})
    if probes:
        fr = man["frames"][0]
        level = [g for g in grid if g["path"] == "cruise" and g["arm"] == "level"
                 and g["traj"] == fr["traj"]][0]
        by_grid["probe_level"] = dict(level, arm="probe_level")
        for key in ("20", "-20", "fast20", "60"):
            by_grid[f"probe_rollramp_{key}"] = dict(level, arm=f"probe_roll_{key}",
                                                    dir=probes[key]["dir"])
    by_clip: dict[str, list[dict]] = {}
    for r in rows:
        by_clip.setdefault(r["clip_id"], []).append(r)

    R_cb = camera_to_body_pitched(np.deg2rad(args.uptilt_deg))
    per_clip, per_frame = [], []
    for cid, rs in sorted(by_clip.items()):
        rs = sorted(rs, key=lambda r: r["frame"])
        g = by_grid.get(cid)
        up = frame_arrays(rs)["up"]
        if g is not None:
            d = ROOT / args.inputs / g["path"] / g["arm"]
            # grid dirs are embedded in the manifest relative to the repo root
            d = ROOT / g["dir"].replace("data/h01_lingbot_inputs/", "data/h01_lingbot_inputs/")
            poses = np.load(ROOT / g["dir"] / "poses.npy")
            thrust = np.load(ROOT / g["dir"] / "thrust.npy")
            n = min(len(up), len(poses))
            yaw = yaw_from_poses(poses[:n], R_cb)
            z = depicted_axis_world(up[:n], R_cb, yaw)
            inj = thrust[:n]
            tilt = tilt_deg(z)
            tilt_inj = tilt_deg(inj)
            lean = z[:, :2]
            lean_inj = inj[:, :2]
            lean_sign = np.sign(lean.mean(0))
            lean_sign_inj = np.sign(lean_inj.mean(0))
            agree = bool(np.all(lean_sign == lean_sign_inj))
            cos = float(np.nanmedian(
                (lean * lean_inj).sum(1) /
                np.maximum(np.linalg.norm(lean, axis=1) * np.linalg.norm(lean_inj, axis=1), 1e-9)))
            per_clip.append({
                "clip_id": cid, "path": g["path"], "arm": g["arm"], "traj": g["traj"],
                "frame": g["frame"], "n_frames": int(n),
                "depicted_tilt_median_deg": float(np.nanmedian(tilt)),
                "depicted_tilt_p90_deg": float(np.nanpercentile(tilt, 90)),
                "injected_tilt_median_deg": float(np.nanmedian(tilt_inj)),
                "injected_tilt_max_deg": float(np.nanmax(tilt_inj)),
                "depicted_lean_x": float(lean[:, 0].mean()),
                "depicted_lean_y": float(lean[:, 1].mean()),
                "injected_lean_x": float(lean_inj[:, 0].mean()),
                "injected_lean_y": float(lean_inj[:, 1].mean()),
                "lean_direction_cos_median": cos,
                "lean_direction_agrees": agree,
                "depicted_tilt_first10_deg": float(np.nanmedian(tilt[:10])),
                "depicted_tilt_last10_deg": float(np.nanmedian(tilt[-10:])),
                "injected_tilt_first10_deg": float(np.nanmedian(tilt_inj[:10])),
                "injected_tilt_last10_deg": float(np.nanmedian(tilt_inj[-10:])),
                "pearson_tilt": (float(np.corrcoef(tilt, tilt_inj)[0, 1])
                                 if np.std(tilt) > 0 and np.std(tilt_inj) > 0 else None),
            })
            for i in range(n):
                per_frame.append({"clip_id": cid, "frame": rs[i]["frame"], "i": i,
                                  "depicted_tilt_deg": float(tilt[i]),
                                  "injected_tilt_deg": float(tilt_inj[i]),
                                  "depicted_zx": float(z[i, 0]), "depicted_zy": float(z[i, 1]),
                                  "injected_zx": float(inj[i, 0]), "injected_zy": float(inj[i, 1])})
        else:
            yaw = np.zeros(len(up))
            z = depicted_axis_world(up, R_cb, yaw)
            tilt = tilt_deg(z)
            per_clip.append({"clip_id": cid, "path": None, "arm": None, "traj": None,
                             "frame": None, "n_frames": int(len(up)),
                             "depicted_tilt_median_deg": float(np.nanmedian(tilt)),
                             "depicted_tilt_p90_deg": float(np.nanpercentile(tilt, 90)),
                             "injected_tilt_median_deg": None, "injected_tilt_max_deg": None,
                             "depicted_lean_x": None, "depicted_lean_y": None,
                             "injected_lean_x": None, "injected_lean_y": None,
                             "lean_direction_cos_median": None,
                             "lean_direction_agrees": None})

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(per_clip).sort_values(["path", "traj", "arm"], na_position="first")
    df.to_csv(out / "clips.csv", index=False)
    pd.DataFrame(per_frame).to_csv(out / "frames.csv", index=False)

    summary: dict = {"depicted": args.depicted, "n_clips": int(len(df)),
                     "uptilt_deg": args.uptilt_deg}
    if "arm" in df and df["arm"].notna().any():
        piv = df.pivot_table(index=["path", "traj", "frame"], columns="arm",
                             values="depicted_tilt_median_deg")
        summary["by_arm"] = {a: {"median_deg": float(df.loc[df.arm == a,
                                                            "depicted_tilt_median_deg"].median()),
                                 "n": int((df.arm == a).sum())} for a in df["arm"].dropna().unique()}
        if {"level", "flatness"}.issubset(piv.columns):
            dlt = (piv["flatness"] - piv["level"]).dropna()
            summary["flatness_minus_level_deg"] = {
                "median": float(dlt.median()), "n_paths": int(len(dlt)),
                "n_positive": int((dlt > 0).sum()),
                "n_at_least_half_injected": int(
                    sum(1 for (p, t, f), v in dlt.items()
                        if v >= 0.5 * float(df[(df.path == p) & (df.traj == t)
                                               & (df.frame == f) & (df.arm == "flatness")]
                                            .injected_tilt_max_deg.iloc[0])))}
        if {"flatness", "mirrored_roll"}.issubset(piv.columns):
            m = df[df.arm.isin(["flatness", "mirrored_roll"])].pivot_table(
                index=["path", "traj", "frame"], columns="arm", values="depicted_lean_x")
            flips = np.sign(m["flatness"]) != np.sign(m["mirrored_roll"])
            summary["mirror_flip"] = {"n": int(flips.notna().sum()),
                                      "n_flipped": int(flips.sum())}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(df.to_string(index=False, max_colwidth=40)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
