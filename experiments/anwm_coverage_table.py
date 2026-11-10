"""Per-step FFP coverage for the delivered level clips (computed offline, not in the manifest).

The level clips were generated before coverage recording existed, so their fill fraction is
recomputed here from the stored commanded path and the local released trajectory, with the same
FFP code the generator used. Written to results/anwm_injected/coverage_level_original.json and
joined into the pairing table by Track A2.

    python -m experiments.anwm_coverage_table
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "anwm"))

from anwm.projection import (project_to_2d_image_seq2seq,  # noqa: E402
                             reproject_depth_to_other_pose_seq2seq)
from src.generation.attitude import pose_with_attitude  # noqa: E402
from src.generation.poses import camera_axis_change, pose_from_position_yaw  # noqa: E402

TRAJ = "3KA7IJSNXE3LKLTL2ARA5RQP4ACBP0_processed"
DATA = ROOT / "data/raw/anwm_full" / TRAJ
CTX = 16


def main() -> int:
    import pickle

    t = {k: np.asarray(v) for k, v in pickle.load(open(DATA / "traj_data.pkl", "rb")).items()}
    R_ec, _ = camera_axis_change(t["pose"], t["yaw"])
    specs = json.load(open(ROOT / "results/anwm_clips/specs.json"))
    inj = {s["id"]: s for s in json.load(open(ROOT / "results/anwm_clips_injected/specs.json"))
           if s["arm"] == "injected_true"}
    out = {}
    for sp in specs:
        if sp["regime"] != "C":
            continue
        pos = np.asarray(sp["positions"], float)
        yaw = np.asarray(sp["yaw"], float)
        ct = int(sp["start_time"])
        ctx = list(range(ct - CTX + 1, ct + 1))
        rgbs = np.stack([cv2.cvtColor(cv2.imread(str(DATA / f"{x}.jpg")), cv2.COLOR_BGR2RGB)
                         for x in ctx])
        row = {"start_time": ct, "maneuver": sp["meta"]["maneuver"],
               "policy": sp["meta"]["policy"]}
        for arm, poses in (("level", pose_from_position_yaw(pos[1:], yaw[1:], R_ec)),
                           ("injected_true", pose_with_attitude(
                               pos[1:], yaw[1:], np.asarray(inj[sp["id"]]["thrust"], float)[1:],
                               R_ec))):
            pts, cols = reproject_depth_to_other_pose_seq2seq(
                K=t["K"], depth_maps=t["depth"][ctx], rgb_imgs=rgbs,
                poses_src=t["pose"][ctx], poses_dst=poses)
            img = project_to_2d_image_seq2seq(K=t["K"], points_3d=pts, colors=cols,
                                              image_size=t["depth"][ctx].shape[-2:])
            cov = [float((img[i].sum(-1) > 0).mean()) for i in range(len(img))]
            row[arm] = {"per_step": cov, "mean": float(np.mean(cov)), "min": float(np.min(cov))}
        out[sp["id"]] = row
        print(f"{sp['id']:32s} level {row['level']['mean']:.3f}/{row['level']['min']:.3f}  "
              f"injected {row['injected_true']['mean']:.3f}/{row['injected_true']['min']:.3f}",
              flush=True)
    p = ROOT / "results/anwm_injected/coverage_level_original.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} clips -> {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
