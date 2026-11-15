"""How different is the injected prior from the level prior? (input side, no generation)

For every injected clip, renders both priors from the same context frames and reports the
per-step pixel difference and correlation. This is the *input* difference that the generated
frames are asked to follow; the output side is `evaluate_injected`.

    python -m experiments.anwm_prior_difference
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


def render(t, ctx, rgbs, poses):
    pts, cols = reproject_depth_to_other_pose_seq2seq(
        K=t["K"], depth_maps=t["depth"][ctx], rgb_imgs=rgbs, poses_src=t["pose"][ctx],
        poses_dst=poses)
    return project_to_2d_image_seq2seq(K=t["K"], points_3d=pts, colors=cols,
                                       image_size=t["depth"][ctx].shape[-2:])


def main() -> int:
    import pickle

    t = {k: np.asarray(v) for k, v in pickle.load(open(DATA / "traj_data.pkl", "rb")).items()}
    R_ec, _ = camera_axis_change(t["pose"], t["yaw"])
    specs = [s for s in json.load(open(ROOT / "results/anwm_clips_injected/specs.json"))
             if s["arm"] == "injected_true" and int(s.get("seed", 0)) == 0]
    out = {}
    for sp in specs:
        pos = np.asarray(sp["positions"], float)
        yaw = np.asarray(sp["yaw"], float)
        ct = int(sp["start_time"])
        ctx = list(range(ct - CTX + 1, ct + 1))
        rgbs = np.stack([cv2.cvtColor(cv2.imread(str(DATA / f"{x}.jpg")), cv2.COLOR_BGR2RGB)
                         for x in ctx])
        lvl = render(t, ctx, rgbs, pose_from_position_yaw(pos[1:], yaw[1:], R_ec))
        inj = render(t, ctx, rgbs, pose_with_attitude(
            pos[1:], yaw[1:], np.asarray(sp["thrust"], float)[1:], R_ec))
        mae, ncc = [], []
        for i in range(len(lvl)):
            a, b = lvl[i].astype(float), inj[i].astype(float)
            m = (a.sum(-1) > 0) & (b.sum(-1) > 0)
            mae.append(float(np.abs(a - b)[m].mean()) if m.any() else 0.0)
            if m.sum() > 100:
                x = a[m].ravel() - a[m].mean()
                y = b[m].ravel() - b[m].mean()
                ncc.append(float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-9)))
            else:
                ncc.append(0.0)
        out[sp["id"]] = {"maneuver": sp["meta"]["maneuver"], "policy": sp["meta"]["policy"],
                         "mae_per_step_255": mae, "mae_mean_255": float(np.mean(mae)),
                         "ncc_per_step": ncc, "ncc_mean": float(np.mean(ncc))}
        print(f"{sp['id']:34s} prior mae {np.mean(mae):6.2f}/255 ncc {np.mean(ncc):.3f}",
              flush=True)
    p = ROOT / "results/anwm_injected/prior_difference.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} clips -> {p.relative_to(ROOT)}")
    import collections
    g = collections.defaultdict(list)
    for v in out.values():
        g[(v["maneuver"], v["policy"])].append((v["mae_mean_255"], v["ncc_mean"]))
    for k, v in sorted(g.items()):
        a = np.array(v)
        print(f"{k[0]+'/'+k[1]:24s} n={len(v):2d} prior mae {a[:,0].mean():6.2f}/255  ncc {a[:,1].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
