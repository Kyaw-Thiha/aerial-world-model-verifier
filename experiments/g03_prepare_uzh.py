"""Stage UZH forward-rig frames with mocap-grade attitude labels for report G03.

The UZH-FPV forward sequences are the transfer rigs (the 45-degree-down mount did not
transfer, report E03). Each frame's label is the world-up direction in the *pinhole* camera
frame, built from the Leica truth attitude and the Kalibr ``T_cam_imu`` mount - no pixel
estimate enters it. Undistortion preserves ray directions, so this is the same vector the
fisheye model would predict.

    PYTHONPATH=. python experiments/g03_prepare_uzh.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SEQUENCES = ["indoor_forward_3", "indoor_forward_5", "indoor_forward_9"]


def main() -> int:
    out_root = Path("data/g03_uzh")
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {"sequences": {}}
    for seq in SEQUENCES:
        src = Path("data/uzh") / seq
        meta = json.loads((src / "meta.json").read_text())
        truth = np.load(src / "truth.npz")
        T = np.asarray(meta["kalibr_imu_cam0"]["T_cam_imu"], dtype=np.float64)
        # T_cam_imu maps imu -> camera. The project's R_cb is camera -> body (the transpose),
        # and up_cam = R_cb^T @ up_body = R_cam_imu @ up_body. Using T's rotation directly as
        # R_cb transposes the mount and put a ~60 deg error into the first UZH label pass.
        R_ci = T[:3, :3]                                   # imu -> camera
        R_cb = R_ci.T                                      # camera -> body (project convention)
        R_bw = Rotation.from_quat(np.asarray(truth["quat"], dtype=np.float64)).as_matrix()
        up_body = np.einsum("nji,j->ni", R_bw, np.array([0.0, 0.0, 1.0]))
        up_cam = np.einsum("ji,nj->ni", R_cb, up_body)     # = R_ci @ up_body
        frames = sorted((src / "frames_pinhole").glob("*.png"))
        if len(frames) != len(up_cam):
            raise SystemExit(f"{seq}: {len(frames)} frames vs {len(up_cam)} labels")
        dst = out_root / seq
        dst.mkdir(parents=True, exist_ok=True)
        for p in frames:
            target = dst / p.name
            if not target.exists():
                target.symlink_to(p.resolve())
        labels = [{"frame": p.name, "up_cam": [float(x) for x in up_cam[i]],
                   "t": float(truth["t"][i])}
                  for i, p in enumerate(frames)]
        (dst / "labels.json").write_text(json.dumps({
            "sequence": seq, "rig": meta["rig"],
            "mount": "kalibr_imu_cam0.T_cam_imu (cam<-imu)",
            "camera": meta["intrinsics_pinhole"],
            "frames": labels}, indent=0))
        manifest["sequences"][seq] = len(labels)
        print(f"{seq}: {len(labels)} frames", flush=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
