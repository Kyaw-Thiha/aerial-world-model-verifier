"""Extract the exact TII frames the verdict used, with mocap-labelled attitude (report G03).

Inputs
------
* ``data/processed/p3a/p3a_v1/<flight>.npz``: the per-frame table Track A built (times,
  positions, quaternions, frame *names*, recorded uptilt). The clips index into it.
* ``data/raw/tii_video_full/camera_<flight>.zip``: the stored (uncompressed) JPEG members.
* ``results/gate_a2/tii/<flight>__identity.jsonl``: frozen GeoCalib gravity + confidence per
  frame, the baseline this report compares against (no GPU needed to recompute it).

Output
------
``data/g03_frames/<flight>/<name>.jpg`` plus ``labels.json``:

* ``up_cam``: the true world-up direction in the camera frame, from the mocap body attitude
  and the analytic camera-to-body mount the verdict uses (``camera_to_body_pitched``);
* ``yaw``, ``t``: the path quantities the verdict needs;
* ``up_cam_base``, ``conf_base``: the frozen reader's per-frame output on the same frame.

The label is a *mocap* label: no pixel estimate enters it. It is the quantity GeoCalib
predicts (gravity, i.e. up, in camera coordinates), so the per-frame error is directly
comparable.

    PYTHONPATH=. python experiments/g03_prepare_frames.py
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.perception.frames import camera_to_body_pitched

FLIGHTS = ["flight-01a-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
           "flight-13a-trackRATM", "flight-01p-ellipse", "flight-07p-lemniscate",
           "flight-13p-trackRATM", "flight-16p-trackRATM"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="data/processed/p3a/p3a_v1")
    ap.add_argument("--zips", default="data/raw/tii_video_full")
    ap.add_argument("--rows", default="results/gate_a2/tii")
    ap.add_argument("--out", default="data/g03_frames")
    args = ap.parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {"frames": 0, "per_flight": {}}

    for flight in FLIGHTS:
        spec = np.load(Path(args.specs) / f"{flight}.npz", allow_pickle=False)
        names = [str(n) for n in spec["names"]]
        t = np.asarray(spec["t"], dtype=np.float64)
        R_bw = Rotation.from_quat(np.asarray(spec["quat"], dtype=np.float64)).as_matrix()
        up_world = np.array([0.0, 0.0, 1.0])
        up_body = np.einsum("nji,j->ni", R_bw, up_world)          # R_bw^T e3
        R_cb = camera_to_body_pitched(np.deg2rad(float(spec["uptilt_deg"])))
        up_cam_true = np.einsum("ji,nj->ni", R_cb, up_body)        # R_cb^T up_body
        yaw = Rotation.from_matrix(R_bw).as_euler("ZYX")[:, 0]

        rows = {}
        path = Path(args.rows) / f"{flight}__identity.jsonl"
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["frame"]] = r

        dst = out_root / flight
        dst.mkdir(parents=True, exist_ok=True)
        zpath = Path(args.zips) / f"camera_{flight}.zip"
        with zipfile.ZipFile(zpath) as z:
            members = {Path(n).name: n for n in z.namelist() if n.lower().endswith(".jpg")}
            labels = []
            for i, name in enumerate(names):
                member = members.get(name)
                if member is None:
                    continue
                target = dst / name
                if not target.exists():
                    with z.open(member) as src, open(target, "wb") as fh:
                        fh.write(src.read())
                r = rows.get(name, {})
                labels.append({
                    "frame": name, "index": i, "t": float(t[i]),
                    "yaw": float(yaw[i]),
                    "up_cam": [float(x) for x in up_cam_true[i]],
                    "up_cam_base": r.get("gravity_up_cam"),
                    "conf_base": r.get("latitude_confidence_mean"),
                })
        (dst / "labels.json").write_text(json.dumps({
            "flight": flight, "uptilt_deg": float(spec["uptilt_deg"]),
            "measured_fps": float(spec["measured_fps"]),
            "note": "up_cam is the mocap-truth world-up in the camera frame under the analytic "
                    "mount the verdict uses; up_cam_base is the frozen GeoCalib estimate",
            "frames": labels}, indent=0))
        manifest["per_flight"][flight] = len(labels)
        manifest["frames"] += len(labels)
        print(f"{flight}: {len(labels)} frames", flush=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("total", manifest["frames"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
