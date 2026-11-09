"""Prepare one UZH-FPV sequence (Task 4): undistort to a pinhole view and pair with truth.

The point of the exercise: UZH-FPV ships a well-posed Kannala-Brandt calibration, so the
same "undistort, then run DPVO" pipeline can be tested where the calibration is not the
suspect. TII's released intrinsics are a pinhole fit of a 175 deg fisheye whose polynomial
inverts only to 56 deg, and DPVO's translation fails there; if it works here, the
calibration is the likely cause, and if it fails here too, the method is.

Input : one ``*_with_gt.zip`` from UZH-FPV (calib.yaml, groundtruth.txt, images/)
Output: ``<out>/frames/*.png`` (undistorted, 8-bit grey), ``<out>/meta.json`` with the
        pinhole intrinsics, per-frame timestamps, and the ground-truth track on the same
        clock as the frame filenames.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_calib(path: Path) -> dict:
    cal = yaml.safe_load(path.read_text())
    keys = {"fx", "fy", "cx", "cy"} | {k for k in cal if str(k).startswith("k")}
    return {k: float(cal[k]) for k in keys}


def load_groundtruth(path: Path) -> np.ndarray:
    """TUM-format rows: timestamp tx ty tz qx qy qz qw."""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        v = [float(x) for x in line.split()]
        if len(v) >= 8:
            rows.append(v[:8])
    return np.asarray(rows, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--out-size", default="480x360", help="WxH of the pinhole view")
    args = ap.parse_args()

    import zipfile

    zpath = Path(args.zip)
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        calib_name = next(n for n in names if n.endswith("calib.yaml"))
        gt_name = next(n for n in names if n.endswith("groundtruth.txt"))
        tmp = Path("/tmp") / "calib.yaml"
        tmp.write_bytes(z.read(calib_name))
        calib = load_calib(tmp)
        tmp_gt = Path("/tmp") / "gt.txt"
        tmp_gt.write_bytes(z.read(gt_name))
        gt = load_groundtruth(tmp_gt)
        imgs = sorted(n for n in names if n.lower().endswith((".png", ".jpg")))[:: args.stride]
        imgs = imgs[: args.max_frames]
        w_out, h_out = (int(x) for x in args.out_size.split("x"))
        K_in = np.array([[calib["fx"], 0, calib["cx"]], [0, calib["fy"], calib["cy"]],
                         [0, 0, 1]], dtype=np.float64)
        dist = np.array([calib.get("k1", 0.0), calib.get("k2", 0.0), calib.get("k3", 0.0),
                         calib.get("k4", 0.0)], dtype=np.float64)
        # keep the same focal in the undistorted view, so the field narrows instead of the
        # image scaling; the calibration is valid over the whole KB4 domain
        K_out = np.array([[calib["fx"], 0, w_out / 2], [0, calib["fy"], h_out / 2],
                          [0, 0, 1]], dtype=np.float64)
        n_written, ts = 0, []
        for i, n in enumerate(imgs):
            buf = np.frombuffer(z.read(n), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            und = cv2.fisheye.undistortImage(img, K_in, dist, Knew=K_out,
                                             new_size=(w_out, h_out))
            cv2.imwrite(str(out / "frames" / f"{i:05d}.png"), und)
            ts.append(int(Path(n).stem))
            n_written += 1

    meta = {"sequence": zpath.stem, "n_frames": n_written, "stride": args.stride,
            "K": K_out.tolist(), "image_hw": [h_out, w_out],
            "source_calib": {"fx": calib["fx"], "fy": calib["fy"], "cx": calib["cx"],
                             "cy": calib["cy"], "k1": float(dist[0]), "k2": float(dist[1]),
                             "k3": float(dist[2]), "k4": float(dist[3]), "model": "KB4"},
            "frame_timestamps_ns": ts,
            "gt_tum": gt.tolist(),
            "gt_note": "TUM rows: timestamp tx ty tz qx qy qz qw, dataset clock (ns)"}
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"{n_written} frames -> {out}; gt rows {len(gt)}; K_out fx={K_out[0,0]:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
