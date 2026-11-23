"""Independent geometric check of a rendered clip: does the horizon sit where the pose says?

GeoCalib is the instrument under test, so it cannot also be the instrument that verifies the
render. This script measures the sky/ground boundary straight from the pixels and compares it
with the boundary the *known* camera pose predicts, column by column. It needs no network, no
GPU and no model.

Predicted horizon: a camera ray ``d_c`` lies on the horizon when its world direction is
horizontal, i.e. ``up_cam . d_c = 0``. With ``d_c = ((x-cx)/f, (y-cy)/f, 1)`` this is linear
in ``y``, so each column has exactly one horizon row.

Measured horizon: the strongest vertical intensity gradient in that column, searched from the
top of the image (the sky/ground transition is the first big edge on an outdoor scene).

    python experiments/i01_horizon_check.py --frames results/i01_spike/frames_blender/feasible \
        --poses results/i01_spike/render/i01_feasible_poses.npz --out results/i01_spike/horizon_blender.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def predicted_horizon_rows(poses: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(M, W) predicted horizon row per column for each camera pose (camera-to-world)."""
    W = int(round(2 * K[0, 2] + 1))
    f = K[0, 0]
    cx, cy = K[0, 2], K[1, 2]
    cols = np.arange(W)
    u = (cols - cx) / f
    out = np.full((len(poses), W), np.nan)
    for i, T in enumerate(poses):
        up = T[:3, :3].T @ np.array([0.0, 0.0, 1.0])   # world up in camera coords
        if abs(up[1]) < 1e-9:
            continue                                   # horizon is vertical in this view
        v = -(up[0] * u + up[2]) / up[1]
        out[i] = cy + f * v
    return out


def measured_horizon_rows(gray: np.ndarray, std_thresh: float = 0.012,
                          run: int = 4) -> np.ndarray:
    """(W,) horizon row per column: where the smooth sky stops being smooth.

    A gradient maximum is the wrong detector here: buildings, trees and the checker pattern
    all produce edges above the sky/ground boundary, and on a rendered clip that detector was
    11 deg off the pose. Local contrast is the property that actually separates the two
    regions -- the sky is a smooth gradient, the ground is textured -- so the boundary is the
    first row (from the top) after which the local standard deviation stays above
    ``std_thresh`` for ``run`` consecutive rows.
    """
    g = np.asarray(gray, dtype=np.float64)
    H, W = g.shape
    # 5x5 local standard deviation via box sums
    pad = np.pad(g, 2, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, (5, 5))
    std = win.std(axis=(-1, -2))
    rows = np.full(W, np.nan)
    for x in range(W):
        col = std[:, x]
        above = col > std_thresh
        for y in range(0, H - run):
            if above[y:y + run].all():
                rows[x] = y + 0.5
                break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--hfov", type=float, default=90.0)
    args = ap.parse_args()

    from PIL import Image

    npz = np.load(args.poses)
    poses = npz["poses"]
    f = 0.5 * args.width / np.tan(np.radians(args.hfov) / 2.0)
    K = np.array([[f, 0, (args.width - 1) / 2], [0, f, (args.height - 1) / 2], [0, 0, 1]])
    pred = predicted_horizon_rows(poses, K)

    rows = []
    for p in sorted(Path(args.frames).glob("*.png")):
        i = int(p.stem)
        gray = np.asarray(Image.open(p).convert("L"), dtype=np.float64) / 255.0
        meas = measured_horizon_rows(gray)
        ok = np.isfinite(meas) & np.isfinite(pred[i])
        # ignore the 10% margin where the render has border artefacts
        margin = int(0.1 * args.width)
        ok[:margin] = False
        ok[-margin:] = False
        if ok.sum() < 20:
            continue
        d_px = meas[ok] - pred[i][ok]
        # angular error of the horizon position: a row offset at the image centre is
        # atan(dy / f) away from where the pose puts it
        d_deg = np.degrees(np.arctan(d_px / f))
        rows.append({"frame": p.name, "n_cols": int(ok.sum()),
                     "median_d_px": float(np.median(d_px)),
                     "median_d_deg": float(np.median(d_deg)),
                     "p90_abs_d_deg": float(np.percentile(np.abs(d_deg), 90))})

    out = {"frames": len(rows),
           "median_d_deg": float(np.median([r["median_d_deg"] for r in rows])) if rows else None,
           "p90_abs_d_deg": float(np.percentile([abs(r["median_d_deg"]) for r in rows], 90))
           if rows else None,
           "per_frame": rows}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_frame"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
