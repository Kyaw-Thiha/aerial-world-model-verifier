"""Diagnostics for the two UZH sequences where every extractor fails (45-deg mount, outdoor).

Four questions, all answerable from stored poses plus one re-run:

1. **Is it the scale?** Sweep the scale factor around the oracle value and see whether the
   magnitude ratio can be brought to 1 while the direction error stays bad.
2. **Is it a constant rotation (mount convention)?** Fit one global rotation on the recovered
   acceleration vectors and report the residual direction error. A convention error collapses
   to zero; a real failure does not.
3. **Is the mount what we think it is?** Report the camera-to-world tilt implied by the truth
   attitude (angle of the body z axis from world up) per sequence.
4. **Is it the stride?** Requires a re-run at a different stride; the numbers land in the same
   report, this script only compares stored runs.

    python experiments/pathrec_diagnostics.py --poses /tmp/cposes2/poses/pi3 \
        --truth /tmp/ctruth3/uzh --out results/pathrec_diag/v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.pathrec.align import umeyama
from src.pathrec.evaluate import up_axis
from src.pathrec.lean import differentiate, direction_error_deg, lean_xy, magnitude_ratio


def fit_rotation(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotation R minimising |a - R b| (Kabsch on zero-mean vectors), plus residual (deg)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    a, b = a[ok], b[ok]
    a0, b0 = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a0.T @ b0)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    resid = a0 - b0 @ r.T
    ang = np.degrees(np.arccos(np.clip(
        (a0 * (b0 @ r.T)).sum(1) / (np.linalg.norm(a0, axis=1) * np.linalg.norm(b0, axis=1)), -1, 1)))
    return r, float(np.median(ang)), resid


def diagnostics(c2w: np.ndarray, true_pos: np.ndarray, true_quat: np.ndarray, fps: float,
                window_s: float = 0.5) -> dict:
    est = c2w[:, :3, 3]
    n = min(len(est), len(true_pos))
    est, true_pos, true_quat = est[:n], true_pos[:n], true_quat[:n]
    rot, scale, trans = umeyama(est, true_pos, with_scale=True)
    dt = 1.0 / fps
    t_e, v_e, a_e = differentiate(scale * est @ rot.T + trans, dt=dt, window_s=window_s)
    t_t, v_t, a_t = differentiate(true_pos, dt=dt, window_s=window_s)
    m = min(len(a_e), len(a_t))
    a_e, a_t = a_e[:m], a_t[:m]
    up = up_axis(true_pos)
    perm = [i for i in range(3) if i != up["axis"]] + [up["axis"]]
    a_e, a_t = a_e[:, perm], a_t[:, perm]

    l_e, l_t = lean_xy(a_e), lean_xy(a_t)
    out = {"n": int(n), "fps": fps, "scale": float(scale),
           "up_axis": up["axis"], "up_unambiguous": up["unambiguous"],
           "dir_err_median_deg": float(np.median(direction_error_deg(l_e, l_t))),
           "mag_ratio": magnitude_ratio(l_e, l_t)}

    # (1) scale sweep: does any scale fix the magnitude?
    for f in (0.5, 0.75, 1.0, 1.5, 2.0):
        l_s = lean_xy(a_e * f)
        out[f"mag_ratio_at_scale_x{f:g}"] = magnitude_ratio(l_s, l_t)
        out[f"dir_err_at_scale_x{f:g}"] = float(np.median(direction_error_deg(l_s, l_t)))

    # (2) one global rotation on the horizontal acceleration vectors
    ok = np.isfinite(l_e).all(1) & np.isfinite(l_t).all(1)
    if ok.sum() >= 10:
        a3e = np.c_[a_e[ok][:, :2], np.zeros(ok.sum())]
        a3t = np.c_[a_t[ok][:, :2], np.zeros(ok.sum())]
        r, resid, _ = fit_rotation(a3t, a3e)
        out["global_rotation_residual_deg"] = resid
        out["global_rotation_yaw_deg"] = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))

    # (3) mount: angle of the truth body z axis from world up
    q = true_quat / np.linalg.norm(true_quat, axis=1, keepdims=True)
    rz = Rotation.from_quat(q).as_matrix()[:, :, 2]
    out["truth_body_z_from_up_deg_median"] = float(np.degrees(np.median(
        np.arccos(np.clip(rz[:, up["axis"]], -1, 1)))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True, help="dir with <seq>.npz files for one model")
    ap.add_argument("--truth", required=True, help="dir with <seq>/truth.npz")
    ap.add_argument("--model", default="pi3")
    ap.add_argument("--out", default="results/pathrec_diag/v1")
    args = ap.parse_args()

    rows = []
    for npz in sorted(Path(args.poses).glob("*.npz")):
        seq = npz.stem
        tp = Path(args.truth) / seq / "truth.npz"
        if not tp.exists():
            continue
        meta_p = Path(args.truth) / seq / "meta.json"
        fps = json.loads(meta_p.read_text())["fps"] if meta_p.exists() else 15.1
        rec = diagnostics(np.load(npz)["c2w"], np.load(tp)["pos"], np.load(tp)["quat"], fps)
        rec.update({"model": args.model, "seq": seq})
        rows.append(rec)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    hdr = ["seq", "dir", "ratio", "rot_resid", "body_z_up", "r@0.5", "r@1", "r@2", "d@2"]
    print(" ".join(f"{h:>10s}" for h in hdr))
    for r in rows:
        print(f"{r['seq']:>10s} {r['dir_err_median_deg']:10.1f} {r['mag_ratio']:10.2f} "
              f"{r.get('global_rotation_residual_deg', float('nan')):10.1f} "
              f"{r['truth_body_z_from_up_deg_median']:10.1f} "
              f"{r['mag_ratio_at_scale_x0.5']:10.2f} {r['mag_ratio_at_scale_x1']:10.2f} "
              f"{r['mag_ratio_at_scale_x2']:10.2f} {r['dir_err_at_scale_x2']:10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
