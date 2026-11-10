"""Scale provenance: what scale the bake-off's magnitude ratio uses, and what a depth prior gives.

The bake-off scores every clip after a **Sim(3)** alignment, i.e. with an **oracle** scale
fitted against ground truth. That is fine for measuring trajectory *shape*, but the verifier
has no mocap, so its magnitude claim must come from a depth prior. This script computes a
prior-only scale and re-scores with it.

Depth-prior scale: for consecutive frames with little rotation, the median scene depth falls
by the forward displacement,

    D[t+1] ~= D[t] - (t . z_cam)          (metres, z_cam = camera optical axis)

so the ratio of the metric depth change to the tracker's forward step in its own units is the
metres-per-unit scale:

    s = median( -(D[t+1] - D[t]) / ((p[t+1] - p[t]) . z_vo) )

Only pairs with |delta R| < 3 deg and |forward step| above the noise floor are used. This is a
weak estimator (it assumes a static scene and a mostly-forward motion), which is why the
spread is reported beside the median.

    python experiments/pathrec_scale_provenance.py \
        --poses /tmp/cp/poses/dpvo --truth /tmp/ct/uzh --depth /tmp/cdepth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.pathrec.align import path_length, umeyama
from src.pathrec.evaluate import up_axis
from src.pathrec.lean import differentiate, direction_error_deg, lean_xy, magnitude_ratio


def depth_scale(c2w: np.ndarray, depth_m: list[float], max_rot_deg: float = 3.0) -> dict:
    """metres-per-unit scale from a metric depth sequence (see module docstring)."""
    p = c2w[:, :3, 3]
    R = c2w[:, :3, :3]
    n = min(len(p), len(depth_m))
    p, R, d = p[:n], R[:n], np.asarray(depth_m[:n], dtype=np.float64)
    ratios, used = [], 0
    for i in range(n - 1):
        if not np.isfinite(d[i]) or not np.isfinite(d[i + 1]):
            continue
        rel = R[i].T @ R[i + 1]
        ang = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
        if ang > max_rot_deg:
            continue
        z = R[i][:, 2]
        fwd = float((p[i + 1] - p[i]) @ z)
        dD = float(d[i + 1] - d[i])
        if abs(fwd) < 1e-4 or not (0.02 < abs(dD) < 5.0):
            continue
        # depth falls when the camera moves forward along +z, so -dD / fwd > 0
        s = -dD / fwd
        if 1e-3 < s < 1e6:
            ratios.append(s)
            used += 1
    if not ratios:
        return {"n": 0, "scale": float("nan"), "spread": float("nan")}
    r = np.asarray(ratios)
    return {"n": used, "scale": float(np.median(r)),
            "spread": float(np.percentile(r, 75) - np.percentile(r, 25))}


def score_with_scale(c2w: np.ndarray, true_pos: np.ndarray, scale: float, dt: float,
                     window_s: float = 0.5) -> dict:
    """Direction and magnitude ratio with a FIXED scale (no truth-fitted scale)."""
    est = c2w[:, :3, 3]
    n = min(len(est), len(true_pos))
    rot, _, _ = umeyama(est[:n], true_pos[:n], with_scale=True)   # rotation only is used
    p_metric = scale * est[:n] @ rot.T
    p_metric += true_pos[:n].mean(0) - p_metric.mean(0)
    _, _, a_e = differentiate(p_metric, dt=dt, window_s=window_s)
    _, v_t, a_t = differentiate(true_pos[:n], dt=dt, window_s=window_s)
    m = min(len(a_e), len(a_t))
    up = up_axis(true_pos[:n])
    perm = [i for i in range(3) if i != up["axis"]] + [up["axis"]]
    l_e, l_t = lean_xy(a_e[:m][:, perm]), lean_xy(a_t[:m][:, perm])
    d = direction_error_deg(l_e, l_t)
    return {"dir_err_deg_median": float(np.median(d)) if d.size else float("nan"),
            "mag_ratio": magnitude_ratio(l_e, l_t),
            "coverage": float((np.degrees(np.arctan(np.linalg.norm(l_t, axis=1))) >= 10).mean()),
            "path_len_ratio": path_length(p_metric) / path_length(true_pos[:n])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--out", default="results/pathrec_scale_provenance/v1")
    args = ap.parse_args()

    depth_path = Path(args.depth)
    depth_by_seq = {}
    if depth_path.is_file():
        for r in json.loads(depth_path.read_text()):
            depth_by_seq[r["seq"]] = r
    rows = []
    for npz in sorted(Path(args.poses).glob("*.npz")):
        if npz.stem.startswith("tii_"):
            continue
        seq = npz.stem
        tp = Path(args.truth) / seq / "truth.npz"
        drec = depth_by_seq.get(seq)
        if not tp.exists() or drec is None:
            continue
        c2w = np.load(npz)["c2w"]
        truth = np.load(tp)
        meta = Path(args.truth) / seq / "meta.json"
        fps = json.loads(meta.read_text())["fps"] if meta.exists() else 15.1
        _, s_oracle, _ = umeyama(c2w[:, :3, 3], truth["pos"][:len(c2w)], with_scale=True)
        ds = depth_scale(c2w, drec["median_depth_m"])
        with_prior = score_with_scale(c2w, truth["pos"], ds["scale"], 1.0 / fps) if np.isfinite(ds["scale"]) else {}
        rows.append({"seq": seq, "scale_oracle": float(s_oracle), "scale_depth": ds["scale"],
                     "depth_pairs": ds["n"], "depth_spread": ds["spread"],
                     "depth_vs_oracle": (ds["scale"] / s_oracle) if np.isfinite(ds["scale"]) else None,
                     "with_depth_scale": with_prior})
        print(f"{seq:22s} oracle {s_oracle:7.3f}  depth {ds['scale']:9.3f} "
              f"(x{ds['scale']/s_oracle:6.3f}, n={ds['n']:3d})  "
              f"dir {with_prior.get('dir_err_deg_median', float('nan')):6.1f}  "
              f"ratio {with_prior.get('mag_ratio', float('nan')):5.2f}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    if rows:
        r = np.asarray([x["depth_vs_oracle"] for x in rows if x["depth_vs_oracle"]], dtype=float)
        print(f"\ndepth/oracle scale ratio: median {np.median(r):.3f}, "
              f"p10 {np.percentile(r,10):.3f}, p90 {np.percentile(r,90):.3f} over {len(r)} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
