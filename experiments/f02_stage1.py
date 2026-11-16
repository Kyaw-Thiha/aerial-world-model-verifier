"""Track F, stage 1: score ViPE and MASt3R-SLAM against Leica truth on the ten UZH sequences.

Uses exactly Track C's bake-off metric (`src.pathrec.evaluate.score_clip`: 0.5 s differentiation
window, k_d = 0.295, tilt >= 10 deg coverage, Sim(3) and SE(3) ATE, direction, magnitude, per
frame rotation) so the table extends E7's, and the same pre-registered pass rule
(`configs/pathrec_prereg.yaml`: direction <= 30 deg, magnitude ratio 0.7-1.4, coverage >= 0.5).

    python experiments/f02_stage1.py --poses-root data/pathrec/poses
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.pathrec.evaluate import score_clip

SEQS = ("indoor_forward_3", "indoor_forward_5", "indoor_forward_6", "indoor_forward_7",
        "indoor_forward_9", "indoor_forward_10", "outdoor_forward_1", "outdoor_forward_3",
        "outdoor_forward_5", "indoor_45_9")
METHODS = ("dpvo", "vipe", "mast3r")


def load_truth(uzh: Path, seq: str) -> dict:
    d = np.load(uzh / seq / "truth.npz")
    t = np.asarray(d["t"], dtype=np.float64)
    q = np.asarray(d["quat"], dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    meta = json.loads((uzh / seq / "meta.json").read_text())
    return {"t": t, "pos": np.asarray(d["pos"], dtype=np.float64), "quat": q,
            "fps": float(meta["fps_measured"])}


def score_method(poses: np.ndarray, truth: dict) -> dict:
    n = min(len(poses), len(truth["pos"]))
    est = poses[:n, :3, 3]
    quat = Rotation.from_matrix(poses[:n, :3, :3]).as_quat()
    sc = score_clip(est, truth["pos"][:n], dt=1.0 / truth["fps"],
                    est_quat=quat, true_quat=truth["quat"][:n], window_s=0.5)
    finite = bool(np.isfinite(poses).all())
    sc.update({"n_used": int(n), "all_finite": finite,
               "path_len_units": float(np.linalg.norm(np.diff(est, axis=0), axis=1).sum())})
    return sc


def passes(sc: dict, crit: dict) -> bool:
    d = sc.get("accel_dir_err_deg_median")
    ratio = sc.get("accel_mag_ratio")
    cov = sc.get("coverage_fraction")
    if d is None or ratio is None or cov is None:
        return False
    lo, hi = crit["accel_magnitude"]["interval"]
    return bool(d <= crit["accel_direction"]["threshold_deg"] and lo <= ratio <= hi
                and cov >= 0.2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses-root", default="data/pathrec/poses")
    ap.add_argument("--uzh", default="data/uzh")
    ap.add_argument("--prereg", default="configs/pathrec_prereg.yaml")
    ap.add_argument("--out", default="results/f02_stage1/scores.json")
    args = ap.parse_args()
    import yaml
    crit = yaml.safe_load(Path(args.prereg).read_text())["criteria"]
    root, uzh = Path(args.poses_root), Path(args.uzh)

    out = {"criteria": crit, "methods": {}}
    for method in METHODS:
        d = root / method
        if not d.exists():
            out["methods"][method] = {"missing": str(d)}
            continue
        rows = {}
        for seq in SEQS:
            p = d / f"{seq}.npz"
            if not p.exists():
                continue
            z = np.load(p)
            poses = np.asarray(z["c2w"], dtype=np.float64)
            truth = load_truth(uzh, seq)
            sc = score_method(poses, truth)
            sc["pass"] = passes(sc, crit)
            rows[seq] = sc
        if rows:
            dirs = [r["accel_dir_err_deg_median"] for r in rows.values()]
            ratios = [r["accel_mag_ratio"] for r in rows.values()]
            out["methods"][method] = {
                "n_sequences": len(rows),
                "n_pass": int(sum(r["pass"] for r in rows.values())),
                "median_dir_err_deg": float(np.median(dirs)),
                "median_mag_ratio": float(np.median(ratios)),
                "rows": rows,
            }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    for method, block in out["methods"].items():
        if "missing" in block:
            print(f"{method:8s} missing {block['missing']}")
            continue
        print(f"{method:8s} {block['n_pass']}/{block['n_sequences']} pass, "
              f"median dir {block['median_dir_err_deg']:.1f} deg, "
              f"median ratio {block['median_mag_ratio']:.2f}")
        for seq, r in block["rows"].items():
            print(f"   {seq:20s} dir={r['accel_dir_err_deg_median']:6.1f} "
                  f"ratio={r['accel_mag_ratio']:5.2f} cov={r['coverage_fraction']:4.2f} "
                  f"ate_sim3={r['ate_sim3_m']:6.2f} pass={r['pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
