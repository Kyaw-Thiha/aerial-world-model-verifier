"""Track F2, part 2: what path error the verifier actually needs.

For every stored path (DPVO, ViPE, MASt3R-SLAM) and every UZH sequence, compute the lean error
the path implies against the truth attitude, in the E05 rule-sized windows, with oracle
attitude. This is deliberately a *path* metric, not a verdict: it asks how far the thrust axis
implied by the recovered path is from the thrust axis the airframe actually had.

Concretely, in the Leica world frame:

    u_path = (s*a + g e3 + k_d*s*v) / ||.||        (spline derivatives, Sim(3) scale s)
    u_true = R_bw e3                               (truth body z)
    lean error = angle(u_path, u_true)             (yaw-invariant)

with k_d = 0.295. Windows are 0.5/f95 s (E05); the per-sequence value is the median of the
window medians. The same script reports Track C's bake-off direction error recomputed from the
same poses (self-contained; cross-checked against F02's table), the rank correlation between
the two path metrics, and (if part 1's results exist) whether either predicts per-clip verdict
quality.

    python experiments/f05_lean_error.py --out results/f05_lean_error
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.pathrec.align import umeyama
from src.pathrec.evaluate import score_clip
from src.pathrec.spline_fit import G0, fit_spline_path

SEQUENCES = ("indoor_forward_3", "indoor_forward_5", "indoor_forward_6",
             "indoor_forward_7", "indoor_forward_9", "indoor_forward_10",
             "outdoor_forward_1", "outdoor_forward_3", "outdoor_forward_5",
             "indoor_45_9")
PATHS = ("dpvo", "vipe", "mast3r", "mapanything", "pi3x")
PRIMARY_SEQS = ("indoor_forward_3", "indoor_forward_5", "indoor_forward_9",
                "indoor_forward_10")
KD = 0.295
N_BOOT = 5000
SEED = 1


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def _py(o):
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def load_truth(uzh: Path, seq: str) -> dict:
    d = np.load(uzh / seq / "truth.npz")
    meta = json.loads((uzh / seq / "meta.json").read_text())
    q = np.asarray(d["quat"], dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return {"t": np.asarray(d["t"], dtype=np.float64),
            "pos": np.asarray(d["pos"], dtype=np.float64),
            "quat": q, "R_bw": Rotation.from_quat(q).as_matrix(),
            "fps": float(meta.get("fps_measured", meta.get("fps")))}


def unwrap_rotation(mats: np.ndarray) -> np.ndarray:
    """Closest proper rotation to the mean of a set of rotations (SVD projection)."""
    m = np.mean(mats, axis=0)
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def path_leans(poses: np.ndarray, truth: dict, window_s: float, k_d: float = KD) -> dict:
    """Windowed lean error of one path against truth attitude (oracle attitude)."""
    n = min(len(poses), len(truth["pos"]))
    P = np.asarray(poses[:n, :3, 3], dtype=np.float64)
    R_wc = np.asarray(poses[:n, :3, :3], dtype=np.float64)
    R_bw = truth["R_bw"][:n]
    t = truth["t"][:n] - truth["t"][0]

    # the pre-registered evaluation transform: the whole-sequence Sim(3) alignment supplies
    # rotation AND scale (using the attitude average for the rotation instead lets DPVO's
    # absolute orientation drift masquerade as a path-shape error; measured 16.2 vs 3.4 deg
    # on indoor_forward_3 -- decisions.md [F2])
    R_align, s, _ = umeyama(P, truth["pos"][:n], with_scale=True)

    sp = fit_spline_path(t, P, window_s)
    a = (R_align @ (s * sp.a.T)).T           # (M,3) m/s^2 in Leica world
    v = (R_align @ (s * sp.v.T)).T           # (M,3) m/s
    up = np.array([0.0, 0.0, 1.0])
    q = a + G0 * up + k_d * v
    nq = np.linalg.norm(q, axis=1)
    ok = np.isfinite(nq) & (nq > 1e-9)
    u_path = np.full_like(q, np.nan)
    u_path[ok] = q[ok] / nq[ok, None]
    u_true = np.einsum("nij,j->ni", R_bw[3:len(t) - 3][:len(sp.t)], up)
    cos = np.clip(np.sum(u_path * u_true, axis=1), -1, 1)
    ang = np.degrees(np.arccos(cos))

    # non-overlapping windows over the spline's interior grid
    dt = float(np.median(np.diff(sp.t)))
    w = max(1, int(round(window_s / dt)))
    meds = []
    for lo in range(0, len(ang) - w + 1, w):
        seg = ang[lo:lo + w]
        if np.isfinite(seg).all():
            meds.append(float(np.median(seg)))
    meds = np.asarray(meds)
    return {"n_frames": int(n), "n_windows": int(len(meds)),
            "window_s": float(window_s), "window_samples": int(w),
            "scale_sim3": float(s),
            "window_median_deg_median": float(np.median(meds)) if len(meds) else float("nan"),
            "window_median_deg_p75": float(np.percentile(meds, 75)) if len(meds) else float("nan"),
            "window_median_deg_p90": float(np.percentile(meds, 90)) if len(meds) else float("nan"),
            "sample_angle_median_deg": float(np.nanmedian(ang)),
            "per_window_deg": meds.tolist()}


def bakeoff_direction(poses: np.ndarray, truth: dict) -> dict:
    """Track C's bake-off direction metric, recomputed from the same poses."""
    n = min(len(poses), len(truth["pos"]))
    q = Rotation.from_matrix(poses[:n, :3, :3]).as_quat()
    sc = score_clip(poses[:n, :3, 3], truth["pos"][:n], dt=1.0 / truth["fps"],
                    est_quat=q, true_quat=truth["quat"][:n], window_s=0.5)
    return {k: sc.get(k) for k in ("accel_dir_err_deg_median", "accel_mag_ratio",
                                   "coverage_fraction", "ate_sim3_m", "step_dir_err_deg_median")}


def rank_corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)) + 1.0
    ry = np.argsort(np.argsort(y)) + 1.0
    return float(np.corrcoef(rx, ry)[0, 1])


def clustered_corr(x, y, groups, n_boot=N_BOOT, seed=SEED) -> dict:
    groups = np.asarray(groups)
    uniq = sorted(set(groups))
    if len(uniq) < 3:
        return {"note": "fewer than 3 clusters"}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == uniq[i]) for i in pick])
        if len(np.unique(idx)) < 3:
            continue
        vals.append(rank_corr(np.asarray(x)[idx], np.asarray(y)[idx]))
    vals = [v for v in vals if np.isfinite(v)]
    if len(vals) < 20:
        return {"note": "too few replicates"}
    return {"lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            "n_boot": len(vals), "n_clusters": len(uniq),
            "note": "sequence-clustered; descriptive below ~10 clusters"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses-root", default="data/pathrec/poses")
    ap.add_argument("--uzh", default="data/uzh")
    ap.add_argument("--clips", default="data/clips_v2_w370ms")
    ap.add_argument("--window-rule", default="results/e_window_rule.json")
    ap.add_argument("--part1", default="results/f04_spline/summary.json")
    ap.add_argument("--out", default="results/f05_lean_error")
    args = ap.parse_args()

    window_rule = json.loads(Path(args.window_rule).read_text())
    uzh = Path(args.uzh)
    out = {"provenance": {"commit": _git_head(),
                          "prereg": "configs/f04_round2_prereg.yaml"},
           "paths": {}}
    seq_rows = []
    for path_name in PATHS:
        root = Path(args.poses_root) / path_name
        if not root.exists():
            out["paths"][path_name] = {"missing": str(root)}
            continue
        block = {}
        for seq in SEQUENCES:
            p = root / f"{seq}.npz"
            if not p.exists():
                continue
            poses = np.load(p)["c2w"]
            truth = load_truth(uzh, seq)
            f95 = float(window_rule["per_sequence"][seq]["f95_hz"])
            lean = path_leans(poses, truth, 0.5 / f95)
            bakeoff = bakeoff_direction(poses, truth)
            block[seq] = {**lean, **{f"bakeoff_{k}": v for k, v in bakeoff.items()},
                          "f95_hz": f95}
            seq_rows.append({"path": path_name, "seq": seq,
                             "lean_median_deg": lean["window_median_deg_median"],
                             "lean_p75_deg": lean["window_median_deg_p75"],
                             "n_windows": lean["n_windows"],
                             "bakeoff_dir_deg": bakeoff["accel_dir_err_deg_median"],
                             "bakeoff_ratio": bakeoff["accel_mag_ratio"],
                             "bakeoff_ate_sim3_m": bakeoff["ate_sim3_m"]})
        out["paths"][path_name] = block

    # cross-path comparison per sequence and the rank correlation of the two metrics
    corr = {}
    xs = [r["lean_median_deg"] for r in seq_rows]
    ys = [r["bakeoff_dir_deg"] for r in seq_rows]
    gs = [r["seq"] for r in seq_rows]
    corr["all_sequences"] = {"spearman": rank_corr(xs, ys),
                             "n": int(sum(np.isfinite(xs) & np.isfinite(ys))),
                             "clustered": clustered_corr(xs, ys, gs)}
    for path_name in PATHS:
        sub = [r for r in seq_rows if r["path"] == path_name]
        corr[path_name] = {"spearman": rank_corr([r["lean_median_deg"] for r in sub],
                                                 [r["bakeoff_dir_deg"] for r in sub]),
                           "n": len(sub)}
    out["correlations"] = corr
    out["sequence_rows"] = seq_rows

    # --- does either path metric predict per-clip verdict quality? (needs part 1)
    part1 = Path(args.part1)
    rows_csv = part1.parent / "rows.csv"
    if rows_csv.exists():
        import csv as _csv
        with open(rows_csv, newline="") as fh:
            all_rows = list(_csv.DictReader(fh))
        per_clip = {}
        for path_name in sorted({r["path"] for r in all_rows}):
            real = {}
            for r in all_rows:
                if (r["path"] == path_name and r["cond"] == "real"
                        and r["arm"] == "estimated_attitude" and r["testable"] == "True"):
                    try:
                        r["rho"] = float(r["rho"])
                    except (TypeError, ValueError):
                        r["rho"] = float("nan")
                    real[(r["seq"], r["idx"])] = r
            per_clip[path_name] = real
        link = {}
        for path_name, reals in per_clip.items():
            x, y, g = [], [], []
            for (seq, idx), r in reals.items():
                if not np.isfinite(r.get("rho", np.nan)):
                    continue
                lean = out["paths"].get(path_name, {}).get(seq, {})
                if not lean:
                    continue
                x.append(lean["window_median_deg_median"])
                y.append(r["rho"])
                g.append(seq)
            link[path_name] = {"spearman_lean_vs_rho": rank_corr(x, y), "n": len(x),
                               "clustered": clustered_corr(x, y, g) if len(x) >= 3 else None}
        out["prediction"] = link
    else:
        out["prediction"] = {"note": f"{rows_csv} absent; per-clip link not computed"}

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "lean_error.json").write_text(json.dumps(_py(out), indent=1))
    import csv
    with open(Path(args.out) / "sequences.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(seq_rows[0].keys()))
        w.writeheader()
        w.writerows(seq_rows)
    for path_name in PATHS:
        b = out["paths"].get(path_name, {})
        if "missing" in b:
            print(f"{path_name}: missing")
            continue
        vals = [v["window_median_deg_median"] for v in b.values()]
        dirs = [v["bakeoff_accel_dir_err_deg_median"] for v in b.values()]
        print(f"{path_name:7s} lean median {np.median(vals):6.2f} deg "
              f"(range {min(vals):5.2f}-{max(vals):6.2f})  bakeoff dir median "
              f"{np.median(dirs):5.2f} deg")
    print("correlations:", json.dumps(_py(corr), indent=1)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
