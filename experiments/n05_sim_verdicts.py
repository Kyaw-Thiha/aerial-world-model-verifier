"""Track N, N05: C2 and coherence on the propagated lean over the 200 simulated dev clips.

Pre-registration ``configs/n05_sim_verdict_prereg.yaml``. The propagated arm uses a LEVEL-BODY
start (u_cam(0) = R_cb^T e3): no oracle attitude, no reader reading anywhere on that arm. The
reader arm is the frozen GeoCalib-224 reader (L01 settings, stride 2).

    PYTHONPATH=. python experiments/n05_sim_verdicts.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.verdict.lean import depicted_lean_from_gravity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/n05"
POSES = ROOT / "data/n05/poses/sim200/scale1"
SCORES = ROOT / "data/n05/scores"
GT = ROOT / "data/n05/gt"
G0 = 9.80665
E3 = np.array([0.0, 0.0, 1.0])
KD_GRID = np.round(np.arange(0.0, 0.401, 0.05), 3)


def ang_lean(a, b):
    def ax(l):
        z = np.clip(1.0 - np.sum(np.asarray(l) ** 2, axis=-1), 0.0, 1.0)
        v = np.concatenate([np.asarray(l), np.sqrt(z)[..., None]], axis=-1)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)
    a = np.atleast_2d(a); b = np.atleast_2d(b)
    return np.degrees(np.arccos(np.clip(np.sum(ax(a) * ax(b), axis=-1), -1.0, 1.0)))


def tier_scores(l_dep, l_req, vreg, mask):
    ok = mask & np.isfinite(l_dep).all(1) & np.isfinite(l_req).all(1) & np.isfinite(vreg).all(1)
    if ok.sum() < 8:
        return float("nan"), float("nan"), int(ok.sum())
    ld, lr, vr = l_dep[ok], l_req[ok], vreg[ok]
    best = float("inf")
    for kd in KD_GRID:
        r = ld - (lr + kd * vr)
        off = np.median(r, axis=0)
        s = float(np.median(ang_lean(ld, lr + kd * vr + off)))
        best = min(best, s)
    rs = []
    for ax in range(2):
        a, b = ld[:, ax], lr[:, ax]
        rs.append(0.0 if (np.std(a) < 1e-12 or np.std(b) < 1e-12)
                  else float(np.corrcoef(a, b)[0, 1]))
    return best, 1.0 - min(rs), int(ok.sum())


def build_clip(row, poses_dir=POSES):
    cid = row["clip_id"]
    gt = np.load(GT / f"{cid}.npz")
    tgt = np.asarray(gt["t"], float)
    lean_req = np.asarray(gt["lean_req"], float)
    vel = np.asarray(gt["vel"], float)
    yaw = np.asarray(gt["yaw"], float)
    # the mount is a mechanical constant from the frozen split (not clip truth); the analytic
    # construction matches the gt mount exactly on every dev clip (checked once, max |diff| = 0)
    from src.perception.frames import camera_to_body_pitched
    R_cb = camera_to_body_pitched(np.deg2rad(float(row["mount"])))
    assert np.max(np.abs(R_cb - np.asarray(gt["R_cb"], float))) < 1e-9
    sc = np.load(SCORES / f"{cid}.npz")
    fidx = np.asarray(sc["frame"], int)
    t = fidx / 30.0
    lean_reader = np.asarray(sc["lean_est"], float)
    # required + velocity regressor at the scored frames
    lr = np.column_stack([np.interp(t, tgt, lean_req[:, k]) for k in range(2)])
    vy = np.column_stack([np.interp(t, tgt, vel[:, 0]), np.interp(t, tgt, vel[:, 1]),
                          np.interp(t, tgt, yaw)])
    c, s = np.cos(-vy[:, 2]), np.sin(-vy[:, 2])
    vreg = np.column_stack([c * vy[:, 0] / G0 + s * vy[:, 1] / G0,
                            -s * vy[:, 0] / G0 + c * vy[:, 1] / G0])
    # propagated: level-body start, DPVO relative rotations
    d = np.load(poses_dir / f"{cid}.npz")
    R = np.asarray(d["c2w"], float)[:, :3, :3]
    ok_idx = fidx[fidx < len(R)]
    l_prop = np.full((len(fidx), 2), np.nan)
    if len(ok_idx):
        u0 = R_cb.T @ E3  # body level at the first scored frame: no oracle, no reader
        ups = np.array([(R[k].T @ R[fidx[0]]) @ u0 for k in ok_idx])
        l_prop[:len(ok_idx)] = depicted_lean_from_gravity(ups, R_cb)
    # mask: finite on both sides and inside the DPVO/reader overlap
    mask = np.zeros(len(fidx), bool)
    mask[:len(ok_idx)] = True
    mask &= np.isfinite(lean_reader).all(1) & np.isfinite(l_prop).all(1)
    c2_r, r_r, n1 = tier_scores(lean_reader, lr, vreg, mask)
    c2_p, r_p, n2 = tier_scores(l_prop, lr, vreg, mask)
    return {"clip_id": cid, "family": row["family"], "scene_seed": row["scene_seed"],
            "mount": row["mount"], "condition": row["condition"], "factor": row["factor"],
            "interval_fraction": row["interval_fraction"], "n": int(mask.sum()),
            "C2_reader": c2_r, "R_reader": r_r, "C2_propagated": c2_p, "R_propagated": r_p}


def main(poses_dir=POSES, tag="") -> int:
    rows = [r for r in csv.DictReader(open(OUT / "clips.csv"))]
    assert all("i02_dev" in r["clip_id"] for r in rows) and len(rows) == 200
    out = []
    for i, r in enumerate(rows):
        try:
            out.append(build_clip(r, poses_dir))
        except Exception as exc:  # noqa: BLE001
            out.append({"clip_id": r["clip_id"], "family": r["family"],
                        "scene_seed": r["scene_seed"], "mount": r["mount"],
                        "condition": r["condition"], "error": str(exc)})
        if i % 20 == 0:
            print(i, flush=True)
    with open(OUT / f"per_clip{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()), extrasaction="ignore")
        w.writeheader(); w.writerows(out)

    d = [r for r in out if "error" not in r]
    seeds = sorted({r["scene_seed"] for r in d})
    feas = [r for r in d if r["condition"] == "feasible"]
    summary = {"n": len(d), "seeds": seeds, "tiers": {}}
    for key in ("C2_reader", "C2_propagated", "R_reader", "R_propagated"):
        # natural leave-one-scene-seed-out threshold on feasible clips
        flags = {}
        for r in d:
            other = [f[key] for f in feas if f["scene_seed"] != r["scene_seed"]]
            other = [v for v in other if np.isfinite(v)]
            thr = float(np.quantile(other, 0.95)) if other else np.nan
            flags[r["clip_id"]] = bool(np.isfinite(r[key]) and np.isfinite(thr) and r[key] > thr)
        nat_feas = sum(flags[r["clip_id"]] for r in feas)
        summary["tiers"][key] = {"natural_feasible_flags": f"{nat_feas}/{len(feas)}"}
        # matched threshold: (k+1)-th largest feasible score, k = reader C2 natural count
        if key == "C2_reader":
            summary["k_match"] = nat_feas
        summary["tiers"][key]["flags_natural"] = flags
    k = summary.get("k_match", 0)
    for key in ("C2_reader", "C2_propagated", "R_reader", "R_propagated"):
        vals = sorted([f[key] for f in feas if np.isfinite(f[key])], reverse=True)
        thr = (vals[k] if 0 <= k < len(vals) else (vals[-1] if vals else np.nan))
        flags = {r["clip_id"]: bool(np.isfinite(r[key]) and r[key] > thr) for r in d}
        summary["tiers"][key]["threshold_matched"] = float(thr)
        summary["tiers"][key]["flags_matched"] = flags

    def cell(sub, key, flags):
        vals = [flags[r["clip_id"]] for r in sub]
        return f"{int(sum(vals))}/{len(vals)}"

    conditions = ["feasible", "tilt_removed", "mirrored_roll", "tilt_scaled", "partial",
                  "thrust_limit"]
    table = {}
    for mode in ("natural", "matched"):
        table[mode] = {}
        for cond in conditions:
            sub = [r for r in d if r["condition"] == cond]
            table[mode][cond] = {key: cell(sub, key, summary["tiers"][key][f"flags_{mode}"])
                                 for key in ("C2_reader", "C2_propagated",
                                             "R_reader", "R_propagated")}
    # by family and mount, natural
    by_cell = {}
    for cond in ("feasible", "tilt_removed", "mirrored_roll"):
        for fam in sorted({r["family"] for r in d}):
            for mount in ("0.0", "15.0", "30.0", "45.0"):
                sub = [r for r in d if r["condition"] == cond and r["family"] == fam
                       and r["mount"] == mount]
                if not sub:
                    continue
                by_cell[f"{cond}|{fam}|{mount}"] = {
                    key: cell(sub, key, summary["tiers"][key]["flags_natural"])
                    for key in ("C2_reader", "C2_propagated", "R_reader", "R_propagated")}
    # scale factors and partial fractions
    split = {}
    for factor in sorted({r["factor"] for r in d if r["condition"] == "tilt_scaled"}):
        sub = [r for r in d if r["condition"] == "tilt_scaled" and r["factor"] == factor]
        split[f"scale_{factor}"] = {key: cell(sub, key, summary["tiers"][key]["flags_natural"])
                                    for key in ("C2_reader", "C2_propagated", "R_reader",
                                                "R_propagated")}
    for frac in sorted({r["interval_fraction"] for r in d if r["condition"] == "partial"}):
        sub = [r for r in d if r["condition"] == "partial" and r["interval_fraction"] == frac]
        split[f"partial_{frac}"] = {key: cell(sub, key, summary["tiers"][key]["flags_natural"])
                                    for key in ("C2_reader", "C2_propagated", "R_reader",
                                                "R_propagated")}
    summary.update({"table": table, "by_cell": by_cell, "split": split,
                    "medians": {cond: {key: float(np.nanmedian([r[key] for r in d
                                                                if r["condition"] == cond]))
                                       for key in ("C2_reader", "C2_propagated",
                                                   "R_reader", "R_propagated")}
                                for cond in conditions},
                    "start": "level body, u_cam(0)=R_cb^T e3; no oracle, no reader",
                    "note": ("natural = leave-one-scene-seed-out 95th percentile of feasible; "
                             "matched = common feasible false-reject count k = reader C2 natural"),
                    "k_match": k})
    (OUT / f"summary{tag}.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(table["natural"], indent=1))
    print("k_match", k)
    print(json.dumps(table["matched"], indent=1))
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", default=str(POSES))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    raise SystemExit(main(Path(a.poses), a.tag))
