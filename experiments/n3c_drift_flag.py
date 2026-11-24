"""Track N3, Part C: a label-free reliability flag for the high-mount drift tail.

Pre-registration `configs/n3c_drift_flag_prereg.yaml`, committed before any flag statistic existed
and before the anchored F0 scores of Part B. Definitions, the 10 % label-free budget and the
success criteria are fixed there.

Reads: the two stored DPVO draws (rotation sequences only), and Part B's per-clip tier scores.
Writes: results/n3/c/flags.csv, summary.json, sensitivity.csv.

Run:  PYTHONPATH=. pixi run python experiments/n3c_drift_flag.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.verdict.anchor import angle_between_deg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/n3/n3c"
N3B = ROOT / "results/n3/n3b"
DRAW1 = ROOT / "data/n05/poses/sim200/scale1"
DRAW2 = ROOT / "data/n05/poses/sim200/scale1_rep2"
MANIFEST = ROOT / "results/n2/inputs/manifest.csv"

BUDGETS = (0.05, 0.10, 0.20)
PRIMARY_BUDGET = 0.10
HIGHER_IS_WORSE = {"jit": True, "spike": True, "rough": True, "cover": True,
                   "draw_dis": True, "draw_dis_oracle": True}
TAIL_DEG = 20.0
TAIL_DEG_2 = 15.0


# ------------------------------------------------------------------ flag statistics
def increments(R: np.ndarray) -> np.ndarray:
    """Per-frame rotation increment |angle(R_k^T R_{k+1})| in degrees."""
    if len(R) < 2:
        return np.array([])
    rel = np.einsum("nij->nji", R[:-1]) @ R[1:]
    # trace-based angle is accurate enough here (increments are small, not near 0 or 180)
    tr = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def rotation_vector_path(R: np.ndarray) -> np.ndarray:
    """log(R_0^T R_k) as a 3-vector per frame (degrees)."""
    rel = np.einsum("nij->nji", R) @ R[0]
    return np.degrees(Rotation.from_matrix(rel).as_rotvec())


def rough_stat(R: np.ndarray, half: int = 4, deg: int = 3) -> float:
    """RMS of the residual of a local degree-``deg`` polynomial fit to the rotation vector."""
    rv = rotation_vector_path(R)
    n = len(rv)
    if n < 2 * half + deg + 1:
        return float("nan")
    resid = np.zeros_like(rv)
    idx = np.arange(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        if hi - lo < deg + 1:
            resid[i] = 0.0
            continue
        x = idx[lo:hi] - i
        for k in range(3):
            coef = np.polyfit(x, rv[lo:hi, k], deg)
            resid[i, k] = rv[i, k] - np.polyval(coef, 0.0)
    return float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))


def flag_stats(clip: str) -> dict:
    d = np.load(DRAW1 / f"{clip}.npz")
    R = np.asarray(d["c2w"], float)[:, :3, :3]
    w = increments(R)
    out = {
        "n_frames": int(len(R)),
        "cover": float(1.0 - float(d["n_pose"]) / max(float(d["n_input"]), 1.0)),
        "rot_total_deg": float(np.sum(w)) if len(w) else float("nan"),
    }
    if len(w) >= 4:
        out["jit"] = float(np.percentile(np.abs(np.diff(w)), 90))
        out["spike"] = float(np.max(np.abs(w - np.median(w))))
    else:
        out["jit"] = float("nan")
        out["spike"] = float("nan")
    out["rough"] = rough_stat(R)
    return out


# ------------------------------------------------------------------ evaluation helpers
def flag_set(rows: list[dict], key: str, budget: float) -> set[str]:
    """Flag the ``budget`` fraction of clips with the WORST statistic (label-free cut)."""
    vals = sorted([(r[key], r["clip_id"]) for r in rows if np.isfinite(r[key])], reverse=True)
    n_take = int(round(budget * len(rows)))
    return {cid for _, cid in vals[:n_take]}


def cut_at(rows: list[dict], key: str) -> float:
    vals = sorted([r[key] for r in rows if np.isfinite(r[key])], reverse=True)
    n_take = int(round(PRIMARY_BUDGET * len(rows)))
    return vals[n_take] if n_take < len(vals) else (vals[-1] if vals else float("nan"))


def matched_cut(scores: dict, feasible: list[str], k: int = 8) -> float | None:
    vals = sorted([scores[c] for c in feasible if np.isfinite(scores.get(c, np.nan))],
                  reverse=True)
    if not vals:
        return None
    kk = min(k, len(vals))
    return vals[kk] if kk < len(vals) else vals[-1]


def rates(scores: dict, rows: list[dict], cut: float, stratum, cond: str) -> str:
    sub = [r for r in rows if stratum(r) and r["condition"] == cond]
    hit = sum(1 for r in sub if np.isfinite(scores.get(r["clip_id"], np.nan))
              and scores[r["clip_id"]] > cut)
    return f"{hit}/{len(sub)}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    b1 = list(csv.DictReader(open(N3B / "per_clip.csv")))
    b2 = {r["clip_id"]: r for r in csv.DictReader(open(N3B / "per_clip_draw2.csv"))}
    man = {r["clip_id"]: r for r in csv.DictReader(open(MANIFEST)) if r["split"] == "dev"}

    rows = []
    for r in b1:
        cid = r["clip_id"]
        rec = {"clip_id": cid, "family": r["family"], "mount": float(r["mount"]),
               "condition": r["condition"],
               "dynamic": int(float(man[cid]["peak_tilt_req_deg"]) >= 15.0),
               "f0_oracle": float(r["F0|oracle"]),
               "f0_reader_start": float(r["F0|reader_start"]),
               "f0_anchor_P3": float(r["F0|anchor1_P3"]),
               "has_anchor_P3": int(np.isfinite(float(r["F0|anchor1_P3"])))}
        rec.update(flag_stats(cid))
        rec["draw_dis"] = abs(float(r["F0|anchor1_P3"]) - float(b2[cid]["F0|anchor1_P3"])) \
            if np.isfinite(float(r["F0|anchor1_P3"])) and \
            np.isfinite(float(b2[cid]["F0|anchor1_P3"])) else float("nan")
        # labelled amendment: the same disagreement statistic on an arm that scores every clip
        rec["draw_dis_oracle"] = abs(float(r["F0|oracle"]) - float(b2[cid]["F0|oracle"]))
        rows.append(rec)
    assert len(rows) == 200 and all(r["clip_id"].startswith("i02_dev") for r in rows)

    fields = list(rows[0].keys())
    with open(OUT / "flags.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    feas = [r for r in rows if r["condition"] == "feasible"]
    # target set, decided by the pre-registered rule
    tail_oracle = [r for r in feas if r["f0_oracle"] > TAIL_DEG]
    tail_anchor = [r for r in feas if np.isfinite(r["f0_anchor_P3"])
                   and r["f0_anchor_P3"] > TAIL_DEG]
    target = tail_anchor if len(tail_anchor) >= 3 else tail_oracle
    target_name = "anchor_P3_tail" if len(tail_anchor) >= 3 else "oracle_tail_fallback"

    # per-flag evaluation at every budget
    sens = []
    for key in ("jit", "spike", "rough", "cover", "draw_dis", "draw_dis_oracle"):
        for budget in BUDGETS:
            flagged = flag_set(rows, key, budget)
            tgt = {r["clip_id"] for r in target}
            forest_feas = [r for r in feas if r["family"] == "forest_ridge"]
            flagged_feas = [r for r in flagged if any(x["clip_id"] == r for x in feas)]
            hits = len(flagged & tgt)
            sens.append({
                "flag": key, "budget": budget, "n_flagged": len(flagged),
                "recall": f"{hits}/{len(tgt)}",
                "hits": hits, "n_target": len(tgt),
                "precision": f"{hits}/{len(flagged)}",
                "flagged_feasible": len(flagged_feas),
                "forest_feasible_flagged": f"{sum(1 for r in forest_feas if r['clip_id'] in flagged)}"
                                            f"/{len(forest_feas)}",
                "flagged_dynamic": f"{sum(1 for r in flagged if next(x for x in rows if x['clip_id'] == r)['dynamic'])}/200",
            })
    with open(OUT / "sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sens[0].keys()))
        w.writeheader()
        w.writerows(sens)

    # after abstaining: the matched 8/80 operating point for the arms that have a tail
    after = {}
    for arm_key, label in (("f0_oracle", "oracle"), ("f0_reader_start", "reader_start"),
                           ("f0_anchor_P3", "anchor1_P3")):
        scores = {r["clip_id"]: r[arm_key] for r in rows}
        for key in ("jit", "spike", "rough", "draw_dis", "draw_dis_oracle"):
            flagged = flag_set(rows, key, PRIMARY_BUDGET)
            if key == "draw_dis":
                flagged = {c for c in flagged if np.isfinite(scores.get(c, np.nan))
                           or True}
            cut_before = matched_cut(scores, [r["clip_id"] for r in feas], 8)
            feas_keep = [r["clip_id"] for r in feas if r["clip_id"] not in flagged]
            cut_after = matched_cut(scores, feas_keep, 8)
            strat = {  # name -> (stratum predicate, condition)
                "feasible": (lambda r: True, "feasible"),
                "tilt_removed": (lambda r: True, "tilt_removed"),
                "mirrored_roll": (lambda r: True, "mirrored_roll"),
                "forest_removed": (lambda r: r["family"] == "forest_ridge", "tilt_removed"),
                "forest_feasible": (lambda r: r["family"] == "forest_ridge", "feasible"),
                "high_mount_feasible": (lambda r: r["mount"] in (30.0, 45.0), "feasible")}
            before = {k: rates(scores, rows, cut_before, fn, cond)
                      for k, (fn, cond) in strat.items()}
            after_ = {k: rates(scores, rows, cut_after, fn, cond)
                      for k, (fn, cond) in strat.items()}
            after[f"{label}|{key}"] = {
                "cut_before_deg": cut_before, "cut_after_deg": cut_after,
                "n_feasible_before": int(sum(1 for c in feas if np.isfinite(scores.get(c["clip_id"], np.nan)))),
                "n_feasible_after": len(feas_keep),
                "flagged": f"{len(flagged)}/200",
                "before": before, "after": after_}
    summary = {
        "config": {"primary_budget": PRIMARY_BUDGET, "budgets": BUDGETS,
                   "tail_deg": TAIL_DEG, "tail_deg_alt": TAIL_DEG_2,
                   "primary_flag": "jit"},
        "tail": {"oracle": {"n": len(tail_oracle),
                            "clips": [r["clip_id"] for r in tail_oracle],
                            "mounts": sorted({r["mount"] for r in tail_oracle})},
                 "anchor_P3": {"n": len(tail_anchor),
                               "clips": [r["clip_id"] for r in tail_anchor],
                               "note": "0 or 1 because the anchored arm abstains on 171/200 "
                                       "clips; that is abstention, not a cured tail"},
                 "reader_start": {"n": int(sum(1 for r in feas if r["f0_reader_start"] > TAIL_DEG))},
                 "fifteen_deg": {
                     "oracle": int(sum(1 for r in feas if r["f0_oracle"] > TAIL_DEG_2)),
                     "reader_start": int(sum(1 for r in feas
                                             if r["f0_reader_start"] > TAIL_DEG_2)),
                     "anchor_P3": int(sum(1 for r in feas if np.isfinite(r["f0_anchor_P3"])
                                          and r["f0_anchor_P3"] > TAIL_DEG_2))}},
        "target": {"name": target_name, "n": len(target),
                   "clips": [r["clip_id"] for r in target]},
        "after_abstaining": after,
        "flag_distribution": {k: {f"p{p}": float(np.nanpercentile([r[k] for r in rows], p))
                                  for p in (50, 75, 90, 95, 99)}
                              for k in ("jit", "spike", "rough", "cover", "draw_dis",
                                        "draw_dis_oracle")},
    }
    # criteria, scored mechanically
    primary = [s for s in sens if s["flag"] == "jit" and s["budget"] == PRIMARY_BUDGET][0]
    c1 = primary["hits"] >= int(np.ceil(2 / 3 * len(target)))
    a = after[f"oracle|jit"]
    c2 = bool(a["cut_after_deg"] is not None and a["cut_before_deg"] is not None
              and a["cut_after_deg"] <= 12.0
              and int(a["after"]["tilt_removed"].split("/")[0])
              >= int(a["before"]["tilt_removed"].split("/")[0]) - 2)
    forest_flagged = int(primary["forest_feasible_flagged"].split("/")[0])
    c3 = forest_flagged <= 8
    summary["criteria"] = {
        "C1_recall": {"criterion": "jit at the 10 % budget flags >= 2/3 of the target set",
                      "value": f"{primary['hits']}/{len(target)}", "pass": bool(c1)},
        "C2_operating_point": {
            "criterion": "after abstaining, oracle matched cut <= 12 deg and removals fall by <= 2",
            "cut_before": a["cut_before_deg"], "cut_after": a["cut_after_deg"],
            "removals_before": a["before"]["tilt_removed"],
            "removals_after": a["after"]["tilt_removed"], "pass": c2},
        "C3_forest_cost": {"criterion": "<= 8 of 26 forest feasible clips abstained",
                           "value": primary["forest_feasible_flagged"], "pass": bool(c3)},
        "all_flags": {f"{s['flag']}|{s['budget']}": {"recall": s["recall"],
                                                    "precision": s["precision"]}
                      for s in sens},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({"target": summary["target"], "tail": summary["tail"],
                      "criteria": summary["criteria"]}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
