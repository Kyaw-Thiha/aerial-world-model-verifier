"""Track E Task 5: the frozen-threshold verdict test on UZH-FPV (commanded-path mode).

Reads the exported clips (shared schema, section 6 of the handoff), rebuilds the path side from
the Leica truth, scores every clip and condition with Track A2's statistic (mount fixed at the
published calibration, drag-free path side, lag nuisance only), and reports:

  * false reject on real clips, detection per edit family, both with Wilson intervals,
  * abstention and its causes, per rig and environment,
  * paired (each edit against its own real clip) and unmatched populations,
  * a leave-one-sequence-out threshold sensitivity (labelled: never the headline).

T2 (configs/e_acceptance_ref.yaml): false reject <= 0.15, detection of tilt removal and
mirrored roll >= 0.80 at thresholds frozen on TII.

    pixi run python experiments/e_verdict.py --clips results/e_clips --root data/uzh
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.depicted.clips import (path_side_from_pos, path_side_with_gt,
                                up_body_from_lean)
from src.depicted.metrics import wilson
from src.depicted.stat_a2 import FROZEN_THRESHOLD_DEG, score_clip_upb
from src.depicted.uzh import all_sequences, load_truth
from src.score.verdict import conformal_quantile

FAMILIES = ("tilt_removal", "scale_0.5", "scale_1.5", "mirror_roll", "yaw_control",
            "null_edit")


def load_clip(npz_path: Path, meta_path: Path) -> dict:
    d = np.load(npz_path)
    meta = json.loads(meta_path.read_text())
    return {"name": meta["clip_id"], "seq": meta["sequence"], "cond": meta["condition"],
            "index": meta["index"], "rig": meta.get("rig", ""), "meta": meta,
            "conf_median": float(np.nanmedian(d["depicted_conf"]))
            if np.isfinite(d["depicted_conf"]).any() else float("nan"),
            "t": d["t"], "pos": d["pos"], "yaw": d["yaw"],
            "lean": d["depicted_lean"], "conf": d["depicted_conf"],
            "valid_frac": d["valid_frac"] if "valid_frac" in d.files else np.ones(len(d["t"]))}


def score_all(clips_dir: Path, root: Path, uptilt_offsets=(0.0,),
              lag_grid_ms: np.ndarray | None = None,
              path_source: str = "dense") -> list[dict]:
    path_cache: dict[str, dict] = {}
    rows = []
    for seq in all_sequences():
        d = clips_dir / seq
        if not d.exists():
            continue
        truth = load_truth(root, seq)
        gt = np.load(root / seq / "truth.npz")["gt"]
        R_cb = truth.R_cb()
        for npz in sorted(d.glob("*.npz")):
            meta_path = npz.with_suffix(".meta.json")
            if not meta_path.exists():
                continue
            clip = load_clip(npz, meta_path)
            lo, hi = clip["index"]
            if path_source == "schema":
                # T6: rebuild the path side from the clip's own pos/t only
                path = path_side_from_pos(clip["pos"], clip["t"], k_d=0.0)
                lo, hi = 0, len(clip["t"])
            else:
                if seq not in path_cache:
                    path_cache[seq] = path_side_with_gt(truth, gt, k_d=0.0)
                path = path_cache[seq]
            up_b = up_body_from_lean(clip["lean"])
            valid = np.asarray(clip["valid_frac"]) >= 0.5
            res = score_clip_upb(
                up_body=up_b, imp_axis=path["required_axis"][lo:hi], yaw=clip["yaw"],
                t=clip["t"], R_cb=R_cb,
                implied_tilt_deg=path["required_tilt_deg"][lo:hi], valid=valid,
                uptilt_offsets_deg=uptilt_offsets, lag_grid_ms=lag_grid_ms)
            rows.append({"seq": seq, "rig": truth.rig, "env": truth.env, "clip": clip["name"],
                         "cond": clip["cond"], "index": [lo, hi],
                         "conf_median": clip["conf_median"], **res})
    return rows


def _distilled(rows: list[dict], thr: float) -> dict:
    """Rates over testable clips at a given threshold."""
    out = {"threshold_deg": thr, "n_clips": len(rows),
           "abstained": sum(1 for r in rows if not r["testable"]),
           "testable": sum(1 for r in rows if r["testable"])}
    out["abstention_rate"] = out["abstained"] / max(out["n_clips"], 1)
    return out


def rejection(rows: list[dict], cond: str, thr: float) -> dict:
    sel = [r for r in rows if r["cond"] == cond and r["testable"] and r["score"] is not None]
    n = len(sel)
    k = sum(1 for r in sel if r["score"] > thr)
    lo, hi = wilson(k, n)
    return {"n": n, "rejected": k, "rate": (k / n) if n else None,
            "wilson_lo": lo, "wilson_hi": hi,
            "median_score_deg": float(np.median([r["score"] for r in sel])) if n else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="results/e_clips")
    ap.add_argument("--root", default="data/uzh")
    ap.add_argument("--out", default="results/e_verdict")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--uptilt-offsets", default="0",
                    help="comma list, degrees; '0' is the pre-registered primary (published mount)")
    ap.add_argument("--gate", action="store_true", help="post-hoc: gate by depicted_conf")
    ap.add_argument("--path-source", default="dense", choices=("dense", "schema"),
                    help="'dense' uses the 500 Hz truth table; 'schema' uses the clip's own pos")
    args = ap.parse_args()
    clips_dir = Path(args.clips)
    root = Path(args.root)
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    offsets = tuple(float(x) for x in args.uptilt_offsets.split(",") if x)

    rows = score_all(clips_dir, root, uptilt_offsets=offsets,
                     path_source=args.path_source)
    if not rows:
        print("no clips found")
        return 1

    thr = FROZEN_THRESHOLD_DEG
    summary: dict = {
        "tag": args.tag, "threshold_deg": thr, "threshold_source": "A2 pooled drag-free lag-only",
        "uptilt_offsets_deg": offsets, "n_rows": len(rows), "path_source": args.path_source,
        "by_condition": {}, "per_rig": {}, "per_env": {}, "abstention": {},
    }
    for cond in ("real",) + FAMILIES:
        summary["by_condition"][cond] = rejection(rows, cond, thr)

    # per rig and environment, real vs the two headline families
    for key, field in (("per_rig", "rig"), ("per_env", "env")):
        groups = sorted({r[field] for r in rows})
        for g in groups:
            sub = [r for r in rows if r[field] == g]
            summary[key][g] = {c: rejection(sub, c, thr)
                               for c in ("real", "tilt_removal", "mirror_roll")}
            summary[key][g]["abstention_rate"] = (
                sum(1 for r in sub if not r["testable"]) / max(len(sub), 1))

    # abstention causes
    causes: dict[str, int] = {}
    for r in rows:
        for reason in r.get("abstain_reasons") or []:
            key = reason.split()[0]
            causes[key] = causes.get(key, 0) + 1
    summary["abstention"] = {"causes": causes,
                             "overall": sum(1 for r in rows if not r["testable"]) / len(rows),
                             "per_condition": {c: (sum(1 for r in rows if r["cond"] == c
                                                      and not r["testable"])
                                                   / max(sum(1 for r in rows if r["cond"] == c), 1))
                                               for c in ("real",) + FAMILIES}}

    # Paired detection: an edit is "detected" when it is rejected while its own real clip was
    # accepted. Clip ids carry the condition as a suffix, so pair on the stem, not the id.
    def pair_key(row: dict) -> str:
        """``<seq>_<clipindex>``: the clip id without its condition suffix.

        Condition names contain underscores (``tilt_removal``), so the suffix has to be
        stripped by its own length rather than by splitting on the last underscore.
        """
        return row["clip"][: -(len(row["cond"]) + 1)]

    real_by_clip = {pair_key(r): r for r in rows if r["cond"] == "real"}
    paired = {}
    for cond in FAMILIES:
        n = k = 0
        for r in rows:
            if r["cond"] != cond or not r["testable"] or r["score"] is None:
                continue
            base = real_by_clip.get(pair_key(r))
            if base is None or not base["testable"] or base["score"] is None:
                continue
            # the population where a detection is meaningful: the real clip was accepted
            if base["score"] > thr:
                continue
            n += 1
            if r["score"] > thr:
                k += 1
        lo, hi = wilson(k, n)
        paired[cond] = {"n": n, "detected": k, "rate": (k / n) if n else None,
                        "wilson_lo": lo, "wilson_hi": hi,
                        "population": "clips whose real condition was accepted"}
    summary["paired"] = paired

    # frozen S3 gate, applied two ways (A2's pre-registered per-condition rule and the
    # pair-level variant it reports as analysis)
    tau = -0.0016656352350220922
    gated = {}
    for cond in ("real",) + FAMILIES:
        sel = [r for r in rows if r["cond"] == cond]
        drop = [r for r in sel if np.isfinite(r.get("conf_median", np.nan))
                and -r["conf_median"] > tau]
        drop_ids = {id(r) for r in drop}
        kept = [r for r in sel if id(r) not in drop_ids]
        gated[cond] = {
            "n": len(sel), "n_dropped": len(drop),
            "drop_rate": len(drop) / max(len(sel), 1),
            "rejection_after_gate": rejection(kept, cond, thr),
        }
    real_gate = {pair_key(r): (np.isfinite(r.get("conf_median", np.nan))
                               and -r["conf_median"] > tau)
                 for r in rows if r["cond"] == "real"}
    pair_gate = {}
    for cond in FAMILIES:
        sel = [r for r in rows if r["cond"] == cond and r["testable"] and r["score"] is not None
               and not real_gate.get(pair_key(r), True)
               and pair_key(r) in real_gate]
        n = len(sel)
        k = sum(1 for r in sel if r["score"] > thr)
        lo, hi = wilson(k, n)
        pair_gate[cond] = {"n": n, "detected": k, "rate": (k / n) if n else None,
                           "wilson_lo": lo, "wilson_hi": hi,
                           "note": "real clip passes the frozen gate; edits inherit the decision"}
    summary["gate_s3"] = {"tau": tau, "per_condition": gated, "pair_level": pair_gate}
    summary["gate_s3"]["real_clips_dropped"] = sum(1 for v in real_gate.values() if v)

    # leave-one-sequence-out threshold sensitivity
    real_scores = {}
    for r in rows:
        if r["cond"] == "real" and r["testable"] and r["score"] is not None:
            real_scores.setdefault(r["seq"], []).append(r["score"])
    loso = {}
    if len(real_scores) >= 3:
        for cond in ("tilt_removal", "mirror_roll", "scale_0.5", "scale_1.5"):
            n = k = 0
            for r in rows:
                if r["cond"] != cond or not r["testable"] or r["score"] is None:
                    continue
                fit = [s for q, v in real_scores.items() if q != r["seq"] for s in v]
                if len(fit) < 5:
                    continue
                t = conformal_quantile(np.asarray(fit), 0.05)
                n += 1
                k += int(r["score"] > t)
            loso[cond] = {"n": n, "detected": k, "rate": (k / n) if n else None,
                          "note": "sensitivity only: threshold refit without the clip's sequence"}
    summary["loso_sensitivity"] = loso

    # T2 statement
    fre = summary["by_condition"]["real"]["rate"]
    det_rm = summary["by_condition"]["tilt_removal"]["rate"]
    det_mr = summary["by_condition"]["mirror_roll"]["rate"]
    summary["t2"] = {
        "false_reject": fre, "false_reject_max": 0.15,
        "detection_tilt_removal": det_rm, "detection_mirror_roll": det_mr, "detection_min": 0.80,
        "passes": bool(fre is not None and det_rm is not None and det_mr is not None
                       and fre <= 0.15 and det_rm >= 0.80 and det_mr >= 0.80),
    }
    summary["rows"] = rows

    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("by_condition", "paired", "abstention", "t2")},
                     indent=1)[:4000])
    print(f"\nwrote {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
