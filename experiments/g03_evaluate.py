"""G03 evaluation: per-frame error and the downstream verdict, frozen reader vs fine-tuned.

Consumes the fold files the Modal app wrote (``results/g03_tilt_reader/folds/*.json``, pulled
from the ``awmv-g-data`` volume). CPU only.

Downstream, only the *depicted* side changes: each Track D schema-v2 clip's ``depicted_lean`` is
rebuilt from the fine-tuned reader's gravity on that clip's own frames (the clip frame names come
from the p3a spec, the same table the clips were built from), and the frozen path side
(``lean_req``, ``vel_req``, ``tilt_req_deg``) is reused unchanged. The statistics are the frozen
clip-level ones: median residual and the coherence tier, plus the frozen abstention rules.

    PYTHONPATH=. python experiments/g03_evaluate.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.perception.frames import camera_to_body_pitched
from src.verdict.abstain import gate_kept_from_conf
from src.verdict.coherence import coherence
from src.verdict.fit import FitOptions, fit_gain
from src.verdict.lean import depicted_lean_from_gravity
from src.verdict.schema import load_clip
from src.verdict.segments import frame_residual_deg

CLIP_ROOT = Path("data/processed/track_d/clips")
GATE = __import__("yaml").safe_load(Path("configs/gate_frozen.yaml").read_text())
SPECS = Path("data/processed/p3a/p3a_v1")
DEV = ["flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse",
       "flight-07p-lemniscate", "flight-13a-trackRATM", "flight-16p-trackRATM"]
HELD = ["flight-13p-trackRATM", "flight-07a-lemniscate"]


def ang(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1),
                                           1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def err_stats(e):
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return {"n": 0}
    s = np.sort(e)
    k = max(1, int(round(0.9 * len(s))))
    return {"n": int(len(e)), "median": float(np.median(e)),
            "trimmed_rmse": float(np.sqrt(np.mean(s[:k] ** 2))),
            "frac_gt10": float(np.mean(e > 10)), "frac_gt20": float(np.mean(e > 20)),
            "p90": float(np.percentile(e, 90))}


def fold_error(rows, key):
    up = np.array([r[key] if r[key] is not None else [np.nan] * 3 for r in rows], float)
    lab = np.array([r["up_cam"] for r in rows], float)
    ok = np.isfinite(up).all(axis=1) & np.isfinite(lab).all(axis=1)
    return ang(lab[ok], up[ok])


def downstream(fold_rows: dict[str, dict], flights: list[str], tag: str) -> dict:
    """Clip statistics with the fine-tuned depicted lean; LOFO conformal threshold on dev."""
    clips = json.loads((SPECS / "clips.json").read_text())["clips"]
    out = {}
    for flight in flights:
        spec = np.load(SPECS / f"{flight}.npz", allow_pickle=False)
        names = [str(n) for n in spec["names"]]
        R_cb = camera_to_body_pitched(np.deg2rad(float(spec["uptilt_deg"])))
        pred = fold_rows.get(flight)
        if pred is None:
            continue
        up_map = {r["frame"]: r["up_cam_ft"] for r in pred}
        rows = []
        for clip in [c for c in clips if c["flight"] == flight]:
            cid = clip["clip_id"]
            d = CLIP_ROOT / flight / f"{cid}__real"
            if not d.exists():
                continue
            clip_obj = load_clip(d)
            lo, hi = clip["index"]
            frame_names = names[int(lo):int(hi)]
            up = np.array([up_map.get(n, [np.nan] * 3) for n in frame_names], float)
            lean_ft = depicted_lean_from_gravity(up, R_cb)
            resid_frozen = frame_residual_deg(clip_obj.depicted_lean, clip_obj.lean_req,
                                              clip_obj.yaw)
            resid_ft = frame_residual_deg(lean_ft, clip_obj.lean_req, clip_obj.yaw)
            resid_rm = frame_residual_deg(np.zeros_like(clip_obj.depicted_lean),
                                          clip_obj.lean_req, clip_obj.yaw)
            ft = type(clip_obj)(t=clip_obj.t, pos=clip_obj.pos, yaw=clip_obj.yaw,
                                depicted_lean=lean_ft, depicted_conf=clip_obj.depicted_conf,
                                meta=clip_obj.meta, lean_req=clip_obj.lean_req,
                                vel_req=clip_obj.vel_req, tilt_req_deg=clip_obj.tilt_req_deg)
            gain = fit_gain(ft, FitOptions(bootstrap=False))
            coh = coherence(ft, lag_ms=gain.lag_ms)
            rows.append({"clip": cid, "flight": flight,
                         "median_frozen": float(np.nanmedian(resid_frozen)),
                         "median_ft": float(np.nanmedian(resid_ft)),
                         "median_removal_frozen": float(np.nanmedian(resid_rm)),
                         "coherence_frozen": None,
                         "coherence_ft": coh.correlation_score,
                         "usable": float(np.mean(np.isfinite(lean_ft).all(axis=1))),
                         "frac_signal": float(np.mean(np.asarray(clip_obj.tilt_req_deg) >= 10)),
                         "n_signal": int(np.sum(np.asarray(clip_obj.tilt_req_deg) >= 10)),
                         "gate_kept": gate_kept_from_conf(clip_obj.depicted_conf, GATE)})
        out[flight] = rows
    return out


def abstain_rate(rows: list[dict]) -> dict:
    """The frozen abstention rules that do not need a bootstrap (CI term excluded, noted)."""
    n_abs = 0
    for r in rows:
        reasons = []
        if r["usable"] < 0.5:
            reasons.append("usable")
        if r["n_signal"] < 12 or r["frac_signal"] < 0.20:
            reasons.append("signal")
        r["abstain_reasons"] = reasons
        n_abs += bool(reasons)
    return {"k": int(n_abs), "n": len(rows), "rate": float(n_abs / len(rows)) if rows else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="results/g03_tilt_reader/folds")
    ap.add_argument("--out", default="results/g03_tilt_reader")
    args = ap.parse_args()
    fold_files = sorted(Path(args.folds).glob("*.json"))
    if not fold_files:
        raise SystemExit("no fold files: pull them from the awmv-g-data volume first")

    out = {"folds": {}, "per_frame": {}, "downstream": {}}
    pooled = {"base": [], "ft": []}
    for ff in fold_files:
        rec = json.loads(ff.read_text())
        key = f"{rec['dataset']}__{rec['holdout']}"
        hold = rec["held_out"]
        fit = rec["fit_sample"]
        e_base_hold = fold_error(hold, "up_cam_base")
        e_ft_hold = fold_error(hold, "up_cam_ft")
        e_base_fit = fold_error(fit, "up_cam_base")
        e_ft_fit = fold_error(fit, "up_cam_ft")
        out["folds"][key] = {
            "holdout": rec["holdout"], "n_train": rec["n_train_frames"],
            "epochs": rec["epochs"], "lr": rec["lr"], "batch": rec["batch"],
            "lm_steps": rec["lm_steps"], "history": rec["history"],
            "held_out_base": err_stats(e_base_hold), "held_out_ft": err_stats(e_ft_hold),
            "fit_sample_base": err_stats(e_base_fit), "fit_sample_ft": err_stats(e_ft_fit)}
        if rec["dataset"] == "tii":
            for r in hold:
                if np.isfinite(e_base_hold).any():
                    pass
            pooled["base"].append(e_base_hold)
            pooled["ft"].append(e_ft_hold)
        print(f"{key}: held-out median {np.nanmedian(e_base_hold):.2f} -> "
              f"{np.nanmedian(e_ft_hold):.2f} deg, >10 deg "
              f"{np.mean(e_base_hold > 10):.3f} -> {np.mean(e_ft_hold > 10):.3f} "
              f"(n={np.isfinite(e_ft_hold).sum()})", flush=True)

    if pooled["base"]:
        b = np.concatenate(pooled["base"])
        f = np.concatenate(pooled["ft"])
        out["per_frame"]["tii_pooled"] = {"baseline": err_stats(b), "finetuned": err_stats(f)}

    # downstream: TII folds only, held-out flight clips
    fold_rows = {}
    for ff in fold_files:
        rec = json.loads(ff.read_text())
        if rec["dataset"] != "tii":
            continue
        fold_rows[rec["holdout"]] = rec["held_out"]
    if fold_rows:
        ds = downstream(fold_rows, [f for f in DEV + HELD if f in fold_rows], "tii")
        out["downstream"]["clips"] = ds
        allc = [r for rows in ds.values() for r in rows]
        # LOFO conformal thresholds on the dev flights' real clips
        # Two calibrations, both leave-one-flight-out: every real clip, and the frozen
        # gate-kept real clips (the frozen S3 gate, applied to the frozen reader's confidence
        # because the fine-tuned pass does not store its own confidence).
        tau = {}
        for target in ds:
            for gated in (False, True):
                fit = [r["median_ft"] for fl, rows in ds.items() if fl != target
                       for r in rows if r["gate_kept"] or not gated]
                fit = [v for v in fit if np.isfinite(v)]
                tau[(target, gated)] = (float(np.quantile(fit, 0.95))
                                        if len(fit) >= 8 else float("nan"))
        for target, rows in ds.items():
            for r in rows:
                for gated in (False, True):
                    if gated and not r["gate_kept"]:
                        r[f"flagged_ft_{gated}"] = None
                        r[f"flagged_removal_ft_{gated}"] = None
                        continue
                    t = tau[(target, gated)]
                    r[f"flagged_ft_{gated}"] = bool(np.isfinite(r["median_ft"])
                                                    and r["median_ft"] > t)
                    r[f"flagged_removal_ft_{gated}"] = bool(
                        np.isfinite(r["median_removal_frozen"])
                        and r["median_removal_frozen"] > t)
        by_flight = {}
        for target, rows in ds.items():
            kept = [r for r in rows if r["gate_kept"]]
            by_flight[target] = {"tau_median_residual": tau[(target, False)],
                                 "tau_median_residual_gated": tau[(target, True)],
                                 "n": len(rows), "n_gate_kept": len(kept),
                                 "flagged_real": int(sum(bool(r["flagged_ft_False"]) for r in rows)),
                                 "flagged_removal": int(sum(bool(r["flagged_removal_ft_False"])
                                                            for r in rows)),
                                 "flagged_real_gated": int(sum(bool(r["flagged_ft_True"])
                                                               for r in kept)),
                                 "flagged_removal_gated": int(sum(
                                     bool(r["flagged_removal_ft_True"]) for r in kept)),
                                 "abstain_ft": abstain_rate(rows),
                                 "median_residual_frozen": float(np.nanmedian(
                                     [r["median_frozen"] for r in rows])),
                                 "median_residual_ft": float(np.nanmedian(
                                     [r["median_ft"] for r in rows]))}
        out["downstream"]["per_flight"] = by_flight
        out["downstream"]["note"] = (
            "threshold = leave-one-flight-out 95th percentile of the fine-tuned clip medians; "
            "removal = depicted lean set to zero, the limit of the pixel tilt-removal edit; "
            "abstention excludes the frozen bootstrap CI rule (documented in the report)")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {Path(args.out) / 'summary.json'}")
    if "tii_pooled" in out["per_frame"]:
        for k in ("baseline", "finetuned"):
            s = out["per_frame"]["tii_pooled"][k]
            print(f"  pooled {k:9s} n={s['n']} median {s['median']:.2f} "
                  f"trimRMSE {s['trimmed_rmse']:.2f} >10 {s['frac_gt10']:.3f} "
                  f">20 {s['frac_gt20']:.3f}")
    for fl, v in out["downstream"].get("per_flight", {}).items():
        print(f"  {fl:24s} tau {v['tau_median_residual']:.1f}/{v['tau_median_residual_gated']:.1f} "
              f"(ungated/gated)  flagged real {v['flagged_real']}/{v['n']} "
              f"(gated {v['flagged_real_gated']}/{v['n_gate_kept']})  removal "
              f"{v['flagged_removal']}/{v['n']} (gated {v['flagged_removal_gated']}/"
              f"{v['n_gate_kept']})  abstain {v['abstain_ft']['k']}/{v['abstain_ft']['n']}  "
              f"median resid {v['median_residual_frozen']:.1f} -> {v['median_residual_ft']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
