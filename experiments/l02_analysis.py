"""Track L, L02 analysis: fold errors, the two error criteria, and the sim-to-real gap.

Reads the fold JSONs written by ``remote/l_tilt_reader.py`` (pulled from ``awmv-l-data``) and
the frozen inputs. CPU only. Nothing here selects or tunes a model; every rule is from
``configs/l02_reader_prereg.yaml``.

    PYTHONPATH=. python experiments/l02_analysis.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.perception.frames import camera_to_body_pitched
from src.verdict.lean import depicted_lean_from_gravity

FOLDS = Path("results/l02/folds")
SIM_NPZ = Path("results/l02/folds")          # not used directly; kept for provenance
GT = Path("data/l01/sim_dev/gt")
REAL_LABELS = Path("data/g03_frames")
MANIFEST = Path("results/i02/manifest.csv")
L01 = Path("results/l01/per_clip.csv")
OUT = Path("results/l02")
FAMILIES = ["urban_checker", "forest_ridge", "industrial"]
DEV = ["flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse",
       "flight-07p-lemniscate", "flight-13a-trackRATM", "flight-16p-trackRATM"]
HELD = ["flight-13p-trackRATM", "flight-07a-lemniscate"]


def angle(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) *
                                           np.linalg.norm(b, axis=1), 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def stats(e):
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    if not e.size:
        return {"n": 0}
    s = np.sort(e)
    k = max(1, int(round(0.9 * len(s))))
    return {"n": int(e.size), "median": float(np.median(e)),
            "trimmed_rmse": float(np.sqrt(np.mean(s[:k] ** 2))),
            "p90": float(np.percentile(e, 90)), "frac_gt10": float(np.mean(e > 10)),
            "frac_gt20": float(np.mean(e > 20))}


def real_truth():
    """flight -> {frame_name: (up_cam, R_cb)} from the G03 prepared labels."""
    out = {}
    for flight in DEV + HELD:
        lab = json.loads((REAL_LABELS / flight / "labels.json").read_text())
        R_cb = camera_to_body_pitched(np.deg2rad(float(lab["uptilt_deg"])))
        out[flight] = ({f["frame"]: np.asarray(f["up_cam"], float) for f in lab["frames"]},
                       R_cb, {f["frame"]: f.get("up_cam_base") for f in lab["frames"]})
    return out


def sim_truth(clip_id: str):
    """frame index -> (up_cam, lean_gt, R_cb) for one simulated clip."""
    row = SIM_META[clip_id]
    z = np.load(GT / f"{clip_id}.npz")
    R = np.asarray(z["R_bw_cam"], float)
    R_cb = np.asarray(z["R_cb"], float)
    hz = float(np.asarray(z["gt_hz"]).ravel()[0])
    lean = np.asarray(z["lean_dep"], float)
    e3 = np.array([0.0, 0.0, 1.0])
    out = {}
    for k in range(int(row["n_frames"])):
        i = int(min(max(round((k / 30.0) * hz), 0), len(R) - 1))
        up = (R[i] @ R_cb).T @ e3
        out[k] = (up, lean[i], R_cb)
    return out


SIM_META = {r["clip_id"]: r for r in csv.DictReader(MANIFEST.open())}


def rows_error(rows, cross=None):
    e, lean_err, fams = [], [], []
    for r in rows:
        if r.get("up_cam_ft") is None:
            continue
        lab = np.asarray(r["up_cam"], float)
        est = np.asarray(r["up_cam_ft"], float)
        e.append(angle(lab[None], est[None])[0])
        if cross is not None:
            lean_err.append(cross(r))
        fams.append(r.get("clip_id"))
    return np.asarray(e), lean_err, fams


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(FOLDS.glob("*.json"))
    if not files:
        raise SystemExit("no fold files: pull them from awmv-l-data /l02/folds first")
    folds = {f.stem: json.loads(f.read_text()) for f in files}
    rtruth = real_truth()
    out = {"folds": {}, "criteria": {}}

    # ---------------- fold-level ----------------
    for key, rec in folds.items():
        arm, hold = key.split("__", 1)
        entry = {"arm": arm, "holdout": hold, "n_train_pool": rec.get("n_train_pool"),
                 "frames_per_epoch": rec.get("frames_per_epoch"),
                 "history": rec.get("history")}
        if rec.get("held_out_real"):
            flights = {}
            for r in rec["held_out_real"]:
                if r.get("flight") is None:
                    continue
                flights.setdefault(r["flight"], []).append(r)
            entry["real"] = {}
            for fl, rs in flights.items():
                e = np.asarray([angle(np.asarray([r["up_cam"]]), np.asarray([r["up_cam_ft"]]))[0]
                                for r in rs if r["up_cam_ft"]])
                entry["real"][fl] = stats(e)
        if rec.get("held_out_sim"):
            # bias per family on the held-out family's frames
            per_fam = {}
            for r in rec["held_out_sim"]:
                cid = r["clip_id"]
                if cid is None:
                    continue
                fam = SIM_META[cid]["family"]
                t = sim_truth(cid)
                up_t, lean_t, R_cb = t[int(r["frame"])]
                lean_est = depicted_lean_from_gravity(np.asarray([r["up_cam_ft"]]), R_cb)[0]
                per_fam.setdefault(fam, []).append(lean_est - lean_t)
            entry["sim"] = {f: {"n": len(v),
                                "bias": np.mean(np.asarray(v), axis=0).tolist(),
                                "bias_magnitude": float(np.linalg.norm(np.mean(np.asarray(v), axis=0))),
                                "median_error": None}
                            for f, v in per_fam.items()}
            e = np.asarray([angle(np.asarray([r["up_cam"]]), np.asarray([r["up_cam_ft"]]))[0]
                            for r in rec["held_out_sim"] if r["up_cam_ft"]])
            entry["sim_error"] = stats(e)
        if rec.get("all_real"):
            by = {}
            for r in rec["all_real"]:
                by.setdefault(r["flight"], []).append(r)
            entry["all_real"] = {}
            for fl, rs in by.items():
                e = np.asarray([angle(np.asarray([r["up_cam"]]), np.asarray([r["up_cam_ft"]]))[0]
                                for r in rs if r["up_cam_ft"]])
                entry["all_real"][fl] = stats(e)
        out["folds"][key] = entry

    # ---------------- S1: held-out real error ----------------
    base = {}
    for fl in HELD:
        src = Path("results/j_inputs/g03_folds") / f"tii__{fl}.json"
        if not src.exists():
            continue
        rec = json.loads(src.read_text())
        e = [angle(np.asarray([r["up_cam"]]), np.asarray([r["up_cam_ft"]]))[0]
             for r in rec["held_out"] if r.get("up_cam_ft")]
        base[fl] = stats(e)
    out["criteria"]["s1_g03_baseline"] = base
    out["criteria"]["s1_arms"] = {}
    for arm in ("real", "sim", "mixed"):
        rec = folds.get(f"{arm}__none")
        if not rec or not rec.get("all_real"):
            continue
        e_all = []
        per = {}
        for fl in HELD:
            e = [angle(np.asarray([r["up_cam"]]), np.asarray([r["up_cam_ft"]]))[0]
                 for r in rec["all_real"] if r.get("flight") == fl and r["up_cam_ft"]]
            per[fl] = stats(e)
            e_all.extend(e)
        s = stats(e_all)
        ref = base
        pooled_base = (np.mean([v["trimmed_rmse"] for v in ref.values()])
                       if ref else float("nan"))
        out["criteria"]["s1_arms"][arm] = {
            "pooled": s, "per_flight": per,
            "reduction_vs_pooled_g03": (1.0 - s["trimmed_rmse"] / pooled_base)
            if ref else None,
        }

    # ---------------- S2: scene-family bias spread ----------------
    if L01.exists():
        frozen = {}
        for fam in FAMILIES:
            rows = [r for r in csv.DictReader(L01.open()) if r["family"] == fam]
            b = np.array([float(r["bias_forward"]) for r in rows] +
                         [0.0])  # keep shape
            bl = np.mean([float(r["bias_left"]) for r in rows])
            frozen[fam] = float(np.hypot(np.mean([float(r["bias_forward"]) for r in rows]), bl))
        out["criteria"]["s2_frozen"] = frozen
        out["criteria"]["s2_frozen_spread"] = float(max(frozen.values()) - min(frozen.values()))
        for arm in ("sim", "mixed"):
            per = {}
            for fam in FAMILIES:
                rec = folds.get(f"{arm}__{fam}")
                if not rec or "sim" not in out["folds"].get(f"{arm}__{fam}", {}):
                    continue
                per[fam] = out["folds"][f"{arm}__{fam}"]["sim"][fam]["bias_magnitude"]
            if len(per) == 3:
                spread = float(max(per.values()) - min(per.values()))
                out["criteria"][f"s2_{arm}"] = {
                    "per_family": per, "spread": spread,
                    "reduction_vs_frozen": (1.0 - spread / out["criteria"]["s2_frozen_spread"])
                    if out["criteria"]["s2_frozen_spread"] > 0 else None}

    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["criteria"], indent=1, default=float)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
