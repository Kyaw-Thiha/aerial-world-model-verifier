"""Track A2: thresholds calibrated in the same mode as the residual they judge (amendment 9).

The infeasibility counts so far used 6.46 deg, which is TII's conformal threshold at
k_d = 0.295 with BOTH nuisances fitted, while the primary ANWM mode is now drag-free with
lag only. That mixes a threshold from one mode with a residual from another. This script
rebuilds the comparison:

  (a) TII conformal threshold under the SAME nuisance mode and k_d, pooled and per fold;
  (b) the ANWM null in that mode: straight-cruise clip residuals (per frame) and the released
      level ground-truth frames (depicted tilt, no path);
  (c) primary threshold = the larger of (a) and the null p99s;
  (d) verdict counts at 6.46 / 8.40 / 12.95 and at the primary, by regime and designed cell.

    PYTHONPATH=. pixi run -e gpu python experiments/a2_modes.py
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.a2_verdict_gated import axis_at, mocap_derivs
from experiments.phase3a_verdict import evaluation, threshold_lookup, thresholds_for
from src.datasets.tii_video import flight_parts
from src.perception.frames import camera_to_body_pitched
from src.perception.gate import clip_signal, kept
from src.physics.flatness import tilt_deg
from src.score.verdict import (clip_index, depicted_axis_world, fit_nuisances,
                               score_clip, speed_bin)
from experiments.a2_anwm_required import frame_required

ROOT = Path(__file__).resolve().parents[1]
FLIGHTS = ["flight-01a-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
           "flight-13a-trackRATM", "flight-01p-ellipse", "flight-07p-lemniscate",
           "flight-13p-trackRATM", "flight-16p-trackRATM"]
CONDS = ["identity", "tilt_k0", "tilt_k05", "tilt_k15", "mirror_roll", "yaw_control"]
K_DS = (0.0, 0.295)
MODES = ("lag_only", "full_parity")
CANDIDATE_TS = (6.46, 8.40, 12.95)


def latest(p):
    return sorted(ROOT.glob(p))[-1]


def jload(p):
    return json.loads(Path(p).read_text())


def load_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def gate_kept_rows(rows: dict[str, dict], names: list[str], cfg: dict,
                   idx: np.ndarray) -> bool:
    """The frozen gate applied to ONE CLIP's frames (not the whole flight)."""
    arrs = {"up": np.stack([np.array(rows[n]["gravity_up_cam"]) if n in rows
                            else np.full(3, np.nan) for n in names]),
            "up_confidence_mean": np.array([rows[n].get("up_confidence_mean", np.nan)
                                            if n in rows else np.nan for n in names]),
            "latitude_confidence_mean": np.array(
                [rows[n].get("latitude_confidence_mean", np.nan)
                 if n in rows else np.nan for n in names]),
            "gravity_uncertainty_raw": np.array(
                [rows[n].get("gravity_uncertainty_raw", np.nan)
                 if n in rows else np.nan for n in names])}
    arrs = {k: v[idx] for k, v in arrs.items()}
    return bool(kept(clip_signal(cfg["signal"], arrs), cfg["tau"]))


def tii_records(gate_root: Path, k_d: float, mode: str, prereg: dict,
                cfg: dict) -> list[dict]:
    """Re-score the Gate 3 clip set under one nuisance mode and drag constant."""
    clips = json.loads((ROOT / "data/processed/p3a/p3a_v1/clips.json").read_text())["clips"]
    out = []
    for flight in FLIGHTS:
        spec = np.load(ROOT / f"data/processed/p3a/p3a_v1/{flight}.npz", allow_pickle=False)
        control, _ = flight_parts(flight)
        d = mocap_derivs(ROOT / f"data/raw/tii/{control}/mocap_{flight}.csv")
        axis = axis_at(d, spec["t"], k_d)
        tilt = tilt_deg(axis)
        from scipy.spatial.transform import Rotation
        R_bw = Rotation.from_quat(spec["quat"]).as_matrix()
        yaw = Rotation.from_matrix(R_bw).as_euler("ZYX")[:, 0]
        names = [str(n) for n in spec["names"]]
        rows_by_cond = {f.name.split("__")[1].replace(".jsonl", ""): {
            r["frame"]: r for r in load_jsonl(f)}
            for f in sorted(gate_root.glob(f"{flight}__*.jsonl"))}
        for cond in CONDS:
            rows = rows_by_cond.get(cond)
            if not rows:
                continue
            up_all = np.stack([np.array(rows[n]["gravity_up_cam"]) if n in rows
                               else np.full(3, np.nan) for n in names])
            valid_all = np.array([float(rows[n].get("valid_frac", 1.0)) >= 0.5
                                  if n in rows else False for n in names])
            for clip in clips:
                if clip["flight"] != flight:
                    continue
                idx = clip_index(clip)
                up, t, tl = up_all[idx], spec["t"][idx], tilt[idx]
                vv = valid_all[idx]
                finite = np.isfinite(up).all(axis=1) & np.isfinite(axis[idx]).all(axis=1)
                mask = finite & (tl >= 10.0)
                nu = fit_nuisances(up, axis[idx], yaw[idx], t, float(spec["uptilt_deg"]),
                                   mask, prereg,
                                   uptilt_offset_override=(None if mode == "full_parity"
                                                           else 0.0))
                r = score_clip(up, axis[idx], yaw[idx], t, float(spec["uptilt_deg"]), tl,
                               vv, prereg, nuisances=nu)
                if not gate_kept_rows(rows, names, cfg, idx):
                    r["testable"] = False      # gate applies before the threshold fit
                    r["gate_kept"] = False
                else:
                    r["gate_kept"] = True
                r.update({"flight": flight, "clip_id": clip["clip_id"], "condition": cond,
                          "bin": speed_bin(clip["speed_median"],
                                           prereg["thresholds"]["speed_bins_mps"]),
                          "track": clip["track"], "control": clip["control"],
                          "tilt_implied_median": float(np.nanmedian(tl))})
                out.append(r)
    return out


def anwm_frames() -> tuple[dict, dict, dict]:
    rows = load_jsonl(ROOT / "results/gate_a2/anwm/depicted.jsonl")
    by_clip: dict[str, list[dict]] = {}
    for r in rows:
        by_clip.setdefault(r["clip_id"], []).append(r)
    man = {m["id"]: m for m in (json.loads(p.read_text()) for p in
                                sorted((ROOT / "data/processed/anwm_clips").glob("*/*/manifest.json")))}
    return by_clip, man, load_jsonl(ROOT / "results/gate_a2/floor/floor_224.jsonl")


def main() -> int:
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results/a2_modes" / run
    out.mkdir(parents=True, exist_ok=True)
    prereg = yaml.safe_load((ROOT / "configs/verdict_preregistration.yaml").read_text())
    gate_root = ROOT / "results/gate_a2/tii"

    cfg = yaml.safe_load((ROOT / "configs/gate_frozen.yaml").read_text())
    doc: dict = {"thresholds": {}, "nulls": {}, "verdicts": {}, "gate": cfg}
    tii_scores: dict[tuple[float, str], list[dict]] = {}
    for k_d in K_DS:
        for mode in MODES:
            recs = tii_records(gate_root, k_d, mode, prereg, cfg)
            tii_scores[(k_d, mode)] = recs
            pooled = thresholds_for(recs, prereg, holdout=None, policy="all_flights")
            folds = {f: thresholds_for(recs, prereg, holdout=f, policy="all_flights")
                     for f in FLIGHTS}
            real = [r for r in recs if r["condition"] == "identity" and r["testable"]]
            doc["thresholds"][f"kd{k_d}_{mode}"] = {
                "tii_conformal_pooled": pooled["pooled"]["threshold"],
                "tii_conformal_pooled_n": pooled["pooled"]["n"],
                "tii_per_fold": {f: folds[f]["pooled"]["threshold"] for f in FLIGHTS},
                "tii_fold_spread": {
                    "min": float(np.min([folds[f]["pooled"]["threshold"] for f in FLIGHTS])),
                    "max": float(np.max([folds[f]["pooled"]["threshold"] for f in FLIGHTS])),
                    "median": float(np.median([folds[f]["pooled"]["threshold"] for f in FLIGHTS]))},
                "real_score_p95": float(np.percentile([r["score"] for r in real], 95)),
                "real_n": len(real),
            }

    # ---- ANWM nulls in the same modes
    by_clip, man, floor_rows = anwm_frames()
    cruise = [c for c in by_clip if (man[c].get("meta") or {}).get("maneuver") == "cruise"]
    gt_up = np.asarray([r["gravity_up_cam"] for r in floor_rows])
    gt_tilt = tilt_deg(depicted_axis_world(gt_up, camera_to_body_pitched(0.0),
                                           np.zeros(len(gt_up))))
    doc["nulls"]["gt_level_frames"] = {
        "n": int(len(gt_tilt)), "median": float(np.median(gt_tilt)),
        "p95": float(np.percentile(gt_tilt, 95)), "p99": float(np.percentile(gt_tilt, 99)),
        "note": "no commanded path exists for these frames, so the null is the depicted tilt "
                "itself (truth is exactly level); mode-independent"}
    for k_d in K_DS:
        for mode in MODES:
            per_frame = []
            for c in cruise:
                rc = by_clip[c]
                dep = tilt_deg(depicted_axis_world(
                    np.asarray([r["gravity_up_cam"] for r in rc]),
                    camera_to_body_pitched(0.0), np.zeros(len(rc))))
                pos = np.asarray(man[c]["commanded_path"]["positions"], float)
                for kind in ("spline",):
                    try:
                        req = frame_required(pos, 1.0, k_d, kind)
                    except ValueError:
                        continue
                    n = min(len(req), len(dep))
                    r = np.abs(dep[:n] - req[:n])
                    m = np.isfinite(r)
                    per_frame.append(r[m])
            v = np.concatenate(per_frame) if per_frame else np.array([])
            doc["nulls"][f"kd{k_d}_{mode}"] = {
                "cruise_n_frames": int(v.size), "cruise_median": float(np.median(v)),
                "cruise_p95": float(np.percentile(v, 95)),
                "cruise_p99": float(np.percentile(v, 99)),
                "note": "residual |required - depicted| on the 7 straight-cruise clips at 1 s; "
                        "lag moves the profile by <1/6 frame so lag_only == none here"}

    # ---- primary threshold per mode
    for k_d in K_DS:
        for mode in MODES:
            t = doc["thresholds"][f"kd{k_d}_{mode}"]
            nl = doc["nulls"][f"kd{k_d}_{mode}"]
            primary = max(t["tii_conformal_pooled"], nl["cruise_p99"],
                          doc["nulls"]["gt_level_frames"]["p99"])
            t["primary_threshold"] = float(primary)
            t["primary_source"] = ("TII conformal" if primary == t["tii_conformal_pooled"]
                                   else ("ANWM cruise p99" if primary == nl["cruise_p99"]
                                         else "GT p99"))
            t["detection_floor_deg"] = float(primary + nl["cruise_median"])

    # ---- verdicts by threshold, regime/cell, using the parity table residuals
    par = pd.read_csv(latest("results/a2_parity/*/parity.csv"))
    verdicts = {}
    for k_d, mode, col in [(0.0, "lag_only", "residual_lag_only"),
                           (0.295, "lag_only", "residual_lag_only"),
                           (0.0, "full_parity", "residual_parity"),
                           (0.295, "full_parity", "residual_parity")]:
        sub = par[(par.k_d == k_d) & par.testable]
        primary = doc["thresholds"][f"kd{k_d}_{mode}"]["primary_threshold"]
        for key, g in sub.groupby([sub.regime, sub.designed.fillna(-1)]):
            reg, des = key
            cells = {"n_clips": int(g.clip_id.nunique()),
                     "by_interpolant": {}}
            for kind, gk in g.groupby("interpolant"):
                d = {"n_clips": int(gk.clip_id.nunique()),
                     "median_residual": float(gk[col].median())}
                for t in list(CANDIDATE_TS) + [primary]:
                    d[f"inf_{t:.2f}"] = int((gk[col] > t).sum())
                cells["by_interpolant"][kind] = d
            verdicts[f"kd{k_d}_{mode}_{reg}_{'native' if des < 0 else int(des)}"] = cells
    doc["verdicts"] = verdicts
    (out / "modes.json").write_text(json.dumps(doc, indent=1, default=float))
    (out / "tii_scores.json").write_text(json.dumps(
        {f"kd{k}_{m}": v for (k, m), v in tii_scores.items()}, indent=1, default=float))

    # ---- print
    print("== thresholds and nulls ==")
    for k in ("kd0.0", "kd0.295"):
        for m in MODES:
            t = doc["thresholds"][f"{k}_{m}"]
            nl = doc["nulls"][f"{k}_{m}"]
            print(f"  {k} {m:11s} TII conf {t['tii_conformal_pooled']:6.2f} "
                  f"(folds {t['tii_fold_spread']['min']:.1f}-{t['tii_fold_spread']['max']:.1f}) "
                  f"cruise p95 {nl['cruise_p95']:5.2f} p99 {nl['cruise_p99']:5.2f} "
                  f"-> primary {t['primary_threshold']:5.2f} [{t['primary_source']}] "
                  f"floor {t['detection_floor_deg']:5.2f}")
    print("  GT level frames:", {k: round(v, 2) for k, v in
                                 doc["nulls"]["gt_level_frames"].items() if k != "note"})
    print("== verdicts (testable clips; counts at 6.46 / 8.40 / 12.95 / primary) ==")
    for k, v in verdicts.items():
        for kind, d in v["by_interpolant"].items():
            keys = [kk for kk in d if kk.startswith("inf_")]
            keys.sort(key=lambda s: float(s.split("_")[-1]))
            print(f"  {k:28s} {kind:6s} n={d['n_clips']:2d} med {d['median_residual']:5.1f} | "
                  + " ".join(f"{kk.split('_')[-1]}->{d[kk]}" for kk in keys))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
