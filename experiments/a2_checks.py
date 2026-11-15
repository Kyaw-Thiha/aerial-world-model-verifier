"""Track A2: the follow-up checks requested after the first write-up.

One pass over everything already computed (gate, verdict, controls, ANWM) plus what was
missing: per-fold thresholds, the null-edit control for generated frames, spline
end-condition sensitivity, excitation-only untestability, and the rigorous peak bounds.

Prints compact tables and writes `checks.json`. Never re-runs the estimator.

    PYTHONPATH=. pixi run -e gpu python experiments/a2_checks.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.a2_anwm_required import frame_required
from src.perception.frames import camera_to_body_pitched
from src.perception.gate import SINGLE_FRAME, clip_signal, kept
from src.physics.flatness import tilt_deg
from src.physics.peaks import (peak_accel_lower_bound, summarise_bounds,
                               thrust_ratio_lower_bound, tilt_lower_bound_deg,
                               peak_horizontal_accel_lower_bound)
from src.score.verdict import depicted_axis_world

ROOT = Path(__file__).resolve().parents[1]
K_DS = (0.0, 0.295)
THRESH = 6.46


def latest(pattern: str) -> Path:
    hits = sorted(ROOT.glob(pattern))
    if not hits:
        raise SystemExit(f"missing: {pattern}")
    return hits[-1]


def jload(p: Path):
    return json.loads(Path(p).read_text())


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/a2_checks")
    args = ap.parse_args()
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / args.out / run
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "configs/gate_frozen.yaml").read_text())
    doc: dict = {"run_id": run, "gate": cfg}

    # ---------------------------------------------------------------- 1. gate
    gv = jload(latest("results/a2_gate/*/gate_validation.json"))
    v = gv["validation"]
    gate = {"n_clips": v["n_clips"], "n_positive": v["n_positive"],
            "signals": {s: {"auroc": x["auroc"]["auroc"], "lo": x["auroc"]["lo"],
                            "hi": x["auroc"]["hi"], "coverage": x["lofo_coverage_overall"],
                            "kept_err": x["kept_median_error_deg"],
                            "dropped_err": x["dropped_median_error_deg"],
                            "eligible": SINGLE_FRAME[s],
                            "sens5": x.get("auroc_label_sensitivity", {}).get("5"),
                            "sens20": x.get("auroc_label_sensitivity", {}).get("20")}
                        for s, x in v["signals"].items()},
            "label_sensitivity": {k: {"n_pos": x["n_positive"],
                                      "per_flight": x["per_flight"]}
                                  for k, x in v["label_sensitivity"].items()},
            "folds": v["signals"][cfg["signal"]]["lofo"],
            "selection": v["selection"]}
    clips = gv["clips"]
    sig = cfg["signal"]

    def gsig(c):
        return c.get(f"identity::{sig}")

    pos = [c for c in clips if c.get("label") == 1]
    gate["positives"] = [{"clip": c["clip_id"], "err_deg": c["median_error_deg"],
                          "signal": gsig(c), "kept_global": bool(gsig(c) <= cfg["tau"])}
                         for c in pos]
    gate["worst"] = [{"clip": c["clip_id"], "err_deg": c["median_error_deg"],
                      "signal": gsig(c), "kept_global": bool(gsig(c) <= cfg["tau"])}
                     for c in sorted(clips, key=lambda c: -(c.get("median_error_deg") or 0))[:8]]
    gate["coverage_per_flight"] = {f: {"coverage": h["coverage"], "tau": h["tau"],
                                       "dropped_pos": h["n_dropped_pos"],
                                       "dropped_neg": h["n_dropped_neg"], "n": h["n"]}
                                   for f, h in gate["folds"].items()}

    # baseline exactness on the ORIGINAL Gate 3 rows
    orig = jload(ROOT / "results/phase3a_verdict/20261109T090957Z_bc3a398/metrics.json")
    base_orig = latest("results/a2_verdict_baseline_origrows/*/metrics.json")
    myb = jload(base_orig)["results"]["kd_0.295"]
    o = {(r["flight"], r["clip_id"], r["condition"]): r for r in orig["records"]}
    m = {(r["flight"], r["clip_id"], r["condition"]): r for r in myb.get("_records", [])}
    common = sorted(set(o) & set(m))
    score_diffs = [abs(o[k]["score"] - m[k]["score"]) for k in common
                   if o[k].get("score") is not None and m[k].get("score") is not None]
    testable_diff = sum(1 for k in common if o[k]["testable"] != m[k]["testable"])
    base_new = latest("results/a2_verdict_gated_baseline/*/metrics.json")
    myn = jload(base_new)["results"]["kd_0.295"]
    n = {(r["flight"], r["clip_id"], r["condition"]): r for r in myn.get("_records", [])}
    gate["baseline"] = {
        "orig_auroc": orig["evaluation"]["family_auroc_pooled"],
        "mine_auroc_same_rows": myb["family_auroc_pooled"],
        "orig_acceptance": orig["evaluation"]["heldout_acceptance"],
        "mine_acceptance_same_rows": myb["heldout_acceptance"],
        "n_records": len(common),
        "n_score_diff_gt_1e-6": int(sum(1 for d in score_diffs if d > 1e-6)),
        "max_score_diff_deg": float(max(score_diffs)) if score_diffs else None,
        "n_testable_diff": int(testable_diff),
        "note": "same rows as Gate 3; differences are nuisance grid ties, not code",
        "re_extracted_rows_auroc": myn["family_auroc_pooled"],
        "re_extracted_rows_acceptance": myn["heldout_acceptance"],
        "n_score_diff_vs_original_rows": int(sum(
            1 for k in common if k in n and o[k].get("score") is not None
            and n[k].get("score") is not None
            and abs(o[k]["score"] - n[k]["score"]) > 1e-6)),
    }
    doc["gate"] = gate

    # ---------------------------------------------------------------- 2. verdict
    gv_run = latest("results/a2_verdict_gated/*/metrics.json")
    pair = latest("results/a2_verdict_gated_pairgate/*/metrics.json")
    verdict = {"per_condition": {}, "pair_level": {}}
    for key, val in jload(gv_run)["results"].items():
        fam = val["families"]
        verdict["per_condition"][key] = {
            "k_d": val["k_d"], "threshold": val["pooled_thresholds"]["pooled"]["threshold"],
            "acceptance": val["heldout_acceptance"], "n": val["heldout_n"],
            "wilson": val["heldout_acceptance_wilson95"],
            "pooled_paired": val["family_auroc_pooled"],
            "pooled_unmatched": val["unmatched_auroc"]["pooled_tilt_families"],
            "yaw_auroc": fam["yaw_control"]["auroc"],
            "yaw_rejection": fam["yaw_control"]["rejection_rate"],
            "families": {c: {"rejection": fam[c]["rejection_rate"], "n": fam[c]["n"],
                             "auroc": fam[c]["auroc"]}
                         for c in ("tilt_k0", "tilt_k05", "tilt_k15", "mirror_roll",
                                   "yaw_control")},
            "mdv": val["mdv"], "mdv80": val["mdv_smallest_rejected80_deg"],
            "abstention": {k: x for k, x in val["abstention_with_gate"].items()
                           if k in ("overall", "excitation", "gate", "trackratm",
                                    "autonomous", "piloted")},
            "per_flight": val["acceptance_per_flight"],
        }
    pv = jload(pair)["results"]["kd_0.295"]
    fam = pv["families"]
    verdict["pair_level"] = {
        "acceptance": pv["heldout_acceptance"], "n": pv["heldout_n"],
        "pooled_paired": pv["family_auroc_pooled"],
        "pooled_unmatched": pv["unmatched_auroc"]["pooled_tilt_families"],
        "yaw_auroc": fam["yaw_control"]["auroc"],
        "families": {c: {"rejection": fam[c]["rejection_rate"], "n": fam[c]["n"]}
                     for c in ("tilt_k0", "tilt_k05", "tilt_k15", "mirror_roll",
                               "yaw_control")}}
    doc["verdict"] = verdict

    # ---------------------------------------------------------------- 3. controls
    dep_rows = load_jsonl(ROOT / "results/gate_a2/anwm/depicted.jsonl")
    floor_rows = load_jsonl(ROOT / "results/gate_a2/floor/floor_224.jsonl")
    edit_rows = load_jsonl(ROOT / "results/gate_a2/edits/edits.jsonl")

    def tilt_of(rows):
        up = np.asarray([r["gravity_up_cam"] for r in rows], dtype=np.float64).reshape(-1, 3)
        return tilt_deg(depicted_axis_world(up, camera_to_body_pitched(0.0),
                                            np.zeros(len(up))))

    by_clip: dict[str, list[dict]] = {}
    for r in dep_rows:
        by_clip.setdefault(r["clip_id"], []).append(r)
    man = {m["id"]: m for m in (json.loads(p.read_text())
                                for p in sorted((ROOT / "data/processed/anwm_clips")
                                                .glob("*/*/manifest.json")))}
    cruise = [c for c in by_clip if (man[c].get("meta") or {}).get("maneuver") == "cruise"]
    seeds = [c for c in by_clip if c.endswith("_s1") or c.endswith("_s2")]
    cruise_frames = np.concatenate([tilt_of(by_clip[c]) for c in cruise])
    floor_tilt = tilt_of(floor_rows)
    transfer = {"gt_clips": 0, "gt_dropped": 0}

    def gate_kept(rows):
        arrs = {"up": np.asarray([r["gravity_up_cam"] for r in rows]).reshape(-1, 3),
                "up_confidence_mean": np.array([r.get("up_confidence_mean", np.nan)
                                                for r in rows]),
                "latitude_confidence_mean": np.array([r.get("latitude_confidence_mean", np.nan)
                                                      for r in rows]),
                "gravity_uncertainty_raw": np.array([r.get("gravity_uncertainty_raw", np.nan)
                                                     for r in rows])}
        return kept(clip_signal(cfg["signal"], arrs), cfg["tau"])

    groups: dict[str, list[dict]] = {}
    for r in floor_rows:
        groups.setdefault(r["traj"], []).append(r)
    for traj, rows in groups.items():
        rows = sorted(rows, key=lambda r: r["frame"])
        for i in range(0, max(0, len(rows) - 15), 16):
            ch = rows[i:i + 16]
            if len(ch) < 16:
                continue
            transfer["gt_clips"] += 1
            transfer["gt_dropped"] += int(not gate_kept(ch))

    # null edit: identity remap vs the raw frames of the same clip
    per_clip_id: dict[str, dict] = {}
    for r in edit_rows:
        per_clip_id.setdefault(r["clip_id"], {}).setdefault(r["condition"], []).append(r)
    null_diff = []
    for cid, conds in per_clip_id.items():
        if "identity" not in conds or cid not in by_clip:
            continue
        a = np.median(tilt_of(conds["identity"]))
        b = np.median(tilt_of(by_clip[cid]))
        null_diff.append({"clip": cid, "identity_remap_deg": float(a),
                          "raw_deg": float(b), "delta_deg": float(a - b)})
    edits = {}
    for cond in sorted({r["condition"] for r in edit_rows}):
        if cond == "identity":
            continue
        axis, ang = cond.split("_")
        ang = float(ang)
        per_clip_delta = []
        for cid, conds in per_clip_id.items():
            if cond not in conds or "identity" not in conds:
                continue
            per_clip_delta.append(float(np.median(tilt_of(conds[cond]))
                                        - np.median(tilt_of(conds["identity"]))))
        err = np.abs(np.array(per_clip_delta) - ang)
        edits[cond] = {"applied": ang, "recovered_median": float(np.median(per_clip_delta)),
                       "mae": float(np.median(err)), "p90_err": float(np.percentile(err, 90)),
                       "frac_within_3": float(np.mean(err <= 3.0)), "n": len(err)}
    doc["controls"] = {
        "noise_floor": {
            "cruise_median_deg": float(np.median(cruise_frames)),
            "cruise_p99_deg": float(np.percentile(cruise_frames, 99)),
            "cruise_max_deg": float(np.max(cruise_frames)),
            "n_cruise_frames": int(cruise_frames.size),
            "seed_median_of_medians_deg": float(np.median(
                [np.median(tilt_of(by_clip[c])) for c in seeds])),
            "gt_level_median_deg": float(np.median(floor_tilt)),
            "gt_level_p99_deg": float(np.percentile(floor_tilt, 99)),
            "reference_deg": 0.57},
        "null_edit": {"n": len(null_diff),
                      "median_abs_delta_deg": float(np.median([abs(x["delta_deg"])
                                                               for x in null_diff])),
                      "max_abs_delta_deg": float(np.max([abs(x["delta_deg"])
                                                         for x in null_diff])),
                      "per_clip": null_diff},
        "edit_recovery": edits,
        "gate_transfer": transfer,
        "anwm_gate_dropped": int(sum(1 for c in by_clip if not gate_kept(by_clip[c]))),
        "anwm_n_clips": len(by_clip),
    }

    # ---------------------------------------------------------------- 4/5/6. ANWM
    frame_csv = latest("results/a2_anwm_score/*/anwm_frame_level.csv")
    ftab = pd.read_csv(frame_csv)
    summary = jload(latest("results/a2_anwm_score/*/summary.json"))
    # excitation-only untestability: kept, not collapsed, but < 12 frames above 10 deg
    per = {}
    for (kind, k_d), g in ftab.groupby(["interpolant", "k_d"]):
        per[f"{kind}_{k_d}"] = {
            "C": {"n": int((g.regime == "C").sum()),
                  "gate_kept": int(g[g.regime == "C"].gate_kept.sum()),
                  "testable": int(g[g.regime == "C"].testable.sum()),
                  "untestable_excitation_only": int(((g.regime == "C") & g.gate_kept
                                                     & ~g.testable).sum()),
                  "infeasible": int(g[g.regime == "C"].infeasible.sum())},
            "A": {"n": int((g.regime == "A").sum()),
                  "gate_kept": int(g[g.regime == "A"].gate_kept.sum()),
                  "testable": int(g[g.regime == "A"].testable.sum()),
                  "untestable_excitation_only": int(((g.regime == "A") & g.gate_kept
                                                     & ~g.testable).sum()),
                  "infeasible": int(g[g.regime == "A"].infeasible.sum())},
        }
    # Regime C cells and the slope of depicted on designed
    dose = {}
    for (kind, k_d), g in ftab.groupby(["interpolant", "k_d"]):
        c = g[(g.regime == "C") & g.designed_tilt_deg.notna()]
        cells = []
        for lvl, gg in c.groupby("designed_tilt_deg"):
            cells.append({"designed": float(lvl), "n": int(len(gg)),
                          "depicted_median": float(gg.depicted_median_deg.median()),
                          "depicted_p10": float(gg.depicted_median_deg.quantile(0.1)),
                          "depicted_p90": float(gg.depicted_median_deg.quantile(0.9)),
                          "required_median": float(gg.required_median_deg.median()),
                          "residual_median": float(gg.residual_median_deg.median()),
                          "testable": int(gg.testable.sum()),
                          "infeasible": int(gg.infeasible.sum())})
        slope_dep = (float(np.polyfit(c.designed_tilt_deg, c.depicted_median_deg, 1)[0])
                     if len(c) > 2 else None)
        slope_req = (float(np.polyfit(c.designed_tilt_deg, c.required_median_deg, 1)[0])
                     if len(c) > 2 else None)
        dose[f"{kind}_{k_d}"] = {"cells": cells, "slope_depicted": slope_dep,
                                 "slope_required": slope_req}
    doc["anwm"] = {"testability": per, "dose": dose,
                   "rule": {"excitation_min_frames_ge10deg": 12,
                            "frames_per_clip": 16,
                            "threshold_deg": THRESH,
                            "gate": {"signal": cfg["signal"], "tau": cfg["tau"]},
                            "collapse": "median frame_std < 0.05 or temporal_delta < 0.02"}}

    # Regime A: end-condition sensitivity (spline bc variants)
    bc_sens = {}
    for bc in ("natural", "not-a-knot", "clamped"):
        rowsA = []
        for cid, pc in [(c, None) for c in by_clip]:
            if man[cid]["regime"] != "A":
                continue
            pos_arr = np.asarray(man[cid]["commanded_path"]["positions"], dtype=np.float64)
            dep = tilt_of(by_clip[cid])
            for k_d in K_DS:
                try:
                    req = frame_required(pos_arr, 1.0, k_d, "spline", bc=bc)
                except ValueError:
                    continue
                n = min(len(req), len(dep))
                ok = np.isfinite(req[:n]) & np.isfinite(dep[:n])
                if ok.sum() < 8:
                    continue
                exc = int((req[:n][ok] >= 10).sum())
                res = float(np.median(np.abs(req[:n][ok] - dep[:n][ok])))
                testable = bool(gate_kept(by_clip[cid]) and exc >= 12)
                rowsA.append({"clip": cid, "k_d": k_d, "testable": testable,
                              "infeasible": bool(testable and res > THRESH),
                              "residual": res, "excitation": exc})
        d = pd.DataFrame(rowsA)
        bc_sens[bc] = {f"kd{k}": {"n": int(len(d[d.k_d == k])),
                                  "testable": int(d[d.k_d == k].testable.sum()),
                                  "infeasible": int(d[d.k_d == k].infeasible.sum()),
                                  "median_residual_testable": (
                                      float(d[(d.k_d == k) & d.testable].residual.median())
                                      if d[(d.k_d == k) & d.testable].shape[0] else None)}
                        for k in K_DS}
    doc["anwm"]["spline_end_conditions"] = bc_sens

    # Regime A: what the testable clips look like
    ta = ftab[(ftab.regime == "A") & (ftab.k_d == 0.295) & (ftab.interpolant == "spline")]
    doc["anwm"]["regime_A_testable"] = [
        {"clip": r.clip_id, "maneuver": r.maneuver, "step_m": man[r.clip_id]
         ["commanded_path"]["step_spacing_m"], "excitation_frames": int(r.excitation_frames),
         "required_median": float(r.required_median_deg),
         "required_max": float(r.required_max_deg),
         "depicted_median": float(r.depicted_median_deg),
         "residual": float(r.residual_median_deg)} for r in ta[ta.testable].itertuples()]

    # drag-free headline
    doc["anwm"]["drag_free"] = {k: v for k, v in per.items() if k.endswith("_0.0")}
    doc["anwm"]["dt_min"] = summary["minimum_consistent_dt_framewise"]["by"]

    # ---------------------------------------------------------------- 7. peak bounds
    peaks = {}
    for cid in sorted(by_clip):
        pos_arr = np.asarray(man[cid]["commanded_path"]["positions"], dtype=np.float64)
        dep = tilt_of(by_clip[cid])
        for dt in (0.25, 0.667, 1.0):
            for k_d in K_DS:
                s = summarise_bounds(pos_arr, dt, k_d, max_vertical_accel=2.0 * 9.81)
                s.update({"clip": cid, "regime": man[cid]["regime"], "dt": dt, "k_d": k_d,
                          "depicted_median": float(np.median(dep))})
                peaks[f"{cid}|{dt}|{k_d}"] = s
    doc["peak_bounds"] = peaks
    by_regime = {}
    for dt in (0.25, 0.667, 1.0):
        for k_d in K_DS:
            for reg in ("A", "B", "C"):
                sel = [p for p in peaks.values()
                       if p["dt"] == dt and p["k_d"] == k_d and p["regime"] == reg]
                if not sel:
                    continue
                twr = np.array([p["thrust_ratio_lower_max"] for p in sel])
                by_regime[f"dt{dt}_kd{k_d}_{reg}"] = {
                    "n": len(sel),
                    "peak_accel_max_median": float(np.median([p["peak_accel_lower_max"]
                                                              for p in sel])),
                    "tilt_lower_max_median": float(np.median([p["tilt_lower_bound_max_deg"]
                                                              for p in sel])),
                    "twr_lower_max_median": float(np.median(twr)),
                    "frac_twr_gt_1": float(np.mean(twr > 1.0)),
                    "frac_twr_gt_2": float(np.mean(twr > 2.0)),
                    "frac_twr_gt_3": float(np.mean(twr > 3.0)),
                }
    doc["peak_bounds_summary"] = by_regime

    (out / "checks.json").write_text(json.dumps(doc, indent=1, default=float))

    # ---------------------------------------------------------------- print
    p = print
    p("== 1. GATE ==")
    for s, x in gate["signals"].items():
        p(f"  {s:28s} AUROC {x['auroc']:.3f} [{x['lo']:.3f},{x['hi']:.3f}] "
          f"cov {x['coverage']:.2f} kept {x['kept_err']:.2f} drop {x['dropped_err']:.2f} "
          f"{'' if x['eligible'] else '(ineligible)'}")
    p("  per-fold tau/coverage:")
    for f, h in gate["coverage_per_flight"].items():
        p(f"    {f:22s} tau {h['tau']:+.5f} cov {h['coverage']:.2f} "
          f"drop+ {h['dropped_pos']} drop- {h['dropped_neg']} n {h['n']}")
    p("  baseline:", json.dumps(gate["baseline"]))
    p("== 2. VERDICT ==")
    for k, x in verdict["per_condition"].items():
        p(f"  {k}: acc {x['acceptance']:.3f} n={x['n']} thr {x['threshold']:.2f} "
          f"paired {x['pooled_paired']:.3f} unmatched {x['pooled_unmatched']:.3f} "
          f"yaw {x['yaw_auroc']:.3f} abst {x['abstention']['overall']:.2f}")
        p("     rej " + " ".join(f"{c}={x['families'][c]['rejection']:.2f}({x['families'][c]['n']})"
                                 for c in x["families"] if x["families"][c]["rejection"] is not None))
    p("  pair-level:", json.dumps(verdict["pair_level"]))
    p("== 3. CONTROLS ==")
    c = doc["controls"]
    p("  floor:", json.dumps(c["noise_floor"]))
    p("  null edit:", {k: c["null_edit"][k] for k in ("n", "median_abs_delta_deg",
                                                      "max_abs_delta_deg")})
    for k, x in c["edit_recovery"].items():
        p(f"  {k}: applied {x['applied']:.0f} rec {x['recovered_median']:.2f} "
          f"mae {x['mae']:.2f} p90 {x['p90_err']:.2f}")
    p("  transfer:", c["gate_transfer"], "anwm dropped", c["anwm_gate_dropped"],
      "of", c["anwm_n_clips"])
    p("== 4/5/6. ANWM ==")
    for k, x in doc["anwm"]["testability"].items():
        p(f"  {k}: A testable {x['A']['testable']}/{x['A']['gate_kept']} "
          f"exc-only-untestable {x['A']['untestable_excitation_only']} infeasible {x['A']['infeasible']} | "
          f"C testable {x['C']['testable']}/{x['C']['gate_kept']} "
          f"exc-only {x['C']['untestable_excitation_only']} infeasible {x['C']['infeasible']}")
    for k, x in doc["anwm"]["dose"].items():
        p(f"  dose {k}: slope depicted {x['slope_depicted']:+.3f} required {x['slope_required']:+.3f}")
        for cell in x["cells"]:
            p(f"     designed {cell['designed']:4.0f} n={cell['n']:2d} dep {cell['depicted_median']:.2f} "
              f"[{cell['depicted_p10']:.2f},{cell['depicted_p90']:.2f}] req {cell['required_median']:.1f} "
              f"res {cell['residual_median']:.1f} testable {cell['testable']} infeas {cell['infeasible']}")
    p("  spline end conditions:", json.dumps(doc["anwm"]["spline_end_conditions"], indent=1))
    p("== 7. PEAK BOUNDS ==")
    for k, x in by_regime.items():
        p(f"  {k}: n={x['n']} peakA {x['peak_accel_max_median']:.1f} "
          f"tiltLB {x['tilt_lower_max_median']:.1f} TWR {x['twr_lower_max_median']:.2f} "
          f">1 {x['frac_twr_gt_1']:.2f} >2 {x['frac_twr_gt_2']:.2f} >3 {x['frac_twr_gt_3']:.2f}")
    p("wrote", out / "checks.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
