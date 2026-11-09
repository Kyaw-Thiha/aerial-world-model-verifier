"""Track A2, Task 3b: score the generated clips (depicted tilt vs the tilt their paths demand).

Inputs are all produced elsewhere and are never recomputed here:
  * depicted tilt — GeoCalib pinhole at 224 px, ``remote/a_gate.py --job anwm``;
  * required tilt — the two interpolators, ``experiments/a2_anwm_required.py``;
  * the gate — ``configs/gate_frozen.yaml`` (frozen before any clip was scored);
  * the verdict threshold — the Task 2 run under amendment 2.

Everything pre-registered in amendment 3 is computed here and nothing else: the transfer
check, the two positive controls, the regime tables, the minimum consistent dt, and the
stopping rule. A clip is "not testable" when the gate drops it, when the rollout has
collapsed, or when the required tilt never reaches the detectability floor at that dt. A
clip is declared INFEASIBLE only when the **lower bound** exceeds the depicted tilt by more
than the verdict threshold.

    PYTHONPATH=. pixi run -e gpu python experiments/a2_anwm_score.py \\
        --depicted results/gate_a2/anwm/depicted.jsonl \\
        --floor results/gate_a2/floor/floor_224.jsonl \\
        --edits results/gate_a2/edits/edits.jsonl \\
        --required results/a2_anwm/<run>/required_tilt.json \\
        --threshold 5.76 --out results/a2_anwm_score
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

from src.perception.frames import camera_to_body_pitched
from src.perception.gate import SINGLE_FRAME, clip_signal, kept
from src.physics.flatness import tilt_deg
from src.score.verdict import depicted_axis_world

ROOT = Path(__file__).resolve().parents[1]
K_DS = (0.0, 0.295)
DTS = (0.25, 0.667, 1.0)
NOISE_FLOOR_REF_DEG = 0.57
REGIMES_NOT_EXPECTED_TESTABLE = {"B", "A15"}


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing input: {path}")
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def by_clip(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["clip_id"], []).append(r)
    return out


def frame_arrays(rows: list[dict]) -> dict:
    up = np.array([r["gravity_up_cam"] for r in rows], dtype=np.float64)
    return {"up": up,
            "gravity_uncertainty_raw": np.array([r.get("gravity_uncertainty_raw", np.nan)
                                                 for r in rows], dtype=np.float64),
            "up_confidence_mean": np.array([r.get("up_confidence_mean", np.nan)
                                            for r in rows], dtype=np.float64),
            "latitude_confidence_mean": np.array([r.get("latitude_confidence_mean", np.nan)
                                                  for r in rows], dtype=np.float64)}


def depicted_tilt_deg(up: np.ndarray, uptilt_deg: float = 0.0) -> np.ndarray:
    """Tilt of the body z axis from world up, per frame, through the recorded mount.

    The optical axis is not the body axis: an exactly level ANWM camera looks 90 deg away
    from up, so measuring the optical axis would report 90 deg for a level frame. This is
    the verifier's own construction (gravity -> body attitude -> body z); yaw is arbitrary
    because tilt does not depend on it. Checked against the Gate 3 noise floor: the 186
    level frames give median 0.579 deg, p99 4.17 deg, matching the 0.57 / 4.20 reported.
    """
    up = np.asarray(up, dtype=np.float64).reshape(-1, 3)
    R_cb = camera_to_body_pitched(np.deg2rad(float(uptilt_deg)))
    z = depicted_axis_world(up, R_cb, np.zeros(len(up)))
    return tilt_deg(z)


def gate_of(cfg: dict, rows: list[dict]) -> tuple[float, bool]:
    if not SINGLE_FRAME[cfg["signal"]]:
        raise SystemExit(f"{cfg['signal']} is not a single-frame signal; amendment 2 forbids "
                         "freezing it, so this config is invalid")
    arrs = frame_arrays(rows)
    val = clip_signal(cfg["signal"], arrs, dt_s=None)
    return val, bool(kept(val, float(cfg["tau"])))


def transfer_check(cfg: dict, floor_rows: list[dict]) -> dict:
    """Apply the frozen gate to released (level) ground-truth frames, 16 per clip."""
    groups: dict[str, list[dict]] = {}
    for r in floor_rows:
        groups.setdefault(r["traj"], []).append(r)
    n_clips = n_dropped = 0
    drops = []
    for traj, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["frame"])
        for i in range(0, max(0, len(rows) - 15), 16):
            chunk = rows[i:i + 16]
            if len(chunk) < 16:
                continue
            val, ok = gate_of(cfg, chunk)
            n_clips += 1
            n_dropped += int(not ok)
            drops.append({"traj": traj, "start": i, "signal": val, "kept": ok})
    frac = n_dropped / n_clips if n_clips else None
    return {"n_clips": n_clips, "n_dropped": n_dropped, "dropped_fraction": frac,
            "clips": drops,
            "verdict": ("gate not transferable to 224 px frames"
                        if frac is not None and frac > 0.5 else "gate transferable")}


def noise_floor(clips: dict[str, list[dict]], meta: dict[str, dict]) -> dict:
    cruise = [cid for cid in clips
              if (meta.get(cid, {}).get("maneuver") == "cruise")]
    seeds = [cid for cid in clips if cid.endswith("_s1") or cid.endswith("_s2")]
    out = {"cruise_clips": {}, "seed_repeats": {}}
    vals = []
    for cid in sorted(cruise):
        d = depicted_tilt_deg(frame_arrays(clips[cid])["up"])
        d = d[np.isfinite(d)]
        out["cruise_clips"][cid] = {"n": int(d.size), "median_deg": float(np.median(d)),
                                    "p90_deg": float(np.percentile(d, 90))}
        vals.append(float(np.median(d)))
    allv = []
    for cid in sorted(seeds):
        d = depicted_tilt_deg(frame_arrays(clips[cid])["up"])
        d = d[np.isfinite(d)]
        out["seed_repeats"][cid] = {"n": int(d.size), "median_deg": float(np.median(d))}
        allv.append(float(np.median(d)))
    out["cruise_median_of_medians_deg"] = float(np.median(vals)) if vals else None
    out["seed_median_of_medians_deg"] = float(np.median(allv)) if allv else None
    out["reference_gt_floor_deg"] = NOISE_FLOOR_REF_DEG
    out["seed_spread_of_medians_deg"] = float(np.std(allv)) if allv else None
    return out


def edit_recovery(rows: list[dict]) -> dict:
    """Recovered tilt change against the applied rotation, per axis and angle."""
    by: dict[tuple[str, float], list[float]] = {}
    per_clip: dict[str, dict[tuple[str, float], list[float]]] = {}
    for r in rows:
        d = depicted_tilt_deg(np.array([r["gravity_up_cam"]], dtype=np.float64))[0]
        if not np.isfinite(d):
            continue
        key = (r["axis"], float(r["applied_deg"]))
        by.setdefault(key, []).append(float(d))
        per_clip.setdefault(r["clip_id"], {}).setdefault(key, []).append(float(d))
    base = {cid: np.median(np.array(v[("none", 0.0)]))
            for cid, v in per_clip.items() if ("none", 0.0) in v}
    out: dict = {"conditions": {}, "stop_rule": {}}
    for (axis, ang), vals in sorted(by.items()):
        if axis == "none":
            continue
        rec = []
        for cid, v in per_clip.items():
            if (axis, ang) in v and cid in base:
                # change relative to the same clip's identity condition, not to zero
                rec.append(np.median(np.array(v[(axis, ang)])) - base[cid])
        rec = np.array(rec, dtype=np.float64)
        err = np.abs(rec - ang)
        out["conditions"][f"{axis}_{ang:g}"] = {
            "n_clips": int(len(rec)), "n_frames": len(vals),
            "applied_deg": ang,
            "recovered_change_median_deg": float(np.median(rec)) if rec.size else None,
            "median_abs_error_deg": float(np.median(err)) if err.size else None,
            "frac_within_3deg": float(np.mean(err <= 3.0)) if err.size else None,
        }
    def mae(prefix: str) -> float | None:
        xs = [v["median_abs_error_deg"] for k, v in out["conditions"].items()
              if k.startswith(prefix) and v["median_abs_error_deg"] is not None]
        return float(np.mean(xs)) if xs else None
    m10 = ([v["median_abs_error_deg"] for k, v in out["conditions"].items()
            if k.endswith("_10")])
    m20 = ([v["median_abs_error_deg"] for k, v in out["conditions"].items()
            if k.endswith("_20")])
    out["stop_rule"]["mae_10_deg"] = float(np.max(m10)) if m10 else None
    out["stop_rule"]["mae_20_deg"] = float(np.max(m20)) if m20 else None
    out["stop_rule"]["passed"] = bool(
        (out["stop_rule"]["mae_10_deg"] is not None)
        and (out["stop_rule"]["mae_20_deg"] is not None)
        and out["stop_rule"]["mae_10_deg"] <= 3.0
        and out["stop_rule"]["mae_20_deg"] <= 5.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depicted", default="results/gate_a2/anwm/depicted.jsonl")
    ap.add_argument("--floor", default="results/gate_a2/floor/floor_224.jsonl")
    ap.add_argument("--edits", default="results/gate_a2/edits/edits.jsonl")
    ap.add_argument("--required", required=True)
    ap.add_argument("--gate-config", default="configs/gate_frozen.yaml")
    ap.add_argument("--clips", default="data/processed/anwm_clips")
    ap.add_argument("--threshold", type=float, required=True,
                    help="verdict threshold refit under amendment 2 (Task 2 run)")
    ap.add_argument("--out", default="results/a2_anwm_score")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.gate_config).read_text())
    req = json.loads(Path(args.required).read_text())
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + git_hash()
    out = ROOT / args.out / run
    out.mkdir(parents=True, exist_ok=True)

    depicted = by_clip(load_jsonl(Path(args.depicted)))
    floor_rows = load_jsonl(Path(args.floor))
    edit_rows = load_jsonl(Path(args.edits))

    meta = {}
    for mf in sorted((ROOT / args.clips).glob("*/*/manifest.json")):
        m = json.loads(mf.read_text())
        meta[m["id"]] = {"regime": m["regime"], "maneuver": (m.get("meta") or {}).get("maneuver"),
                         "policy": (m.get("meta") or {}).get("policy"),
                         "designed": m.get("designed_required_tilt_deg"),
                         "step": m["commanded_path"]["step_spacing_m"],
                         "visual": m.get("visual_integrity") or {}}

    # ---- controls first: they decide whether anything below may be interpreted
    transfer = transfer_check(cfg, floor_rows)
    floor_stats = noise_floor(depicted, meta)
    edits = edit_recovery(edit_rows)
    controls = {"gate_transfer": transfer, "noise_floor": floor_stats, "edit_recovery": edits}
    (out / "controls.json").write_text(json.dumps(controls, indent=1, default=float))

    interpret = bool(edits["stop_rule"]["passed"]) and transfer["dropped_fraction"] is not None \
        and transfer["dropped_fraction"] <= 0.5

    # ---- per clip: depicted side, gate, testability
    per_clip: dict[str, dict] = {}
    for cid, rows in depicted.items():
        arrs = frame_arrays(rows)
        tilt = depicted_tilt_deg(arrs["up"])
        val, ok = gate_of(cfg, rows)
        vis = meta.get(cid, {}).get("visual", {})
        med_std = float(np.median(vis["frame_std"])) if vis.get("frame_std") else np.nan
        med_td = float(np.median(vis["temporal_delta"])) if vis.get("temporal_delta") else np.nan
        collapsed = bool((np.isfinite(med_std) and med_std < 0.05)
                         or (np.isfinite(med_td) and med_td < 0.02))
        per_clip[cid] = {
            "regime": meta.get(cid, {}).get("regime"), "maneuver": meta.get(cid, {}).get("maneuver"),
            "policy": meta.get(cid, {}).get("policy"),
            "designed_tilt_deg": meta.get(cid, {}).get("designed"),
            "step_spacing_m": meta.get(cid, {}).get("step"),
            "depicted_median_deg": float(np.nanmedian(tilt)),
            "depicted_p90_deg": float(np.nanpercentile(tilt, 90)),
            "depicted_per_frame": [float(x) for x in tilt],
            "gate_signal": float(val) if np.isfinite(val) else None,
            "gate_kept": ok, "collapsed": collapsed,
            "frame_std_median": med_std, "temporal_delta_median": med_td,
        }

    # ---- required side, joined
    req_rows = {(r["clip_id"], r.get("dt_s"), r["k_d"]): r for r in req["rows"]}
    table = []
    for cid, pc in per_clip.items():
        for dt in DTS:
            for k_d in K_DS:
                r = req_rows.get((cid, dt, k_d))
                if r is None:
                    continue
                lower, upper = r["lower"], r["upper"]
                detect = lower["max_deg"] is not None and lower["max_deg"] >= 10.0
                testable = pc["gate_kept"] and not pc["collapsed"] and detect
                viol = (None if lower["median_deg"] is None
                        else lower["median_deg"] - pc["depicted_median_deg"])
                table.append({
                    "clip_id": cid, "regime": pc["regime"], "maneuver": pc["maneuver"],
                    "designed_tilt_deg": pc["designed_tilt_deg"], "dt_s": dt, "k_d": k_d,
                    "depicted_median_deg": pc["depicted_median_deg"],
                    "lower_median_deg": lower["median_deg"],
                    "lower_p90_deg": lower["p90_deg"], "lower_max_deg": lower["max_deg"],
                    "upper_median_deg": upper["median_deg"], "upper_max_deg": upper["max_deg"],
                    "violation_lower_median_deg": viol,
                    "near_floor": bool(pc["depicted_median_deg"]
                                       <= 3.0 * max(NOISE_FLOOR_REF_DEG,
                                                    floor_stats["cruise_median_of_medians_deg"]
                                                    or NOISE_FLOOR_REF_DEG)),
                    "gate_kept": pc["gate_kept"], "collapsed": pc["collapsed"],
                    "detectable": bool(detect), "testable": bool(testable),
                    "infeasible": bool(testable and viol is not None
                                       and viol > args.threshold),
                })
    tab = pd.DataFrame(table)
    tab.to_csv(out / "anwm_scored.csv", index=False)

    # ---- tables
    summary: dict = {"threshold_deg": args.threshold, "interpretable": interpret,
                     "gate": cfg, "n_clips": len(per_clip)}

    if interpret:
        # Regime A share, per dt and k_d
        share = {}
        for dt in DTS:
            for k_d in K_DS:
                s = tab[(tab.dt_s == dt) & (tab.k_d == k_d)]
                a = s[s.regime == "A"]
                share[f"dt{dt}_kd{k_d}"] = {
                    "n_A": int(len(a)),
                    "n_testable": int(a.testable.sum()),
                    "n_infeasible": int(a.infeasible.sum()),
                    "n_infeasible_near_floor": int((a.infeasible & a.near_floor).sum()),
                    "share_infeasible_of_testable": (float(a.infeasible.sum() / a.testable.sum())
                                                     if a.testable.sum() else None),
                    "share_infeasible_of_all_A": (float(a.infeasible.sum() / len(a))
                                                  if len(a) else None),
                }
        summary["regime_A_share"] = share

        # Regime C dose-response (1 s, both k_d)
        dose = {}
        for k_d in K_DS:
            s = tab[(tab.dt_s == 1.0) & (tab.k_d == k_d) & (tab.regime == "C")]
            g = s.groupby("designed_tilt_deg").agg(
                n=("clip_id", "size"),
                depicted_median=("depicted_median_deg", "median"),
                depicted_p10=("depicted_median_deg", lambda x: np.percentile(x, 10)),
                depicted_p90=("depicted_median_deg", lambda x: np.percentile(x, 90)),
                required_lower_median=("lower_median_deg", "median"),
                n_testable=("testable", "sum"),
                n_infeasible=("infeasible", "sum"))
            dose[f"kd_{k_d}"] = json.loads(g.reset_index().to_json(orient="records"))
        summary["regime_C_dose_response"] = dose

        # minimum consistent dt
        dtmin = {}
        for k_d in K_DS:
            vals = []
            for r in req["rows"]:
                if r.get("dt_s") is not None or r["k_d"] != k_d:
                    continue
                cid = r["clip_id"]
                pc = per_clip.get(cid)
                if pc is None or not pc["gate_kept"] or pc["collapsed"]:
                    continue
                sweep = np.array(r["sweep_lower_median_deg"], dtype=np.float64)
                dts = np.array(r["sweep_dt"])
                ok = np.abs(sweep - pc["depicted_median_deg"]) <= args.threshold
                if not ok.any():
                    vals.append({"clip_id": cid, "regime": pc["regime"],
                                 "dt_min_s": None, "note": ">5.00"})
                    continue
                i = int(np.argmax(ok))
                dt_min = float(dts[i])
                note = "<=0.10" if i == 0 else ""
                vals.append({"clip_id": cid, "regime": pc["regime"], "dt_min_s": dt_min,
                             "implied_speed_ms": (pc["step_spacing_m"] / dt_min
                                                  if dt_min else None), "note": note})
            d = pd.DataFrame(vals)
            dist = {}
            for regime in ("A", "B", "C"):
                dd = d[d.regime == regime] if len(d) else d
                dist[regime] = {"n": int(len(dd)),
                                "n_none": int(dd.dt_min_s.isna().sum()) if len(dd) else 0,
                                "median": float(dd.dt_min_s.median()) if len(dd) and
                                dd.dt_min_s.notna().any() else None,
                                "p10": float(dd.dt_min_s.quantile(0.1)) if len(dd) and
                                dd.dt_min_s.notna().any() else None,
                                "p90": float(dd.dt_min_s.quantile(0.9)) if len(dd) and
                                dd.dt_min_s.notna().any() else None}
            dtmin[f"kd_{k_d}"] = {"distribution": dist, "per_clip": vals}
            if len(d):
                d.to_csv(out / f"dt_min_kd{k_d}.csv", index=False)
        summary["minimum_consistent_dt"] = dtmin
    else:
        summary["not_interpreted_reason"] = (
            "positive control failed or gate not transferable; see controls.json")

    # ---- testability bookkeeping (always reported)
    expect = {}
    for cid, pc in per_clip.items():
        key = "B" if pc["regime"] == "B" else ("A15" if cid == "A15" else None)
        if key:
            expect[cid] = {"expected_not_testable": True,
                           "gate_kept": pc["gate_kept"],
                           "collapsed": pc["collapsed"],
                           "depicted_median_deg": pc["depicted_median_deg"]}
    # a clip is "not testable" for one of three reasons; keep them apart so a gate drop is
    # never confused with "the path asks for nothing detectable"
    for cid, row in expect.items():
        for k_d in K_DS:
            at = tab[(tab.clip_id == cid) & (tab.dt_s == 1.0) & (tab.k_d == k_d)]
            row[f"detectable_at_dt1_kd{k_d}"] = (bool(at.detectable.iloc[0])
                                                 if len(at) else None)
    summary["expected_not_testable"] = expect
    summary["testable_counts"] = {
        "gate_kept_and_not_collapsed_by_regime": {
            r: int(sum(1 for p in per_clip.values() if p["regime"] == r
                       and p["gate_kept"] and not p["collapsed"])) for r in ("A", "B", "C")},
        "detectable_at_dt1_by_regime": {
            f"kd_{k_d}": {r: int(tab[(tab.regime == r) & (tab.dt_s == 1.0)
                                    & (tab.k_d == k_d)].detectable.sum()) for r in ("A", "B", "C")}
            for k_d in K_DS},
        "gate_dropped": int(sum(1 for p in per_clip.values() if not p["gate_kept"])),
        "collapsed": int(sum(1 for p in per_clip.values() if p["collapsed"])),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (out / "per_clip.json").write_text(json.dumps(per_clip, indent=1, default=float))
    print(json.dumps({k: summary[k] for k in ("threshold_deg", "interpretable",
                                              "testable_counts")}, indent=2, default=float))
    if interpret:
        print("regime A share:", json.dumps(summary["regime_A_share"], indent=1))
    else:
        print("NOT INTERPRETED:", summary.get("not_interpreted_reason"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
