"""Track D: evaluate the gain statistic against the current median-residual statistic.

Protocol is pre-registered in ``configs/gain_prereg.yaml`` and the targets in
``configs/prototype_acceptance_spec.yaml``.  This script only measures.

    PYTHONPATH=. pixi run python experiments/d_eval.py --tag d01

Outputs ``results/d_eval/<tag>/``: metrics.json, per-clip CSVs, operating curves,
ablations, the synthetic gain sweep and plots.  ``--reuse DIR`` reloads a previous run's
fitted results instead of refitting.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.perception.gate import clip_signal, kept
from src.score.verdict import (clip_index, conformal_quantile, fit_nuisances,
                               score_clip, shift_series, speed_bin,
                               wilson_interval)
from src.verdict import (Bands, FitOptions, GainResult, assess, fit_bands, fit_gain,
                         flag, gate_kept_from_conf, load_clip)
from src.verdict.decide import SIMPLICITY, conformal_band, fit_joint_band
from src.physics.flatness import tilt_deg
from src.datasets.tii_video import flight_parts
from experiments.a2_verdict_gated import axis_at, mocap_derivs

ROOT = Path(__file__).resolve().parents[1]
FLIGHTS = ["flight-01a-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
           "flight-13a-trackRATM", "flight-01p-ellipse", "flight-07p-lemniscate",
           "flight-13p-trackRATM", "flight-16p-trackRATM"]
TRACK = {"flight-13a-trackRATM", "flight-13p-trackRATM", "flight-16p-trackRATM"}
CONDITIONS = ["real", "null_edit", "tilt_removal", "scale_0.5", "scale_1.5",
              "mirror_roll", "yaw_control", "delay_ms:100", "delay_ms:200",
              "delay_ms:300"]
EDITS = ["tilt_removal", "scale_0.5", "scale_1.5", "mirror_roll", "yaw_control"]
RULES = ("R1_scalar_g1", "R2_axis_gains", "R3_scalar_or_axis")
STORED = {"real": "identity", "null_edit": "identity", "tilt_removal": "tilt_k0",
          "scale_0.5": "tilt_k05", "scale_1.5": "tilt_k15",
          "mirror_roll": "mirror_roll", "yaw_control": "yaw_control"}


# ------------------------------------------------------------------ io helpers
def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def jload(p: Path):
    return json.loads(Path(p).read_text())


def load_all_clips(root: Path) -> dict[tuple[str, str, str], object]:
    out = {}
    for path in sorted(root.glob("*/*/verdict_clip.npz")):
        clip = load_clip(path)
        out[(clip.flight, clip.clip_id, clip.condition)] = clip
    return out


def result_from_dict(d: dict) -> GainResult:
    fields = set(GainResult.__dataclass_fields__)
    return GainResult(**{k: v for k, v in d.items() if k in fields})


# ------------------------------------------------------------------ gain fitting
def fit_all(clips: dict, opts: FitOptions, conditions: list[str] | None = None,
            quiet: bool = False) -> dict[tuple[str, str, str], GainResult]:
    keys = [k for k, c in clips.items() if conditions is None or c.condition in conditions]
    out = {}
    for i, key in enumerate(keys):
        out[key] = fit_gain(clips[key], opts)
        if not quiet and i and i % 100 == 0:
            print(f"  fitted {i}/{len(keys)}", flush=True)
    return out


def save_results(path: Path, results: dict) -> None:
    path.write_text(json.dumps({f"{k[0]}|{k[1]}|{k[2]}": r.as_dict()
                                for k, r in results.items()}, indent=1, default=float))


def load_results(path: Path) -> dict:
    raw = jload(path)
    out = {}
    for key, d in raw.items():
        f, cid, cond = key.split("|")
        out[(f, cid, cond)] = result_from_dict(d)
    return out


def real_testable(clips: dict, results: dict, abstain_cfg: dict,
                  gate_cfg: dict | None) -> dict[tuple[str, str], dict]:
    """Pair-level testability: the REAL clip's decision is inherited by its edits."""
    out = {}
    for (flight, cid, cond), clip in clips.items():
        if cond != "real":
            continue
        res = results[(flight, cid, "real")]
        gk = gate_kept_from_conf(clip.depicted_conf, gate_cfg) if gate_cfg else True
        ab = assess(res, abstain_cfg, gate_kept=gk)
        out[(flight, cid)] = {"testable": ab.testable, "reasons": ab.reasons,
                             "gate_kept": gk, "flight": flight}
    return out


def _real_results(results: dict, testable: dict, flights: list[str]) -> list[GainResult]:
    out = []
    for (flight, cid, cond), res in results.items():
        if cond != "real" or flight not in flights:
            continue
        if testable.get((flight, cid), {}).get("testable"):
            out.append(res)
    return out


# ------------------------------------------------------------------ evaluation
def gain_operating_point(results: dict, testable: dict, rule: str, alpha: float
                         ) -> dict:
    """Leave-one-flight-out held-out rates at one conformal band."""
    frr, det = [], {c: [] for c in EDITS}
    fold_thresholds = []
    for flight in FLIGHTS:
        fit = _real_results(results, testable, [f for f in FLIGHTS if f != flight])
        bands = fit_bands(fit, rule, alpha)
        fold_thresholds.append({"flight": flight,
                                "scalar": bands.scalar.as_dict() if bands.scalar else None,
                                "axis": bands.axis.as_dict() if bands.axis else None,
                                "joint": bands.joint.as_dict() if bands.joint else None})
        for (f, cid, cond), res in results.items():
            if f != flight or not testable.get((f, cid), {}).get("testable"):
                continue
            if cond == "real":
                frr.append(bool(flag(res, rule, bands)))
            elif cond in EDITS:
                det[cond].append(bool(flag(res, rule, bands)))
    return {"frr": frr, "det": det, "fold_thresholds": fold_thresholds}


def gain_curve(results: dict, testable: dict, rule: str, alphas: np.ndarray) -> dict:
    rows = []
    for alpha in alphas:
        op = gain_operating_point(results, testable, rule, float(alpha))
        rows.append({"alpha": float(alpha), "frr": _rate(op["frr"]),
                     **{f"det_{c}": _rate(op["det"][c]) for c in EDITS}})
    return {"rows": rows}


def baseline_records(prereg: dict, gate_cfg: dict) -> list[dict]:
    """The current median-residual statistic at the primary mode (drag-free, lag-only).

    Copied structurally from ``experiments/a2_modes.py`` but keeping ``gate_kept`` and the
    excitation testability separate, so pair-level gate inheritance can be applied.
    """
    clips = jload(ROOT / "data/processed/p3a/p3a_v1/clips.json")["clips"]
    out = []
    for flight in FLIGHTS:
        spec = np.load(ROOT / f"data/processed/p3a/p3a_v1/{flight}.npz", allow_pickle=False)
        control, _ = flight_parts(flight)
        d = mocap_derivs(ROOT / f"data/raw/tii/{control}/mocap_{flight}.csv")
        axis = axis_at(d, spec["t"], 0.0)
        tilt = tilt_deg(axis)
        from scipy.spatial.transform import Rotation
        R_bw = Rotation.from_quat(spec["quat"]).as_matrix()
        yaw = Rotation.from_matrix(R_bw).as_euler("ZYX")[:, 0]
        names = [str(n) for n in spec["names"]]
        rows_by_cond = {}
        for stored in set(STORED.values()):
            path = ROOT / f"results/gate_a2/tii/{flight}__{stored}.jsonl"
            rows_by_cond[stored] = {r["frame"]: r for r in
                                    (json.loads(x) for x in path.read_text().splitlines()
                                     if x.strip())}
        up_by_cond, valid_by_cond = {}, {}
        for stored, rows in rows_by_cond.items():
            up_by_cond[stored] = np.stack([
                np.array(rows[n]["gravity_up_cam"]) if n in rows else np.full(3, np.nan)
                for n in names])
            valid_by_cond[stored] = np.array(
                [float(rows[n].get("valid_frac", 1.0)) >= 0.5 if n in rows else False
                 for n in names])
        for clip in [c for c in clips if c["flight"] == flight]:
            idx = clip_index(clip)
            t = spec["t"][idx]
            for schema_cond in ["real", "tilt_removal", "scale_0.5", "scale_1.5",
                                "mirror_roll", "yaw_control"]:
                stored = STORED[schema_cond]
                rows = rows_by_cond[stored]
                up, vv = up_by_cond[stored][idx], valid_by_cond[stored][idx]
                finite = np.isfinite(up).all(axis=1) & np.isfinite(axis[idx]).all(axis=1)
                mask = finite & (tilt[idx] >= 10.0)
                nu = fit_nuisances(up, axis[idx], yaw[idx], t, float(spec["uptilt_deg"]),
                                   mask, prereg, uptilt_offset_override=0.0)
                r = score_clip(up, axis[idx], yaw[idx], t, float(spec["uptilt_deg"]),
                               tilt[idx], vv, prereg, nuisances=nu)
                gk = _gate_clip(rows, names, gate_cfg, idx)
                out.append({"flight": flight, "clip_id": clip["clip_id"],
                            "condition": schema_cond, "score": r.get("score"),
                            "testable_excitation": bool(r.get("testable")),
                            "gate_kept": gk, "bin": speed_bin(
                                clip["speed_median"], prereg["thresholds"]["speed_bins_mps"]),
                            "track": clip["track"], "control": clip["control"],
                            "speed": clip["speed_median"],
                            "tilt_implied_median": float(np.nanmedian(tilt[idx]))})
            # delays: the video lags the path, modelled by shifting the implied axis back
            for delay in (100, 200, 300):
                stored = STORED["real"]
                rows = rows_by_cond[stored]
                up, vv = up_by_cond[stored][idx], valid_by_cond[stored][idx]
                imp = shift_series(axis[idx], t, -delay / 1000.0)
                finite = np.isfinite(up).all(axis=1) & np.isfinite(imp).all(axis=1)
                mask = finite & (tilt[idx] >= 10.0)
                nu = fit_nuisances(up, imp, yaw[idx], t, float(spec["uptilt_deg"]),
                                   mask, prereg, uptilt_offset_override=0.0)
                r = score_clip(up, imp, yaw[idx], t, float(spec["uptilt_deg"]),
                               tilt[idx], vv, prereg, nuisances=nu)
                gk = _gate_clip(rows, names, gate_cfg, idx)
                out.append({"flight": flight, "clip_id": clip["clip_id"],
                            "condition": f"delay_ms:{delay}", "score": r.get("score"),
                            "testable_excitation": bool(r.get("testable")),
                            "gate_kept": gk, "bin": speed_bin(
                                clip["speed_median"], prereg["thresholds"]["speed_bins_mps"]),
                            "track": clip["track"], "control": clip["control"],
                            "speed": clip["speed_median"],
                            "tilt_implied_median": float(np.nanmedian(tilt[idx]))})
    return out


def _gate_clip(rows: dict, names: list[str], cfg: dict, idx: np.ndarray) -> bool:
    arrs = {
        "up": np.stack([np.array(rows[n]["gravity_up_cam"]) if n in rows
                        else np.full(3, np.nan) for n in names]),
        "up_confidence_mean": np.array([rows[n].get("up_confidence_mean", np.nan)
                                        if n in rows else np.nan for n in names]),
        "latitude_confidence_mean": np.array(
            [rows[n].get("latitude_confidence_mean", np.nan) if n in rows else np.nan
             for n in names]),
        "gravity_uncertainty_raw": np.array(
            [rows[n].get("gravity_uncertainty_raw", np.nan) if n in rows else np.nan
             for n in names]),
    }
    arrs = {k: v[idx] for k, v in arrs.items()}
    return bool(kept(clip_signal(cfg["signal"], arrs), cfg["tau"]))


def baseline_testable(records: list[dict]) -> dict[tuple[str, str], dict]:
    """Pair-level: real clip must pass its own gate and the excitation rule."""
    out = {}
    for r in records:
        if r["condition"] != "real":
            continue
        ok = bool(r["testable_excitation"] and r["gate_kept"] and
                  np.isfinite(r.get("score", np.nan)))
        out[(r["flight"], r["clip_id"])] = {"testable": ok, "flight": r["flight"]}
    return out


def baseline_operating_point(records: list[dict], testable: dict, alpha: float) -> dict:
    real = {(r["flight"], r["clip_id"]): r for r in records if r["condition"] == "real"}
    frr, det = [], {c: [] for c in EDITS}
    for flight in FLIGHTS:
        fit = [r for (f, cid), r in real.items()
               if f != flight and testable.get((f, cid), {}).get("testable")]
        thr = conformal_quantile([r["score"] for r in fit], alpha)
        for r in records:
            key = (r["flight"], r["clip_id"])
            if r["flight"] != flight or not testable.get(key, {}).get("testable"):
                continue
            if not np.isfinite(r.get("score", np.nan)):
                continue
            hit = bool(r["score"] > thr)
            if r["condition"] == "real":
                frr.append(hit)
            elif r["condition"] in EDITS:
                det[r["condition"]].append(hit)
    return {"frr": frr, "det": det, "thresholds": None}


def baseline_curve(records: list[dict], testable: dict, alphas: np.ndarray) -> dict:
    rows = []
    for alpha in alphas:
        op = baseline_operating_point(records, testable, float(alpha))
        rows.append({"alpha": float(alpha), "frr": _rate(op["frr"]),
                     **{f"det_{c}": _rate(op["det"][c]) for c in EDITS}})
    return {"rows": rows}


def _rate(values: list[bool]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _wilson(values: list[bool]) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    return list(wilson_interval(int(np.sum(values)), len(values)))


# ------------------------------------------------------------------ selection
def _criterion(det_removal: float, det_mirror: float, frr: float) -> tuple:
    under = frr <= 0.10
    detection = np.nanmean([det_removal, det_mirror])
    return (0 if under else 1, -detection if under else frr, -detection)


def select_rule(results: dict, testable: dict, fit_flights: list[str]) -> dict:
    """Inner LOFO over the fit flights only; choose by the pre-registered criterion."""
    scores = {}
    for rule in RULES:
        frr, det = [], {c: [] for c in EDITS}
        for inner in fit_flights:
            inner_fit = [f for f in fit_flights if f != inner]
            fit = _real_results(results, testable, inner_fit)
            if len(fit) < 8:
                continue
            bands = fit_bands(fit, rule, 0.05)
            for (f, cid, cond), res in results.items():
                if f != inner or not testable.get((f, cid), {}).get("testable"):
                    continue
                if cond == "real":
                    frr.append(bool(flag(res, rule, bands)))
                elif cond in EDITS:
                    det[cond].append(bool(flag(res, rule, bands)))
        frr_rate = _rate(frr)
        scores[rule] = {"frr": frr_rate, "n_frr": len(frr),
                        **{f"det_{c}": _rate(det[c]) for c in EDITS}}
    ranked = sorted(RULES, key=lambda r: _criterion(
        scores[r]["det_tilt_removal"], scores[r]["det_mirror_roll"], scores[r]["frr"]))
    return {"selected": ranked[0], "scores": scores}


def lofo_with_selection(results: dict, testable: dict) -> dict:
    fold_rules, frr, det = [], [], {c: [] for c in EDITS}
    for flight in FLIGHTS:
        fit_flights = [f for f in FLIGHTS if f != flight]
        sel = select_rule(results, testable, fit_flights)
        rule = sel["selected"]
        fit = _real_results(results, testable, fit_flights)
        bands = fit_bands(fit, rule, 0.05)
        fold_rules.append({"held_out": flight, "rule": rule,
                           "inner": sel["scores"]})
        for (f, cid, cond), res in results.items():
            if f != flight or not testable.get((f, cid), {}).get("testable"):
                continue
            if cond == "real":
                frr.append(bool(flag(res, rule, bands)))
            elif cond in EDITS:
                det[cond].append(bool(flag(res, rule, bands)))
    return {"fold_rules": fold_rules, "frr": frr, "det": det}


# ------------------------------------------------------------------ synthetic sweep
def synthetic_sweep(clips: dict, results: dict, testable: dict, rule: str,
                    alpha: float = 0.05) -> list[dict]:
    ks = [0.0, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0]
    rows = []
    fit = _real_results(results, testable, FLIGHTS)
    bands = fit_bands(fit, rule, alpha)
    for (flight, cid, cond), clip in clips.items():
        if cond != "real" or not testable.get((flight, cid), {}).get("testable"):
            continue
        res = results[(flight, cid, "real")]
        b = np.asarray(res.intercept, dtype=np.float64)
        for k in ks:
            syn = replace(clip)
            syn.depicted_lean = b + k * (clip.depicted_lean - b)
            # no bootstrap: the abstention set is fixed to the real clip by design
            r = fit_gain(syn, FitOptions(bootstrap=False))
            rows.append({"flight": flight, "clip_id": cid, "k": k,
                         "gain": r.gain, "gx": r.axis_gains[0],
                         "gy": r.axis_gains[1], "flagged": bool(flag(r, rule, bands))})
    return rows


# ------------------------------------------------------------------ ablations
def ablation_rows(clips: dict, abstain_cfg: dict, gate_cfg: dict) -> list[dict]:
    variants = {
        "primary": FitOptions(),
        "drag_free": FitOptions(drag="free"),
        "drag_fixed_0.295": FitOptions(drag="fixed"),
        "no_intercept": FitOptions(intercept=False),
        "huber": FitOptions(estimator="huber"),
        "smoothing_0.25s": FitOptions(smoothing_s=0.25),
        "no_lag_search": FitOptions(lag_search=False),
    }
    conds = ["real", "tilt_removal", "scale_0.5", "scale_1.5", "mirror_roll",
             "yaw_control"]
    rows = []
    for name, base in variants.items():
        opts = FitOptions(**{**base.__dict__, "bootstrap": False})
        print(f"  ablation {name}", flush=True)
        res = fit_all(clips, opts, conditions=conds, quiet=True)
        # testability is held fixed at the primary real decision
        for rule in ("R1_scalar_g1", "R3_scalar_or_axis"):
            frr, det = [], {c: [] for c in EDITS}
            for flight in FLIGHTS:
                fit = _real_results(res, PRIMARY_TESTABLE, [f for f in FLIGHTS
                                                            if f != flight])
                if len(fit) < 8:
                    continue
                bands = fit_bands(fit, rule, 0.05)
                for (f, cid, cond), r in res.items():
                    if f != flight or not PRIMARY_TESTABLE.get((f, cid), {}).get("testable"):
                        continue
                    if cond == "real":
                        frr.append(bool(flag(r, rule, bands)))
                    elif cond in EDITS:
                        det[cond].append(bool(flag(r, rule, bands)))
            rows.append({"variant": name, "rule": rule, "frr": _rate(frr),
                         "n_frr": len(frr),
                         **{f"det_{c}": _rate(det[c]) for c in EDITS},
                         **{f"n_{c}": len(det[c]) for c in EDITS}})
    return rows


# ------------------------------------------------------------------ reporting
def _group_summary(results: dict, testable: dict, rule: str, alpha: float = 0.05
                   ) -> dict:
    op = gain_operating_point(results, testable, rule, alpha)
    by_flight = {}
    for flight in FLIGHTS:
        frr, det = [], {c: [] for c in EDITS}
        fit = _real_results(results, testable, [f for f in FLIGHTS if f != flight])
        bands = fit_bands(fit, rule, alpha)
        for (f, cid, cond), res in results.items():
            if f != flight or not testable.get((f, cid), {}).get("testable"):
                continue
            if cond == "real":
                frr.append(bool(flag(res, rule, bands)))
            elif cond in EDITS:
                det[cond].append(bool(flag(res, rule, bands)))
        by_flight[flight] = {"frr": _rate(frr), "n_real": len(frr),
                             **{f"det_{c}": _rate(det[c]) for c in EDITS},
                             **{f"n_{c}": len(det[c]) for c in EDITS}}
    return {"frr": _rate(op["frr"]), "frr_wilson": _wilson(op["frr"]),
            "n_real": len(op["frr"]),
            **{f"det_{c}": _rate(op["det"][c]) for c in EDITS},
            **{f"det_{c}_wilson": _wilson(op["det"][c]) for c in EDITS},
            **{f"n_{c}": len(op["det"][c]) for c in EDITS},
            "by_flight": by_flight}


def _interp_at(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    order = np.argsort(x[ok])
    return float(np.interp(xq, x[ok][order], y[ok][order]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/processed/track_d/clips")
    ap.add_argument("--out", default="results/d_eval")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--reuse", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="skip bootstrap and ablations (development only)")
    args = ap.parse_args()

    prereg = yaml.safe_load((ROOT / "configs/gain_prereg.yaml").read_text())
    verdict_prereg = yaml.safe_load(
        (ROOT / "configs/verdict_preregistration.yaml").read_text())
    gate_cfg = yaml.safe_load((ROOT / "configs/gate_frozen.yaml").read_text())
    abstain_cfg = prereg["abstention"]
    run = args.tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / args.out / run
    out.mkdir(parents=True, exist_ok=True)

    print("loading clips", flush=True)
    clips = load_all_clips(ROOT / args.clips)
    print(f"  {len(clips)} clip-condition files", flush=True)

    cache_dir = ROOT / args.reuse if args.reuse else None
    if cache_dir and (cache_dir / "primary_results.json").exists():
        print("reusing fitted results", flush=True)
        results = load_results(cache_dir / "primary_results.json")
    else:
        opts = FitOptions(bootstrap=not args.quick, n_boot=400)
        print("fitting primary statistic", flush=True)
        results = fit_all(clips, opts)
        save_results(out / "primary_results.json", results)

    global PRIMARY_TESTABLE
    PRIMARY_TESTABLE = real_testable(clips, results, abstain_cfg, gate_cfg)
    n_test = sum(1 for v in PRIMARY_TESTABLE.values() if v["testable"])
    print(f"  testable real clips: {n_test}/{len(PRIMARY_TESTABLE)}", flush=True)

    # candidate rules at the pre-registered operating point
    print("evaluating candidates", flush=True)
    candidates = {}
    for rule in RULES:
        candidates[rule] = _group_summary(results, PRIMARY_TESTABLE, rule)
        candidates[rule]["selection_inner"] = select_rule(
            results, PRIMARY_TESTABLE, FLIGHTS)
    selected = min(RULES, key=lambda r: _criterion(
        candidates[r]["det_tilt_removal"], candidates[r]["det_mirror_roll"],
        candidates[r]["frr"]))
    print(f"  pooled selection: {selected}", flush=True)
    lofo_sel = lofo_with_selection(results, PRIMARY_TESTABLE)

    # operating curves
    alphas = np.round(np.arange(0.0, 0.5001, 0.025), 4)
    print("operating curves", flush=True)
    curves = {rule: gain_curve(results, PRIMARY_TESTABLE, rule, alphas) for rule in RULES}

    # baseline
    print("baseline records", flush=True)
    base = baseline_records(verdict_prereg, gate_cfg)
    base_test = baseline_testable(base)
    base_summary = _group_summary_baseline(base, base_test, 0.05)
    base_curve = baseline_curve(base, base_test, alphas)

    # synthetic sweep and ablations
    print("synthetic sweep", flush=True)
    synth = synthetic_sweep(clips, results, PRIMARY_TESTABLE, selected)
    ablations = [] if args.quick else ablation_rows(clips, abstain_cfg, gate_cfg)

    # matched false-reject comparison.  The curves are discrete (a conformal threshold
    # moves to the next real score), so at a target false-reject rate we report the
    # range of detections over curve points within 2 points of that rate rather than a
    # single interpolated number.  The range is the honest statement.
    matched = {}
    for family in ("tilt_removal", "mirror_roll", "scale_0.5", "scale_1.5"):
        frr_gain = candidates[selected]["frr"]
        frr_base = base_summary["frr"]

        def _range(rows, target, tol=0.02):
            vals = [r[f"det_{family}"] for r in rows
                    if np.isfinite(r["frr"]) and abs(r["frr"] - target) <= tol]
            if not vals:
                return {"min": float("nan"), "max": float("nan"), "n_points": 0}
            return {"min": float(np.min(vals)), "max": float(np.max(vals)),
                    "n_points": len(vals)}

        matched[family] = {
            "gain_at_own_operating_point": candidates[selected][f"det_{family}"],
            "gain_frr": frr_gain,
            "baseline_at_gain_frr": _range(base_curve["rows"], frr_gain),
            "gain_at_baseline_frr": _range(curves[selected]["rows"], frr_base),
            "baseline_frr": frr_base,
            "baseline_own_operating_point": base_summary[f"det_{family}"],
        }

    # per-clip tables
    rows = []
    for (flight, cid, cond), res in results.items():
        rows.append({"flight": flight, "clip_id": cid, "condition": cond,
                     "track": "trackRATM" if flight in TRACK else "other",
                     "gain": res.gain, "gain_lo": res.gain_ci[0],
                     "gain_hi": res.gain_ci[1], "gx": res.axis_gains[0],
                     "gy": res.axis_gains[1], "drag": res.drag,
                     "r2": res.r2, "lag_ms": res.lag_ms,
                     "excitation_deg": res.excitation_deg,
                     "required_tilt_deg": res.required_tilt_deg,
                     "depicted_tilt_deg": res.depicted_tilt_deg,
                     "n_used": res.n_used, "cond_number": res.cond_number,
                     "testable_pair": PRIMARY_TESTABLE.get((flight, cid), {}).get(
                         "testable"),
                     "abstain_reasons": "; ".join(
                         PRIMARY_TESTABLE.get((flight, cid), {}).get("reasons", []))})
    pd.DataFrame(rows).to_csv(out / "gain_table.csv", index=False)
    pd.DataFrame([{**r, "testable_pair": base_test.get((r["flight"], r["clip_id"]),
                                                       {}).get("testable")}
                  for r in base]).to_csv(out / "baseline_table.csv", index=False)
    pd.DataFrame(ablations).to_csv(out / "ablations.csv", index=False)
    pd.DataFrame(synth).to_csv(out / "synthetic_sweep.csv", index=False)
    pd.DataFrame([{**r, "method": "gain"} for r in curves[selected]["rows"]]
                 + [{**r, "method": "baseline"} for r in base_curve["rows"]]
                 ).to_csv(out / "operating_curves.csv", index=False)

    metrics = {
        "run": run, "git": git_hash(),
        "prereg": {"path": "configs/gain_prereg.yaml",
                   "sha256": _sha(ROOT / "configs/gain_prereg.yaml")},
        "n_clip_files": len(clips), "n_testable_real": n_test,
        "candidates": candidates, "selected_pooled": selected,
        "lofo_selection": lofo_sel,
        "nested_summary": {"frr": _rate(lofo_sel["frr"]),
                           "frr_wilson": _wilson(lofo_sel["frr"]),
                           "n_real": len(lofo_sel["frr"]),
                           **{f"det_{c}": _rate(lofo_sel["det"][c]) for c in EDITS},
                           **{f"det_{c}_wilson": _wilson(lofo_sel["det"][c])
                              for c in EDITS},
                           **{f"n_{c}": len(lofo_sel["det"][c]) for c in EDITS}},
        "selected_summary": _group_summary(results, PRIMARY_TESTABLE, selected),
        "baseline": base_summary,
        "matched_false_reject": matched,
        "operating_curves": curves, "baseline_curve": base_curve,
        "synthetic_sweep": synth, "ablations": ablations,
        "abstention": {
            "overall": 1.0 - n_test / max(len(PRIMARY_TESTABLE), 1),
            "by_flight": {f: {"n": sum(1 for (ff, _), v in PRIMARY_TESTABLE.items()
                                     if ff == f),
                              "testable": sum(1 for (ff, _), v in PRIMARY_TESTABLE.items()
                                              if ff == f and v["testable"])}
                          for f in FLIGHTS},
            "reasons": _reason_counts(PRIMARY_TESTABLE),
        },
        "drag_summary": _drag_summary(results, PRIMARY_TESTABLE),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    (out / "table.md").write_text(_table_md(metrics))
    _plots(out, results, PRIMARY_TESTABLE, curves, base_curve, synth, selected)
    print("wrote", out, flush=True)
    return 0


# ------------------------------------------------------------------ summaries
PRIMARY_TESTABLE: dict = {}


def _group_summary_baseline(records: list[dict], testable: dict, alpha: float) -> dict:
    op = baseline_operating_point(records, testable, alpha)
    by_flight = {}
    for flight in FLIGHTS:
        fit = [r for r in records if r["condition"] == "real" and r["flight"] != flight
               and testable.get((r["flight"], r["clip_id"]), {}).get("testable")]
        thr = conformal_quantile([r["score"] for r in fit], alpha)
        frr, det = [], {c: [] for c in EDITS}
        for r in records:
            if r["flight"] != flight or not testable.get((r["flight"], r["clip_id"]),
                                                         {}).get("testable"):
                continue
            if not np.isfinite(r.get("score", np.nan)):
                continue
            hit = bool(r["score"] > thr)
            if r["condition"] == "real":
                frr.append(hit)
            elif r["condition"] in EDITS:
                det[r["condition"]].append(hit)
        by_flight[flight] = {"frr": _rate(frr), "n_real": len(frr),
                             **{f"det_{c}": _rate(det[c]) for c in EDITS},
                             **{f"n_{c}": len(det[c]) for c in EDITS}}
    return {"frr": _rate(op["frr"]), "frr_wilson": _wilson(op["frr"]),
            "n_real": len(op["frr"]),
            **{f"det_{c}": _rate(op["det"][c]) for c in EDITS},
            **{f"det_{c}_wilson": _wilson(op["det"][c]) for c in EDITS},
            **{f"n_{c}": len(op["det"][c]) for c in EDITS},
            "abstention": 1.0 - len(op["frr"]) / max(len(testable), 1),
            "by_flight": by_flight}


def _reason_counts(testable: dict) -> dict:
    counts: dict[str, int] = {}
    for v in testable.values():
        if v["testable"]:
            continue
        for reason in (v["reasons"] or ["unspecified"]):
            key = reason.split()[0]
            counts[key] = counts.get(key, 0) + 1
    return counts


def _drag_summary(results: dict, testable: dict) -> dict:
    out = {}
    for label, sel in (("all", lambda f: True), ("trackratm", lambda f: f in TRACK),
                       ("non_trackratm", lambda f: f not in TRACK),
                       ("piloted", lambda f: "p" in f.split("-")[1]),
                       ("autonomous", lambda f: "a" in f.split("-")[1])):
        d = [res.drag for (f, cid, cond), res in results.items()
             if cond == "real" and sel(f) and testable.get((f, cid), {}).get("testable")
             and np.isfinite(res.drag)]
        out[label] = {"n": len(d),
                      "median": float(np.median(d)) if d else float("nan"),
                      "iqr": [float(np.percentile(d, 25)), float(np.percentile(d, 75))]
                      if d else [float("nan"), float("nan")]}
    return out


def _sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _table_md(m: dict) -> str:
    def pct(x):
        return "—" if x is None or not np.isfinite(x) else f"{100 * x:.1f}"
    lines = ["# Track D gain-test evaluation", "",
             f"run `{m['run']}` · git `{m['git']}` · testable real clips "
             f"{m['n_testable_real']}", "",
             "## Candidate rules (pooled, held-out at alpha 0.05)", "",
             "| rule | FRR | removal | mirror | x0.5 | x1.5 | yaw | n real |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for rule in RULES:
        c = m["candidates"][rule]
        lines.append(f"| {rule} | {pct(c['frr'])} | {pct(c['det_tilt_removal'])} | "
                     f"{pct(c['det_mirror_roll'])} | {pct(c['det_scale_0.5'])} | "
                     f"{pct(c['det_scale_1.5'])} | {pct(c['det_yaw_control'])} | "
                     f"{c['n_real']} |")
    lines += ["", f"Pooled selection: **{m['selected_pooled']}**", "",
              "## Selected rule vs baseline", "",
              "| quantity | gain (pooled rule) | gain (nested selection) | baseline |",
              "| --- | --- | --- | --- |"]
    s, b = m["selected_summary"], m["baseline"]
    n = m.get("nested_summary", {})
    for key in ("frr", "det_tilt_removal", "det_mirror_roll", "det_scale_0.5",
                "det_scale_1.5", "det_yaw_control"):
        lines.append(f"| {key} | {pct(s[key])} | {pct(n.get(key))} | {pct(b[key])} |")
    lines += ["", "## Matched false-reject", "",
              "| family | gain @ own | baseline @ gain FRR | gain @ baseline FRR | baseline @ own | yaw control @ own |",
              "| --- | --- | --- | --- | --- | --- |"]
    for fam, v in m["matched_false_reject"].items():
        br = v["baseline_at_gain_frr"]
        gr = v["gain_at_baseline_frr"]
        lines.append(f"| {fam} | {pct(v['gain_at_own_operating_point'])} | "
                     f"{pct(br['min'])}-{pct(br['max'])} | "
                     f"{pct(gr['min'])}-{pct(gr['max'])} | "
                     f"{pct(v['baseline_own_operating_point'])} | "
                     f"{pct(m['selected_summary']['det_yaw_control'])} |")
    lines += ["", "Track flights are **not validated** in any table above.", "",
              "## Abstention", ""]
    a = m["abstention"]
    lines.append(f"overall {100 * a['overall']:.1f} % (baseline {100 * b['abstention']:.1f} %)")
    lines.append("")
    for f, v in a["by_flight"].items():
        lines.append(f"- {f}: {v['testable']}/{v['n']} testable")
    lines += ["", f"Abstention reasons: {a['reasons']}", "",
              "## Fitted drag (1/s, primary, real)", ""]
    for k, v in m["drag_summary"].items():
        lines.append(f"- {k}: median {v['median']:.3f} (IQR {v['iqr'][0]:.3f}-"
                     f"{v['iqr'][1]:.3f}, n={v['n']})")
    return "\n".join(lines) + "\n"


def _plots(out: Path, results: dict, testable: dict, curves: dict, base_curve: dict,
           synth: list[dict], selected: str) -> None:
    # gain distributions by condition
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key in zip(axes, ("gain", "axis_gains")):
        data, labels = [], []
        for cond in ("real", "tilt_removal", "scale_0.5", "scale_1.5", "mirror_roll",
                     "yaw_control"):
            vals = []
            for (f, cid, c), r in results.items():
                if c != cond or not testable.get((f, cid), {}).get("testable"):
                    continue
                vals.append(r.gain if key == "gain" else r.axis_gains[1])
            data.append(vals)
            labels.append(cond)
        keep = [(d, l) for d, l in zip(data, labels) if len(d)]
        if keep:
            ax.boxplot([k[0] for k in keep], tick_labels=[k[1] for k in keep],
                       showfliers=False)
        ax.axhline(1.0, color="k", ls=":")
        ax.axhline(0.0, color="k", ls=":")
        ax.set_title("G1" if key == "gain" else "roll-axis gain gy")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "gain_distributions.png", dpi=150)
    plt.close(fig)

    # operating curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, fam in zip(axes, ("tilt_removal", "mirror_roll")):
        for rule, style in ((selected, "-o"), ("R1_scalar_g1", "--")):
            x = [r["frr"] for r in curves[rule]["rows"]]
            y = [r[f"det_{fam}"] for r in curves[rule]["rows"]]
            ax.plot(x, y, style, ms=3, label=f"gain {rule}")
        x = [r["frr"] for r in base_curve["rows"]]
        y = [r[f"det_{fam}"] for r in base_curve["rows"]]
        ax.plot(x, y, "-s", ms=3, label="baseline")
        ax.set_xlabel("false-reject rate")
        ax.set_ylabel(f"detection, {fam}")
        ax.set_xlim(0, 0.6)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "operating_curves.png", dpi=150)
    plt.close(fig)

    # synthetic sweep
    if synth:
        df = pd.DataFrame(synth)
        fig, ax = plt.subplots(figsize=(6, 4))
        g = df.groupby("k")["flagged"].mean()
        ax.plot(g.index, g.values, "-o")
        ax.set_xlabel("synthetic gain k")
        ax.set_ylabel("flag rate")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "synthetic_sweep.png", dpi=150)
        plt.close(fig)

    # drag per flight
    fig, ax = plt.subplots(figsize=(8, 4))
    data, labels = [], []
    for f in FLIGHTS:
        d = [r.drag for (ff, cid, c), r in results.items()
             if ff == f and c == "real" and testable.get((ff, cid), {}).get("testable")]
        data.append(d)
        labels.append(f.replace("flight-", "").replace("-", "\n"))
    keep = [(d, l) for d, l in zip(data, labels) if len(d)]
    if keep:
        ax.boxplot([k[0] for k in keep], tick_labels=[k[1] for k in keep],
                   showfliers=False)
    ax.axhline(0.295, color="k", ls=":", label="Phase 0 k_d")
    ax.set_ylabel("fitted drag (1/s)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "drag_by_flight.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
