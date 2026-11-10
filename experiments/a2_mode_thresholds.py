#!/usr/bin/env python3
"""Amendment 9: mode-consistent thresholds, the verdict-vs-threshold table, and the injection floor.

Why: every earlier ANWM verdict used a fixed 6.46 deg cut, but that number is the TII conformal threshold for the
**full-parity** fit at ``k_d`` 0.295. Amendment 8 makes drag-free + lag-only primary, and in that mode the TII
threshold is larger (14.73 deg drag-free). Applying 6.46 to lag-only residuals therefore over-counts infeasibility --
most visibly the designed-10 family, which reads 12/12 infeasible at 6.46 and 0/12 at its own threshold.

This script regenerates, from the stored runs, everything the amendment quotes:

* TII held-out conformal thresholds per nuisance mode and ``k_d`` (pooled + per-fold spread);
* the ANWM null (cruise clips) p95/p99 in the primary mode, and the released-frame noise floor;
* the verdict-vs-threshold table for 6.46 / 8.40 / 12.95 / primary, on the primary-mode residuals;
* the same table with each mode's own threshold (the consistent version);
* the detection floor in degrees of required tilt per mode.

    PYTHONPATH=. pixi run python experiments/a2_mode_thresholds.py \\
        --lag-run /tmp/a2_lag_kd0 --lag-run /tmp/a2_lag_rest \\
        --out results/a2_mode_thresholds
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED = (6.46, 8.40, 12.95)
DEPICTED_MEDIAN_DEG = 0.79  # report 08, all 82 clips


def pct(values, q):
    values = sorted(values)
    if not values:
        return None
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def load_tii_runs(paths):
    """``{k_d: {pooled, folds, spread, acceptance, n}}`` from a2_verdict_gated metrics."""
    out = {}
    for path in paths:
        for f in glob.glob(str(Path(path) / "**" / "metrics.json"), recursive=True) or                 glob.glob(str(Path(path) / "metrics.json")):
            d = json.load(open(f))
            mode = d["config"].get("nuisances", "both")  # older runs predate the flag
            for key, ev in d["results"].items():
                kd = float(key.replace("kd_", ""))
                folds = []
                for fo in ev.get("fold_thresholds", []):
                    vals = [v["threshold"] for v in fo["thresholds"]["per_bin"].values()
                            if v["threshold"] == v["threshold"]]
                    if vals:
                        folds.append(max(vals))
                out[(mode, kd)] = {
                    "threshold": ev["pooled_thresholds"]["pooled"]["threshold"],
                    "folds": folds, "fold_min": min(folds) if folds else None,
                    "fold_median": st.median(folds) if folds else None,
                    "fold_max": max(folds) if folds else None,
                    "acceptance": ev["heldout_acceptance"], "n": ev["heldout_n"],
                }
    return out


def load_parity(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ("residual_parity", "residual_lag_only", "residual_no_fit", "designed"):
            if r.get(k):
                r[k] = float(r[k])
        r["testable"] = r["testable"] == "True"
        r["k_d"] = float(r["k_d"])
    return rows


def cells(rows, *, k_d, residual_column, threshold):
    """Per-cell testable / infeasible counts and residual median at one threshold."""
    out = {}
    for r in rows:
        if r["k_d"] != k_d or not r["testable"]:
            continue
        res = r[residual_column]
        if r["regime"] == "A":
            key = "A"
        elif r["regime"] == "C":
            key = f"C_{int(r['designed'])}deg"
        else:
            continue
        d = out.setdefault(key, {"n": 0, "infeasible": 0, "residuals": []})
        d["n"] += 1
        d["infeasible"] += int(res > threshold)
        d["residuals"].append(res)
    for d in out.values():
        d["median_deg"] = st.median(d["residuals"])
        d.pop("residuals")
    return dict(sorted(out.items()))


def null_cruise(rows, *, k_d, residual_column):
    # the null is the level-cruise residual itself; testability is a separate property
    vals = [r[residual_column] for r in rows
            if r["k_d"] == k_d and r["maneuver"] == "cruise"]
    return {"n": len(vals), "median": st.median(vals) if vals else None,
            "p95": pct(vals, 0.95), "p99": pct(vals, 0.99), "max": max(vals) if vals else None}


def released_frame_floor():
    hits = sorted(glob.glob(str(ROOT / "results/anwm_scoring/*/noise_floor.json")))
    if not hits:
        return None
    d = json.load(open(hits[-1]))
    for size in d.get("sizes", {}).values():
        if "p99" in size:
            return {"median": size["median"], "p95": size.get("p95"), "p99": size["p99"],
                    "source": hits[-1]}
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lag-run", action="append", required=True,
                    help="a2_verdict_gated out dir (repeatable; lag-only and parity both accepted)")
    ap.add_argument("--parity-csv", default=None)
    ap.add_argument("--checks", default=None)
    ap.add_argument("--out", default="results/a2_mode_thresholds")
    args = ap.parse_args()

    parity_csv = args.parity_csv or sorted(glob.glob(str(ROOT / "results/a2_parity/*/parity.csv")))[-1]
    checks_path = args.checks or sorted(glob.glob(str(ROOT / "results/a2_checks/*/checks.json")))[-1]
    tii = load_tii_runs(args.lag_run)
    # the parity thresholds for k_d 0/0.15/0.295 are the published ones; keep them if the run is absent
    for kd, th in ((0.295, 6.46), (0.15, 8.40), (0.0, 12.95)):
        tii.setdefault(("both", kd), {"threshold": th, "folds": [], "fold_min": None,
                                      "fold_median": None, "fold_max": None,
                                      "acceptance": None, "n": None, "published": True})
    rows = load_parity(parity_csv)
    checks = json.load(open(checks_path))
    required = {int(c["designed"]): c["required_median"]
                for c in checks["anwm"]["dose"]["spline_0.0"]["cells"]}

    primary = tii[("lag_only", 0.0)]
    lag_0295 = tii[("lag_only", 0.295)]
    null = null_cruise(rows, k_d=0.0, residual_column="residual_lag_only")
    frame_floor = released_frame_floor()
    primary_threshold = max([primary["threshold"]] + [v for v in (null["p99"],) if v is not None])

    fixed_table = {f"{th:.2f}": cells(rows, k_d=0.0, residual_column="residual_lag_only", threshold=th)
                   for th in FIXED}
    fixed_table[f"{primary_threshold:.2f} (primary)"] = cells(
        rows, k_d=0.0, residual_column="residual_lag_only", threshold=primary_threshold)

    consistent = {
        "lag_only kd0.0 @ primary": {"threshold": primary_threshold,
                                     "cells": cells(rows, k_d=0.0, residual_column="residual_lag_only",
                                                    threshold=primary_threshold)},
        "lag_only kd0.295 @ own": {"threshold": lag_0295["threshold"],
                                   "cells": cells(rows, k_d=0.295, residual_column="residual_lag_only",
                                                  threshold=lag_0295["threshold"])},
        "parity kd0.295 @ own": {"threshold": tii[("both", 0.295)]["threshold"],
                                 "cells": cells(rows, k_d=0.295, residual_column="residual_parity",
                                                threshold=tii[("both", 0.295)]["threshold"])},
        "parity kd0.0 @ own": {"threshold": tii[("both", 0.0)]["threshold"],
                               "cells": cells(rows, k_d=0.0, residual_column="residual_parity",
                                              threshold=tii[("both", 0.0)]["threshold"])},
    }

    floors = {}
    for name, info in (("lag_only kd0.0 (primary)", primary), ("lag_only kd0.15", tii[("lag_only", 0.15)]),
                       ("lag_only kd0.295", lag_0295), ("parity kd0.295", tii[("both", 0.295)]),
                       ("parity kd0.15", tii[("both", 0.15)]), ("parity kd0.0", tii[("both", 0.0)])):
        floors[name] = {"threshold_deg": info["threshold"],
                        "required_tilt_floor_deg": info["threshold"] + DEPICTED_MEDIAN_DEG}

    table = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tii_thresholds": {f"{mode} kd{kd}": v for (mode, kd), v in sorted(tii.items())},
        "anwm_null_cruise_lag_only": null,
        "released_frame_floor": frame_floor,
        "primary_threshold": primary_threshold,
        "fixed_thresholds_on_primary_mode": fixed_table,
        "mode_consistent": consistent,
        "required_tilt_median_by_cell": required,
        "detection_floors": floors,
        "sources": {"parity_csv": parity_csv, "checks": checks_path},
    }
    out = Path(args.out) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.json").write_text(json.dumps(table, indent=2) + "\n")
    (out / "table.md").write_text(render(table))
    print(render(table))
    print(f"\nwrote {out / 'table.json'}")
    return 0


def render(t: dict) -> str:
    lines = ["## TII thresholds by mode", "",
             "| mode | k_d | pooled | per-fold min / median / max | held-out acceptance |",
             "| --- | --- | --- | --- | --- |"]
    for name, v in t["tii_thresholds"].items():
        folds = ("—" if not v["folds"] else
                 f"{v['fold_min']:.2f} / {v['fold_median']:.2f} / {v['fold_max']:.2f}")
        acc = "—" if v["acceptance"] is None else f"{v['acceptance']:.3f}"
        lines.append(f"| {name} | {v['threshold']:.2f} | {folds} | {acc} |")
    lines += ["", "## Verdict vs threshold, on the primary-mode (drag-free lag-only) residuals", "",
              "| threshold | Regime A infeasible/testable | C 10 deg | C 20 deg |", "| --- | --- | --- | --- |"]
    for th, cells in t["fixed_thresholds_on_primary_mode"].items():
        def fmt(key):
            c = cells.get(key)
            return "—" if not c else f"{c['infeasible']}/{c['n']} (median {c['median_deg']:.1f})"
        lines.append(f"| {th} | {fmt('A')} | {fmt('C_10deg')} | {fmt('C_20deg')} |")
    lines += ["", "## Mode-consistent verdicts (each mode at its own threshold)", "",
              "| mode | threshold | Regime A | C 10 deg | C 20 deg |", "| --- | --- | --- | --- | --- |"]
    for name, v in t["mode_consistent"].items():
        def fmt2(key):
            c = v["cells"].get(key)
            return "—" if not c else f"{c['infeasible']}/{c['n']}"
        lines.append(f"| {name} | {v['threshold']:.2f} | {fmt2('A')} | {fmt2('C_10deg')} | {fmt2('C_20deg')} |")
    lines += ["", "## Detection floor (required tilt, deg)", "",
              "| mode | verdict threshold | required tilt at the floor |", "| --- | --- | --- |"]
    for name, v in t["detection_floors"].items():
        lines.append(f"| {name} | {v['threshold_deg']:.2f} | {v['required_tilt_floor_deg']:.2f} |")
    null = t["anwm_null_cruise_lag_only"]
    fmt = lambda x: "—" if x is None else f"{x:.2f}"
    lines += ["", f"Null (cruise, lag-only, k_d 0.0): n={null['n']}, median {fmt(null['median'])}, "
                  f"p95 {fmt(null['p95'])}, p99 {fmt(null['p99'])}; "
                  f"required tilt by cell {t['required_tilt_median_by_cell']}."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
