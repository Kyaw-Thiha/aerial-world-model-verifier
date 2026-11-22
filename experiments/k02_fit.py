"""Track K, Part 2: fit M0-M5 offline and evaluate leave-one-flight-out / cross-domain.

CPU only. Reads mocap and Leica truth; writes ``results/k02_fit/``. The model
definitions, parameter bounds, lag grid, metrics and selection rule are the
pre-registered ones in ``configs/k02_model_prereg.yaml``.

    PYTHONPATH=. python experiments/k02_fit.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.airframe.fit import (evaluate_fixed, fit_lag, fit_m5, fit_model, lofo, residual,
                              shift_kinematics)
from src.airframe.kinematics import load_all
from src.airframe.models import MODELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/k02_fit"
PRIMARY = ("M0", "M1", "M2", "M3", "M4")


def fit_full(units, model):
    """Fit parameters on all units; M5 fits M1 first, as pre-registered."""
    if model == "M5":
        base, lags, _ = fit_model(units, "M1")
        params, info = fit_m5(units, lags, base)
        params["m5_alpha"] = info["alpha"]
        return params, info
    params, lags, info = fit_model(units, model)
    return params, info


def main() -> int:
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT / run
    out_dir.mkdir(parents=True, exist_ok=True)
    tii = load_all("tii")
    uzh = load_all("uzh", rig="forward")
    uzh45 = load_all("uzh", rig="45")
    doc = {"run": run, "n_tii": len(tii), "n_uzh_forward": len(uzh),
           "n_uzh_45": len(uzh45), "lofo": {}, "cross": {}, "constructions": {
               "fitting_objective": "mean squared lean-vector residual (deg^2), pooled",
               "lag_grid_s": [-0.150, 0.150, 0.025],
               "subsample": 4}}

    for model in MODELS:
        print(f"LOFO {model} ...", flush=True)
        doc["lofo"][model] = lofo(tii, model)

    base_rms = doc["lofo"]["M0"]["pooled_rms_frame_weighted"]
    for model in MODELS:
        r = doc["lofo"][model]["pooled_rms_frame_weighted"]
        doc["lofo"][model]["reduction_vs_M0"] = float(1.0 - r / base_rms)
        doc["lofo"][model]["reduction_vs_M0_mean"] = float(
            1.0 - doc["lofo"][model]["pooled_rms_mean"] / doc["lofo"]["M0"]["pooled_rms_mean"])

    # ---- pre-registered parsimony selection among M0-M4
    best = min(PRIMARY, key=lambda m: doc["lofo"][m]["pooled_rms_frame_weighted"])
    best_rms = doc["lofo"][best]["pooled_rms_frame_weighted"]
    selected = next((m for m in PRIMARY
                     if doc["lofo"][m]["pooled_rms_frame_weighted"] <= best_rms + 0.25), best)
    doc["selection"] = {"best": best, "best_rms": best_rms, "selected": selected,
                        "rule": "lowest complexity within 0.25 deg of the best M0-M4 LOFO RMS",
                        "meets_25pct": bool(doc["lofo"][selected]["reduction_vs_M0"] >= 0.25)}
    print("selection", doc["selection"], flush=True)

    # ---- cross-domain with parameters fitted on the whole source domain
    for model in PRIMARY + ("M5",):
        params, info = fit_full(tii, model)
        doc["cross"][f"tii_to_uzh/{model}"] = {
            "fit": "all 8 TII flights", "params": _jsonable(params),
            "info": _jsonable(info),
            "eval": evaluate_fixed(uzh, model, params),
        }
        doc["cross"][f"tii_to_uzh45/{model}"] = {
            "fit": "all 8 TII flights", "eval": evaluate_fixed(uzh45, model, params)}
        print(f"cross TII->UZH {model}", doc["cross"][f"tii_to_uzh/{model}"]["eval"][
            "pooled_rms_frame_weighted"], flush=True)
    for model in PRIMARY + ("M5",):
        params, info = fit_full(uzh, model)
        doc["cross"][f"uzh_to_tii/{model}"] = {
            "fit": "9 UZH forward sequences", "params": _jsonable(params),
            "info": _jsonable(info), "eval": evaluate_fixed(tii, model, params)}
        print(f"cross UZH->TII {model}", doc["cross"][f"uzh_to_tii/{model}"]["eval"][
            "pooled_rms_frame_weighted"], flush=True)

    # ---- in-domain full fit (for the delivered schema v3 and provenance)
    doc["full_fit"] = {}
    for model in PRIMARY + ("M5",):
        params, info = fit_full(tii, model)
        doc["full_fit"][model] = {"params": _jsonable(params), "info": _jsonable(info),
                                  "in_sample": evaluate_fixed(tii, model, params)}

    (out_dir / "fit.json").write_text(json.dumps(doc, indent=1, default=_jsonable))
    _tables(doc, out_dir)
    print("wrote", out_dir)
    return 0


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def _tables(doc, out_dir: Path) -> None:
    rows = ["model,n_units,pooled_rms_mean,pooled_rms_frame_weighted,reduction_vs_M0"]
    for m in MODELS:
        r = doc["lofo"][m]
        rows.append(f"{m},{r['n_units']},{r['pooled_rms_mean']:.4f},"
                    f"{r['pooled_rms_frame_weighted']:.4f},{r['reduction_vs_M0']:.4f}")
    (out_dir / "lofo_summary.csv").write_text("\n".join(rows) + "\n")

    rows = ["model,unit,rms_deg,median_deg,lean_rms_deg,lag_s,n"]
    for m in MODELS:
        for u in doc["lofo"][m]["units"]:
            rows.append(f"{m},{u['name']},{u['rms_deg']:.4f},{u['median_deg']:.4f},"
                        f"{u['lean_rms_deg']:.4f},{u['lag_s']:.4f},{u['n']}")
    (out_dir / "lofo_per_unit.csv").write_text("\n".join(rows) + "\n")

    rows = ["setting,model,unit,rms_deg,median_deg,lag_s,n"]
    for key, rec in doc["cross"].items():
        if "eval" not in rec:
            continue
        for u in rec["eval"]["units"]:
            rows.append(f"{key},{key.split('/')[-1]},{u['name']},{u['rms_deg']:.4f},"
                        f"{u['median_deg']:.4f},{u['lag_s']:.4f},{u['n']}")
    (out_dir / "cross_per_unit.csv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
