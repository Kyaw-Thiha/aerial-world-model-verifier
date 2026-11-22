"""Track M follow-up: coverage restricted to dynamic clips (peak required tilt >= 15 deg).

Pre-registered in ``configs/m05_dynamic_coverage_prereg.yaml`` (committed at ``c6df271`` before
any count was taken).  CPU only, zero GPU spend; no threshold is refitted.

Real clips: ``results/m03/reason_codes_real.csv`` (one row per clip and condition, with the
frozen gate decisions) joined to ``results/m01/signal_by_clip.csv`` for the peak required tilt.
Simulated clips: ``results/m03/reason_codes_sim.csv``, which carries the same columns.

    PYTHONPATH=. pixi run python experiments/m05_dynamic_coverage.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/m05"
THRESHOLD = 15.0
CODES = ("low_confidence_S3", "weak_signal", "near_vertical", "readers_disagree")
CONFIGS = ("G0_S3_only", "GA", "GB", "GAB")


def _codes(s: str) -> list[str]:
    return [c for c in str(s).split(";") if c and c != "none"]


def _keep_from_codes(codes: list[str]) -> bool:
    return not codes


def real_table() -> pd.DataFrame:
    rc = pd.read_csv(ROOT / "results/m03/reason_codes_real.csv")
    rc = rc[rc.cond == "real"].copy()
    sig = pd.read_csv(ROOT / "results/m01/signal_by_clip.csv")[["clip_id", "tilt_max"]]
    rc = rc.merge(sig, on="clip_id", how="left", suffixes=("", "_sig"))
    rc["peak_tilt"] = rc["tilt_max_sig"]
    rc["dynamic"] = rc.peak_tilt >= THRESHOLD
    rc["track"] = rc.flight.str.extract(r"(ellipse|lemniscate|trackRATM)")[0]
    rc["all_codes"] = rc.reason_codes
    # per-configuration decisions are the reason subsets the reason-code table already encodes
    rc["G0_S3_only"] = ~rc.all_codes.apply(lambda s: "low_confidence_S3" in _codes(s))
    rc["GA"] = rc.G0_S3_only & ~rc.all_codes.apply(
        lambda s: bool({"weak_signal", "near_vertical"} & set(_codes(s))))
    rc["GB"] = rc.G0_S3_only & ~rc.all_codes.apply(
        lambda s: "readers_disagree" in _codes(s))
    rc["GAB"] = rc.G0_S3_only & ~rc.all_codes.apply(
        lambda s: bool({"weak_signal", "near_vertical", "readers_disagree"} & set(_codes(s))))
    return rc


def sim_table() -> pd.DataFrame:
    rc = pd.read_csv(ROOT / "results/m03/reason_codes_sim.csv")
    rc["dynamic"] = rc.tilt_max >= THRESHOLD
    rc["G0_S3_only"] = pd.Series(True, index=rc.index)
    for cfg, keys in (("GA", {"weak_signal", "near_vertical"}),
                      ("GB", {"readers_disagree"}),
                      ("GAB", {"weak_signal", "near_vertical", "readers_disagree"})):
        rc[cfg] = ~rc.reason_codes.apply(lambda s: bool(keys & set(_codes(s))))
    rc["nested"] = rc.family
    return rc


def coverage(df: pd.DataFrame) -> dict:
    out = {}
    n = int(df.dynamic.sum())
    for cfg in CONFIGS:
        k = int(df[df.dynamic][cfg].sum())
        out[cfg] = {"kept": k, "n": n, "coverage": (k / n) if n else float("nan")}
    col = "all_codes" if "all_codes" in df else "reason_codes"
    dyn = df[df.dynamic]
    if len(dyn):
        codes = {code: int(dyn[col].apply(lambda s: code in _codes(s)).sum())
                 for code in CODES}
    else:
        codes = {}
    out["reason_counts"] = {k: v for k, v in codes.items() if v}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"threshold_deg": THRESHOLD,
           "prereg": "configs/m05_dynamic_coverage_prereg.yaml (c6df271)"}

    real = real_table()
    real.to_csv(OUT / "real_dynamic.csv", index=False)
    res["real"] = {
        "all": coverage(real),
        "s3_kept": coverage(real[real.G0_S3_only]),
        "by_flight": {f: coverage(g) for f, g in real.groupby("flight")},
        "by_track": {t: coverage(g) for t, g in real.groupby("track")},
        "n_dynamic": int(real.dynamic.sum()),
        "n_total": int(len(real)),
    }

    sim = sim_table()
    sim.to_csv(OUT / "sim_dynamic.csv", index=False)
    res["sim"] = {
        "all": coverage(sim),
        "by_family": {f: coverage(g) for f, g in sim.groupby("family")},
        "by_mount": {str(m): coverage(g) for m, g in sim.groupby("mount")},
        "by_defect": {d: coverage(g) for d, g in sim.groupby("defect")},
        "n_dynamic": int(sim.dynamic.sum()),
        "n_total": int(len(sim)),
    }
    (OUT / "dynamic_coverage.json").write_text(json.dumps(res, indent=1))

    for label, block in (("REAL", res["real"]), ("SIM", res["sim"])):
        print(f"\n== {label}: {block['n_dynamic']} dynamic of {block['n_total']}")
        for scope in ("all", "s3_kept"):
            if scope not in block:
                continue
            row = block[scope] if scope == "all" else block["s3_kept"]
            print(f"-- {scope}")
            for cfg in CONFIGS:
                e = row[cfg]
                print(f"   {cfg:12s} {e['kept']:3d}/{e['n']:<3d} = {e['coverage']:.3f}")
            print("   reasons:", row["reason_counts"])
        for key in ("by_family", "by_track", "by_mount", "by_defect"):
            if key not in block:
                continue
            print(f"-- {key}")
            for name, e in block[key].items():
                cs = " ".join(f"{cfg}={e[cfg]['kept']}/{e[cfg]['n']}" for cfg in CONFIGS)
                print(f"   {str(name):22s} {cs}")


if __name__ == "__main__":
    main()
