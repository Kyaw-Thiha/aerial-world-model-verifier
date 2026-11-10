"""Track A2: parity with the TII verdict — the same two nuisance fits on generated clips.

The verdict fits, once per clip and never per frame, a mount uptilt offset in ±7 deg (step
1) and a time lag in ±150 ms (step 25 ms), minimising the median residual. The ANWM
analysis as first written fitted nothing, on the argument that the render is exactly level
and a lag has no meaning at 1 s steps. That argument is worth testing rather than asserting:
the same grid is run here on every clip, so the ANWM residuals are directly comparable with
the TII ones and the cost of the nuisances is measured.

    uptilt delta   : R_cb = camera_to_body_pitched(uptilt + delta), applied to the depicted
                     side (the camera is exactly level, so uptilt = 0 and delta is the fit)
    lag            : the required profile is sampled at t + lag (linear), as in the verdict

    PYTHONPATH=. pixi run -e gpu python experiments/a2_parity.py
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
from src.perception.gate import clip_signal, kept
from src.physics.flatness import tilt_deg
from src.score.verdict import depicted_axis_world

ROOT = Path(__file__).resolve().parents[1]
K_DS = (0.0, 0.295)
THRESH = 6.46
UPTILT_GRID = np.arange(-7.0, 7.0001, 1.0)
LAG_GRID_MS = np.arange(-150.0, 150.0001, 25.0)
DT = 1.0


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def depicted_tilt(rows: list[dict], uptilt_deg: float = 0.0) -> np.ndarray:
    up = np.asarray([r["gravity_up_cam"] for r in rows], dtype=np.float64)
    z = depicted_axis_world(up, camera_to_body_pitched(np.deg2rad(uptilt_deg)),
                            np.zeros(len(up)))
    return tilt_deg(z)


def fit_nuisances(dep_by_delta: dict[float, np.ndarray], req: np.ndarray,
                  t: np.ndarray) -> dict:
    """Grid search exactly as the verdict does it: minimise the median residual."""
    best = {"loss": np.inf, "uptilt_offset_deg": 0.0, "lag_ms": 0.0}
    for delta, dep in dep_by_delta.items():
        for lag_ms in LAG_GRID_MS:
            rq = np.interp(t + lag_ms / 1000.0, t, req)
            r = np.abs(dep - rq)
            m = np.isfinite(r)
            if m.sum() < 8:
                continue
            loss = float(np.median(r[m]))
            if loss < best["loss"]:
                best = {"loss": loss, "uptilt_offset_deg": float(delta),
                        "lag_ms": float(lag_ms)}
    return best


def residual_no_fit(dep: np.ndarray, req: np.ndarray) -> float:
    r = np.abs(dep - req)
    m = np.isfinite(r)
    return float(np.median(r[m])) if m.sum() >= 8 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depicted", default="results/gate_a2/anwm/depicted.jsonl")
    ap.add_argument("--clips", default="data/processed/anwm_clips")
    ap.add_argument("--gate-config", default="configs/gate_frozen.yaml")
    ap.add_argument("--out", default="results/a2_parity")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.gate_config).read_text())
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + git_hash()
    out = ROOT / args.out / run
    out.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(Path(args.depicted))
    by_clip: dict[str, list[dict]] = {}
    for r in rows:
        by_clip.setdefault(r["clip_id"], []).append(r)
    man = {m["id"]: m for m in (json.loads(p.read_text())
                                for p in sorted((ROOT / args.clips).glob("*/*/manifest.json")))}

    recs = []
    for cid in sorted(by_clip):
        rows_c = by_clip[cid]
        arrs = {"up": np.asarray([r["gravity_up_cam"] for r in rows_c]).reshape(-1, 3),
                "up_confidence_mean": np.array([r.get("up_confidence_mean", np.nan)
                                                for r in rows_c]),
                "latitude_confidence_mean": np.array([r.get("latitude_confidence_mean", np.nan)
                                                      for r in rows_c]),
                "gravity_uncertainty_raw": np.array([r.get("gravity_uncertainty_raw", np.nan)
                                                     for r in rows_c])}
        gk = bool(kept(clip_signal(cfg["signal"], arrs), cfg["tau"]))
        dep_by_delta = {float(d): depicted_tilt(rows_c, float(d)) for d in UPTILT_GRID}
        dep0 = dep_by_delta[0.0]
        pos = np.asarray(man[cid]["commanded_path"]["positions"], dtype=np.float64)
        t = np.arange(len(dep0)) * DT
        for kind in ("spline", "step"):
            for k_d in K_DS:
                try:
                    req = frame_required(pos, DT, k_d, kind)
                except ValueError:
                    continue
                n = min(len(req), len(dep0))
                rq, dp = req[:n], dep0[:n]
                tt = t[:n]
                nofit = residual_no_fit(dp, rq)
                fit = fit_nuisances({d: v[:n] for d, v in dep_by_delta.items()}, rq, tt)
                exc = int((rq[np.isfinite(rq)] >= 10.0).sum())
                rec = {"clip_id": cid, "regime": man[cid]["regime"],
                       "maneuver": (man[cid].get("meta") or {}).get("maneuver"),
                       "designed": man[cid].get("designed_required_tilt_deg"),
                       "interpolant": kind, "k_d": k_d, "gate_kept": gk,
                       "excitation_frames": exc,
                       "residual_no_fit": nofit,
                       "residual_parity": fit["loss"],
                       "uptilt_offset_deg": fit["uptilt_offset_deg"],
                       "lag_ms": fit["lag_ms"],
                       "absorbed_deg": nofit - fit["loss"],
                       "depicted_median": float(np.median(dp)),
                       "required_median": float(np.nanmedian(rq))}
                rec["testable"] = bool(gk and exc >= 12)
                rec["infeasible_parity"] = bool(rec["testable"]
                                                and rec["residual_parity"] > THRESH)
                rec["infeasible_nofit"] = bool(rec["testable"] and nofit > THRESH)
                recs.append(rec)

    df = pd.DataFrame(recs)
    df.to_csv(out / "parity.csv", index=False)

    summary: dict = {"threshold_deg": THRESH, "grid": {"uptilt": UPTILT_GRID.tolist(),
                                                       "lag_ms": LAG_GRID_MS.tolist()},
                     "by_cell": {}}
    for (kind, k_d, regime), g in df.groupby(["interpolant", "k_d", "regime"]):
        summary["by_cell"][f"{kind}_kd{k_d}_{regime}"] = {
            "n": int(len(g)), "testable": int(g.testable.sum()),
            "infeasible_parity": int(g.infeasible_parity.sum()),
            "infeasible_nofit": int(g.infeasible_nofit.sum()),
            "median_residual_parity": (float(g[g.testable].residual_parity.median())
                                       if g.testable.any() else None),
            "median_residual_nofit": (float(g[g.testable].residual_no_fit.median())
                                      if g.testable.any() else None),
            "median_absorbed_deg": float(g.absorbed_deg.median()) if len(g) else None,
            "uptilt_at_bound_frac": float(np.mean(np.abs(g.uptilt_offset_deg) >= 6.99)),
            "lag_at_bound_frac": float(np.mean(np.abs(g.lag_ms) >= 149.0)),
        }
    # straight-cruise null, with and without the fits
    cruise = df[df.maneuver == "cruise"]
    null = {}
    for (kind, k_d), g in cruise.groupby(["interpolant", "k_d"]):
        null[f"{kind}_kd{k_d}"] = {
            "n": int(len(g)),
            "median_no_fit": float(g.residual_no_fit.median()),
            "median_parity": float(g.residual_parity.median()),
            "p90_parity": float(g.residual_parity.quantile(0.9)),
            "max_parity": float(g.residual_parity.max()),
            "median_absorbed": float(g.absorbed_deg.median())}
    summary["null_cruise"] = null
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    def f(x, width=6, nd=2):
        return " " * width if x is None else f"{x:{width}.{nd}f}"

    print("cell                                   n test parity infeas fit infeas nofit  medPar medNoFit absorbed")
    for k, v in summary["by_cell"].items():
        print(f"{k:38s} {v['n']:3d} {v['testable']:4d} {v['infeasible_parity']:6d} "
              f"{v['infeasible_nofit']:11d} {f(v['median_residual_parity'])} "
              f"{f(v['median_residual_nofit'], 7)} {v['median_absorbed_deg']:8.2f}")
    print("\nstraight-cruise null:", json.dumps(null, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
