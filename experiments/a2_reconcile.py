"""Track A2: reconcile the pair table with the gated table, and recompute dt_min/verdicts
under the parity nuisances.

Part 1 decomposes the rejection difference into its two causes -- the population (testable
vs gate-only pairs) and the threshold (held-out conformal per clip vs a fixed 6.46 deg) --
because the published pair table changed both at once and therefore cannot be compared with
the pre-registered table.

Part 2 recomputes the minimum consistent dt with the nuisance fits, which lowers every
residual and must shorten it, and reports drag-free as primary with both settings shown.

    PYTHONPATH=. pixi run -e gpu python experiments/a2_reconcile.py
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.a2_anwm_required import frame_required
from experiments.phase3a_verdict import threshold_lookup, thresholds_for
from src.perception.frames import camera_to_body_pitched
from src.perception.gate import clip_signal, kept
from src.physics.flatness import tilt_deg
from src.score.verdict import depicted_axis_world

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("tilt_k0", "tilt_k05", "tilt_k15", "mirror_roll", "yaw_control")
FIXED_T = 6.46
UPTILT = np.arange(-7.0, 7.0001, 1.0)
LAGS = np.arange(-150.0, 150.0001, 25.0)


def latest(p):
    return sorted(ROOT.glob(p))[-1]


def jload(p):
    return json.loads(Path(p).read_text())


def load_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


# ------------------------------------------------------------------ part 1
def reconciliation() -> dict:
    run = jload(latest("results/a2_verdict_gated/*/metrics.json"))
    prereg = yaml.safe_load((ROOT / "configs/verdict_preregistration.yaml").read_text())
    out = {}
    for key, val in run["results"].items():
        recs = val["_records"]
        thr = thresholds_for(recs, prereg, holdout=None, policy="all_flights")
        folds = {f["flight"]: f["thresholds"] for f in val["fold_thresholds"]}
        real = {(r["flight"], r["clip_id"]): r for r in recs if r["condition"] == "identity"}

        def heldout(r):
            return threshold_lookup(folds.get(r["flight"], thr), r["bin"])

        cells = {}
        for cond in FAMILIES:
            for pop in ("testable", "gate_only"):
                for tmode in ("heldout", "fixed"):
                    pos, neg = [], []
                    for r in recs:
                        if r["condition"] != cond or "score" not in r:
                            continue
                        p = real.get((r["flight"], r["clip_id"]))
                        if p is None or "score" not in p:
                            continue
                        if pop == "testable":
                            ok = bool(p["testable"] and r["testable"])
                        else:
                            ok = bool(p.get("gate_kept") and r.get("gate_kept"))
                        if not ok:
                            continue
                        pos.append(p)
                        neg.append(r)
                    if not neg:
                        cells[f"{cond}|{pop}|{tmode}"] = None
                        continue
                    if tmode == "fixed":
                        rej = float(np.mean([r["score"] > FIXED_T for r in neg]))
                    else:
                        rej = float(np.mean([r["score"] > heldout(r) for r in neg]))
                    cells[f"{cond}|{pop}|{tmode}"] = {
                        "n": len(neg), "rejection": rej,
                        "real_median": float(np.median([p["score"] for p in pos])),
                        "neg_median": float(np.median([r["score"] for r in neg])),
                        "median_threshold": float(np.median([heldout(r) for r in neg]))}
        out[key] = cells
    return out


# ------------------------------------------------------------------ part 2
def fit_residual(dep_by_delta: dict[float, np.ndarray], req: np.ndarray, t: np.ndarray,
                 lag_seconds: bool = True) -> dict:
    best = {"loss": np.inf, "delta": 0.0, "lag_ms": 0.0}
    lag_ms_values = LAGS if lag_seconds else np.array([0.0])
    for lag_ms in lag_ms_values:
        rq = np.interp(t + lag_ms / 1000.0, t, req) if lag_seconds else req
        for delta, dep in dep_by_delta.items():
            r = np.abs(dep - rq)
            m = np.isfinite(r)
            if m.sum() < 8:
                continue
            loss = float(np.median(r[m]))
            if loss < best["loss"]:
                best = {"loss": loss, "delta": float(delta), "lag_ms": float(lag_ms)}
    return best


def main() -> int:
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results/a2_reconcile" / run
    out.mkdir(parents=True, exist_ok=True)
    doc: dict = {}
    doc["pair_reconciliation"] = reconciliation()

    cfg = yaml.safe_load((ROOT / "configs/gate_frozen.yaml").read_text())
    rows = load_jsonl(ROOT / "results/gate_a2/anwm/depicted.jsonl")
    by_clip: dict[str, list[dict]] = {}
    for r in rows:
        by_clip.setdefault(r["clip_id"], []).append(r)
    man = {m["id"]: m for m in (json.loads(p.read_text()) for p in
                                sorted((ROOT / "data/processed/anwm_clips").glob("*/*/manifest.json")))}

    grid = np.round(np.arange(0.10, 5.0001, 0.05), 3)
    dtmin: dict = {}
    for k_d in (0.0, 0.295):
        for kind in ("spline", "step"):
            recs = []
            for cid in sorted(by_clip):
                rc = by_clip[cid]
                arrs = {"up": np.asarray([r["gravity_up_cam"] for r in rc]).reshape(-1, 3),
                        "up_confidence_mean": np.array([r.get("up_confidence_mean", np.nan)
                                                        for r in rc]),
                        "latitude_confidence_mean": np.array(
                            [r.get("latitude_confidence_mean", np.nan) for r in rc]),
                        "gravity_uncertainty_raw": np.array(
                            [r.get("gravity_uncertainty_raw", np.nan) for r in rc])}
                if not kept(clip_signal(cfg["signal"], arrs), cfg["tau"]):
                    continue
                dep_by_delta = {}
                for d in UPTILT:
                    up = np.asarray([r["gravity_up_cam"] for r in rc])
                    z = depicted_axis_world(up, camera_to_body_pitched(np.deg2rad(float(d))),
                                            np.zeros(len(up)))
                    dep_by_delta[float(d)] = tilt_deg(z)
                dep0 = dep_by_delta[0.0]
                pos = np.asarray(man[cid]["commanded_path"]["positions"], float)
                found_parity, found_nofit, exc1 = None, None, None
                for dt in grid:
                    try:
                        req = frame_required(pos, float(dt), k_d, kind)
                    except ValueError:
                        continue
                    n = min(len(req), len(dep0))
                    ok = np.isfinite(req[:n]) & np.isfinite(dep0[:n])
                    if ok.sum() < 8:
                        continue
                    if exc1 is None and abs(dt - 1.0) < 1e-9:
                        exc1 = int((req[:n][ok] >= 10).sum())
                    t = np.arange(n) * float(dt)
                    if found_parity is None:
                        fit = fit_residual({k: v[:n] for k, v in dep_by_delta.items()},
                                           req[:n], t)
                        if fit["loss"] <= FIXED_T:
                            found_parity = float(dt)
                    if found_nofit is None:
                        r = np.abs(dep0[:n] - req[:n])
                        m = np.isfinite(r)
                        if m.sum() >= 8 and np.median(r[m]) <= FIXED_T:
                            found_nofit = float(dt)
                    if found_parity is not None and found_nofit is not None:
                        break
                recs.append({"clip": cid, "regime": man[cid]["regime"],
                             "dt_min_parity": found_parity, "dt_min_nofit": found_nofit,
                             "testable_at_1s": bool(exc1 is not None and exc1 >= 12),
                             "step": man[cid]["commanded_path"]["step_spacing_m"]})
            df = pd.DataFrame(recs)
            dist = {}
            for reg in ("A", "B", "C"):
                g = df[df.regime == reg]
                if not len(g):
                    continue
                dist[reg] = {
                    "n": int(len(g)),
                    "testable_at_1s": int(g.testable_at_1s.sum()),
                    "median_parity": float(g.dt_min_parity.median()) if g.dt_min_parity.notna().any() else None,
                    "median_nofit": float(g.dt_min_nofit.median()) if g.dt_min_nofit.notna().any() else None,
                    "median_parity_testable_only": float(
                        g[g.testable_at_1s].dt_min_parity.median())
                    if g[g.testable_at_1s].dt_min_parity.notna().any() else None,
                    "median_nofit_testable_only": float(
                        g[g.testable_at_1s].dt_min_nofit.median())
                    if g[g.testable_at_1s].dt_min_nofit.notna().any() else None,
                    "n_parity_gt_nofit": int((g.dt_min_parity > g.dt_min_nofit).sum()),
                }
            dtmin[f"{kind}_kd{k_d}"] = {"distribution": dist, "per_clip": recs}
    doc["dt_min_parity"] = dtmin
    (out / "reconcile.json").write_text(json.dumps(doc, indent=1, default=float))

    # ---- print
    def f(x, nd=2):
        return "n/a" if x is None else f"{x:.{nd}f}"

    print("== pair-table decomposition (rejection) ==")
    for key, cells in doc["pair_reconciliation"].items():
        print(f"  {key}")
        for fam in FAMILIES:
            row = []
            for pop in ("testable", "gate_only"):
                for tm in ("heldout", "fixed"):
                    c = cells.get(f"{fam}|{pop}|{tm}")
                    row.append(f"{pop[:4]}/{tm[:3]} {f(c['rejection']) if c else ' n/a '}"
                               f"(n={c['n'] if c else 0})")
            print(f"    {fam:12s} " + "  ".join(row))
    print("== dt_min with and without parity ==")
    for k, v in dtmin.items():
        print(f"  {k}: " + " ".join(
            f"{r} n={x['n']} test@1s={x['testable_at_1s']} "
            f"parity {f(x['median_parity'])} (testable {f(x['median_parity_testable_only'])}) "
            f"nofit {f(x['median_nofit'])} (testable {f(x['median_nofit_testable_only'])})"
            for r, x in v["distribution"].items()))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
