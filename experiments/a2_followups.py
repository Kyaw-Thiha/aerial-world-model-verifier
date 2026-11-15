"""Track A2 follow-ups: parity cells, gate selection effect, gate-only pairs, censored dt_min.

    PYTHONPATH=. pixi run -e gpu python experiments/a2_followups.py
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
from src.perception.frames import camera_to_body_pitched
from src.perception.gate import clip_signal, kept
from src.physics.flatness import tilt_deg
from src.score.verdict import auroc_detect, depicted_axis_world

ROOT = Path(__file__).resolve().parents[1]
K_DS = (0.0, 0.295)
THRESH = 6.46


def latest(pattern: str) -> Path:
    hits = sorted(ROOT.glob(pattern))
    if not hits:
        raise SystemExit(f"missing {pattern}")
    return hits[-1]


def jload(p):
    return json.loads(Path(p).read_text())


def load_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def main() -> int:
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results/a2_followups" / run
    out.mkdir(parents=True, exist_ok=True)
    doc: dict = {}

    # ---------------------------------------------------------- 1. Regime C cells, no-fit vs parity
    par = pd.read_csv(latest("results/a2_parity/*/parity.csv"))
    cells = {}
    for (kind, k_d, designed), g in par[(par.regime == "C") & par.designed.notna()].groupby(
            ["interpolant", "k_d", "designed"]):
        key = f"{kind}_kd{k_d}"
        cells.setdefault(key, []).append({
            "designed": float(designed), "n": int(len(g)),
            "testable": int(g.testable.sum()),
            "depicted_median": float(g.depicted_median.median()),
            "required_median": float(g.required_median.median()),
            "residual_nofit": float(g.residual_no_fit.median()),
            "residual_parity": float(g.residual_parity.median()),
            "infeasible_nofit": int(g.infeasible_nofit.sum()),
            "infeasible_parity": int(g.infeasible_parity.sum()),
            "absorbed": float(g.absorbed_deg.median())})
    # which clips flip between k_d 0 and 0.295 (parity and no-fit)
    flips = {"parity": [], "nofit": []}
    idx = par.set_index(["clip_id", "interpolant", "k_d"])
    for (cid, kind) in sorted({(c, k) for c, k, _ in idx.index}):
        try:
            a = idx.loc[(cid, kind, 0.0)]
            b = idx.loc[(cid, kind, 0.295)]
        except KeyError:
            continue
        if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
            continue
        for stat in ("parity", "nofit"):
            if a[f"infeasible_{stat}"] != b[f"infeasible_{stat}"]:
                flips[stat].append({"clip": cid, "interpolant": kind,
                                    "regime": a["regime"], "designed": a["designed"],
                                    "kd0": bool(a[f"infeasible_{stat}"]),
                                    "kd295": bool(b[f"infeasible_{stat}"]),
                                    "res_kd0": float(a[f"residual_{'parity' if stat == 'parity' else 'no_fit'}"]),
                                    "res_kd295": float(b[f"residual_{'parity' if stat == 'parity' else 'no_fit'}"]),
                                    "testable_kd0": bool(a["testable"]),
                                    "testable_kd295": bool(b["testable"])})
    doc["regime_C_cells"] = cells
    doc["k_d_flips"] = flips

    # ---------------------------------------------------------- 2. gate selection effect
    gv = jload(latest("results/a2_gate/*/gate_validation.json"))
    clips = gv["clips"]
    sigs = [s for s in ("S1_gravity_uncertainty", "S2_up_confidence",
                        "S3_latitude_confidence", "S6_uncertainty_instability")]
    labels = np.array([c["label"] for c in clips if c.get("label") is not None], float)
    flights = np.array([c["flight"] for c in clips if c.get("label") is not None])
    vals = {s: np.array([c[f"identity::{s}"] if c.get(f"identity::{s}") is not None else np.nan
                         for c in clips if c.get("label") is not None], float) for s in sigs}

    def auroc(v, y):
        """P(signal of a label-1 clip > signal of a label-0 clip)."""
        m = np.isfinite(v) & np.isfinite(y)
        return auroc_detect(v[m][y[m] == 0], v[m][y[m] == 1])

    obs = {s: auroc(vals[s], labels) for s in sigs}
    rng = np.random.default_rng(0)
    perm_max = []
    uniq = np.unique(flights)
    for _ in range(2000):
        y = labels.copy()
        for f in uniq:                      # permute within flight: keeps positives per flight
            m = flights == f
            y[m] = rng.permutation(y[m])
        perm_max.append(max(auroc(vals[s], y) for s in sigs))
    perm_max = np.array(perm_max)
    best = max(obs, key=obs.get)
    doc["gate_selection"] = {
        "n_candidates": len(sigs), "n_positive": int(labels.sum()),
        "observed": obs, "best": best, "best_auroc": obs[best],
        "perm_max_median": float(np.median(perm_max)),
        "perm_max_p95": float(np.percentile(perm_max, 95)),
        "p_value_max_stat": float(np.mean(perm_max >= obs[best])),
        "note": "labels permuted within flight; the statistic is the maximum AUROC over the "
                "four eligible signals, so the winner's curvature is included",
    }
    # false drops per fold (good clips dropped), from the frozen-gate folds
    folds = gv["validation"]["signals"]["S3_latitude_confidence"]["lofo"]
    doc["gate_false_drops"] = {f: {"good_dropped": h["n_dropped_neg"],
                                   "bad_dropped": h["n_dropped_pos"],
                                   "n": h["n"], "coverage": h["coverage"]}
                               for f, h in folds.items()}

    # ---------------------------------------------------------- 3. pairs where both pass the gate
    gvrun = jload(latest("results/a2_verdict_gated/*/metrics.json"))["results"]
    pairs_out = {}
    for key, val in gvrun.items():
        recs = val["_records"]
        real = {(r["flight"], r["clip_id"]): r for r in recs if r["condition"] == "identity"}
        fam_out = {}
        for cond in ("tilt_k0", "tilt_k05", "tilt_k15", "mirror_roll", "yaw_control"):
            pos, neg = [], []
            for r in recs:
                if r["condition"] != cond:
                    continue
                p = real.get((r["flight"], r["clip_id"]))
                if p is None:
                    continue
                if not (p.get("gate_kept") and r.get("gate_kept")):
                    continue
                if "score" not in p or "score" not in r:
                    continue          # excitation abstained: no score exists, gate or not
                pos.append(p["score"])
                neg.append(r["score"])
            pos, neg = np.array(pos, float), np.array(neg, float)
            fam_out[cond] = {"n_pairs": int(len(pos)),
                         "auroc": (float(auroc_detect(pos, neg)) if len(pos) else None),
                         "rejection_at_6_46": (float(np.mean(neg > THRESH)) if len(neg) else None),
                         "real_median": float(np.median(pos)) if len(pos) else None,
                         "neg_median": float(np.median(neg)) if len(neg) else None}
        pairs_out[key] = fam_out
    doc["gate_only_pairs"] = pairs_out

    # ---------------------------------------------------------- 4. censored dt_min
    cfg = yaml.safe_load((ROOT / "configs/gate_frozen.yaml").read_text())
    rows = load_jsonl(ROOT / "results/gate_a2/anwm/depicted.jsonl")
    by_clip: dict[str, list[dict]] = {}
    for r in rows:
        by_clip.setdefault(r["clip_id"], []).append(r)
    man = {m["id"]: m for m in (json.loads(p.read_text()) for p in
                                sorted((ROOT / "data/processed/anwm_clips").glob("*/*/manifest.json")))}
    grid = np.round(np.arange(0.10, 5.0001, 0.05), 3)

    def retained(cid):
        rc = by_clip[cid]
        arrs = {"up": np.asarray([r["gravity_up_cam"] for r in rc]).reshape(-1, 3),
                "up_confidence_mean": np.array([r.get("up_confidence_mean", np.nan) for r in rc]),
                "latitude_confidence_mean": np.array([r.get("latitude_confidence_mean", np.nan)
                                                      for r in rc]),
                "gravity_uncertainty_raw": np.array([r.get("gravity_uncertainty_raw", np.nan)
                                                     for r in rc])}
        return bool(kept(clip_signal(cfg["signal"], arrs), cfg["tau"]))

    def dep(cid):
        up = np.asarray([r["gravity_up_cam"] for r in by_clip[cid]])
        return tilt_deg(depicted_axis_world(up, camera_to_body_pitched(0.0),
                                            np.zeros(len(up))))

    cens = {}
    for k_d in K_DS:
        for kind in ("spline", "step"):
            recs = []
            for cid in sorted(by_clip):
                if not retained(cid):
                    continue
                d = dep(cid)
                pos = np.asarray(man[cid]["commanded_path"]["positions"], float)
                found, exc_at_1 = None, None
                for dt in grid:
                    try:
                        req = frame_required(pos, float(dt), k_d, kind)
                    except ValueError:
                        continue
                    n = min(len(req), len(d))
                    ok = np.isfinite(req[:n]) & np.isfinite(d[:n])
                    if ok.sum() < 8:
                        continue
                    if exc_at_1 is None and abs(dt - 1.0) < 1e-9:
                        exc_at_1 = int((req[:n][ok] >= 10).sum())
                    if np.median(np.abs(req[:n][ok] - d[:n][ok])) <= THRESH:
                        found = float(dt)
                        break
                recs.append({"clip": cid, "regime": man[cid]["regime"],
                             "dt_min": found,
                             "status": ("left_censored" if found is not None and found <= 0.1000001
                                        else ("observed" if found is not None else "right_censored")),
                             "testable_at_1s": bool(exc_at_1 is not None and exc_at_1 >= 12),
                             "step": man[cid]["commanded_path"]["step_spacing_m"]})
            df = pd.DataFrame(recs)
            dist = {}
            for reg in ("A", "B", "C"):
                g = df[df.regime == reg]
                if not len(g):
                    continue
                dist[reg] = {
                    "n": int(len(g)),
                    "observed": int((g.status == "observed").sum()),
                    "left_censored": int((g.status == "left_censored").sum()),
                    "right_censored": int((g.status == "right_censored").sum()),
                    "untestable_at_1s": int((~g.testable_at_1s).sum()),
                    "median_observed": float(g[g.status == "observed"].dt_min.median())
                    if (g.status == "observed").any() else None,
                    "median_observed_testable_only": float(
                        g[(g.status == "observed") & g.testable_at_1s].dt_min.median())
                    if ((g.status == "observed") & g.testable_at_1s).any() else None,
                }
            cens[f"{kind}_kd{k_d}"] = {"distribution": dist, "per_clip": recs}
    doc["dt_min_censored"] = cens
    (out / "followups.json").write_text(json.dumps(doc, indent=1, default=float))

    def f(x, nd=2):
        return "n/a" if x is None else f"{x:.{nd}f}"

    print("== Regime C cells (nofit -> parity) ==")
    for k, v in cells.items():
        print(f"  {k}")
        for c in v:
            print(f"    designed {c['designed']:4.0f} n={c['n']:2d} dep {c['depicted_median']:.2f} "
                  f"req {c['required_median']:5.1f} nofit {c['residual_nofit']:5.1f}"
                  f"->{c['infeasible_nofit']:2d}/{c['testable']:2d}  parity {c['residual_parity']:5.1f}"
                  f"->{c['infeasible_parity']:2d}/{c['testable']:2d}  absorbed {c['absorbed']:4.1f}")
    print("== k_d flips ==")
    for stat, lst in flips.items():
        print(f"  {stat}: {len(lst)} clips flip")
        for x in lst:
            print(f"    {x['clip']:22s} {x['regime']} designed={x['designed']} "
                  f"kd0={x['kd0']} kd295={x['kd295']} res {x['res_kd0']:.1f}->{x['res_kd295']:.1f} "
                  f"testable {x['testable_kd0']}->{x['testable_kd295']}")
    print("== gate selection ==", json.dumps(doc["gate_selection"], indent=1))
    print("== gate-only pairs ==")
    for k, v in pairs_out.items():
        print(f"  {k}: " + " ".join(
            f"{c}={f(x['rejection_at_6_46'],2)}(n={x['n_pairs']},auc={f(x['auroc'],2)})"
            for c, x in v.items()))
    print("== censored dt_min ==")
    for k, v in cens.items():
        print(f"  {k}: " + " ".join(
            f"{r}: obs {x['observed']} left {x['left_censored']} right {x['right_censored']} "
            f"un1s {x['untestable_at_1s']} med {f(x['median_observed'])} "
            f"(testable-only {f(x['median_observed_testable_only'])})"
            for r, x in v["distribution"].items()))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
