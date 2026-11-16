"""G02 - the combined verdict score: max of standardised tiers, calibrated conformally.

Pre-registration: ``configs/g02_combined_prereg.yaml``.  Reads stored TII schema-v2 clips plus
Track D's frozen per-clip gain/coherence tables; no GPU.

Tiers (frozen definitions, never refit):

* ``residual``: clip median of the per-frame depicted-vs-required thrust-axis angle (deg);
* ``coherence``: ``1 - min(r_x, r_y)`` from ``src.verdict.coherence``, at the clip's own
  gain-fit lag (the frozen tier's convention);
* ``sign``: ``1 - min(s_x, s_y)``;
* ``gain``: the frozen JointBand statistic ``max(|gx-c|/s, |gy-c|/s)`` with the per-fold
  centres and scales of ``configs/gain_frozen.yaml``.

Attack path: each tier is standardised on the fit flights of its fold (median, robust scale),
the score is the maximum of the z-scores, and the threshold is the leave-one-flight-out
conformal 95th percentile of that score.  The minimum-detectable-violation curve scales the
real clip's depicted lean vector, ``l_dep -> s * l_dep``, which is a synthetic manipulation of
the depicted side (no pixels re-estimated) and is declared as such.

    python experiments/g02_combined_score.py --population dev
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from src.verdict.coherence import coherence
from src.verdict.fit import FitOptions, _shift_series, fit_gain
from src.verdict.schema import Clip, load_clip
from src.verdict.segments import frame_residual_deg

CONDITIONS = ("real", "tilt_removal", "mirror_roll", "scale_0.5", "scale_1.5",
              "yaw_control", "null_edit")
S_GRID = (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0)
CLIP_ROOT = Path("data/processed/track_d/clips")
ALPHA = 0.05
SEED = 1
N_BOOT = 5000
GAIN_CFG = yaml.safe_load(Path("configs/gain_frozen.yaml").read_text())
PREREG = Path("configs/g02_combined_prereg.yaml")


def dev_and_heldout() -> tuple[list[str], list[str]]:
    cfg = yaml.safe_load(PREREG.read_text())
    return list(cfg["population"]["dev_flights"]), list(cfg["population"]["held_out_flights"])


def stored_tables() -> tuple[dict, dict]:
    """Track D's frozen per-clip gain and coherence rows, keyed (flight, clip_id, condition)."""
    import csv

    gain, coh = {}, {}
    with open("results/d_eval/d02/gain_table.csv") as fh:
        for row in csv.DictReader(fh):
            gain[(row["flight"], row["clip_id"], row["condition"])] = row
    with open("results/d_coherence/d01/coherence_table.csv") as fh:
        for row in csv.DictReader(fh):
            coh[(row["flight"], row["clip_id"], row["condition"])] = row
    return gain, coh


def apply_lag(clip: Clip, lag_ms: float) -> Clip:
    """Move the required side onto the depicted side by the frozen clip-level lag."""
    if not np.isfinite(lag_ms) or lag_ms == 0:
        return clip
    return replace(clip,
                   lean_req=_shift_series(clip.lean_req, clip.t, lag_ms / 1000.0),
                   vel_req=_shift_series(clip.vel_req, clip.t, lag_ms / 1000.0))


def gain_statistic(clip: Clip, lag_ms: float, fold: str) -> tuple[float, float, float]:
    """Frozen JointBand statistic from a bootstrap-free fit at the frozen lag."""
    opts = FitOptions(drag="fitted", intercept=True, lag_search=False, bootstrap=False)
    res = fit_gain(apply_lag(clip, lag_ms), opts)
    band = GAIN_CFG["bands_per_fold"].get(fold, GAIN_CFG["bands"])["axis"]
    if band["center"]["gx"] is None:
        return float("nan"), float("nan"), float("nan")
    zx = abs((res.axis_gains[0] - band["center"]["gx"]) / band["scale"]["gx"])
    zy = abs((res.axis_gains[1] - band["center"]["gy"]) / band["scale"]["gy"])
    return float(max(zx, zy)), float(res.axis_gains[0]), float(res.axis_gains[1])


def tiers(clip: Clip, lag_ms: float, fold: str) -> dict:
    resid = frame_residual_deg(clip.depicted_lean, clip.lean_req, clip.yaw)
    ok = np.isfinite(resid)
    coh = coherence(clip, lag_ms=float(lag_ms) if np.isfinite(lag_ms) else 0.0)
    m, gx, gy = gain_statistic(clip, lag_ms, fold)
    return {"residual": float(np.nanmedian(resid[ok])) if ok.any() else float("nan"),
            "coherence": coh.correlation_score, "sign": coh.sign_score, "gain": m,
            "gx": gx, "gy": gy}


def scale_depicted(clip: Clip, s: float) -> Clip:
    return replace(clip, depicted_lean=clip.depicted_lean * s)


def auc(neg: np.ndarray, pos: np.ndarray) -> float:
    """Rank AUC: probability a positive scores above a negative (ties count half)."""
    neg = np.asarray([v for v in neg if np.isfinite(v)])
    pos = np.asarray([v for v in pos if np.isfinite(v)])
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    allv = np.concatenate([neg, pos])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[len(neg):].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def robust(vals: np.ndarray) -> tuple[float, float]:
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if len(v) < 8:
        return float("nan"), float("nan")
    lo, hi = np.quantile(v, [0.025, 0.975])
    return float(np.median(v)), float(max((hi - lo) / 3.92, 1e-6))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", choices=("dev", "heldout"), default="dev")
    ap.add_argument("--confirm-once", action="store_true")
    ap.add_argument("--out", default="results/g02_combined")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dev, held = dev_and_heldout()
    if args.population == "heldout" and not args.confirm_once:
        raise SystemExit("the held-out flights are scored once: pass --confirm-once")
    flights = dev if args.population == "dev" else held
    gain_tbl, coh_tbl = stored_tables()

    # ---------------------------------------------------------------- scores
    records = []
    for f in flights:
        for d in sorted((CLIP_ROOT / f).iterdir()):
            if not d.name.endswith("__real"):
                continue
            cid = d.name[: -len("__real")]
            key = (f, cid, "real")
            if key not in gain_tbl:
                continue
            row = gain_tbl[key]
            testable = row["testable_pair"] in ("True", "true")
            lag = float(row["lag_ms"]) if row["lag_ms"] not in ("", "nan") else float("nan")
            base = load_clip(d)
            rec = {"clip": cid, "flight": f, "testable": testable, "lag_ms": lag,
                   "gx_stored": float(row["gx"]), "gy_stored": float(row["gy"]),
                   "variants": {}}
            for cond in CONDITIONS:
                clip = load_clip(CLIP_ROOT / f / f"{cid}__{cond}")
                rec["variants"][cond] = tiers(clip, lag, f)
            for s in S_GRID:
                rec["variants"][f"scale_s{s:g}"] = tiers(scale_depicted(base, s), lag, f)
            records.append(rec)

    testable = [r for r in records if r["testable"]]
    print(f"{args.population}: {len(records)} real clips, {len(testable)} testable "
          f"(frozen abstention), {len(flights)} flights")

    # validation of the tier implementation against Track D's frozen tables
    dgx = [abs(r["variants"]["real"]["gx"] - r["gx_stored"]) for r in records
           if np.isfinite(r["variants"]["real"]["gx"])]
    dgy = [abs(r["variants"]["real"]["gy"] - r["gy_stored"]) for r in records
           if np.isfinite(r["variants"]["real"]["gy"])]
    dcoh = []
    for r in records:
        key = (r["flight"], r["clip"], "mirror_roll")
        if key in coh_tbl:
            mine = r["variants"]["mirror_roll"]["coherence"]
            theirs = float(coh_tbl[key]["correlation_score"])
            if np.isfinite(mine) and np.isfinite(theirs):
                dcoh.append(abs(mine - theirs))
    print(f"implementation check vs frozen tables: max|dgx| {max(dgx):.2e}, "
          f"max|dgy| {max(dgy):.2e}, max|dcoh| {max(dcoh):.2e} (n={len(dcoh)})")

    # ---------------------------------------------------------------- calibration
    names = {"G_max3": ("residual", "coherence", "gain"),
             "G_max2_no_resid": ("coherence", "gain"),
             "G_max3_sign": ("residual", "coherence", "sign", "gain")}
    out = {"population": args.population, "flights": flights,
           "n_real": len(records), "n_testable": len(testable), "alpha": ALPHA,
           "n_boot": N_BOOT, "seed": SEED,
           "validation": {"max_dgx": max(dgx), "max_dgy": max(dgy), "max_dcoh": max(dcoh)},
           "combined": {}, "single_tier": {}, "auc": {}, "mdv": {}, "rates": {}}

    for name, comps in names.items():
        taus, params = {}, {}
        for fold in flights:
            fit = [r for r in testable if r["flight"] != fold]
            center, scale = {}, {}
            for t in comps:
                center[t], scale[t] = robust(np.array([r["variants"]["real"][t] for r in fit],
                                                      dtype=float))
            params[fold] = {"center": center, "scale": scale}
            stat = []
            for r in fit:
                zs = [(r["variants"]["real"][t] - center[t]) / scale[t] for t in comps]
                if np.isfinite(zs).all():
                    stat.append(max(zs))
            taus[fold] = float(np.quantile(stat, 1 - ALPHA)) if len(stat) >= 8 else float("nan")
        out["combined"][name] = {"taus": taus, "params": params}
        for r in testable:
            fold = r["flight"]
            for cond, tv in r["variants"].items():
                zs = [(tv[t] - params[fold]["center"][t]) / params[fold]["scale"][t]
                      for t in comps]
                r[f"{cond}::{name}"] = (float(max(zs)) if np.isfinite(zs).all()
                                        else float("nan"))

    # Fisher combination of one-sided conformal p-values
    for r in testable:
        fold = r["flight"]
        for cond, tv in r["variants"].items():
            tot, ok = 0.0, True
            for t in ("residual", "coherence", "gain"):
                fit = np.array([x["variants"]["real"][t] for x in testable
                                if x["flight"] != fold], dtype=float)
                fit = fit[np.isfinite(fit)]
                if len(fit) < 8 or not np.isfinite(tv[t]):
                    ok = False
                    break
                p = (1.0 + float(np.sum(fit >= tv[t]))) / (len(fit) + 1.0)
                tot += -2.0 * np.log(p)
            r[f"{cond}::G_fisher"] = float(tot) if ok else float("nan")
    fisher_taus = {}
    for fold in flights:
        vals = [r["real::G_fisher"] for r in testable
                if r["flight"] != fold and np.isfinite(r["real::G_fisher"])]
        fisher_taus[fold] = (float(np.quantile(vals, 1 - ALPHA)) if len(vals) >= 8
                             else float("nan"))
    out["combined"]["G_fisher"] = {"taus": fisher_taus, "params": {}}

    # single tiers, same conformal rule
    for t in ("residual", "coherence", "sign", "gain"):
        taus = {}
        for fold in flights:
            vals = np.array([r["variants"]["real"][t] for r in testable
                             if r["flight"] != fold], dtype=float)
            vals = vals[np.isfinite(vals)]
            taus[fold] = float(np.quantile(vals, 1 - ALPHA)) if len(vals) >= 8 else float("nan")
        out["single_tier"][t] = {"taus": taus}

    # ---------------------------------------------------------------- rates
    methods = list(names) + ["G_fisher"]
    cells = CONDITIONS + tuple(f"scale_s{s:g}" for s in S_GRID)
    for m in methods + ["residual", "coherence", "sign", "gain"]:
        taus = (out["combined"][m]["taus"] if m in out["combined"]
                else out["single_tier"][m]["taus"])
        for cond in cells:
            flags = []
            for r in testable:
                v = (r.get(f"{cond}::{m}") if m in out["combined"]
                     else r["variants"][cond][m])
                flags.append(bool(np.isfinite(v) and v > taus[r["flight"]]))
            out["rates"][f"{cond}|{m}"] = {"k": int(sum(flags)), "n": len(flags),
                                           "rate": float(np.mean(flags)) if flags else None}

    # ---------------------------------------------------------------- AUC
    def auc_pool(rs, cond, m):
        if m in out["combined"]:
            return auc([r[f"real::{m}"] for r in rs], [r.get(f"{cond}::{m}") for r in rs])
        return auc([r["variants"]["real"][m] for r in rs], [r["variants"][cond][m] for r in rs])

    rng = np.random.default_rng(SEED)
    by_flight = {f: [r for r in testable if r["flight"] == f] for f in flights}
    for cond in cells:
        for m in methods + ["residual", "coherence", "gain"]:
            val = auc_pool(testable, cond, m)
            boots = []
            for _ in range(N_BOOT):
                pick = rng.choice(flights, size=len(flights), replace=True)
                rs = [r for f in pick for r in by_flight[f]]
                a = auc_pool(rs, cond, m)
                if np.isfinite(a):
                    boots.append(a)
            lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots
                      else (float("nan"), float("nan")))
            out["auc"][f"{cond}|{m}"] = {"auc": val, "lo": float(lo), "hi": float(hi),
                                         "n_boot": len(boots)}

    # ---------------------------------------------------------------- MDV
    for m in methods + ["residual", "coherence", "gain"]:
        curve = []
        for s in S_GRID:
            row = out["rates"][f"scale_s{s:g}|{m}"]
            curve.append({"s": s, "violation": 1 - s, "detection": row["rate"],
                          "k": row["k"], "n": row["n"]})
        out["mdv"][m] = {"curve": curve,
                         "mdv50": next((c["violation"] for c in curve
                                        if (c["detection"] or 0) >= 0.5), None)}

    out_dir = Path(args.out) / args.population
    if args.tag:
        out_dir = out_dir.with_name(out_dir.name + "_" + args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    cols = [f"{c}::{m}" for c in CONDITIONS for m in methods]
    with open(out_dir / "per_clip.csv", "w") as fh:
        fh.write(",".join(["clip", "flight", "testable"] + cols) + "\n")
        for r in testable:
            fh.write(",".join([r["clip"], r["flight"], str(r["testable"])]
                              + [repr(r.get(c, "")) for c in cols]) + "\n")

    # ---------------------------------------------------------------- print
    def fmt(key: str) -> str:
        v = out["rates"][key]
        return f"{v['rate']:.3f} ({v['k']}/{v['n']})" if v["rate"] is not None else "n/a"

    print("\nmethod            " + "".join(f"{c[:12]:>15s}" for c in CONDITIONS))
    for m in methods + ["residual", "coherence", "gain"]:
        print(f"{m:18s}" + "".join(f"{fmt(c + '|' + m):>15s}" for c in CONDITIONS))
    print("\nAUC (LOFO score, real vs condition), flight-clustered spread (descriptive)")
    for m in methods + ["residual", "coherence", "gain"]:
        print(f"  {m:18s}" + "".join(
            f" {c[:8]} {out['auc'][c + '|' + m]['auc']:.3f}"
            f"[{out['auc'][c + '|' + m]['lo']:.2f},{out['auc'][c + '|' + m]['hi']:.2f}]"
            for c in ("tilt_removal", "mirror_roll", "scale_0.5", "scale_1.5")))
    print("\nminimum detectable violation at 50 % detection (1 - s)")
    for m in methods + ["residual", "coherence", "gain"]:
        print(f"  {m:18s} mdv50 = {out['mdv'][m]['mdv50']}")
    print("\nMDV curves (detection vs 1-s)")
    for m in ("G_max3", "residual", "coherence", "gain"):
        print(f"  {m:10s} " + " ".join(
            f"{c['violation']:.2f}:{(c['detection'] or 0):.2f}" for c in out["mdv"][m]["curve"]))
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
