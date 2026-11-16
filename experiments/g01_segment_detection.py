"""G01 - segment-level detection: sliding windows, scan statistic, calibrated detections.

Pre-registration: ``configs/g01_segment_prereg.yaml`` (committed before this ran).
Amendment: ``configs/g01_segment_amendment_a.yaml`` (threshold poisoning by one broken
flight; the frozen S3 gate becomes the calibration population for the amended primary).
Reads stored TII schema-v2 clips only; no GPU, no pixel re-estimation.

Partial edits are composed on the depicted side: inside the injected interval the depicted
lean of the same clip's full-clip edit is used, outside it the real clip's. The depicted
reader is per-frame (GeoCalib on one frame; the clip files record no temporal smoothing), so
this is the array a partial re-run would produce.

    python experiments/g01_segment_detection.py --population dev
    python experiments/g01_segment_detection.py --population heldout --confirm-once
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from src.verdict.abstain import gate_kept_from_conf
from src.verdict.schema import Clip, load_clip
from src.verdict.segments import (HOP_FRAMES, TIERS, WINDOW_FRAMES, clip_max,
                                  clip_max_persistent, clip_median_whole,
                                  conformal_threshold, flagged_windows,
                                  flagged_windows_persistent, interval_scores,
                                  merge_intervals, window_statistics)

FAMILIES = ("tilt_removal", "mirror_roll", "scale_0.5", "scale_1.5")
CONTROLS = ("yaw_control", "null_edit")
FRACTIONS = (0.10, 0.25, 0.50)
PERSISTENCE = (1, 2)
CLIP_ROOT = Path("data/processed/track_d/clips")
PREREG = Path("configs/g01_segment_prereg.yaml")
ALPHA = 0.05
SEED = 1
N_BOOT = 5000
GATE = yaml.safe_load(Path("configs/gate_frozen.yaml").read_text())


# ------------------------------------------------------------------ population
def dev_and_heldout() -> tuple[list[str], list[str]]:
    cfg = yaml.safe_load(PREREG.read_text())
    return list(cfg["population"]["dev_flights"]), list(cfg["population"]["held_out_flights"])


def real_clip_ids(flights: list[str]) -> list[str]:
    out = []
    for f in flights:
        for d in sorted((CLIP_ROOT / f).iterdir()):
            if d.name.endswith("__real"):
                out.append(d.name[: -len("__real")])
    return out


def load_conditions(clip_id: str) -> dict[str, Clip]:
    flight = clip_id.split("#")[0]
    out = {}
    for cond in ("real",) + FAMILIES + CONTROLS:
        d = CLIP_ROOT / flight / f"{clip_id}__{cond}"
        if not d.exists():
            raise SystemExit(f"missing condition {cond} for {clip_id}")
        out[cond] = load_clip(d)
    return out


def assert_composable(conds: dict[str, Clip]) -> None:
    """The path side must be identical across conditions, or composition is meaningless."""
    base = conds["real"]
    for name, c in conds.items():
        if not (np.allclose(c.t, base.t) and np.allclose(c.pos, base.pos)
                and np.allclose(c.yaw, base.yaw)):
            raise SystemExit(f"{name}: path side differs from real")
        if c.lean_req is None or not np.allclose(c.lean_req, base.lean_req, equal_nan=True):
            raise SystemExit(f"{name}: lean_req differs from real")


def inject(real: Clip, edit: Clip, frac: float) -> tuple[Clip, np.ndarray]:
    """Swap the depicted lean inside a centred interval; return the clip and the true mask."""
    n = real.n_frames
    if frac >= 1.0:
        return edit, np.ones(n, dtype=bool)
    m = max(1, int(round(frac * n)))
    lo = (n - m) // 2
    mask = np.zeros(n, dtype=bool)
    mask[lo:lo + m] = True
    dep = real.depicted_lean.copy()
    dep[mask] = edit.depicted_lean[mask]
    conf = real.depicted_conf.copy()
    conf[mask] = edit.depicted_conf[mask]
    meta = dict(real.meta)
    meta["condition"] = f"partial:{edit.condition}:{frac:g}"
    meta["injected_index"] = [int(lo), int(lo + m)]
    return replace(real, depicted_lean=dep, depicted_conf=conf, meta=meta), mask


# ------------------------------------------------------------------ statistics
def variant_stats(clip: Clip, length: int, hop: int) -> dict:
    stats = window_statistics(clip, length=length, hop=hop)
    return {"stats": stats,
            "max": {t: clip_max(stats, t) for t in TIERS},
            "max_p2": {t: clip_max_persistent(stats, t, 2) for t in TIERS},
            "whole": clip_median_whole(clip),
            "n_windows": len(stats),
            "abstained": all(s.abstained for s in stats)}


def fold_thresholds(rows: list[dict], flights: list[str], key: str,
                    alpha: float = ALPHA) -> dict:
    """Leave-one-flight-out conformal threshold on ``key`` from the fit flights' real clips."""
    out = {}
    for f in flights:
        fit = [r[key] for r in rows if r["flight"] != f and np.isfinite(r.get(key, np.nan))]
        out[f] = conformal_threshold(fit, alpha)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - s) / d), float((c + s) / d)


def clustered_rate(flags_by_flight: dict[str, list[bool]], rng_seed: int = SEED,
                   n_boot: int = N_BOOT) -> dict:
    """Pooled rate with a flight-clustered bootstrap (descriptive: few clusters)."""
    vals = {f: np.asarray(v, dtype=float) for f, v in flags_by_flight.items() if v}
    allv = np.concatenate(list(vals.values())) if vals else np.array([])
    k, n = int(np.nansum(allv)), int(len(allv))
    out = {"k": k, "n": n, "rate": float(k / n) if n else float("nan"),
           "wilson": wilson(k, n), "n_clusters": len(vals)}
    if len(vals) < 2:
        out["boot"] = (float("nan"), float("nan"))
        out["clusters_few_warning"] = True
        return out
    rng = np.random.default_rng(rng_seed)
    keys = list(vals)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        draws.append(float(np.mean(np.concatenate([vals[keys[i]] for i in pick]))))
    out["boot"] = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
    out["clusters_few_warning"] = len(vals) < 10
    return out


def rate_by_flight(records: list[dict], key: str, flights: list[str]) -> dict:
    return clustered_rate({f: [bool(r["flags"][key]) for r in records if r["flight"] == f]
                           for f in flights})


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", choices=("dev", "heldout"), default="dev")
    ap.add_argument("--confirm-once", action="store_true",
                    help="required for the held-out block: it is scored exactly once")
    ap.add_argument("--out", default="results/g01_segments")
    ap.add_argument("--length", type=int, default=WINDOW_FRAMES)
    ap.add_argument("--hop", type=int, default=HOP_FRAMES)
    ap.add_argument("--calib-population", choices=("all", "gate_kept"), default="all")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dev, held = dev_and_heldout()
    if args.population == "heldout" and not args.confirm_once:
        raise SystemExit("the held-out flights are scored once, on purpose: pass --confirm-once")
    flights = dev if args.population == "dev" else held
    clip_ids = real_clip_ids(flights)
    print(f"population {args.population}: {len(clip_ids)} real clips over {len(flights)} "
          f"flights; window {args.length} hop {args.hop}, calibration "
          f"{args.calib_population}")

    real_records, variant_records = [], []
    for cid in clip_ids:
        conds = load_conditions(cid)
        assert_composable(conds)
        flight = conds["real"].flight
        real = variant_stats(conds["real"], args.length, args.hop)
        real_records.append({"clip": cid, "flight": flight,
                             "n_frames": conds["real"].n_frames, **real,
                             "gate_kept": gate_kept_from_conf(conds["real"].depicted_conf, GATE)})
        for fam in FAMILIES + CONTROLS:
            for frac in FRACTIONS + (1.0,):
                clip, truth = inject(conds["real"], conds[fam], frac)
                v = variant_stats(clip, args.length, args.hop)
                variant_records.append({"clip": cid, "flight": flight, "family": fam,
                                        "fraction": frac, "truth": truth, **v})

    calib = ([r for r in real_records if r["gate_kept"]]
             if args.calib_population == "gate_kept" else real_records)
    keep = {r["clip"] for r in calib}
    print(f"calibration population: {args.calib_population} ({len(calib)} real clips, "
          f"{len(real_records) - len(calib)} excluded)")

    out = {"population": args.population, "flights": flights,
           "n_real_clips": len(clip_ids), "calibration_population": args.calib_population,
           "n_calibration_clips": len(calib), "window_frames": args.length,
           "hop_frames": args.hop, "alpha": ALPHA, "n_boot": N_BOOT, "seed": SEED,
           "n_windows_median": float(np.median([r["n_windows"] for r in real_records])),
           "n_abstained_clips": int(sum(1 for r in real_records if r["abstained"])),
           "blocks": {}}

    for p in PERSISTENCE:
        # thresholds on the persistence statistic itself
        for r in calib:
            for t in TIERS:
                r[f"stat_{t}_p{p}"] = r["max"][t] if p == 1 else r["max_p2"][t]
        taus = {t: fold_thresholds(calib, flights, f"stat_{t}_p{p}") for t in TIERS}
        tau_whole = fold_thresholds(calib, flights, "whole")

        def stat_of(rec, tier):
            return rec["max"][tier] if p == 1 else rec["max_p2"][tier]

        for rec in real_records + variant_records:
            rec.setdefault("flags", {})
            for t in TIERS:
                rec["flags"][f"{t}_p{p}"] = bool(np.isfinite(stat_of(rec, t))
                                                 and stat_of(rec, t) > taus[t][rec["flight"]])
            rec["flags"][f"whole_p{p}"] = bool(np.isfinite(rec["whole"])
                                               and rec["whole"] > tau_whole[rec["flight"]])

        scored_real = [r for r in real_records if r["clip"] in keep]
        block = {"taus": taus, "tau_whole": tau_whole,
                 "real_false_reject": {f"{t}_p{p}": rate_by_flight(scored_real, f"{t}_p{p}", flights)
                                       for t in TIERS + ("whole",)},
                 "per_flight_false_reject": {
                     f: {"n": sum(1 for r in scored_real if r["flight"] == f),
                         "residual": int(sum(1 for r in scored_real if r["flight"] == f
                                             and r["flags"][f"residual_p{p}"]))}
                     for f in flights},
                 "false_segments_per_minute": {}, "rates": {}, "incremental": {}}

        # false segments per minute on the scored real clips
        minutes, seg_counts = 0.0, {t: 0 for t in TIERS}
        for cid in [i for i in clip_ids if i in keep]:
            conds_real = load_clip(CLIP_ROOT / cid.split("#")[0] / f"{cid}__real")
            stats = window_statistics(conds_real, args.length, args.hop)
            minutes += float(conds_real.t[-1] - conds_real.t[0])
            for t in TIERS:
                mask = (flagged_windows_persistent(stats, t, taus[t][cid.split("#")[0]], p)
                        if p > 1 else flagged_windows(stats, t, taus[t][cid.split("#")[0]]))
                seg_counts[t] += len(merge_intervals(mask))
        block["false_segments_per_minute"] = {t: seg_counts[t] / (minutes / 60.0) for t in TIERS}
        block["real_minutes"] = minutes / 60.0

        # interval metrics + aggregate rates
        for v in variant_records:
            flight = v["flight"]
            v["interval"] = {}
            for t in TIERS:
                stats = v["stats"]
                mask = (flagged_windows_persistent(stats, t, taus[t][flight], p) if p > 1
                        else flagged_windows(stats, t, taus[t][flight]))
                sc = interval_scores(mask, v["truth"])
                sc["severity"] = float(stat_of(v, t))
                v["interval"][t] = sc

        real_flag = {r["clip"]: r["flags"][f"residual_p{p}"] for r in real_records}
        for fam in FAMILIES + CONTROLS:
            for frac in FRACTIONS + (1.0,):
                key = f"{fam}@{frac:g}"
                entry = {}
                for t in TIERS + ("whole",):
                    recs = [v for v in variant_records
                            if v["family"] == fam and v["fraction"] == frac and v["clip"] in keep]
                    entry[f"{t}_p{p}"] = clustered_rate(
                        {f: [bool(v["flags"][f"{t}_p{p}"]) for v in recs if v["flight"] == f]
                         for f in flights})
                    # post-hoc diagnostic: detection among clips that pass as real
                    clean = [v for v in recs if not real_flag[v["clip"]]]
                    entry[f"{t}_p{p}_incremental"] = clustered_rate(
                        {f: [bool(v["flags"][f"{t}_p{p}"]) for v in clean if v["flight"] == f]
                         for f in flights})
                recs = [v for v in variant_records
                        if v["family"] == fam and v["fraction"] == frac and v["clip"] in keep]
                entry["iou_residual_mean"] = float(np.nanmean(
                    [v["interval"]["residual"]["iou"] for v in recs]))
                entry["f1_residual_mean"] = float(np.nanmean(
                    [v["interval"]["residual"]["f1"] for v in recs]))
                entry["iou_coherence_mean"] = float(np.nanmean(
                    [v["interval"]["coherence"]["iou"] for v in recs]))
                onsets = [v["interval"]["residual"]["onset_err_frames"] for v in recs
                          if v["flags"][f"residual_p{p}"]]
                entry["onset_err_frames_median"] = (float(np.nanmedian(onsets)) if onsets else None)
                entry["onset_err_frames_all"] = [None if not np.isfinite(x) else float(x)
                                                 for x in onsets]
                block["rates"][key] = entry
        out["blocks"][f"p{p}"] = block

    out_dir = Path(args.out) / args.population
    if args.tag:
        out_dir = out_dir.with_name(out_dir.name + f"_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=1, default=float))

    lines = ["clip,flight,family,fraction,p,flag_res,flag_coh,flag_sign,flag_whole,"
             "iou_res,f1_res,onset_err_frames,severity_res"]
    for p in PERSISTENCE:
        for v in variant_records:
            lines.append(",".join(str(x) for x in [
                v["clip"], v["flight"], v["family"], v["fraction"], p,
                v["flags"][f"residual_p{p}"], v["flags"][f"coherence_p{p}"],
                v["flags"][f"sign_p{p}"], v["flags"][f"whole_p{p}"],
                round(v["interval"]["residual"]["iou"], 4),
                round(v["interval"]["residual"]["f1"], 4),
                v["interval"]["residual"]["onset_err_frames"],
                round(v["interval"]["residual"]["severity"], 3)]))
    (out_dir / "variants.csv").write_text("\n".join(lines) + "\n")

    # ---- print
    def fmt(b: dict) -> str:
        lo, hi = b["boot"]
        star = "*" if b.get("clusters_few_warning") else ""
        return f"{b['rate']:.3f} ({b['k']}/{b['n']}) [{lo:.2f},{hi:.2f}]{star}"

    for p in PERSISTENCE:
        b = out["blocks"][f"p{p}"]
        print(f"\n=== persistence {p} ===")
        print("tau residual per fold: " + ", ".join(
            f"{k.replace('flight-','')}={v:.1f}" for k, v in b["taus"]["residual"].items()))
        print("tau whole median per fold: " + ", ".join(
            f"{k.replace('flight-','')}={v:.1f}" for k, v in b["tau_whole"].items()))
        print("real clips (false reject): " + "  ".join(
            f"{t} {fmt(b['real_false_reject'][t])}" for t in
            ("residual_p1" if p == 1 else "residual_p2", "coherence_p%d" % p,
             "sign_p%d" % p, "whole_p%d" % p)))
        print("false segments per real minute: " + ", ".join(
            f"{t} {v:.2f}" for t, v in b["false_segments_per_minute"].items()))
        for fam in FAMILIES + CONTROLS:
            for frac in FRACTIONS + (1.0,):
                e = b["rates"][f"{fam}@{frac:g}"]
                print(f"  {fam:13s} f={frac:4.2f} resid {fmt(e['residual_p%d' % p])} "
                      f"incr {fmt(e['residual_p%d_incremental' % p])} "
                      f"whole {fmt(e['whole_p%d' % p])} coh {fmt(e['coherence_p%d' % p])} "
                      f"IoU {e['iou_residual_mean']:.2f} F1 {e['f1_residual_mean']:.2f} "
                      f"onset {e['onset_err_frames_median']}")
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
