"""G04 - path-demand windows: select windows from the path alone, then test them.

Pre-registration: ``configs/g04_path_demand_prereg.yaml`` (committed before this ran).
Reads stored TII schema-v2 clips; no GPU, no pixel re-estimation.

Two selections, both computed from ``lean_req``/``tilt_req_deg`` only:

* **demand** - at least 6 of the 8 window frames require at least 15 deg of tilt. The statistic
  is the frozen median residual (depicted vs required thrust axis).
* **quiet** - all 8 frames require at most 3 deg. The statistic is the median depicted tilt:
  tilt the path does not ask for.

Everything else (8-frame windows, hop 4, max-over-windows conformal calibration on the gate-kept
population, partial-edit composition, the metrics) is round 1's, unchanged, so the ablation
isolates the selection.

    PYTHONPATH=. python experiments/g04_path_demand.py
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
from src.verdict.segments import (HOP_FRAMES, WINDOW_FRAMES, clip_median_whole,
                                  conformal_threshold, frame_residual_deg, interval_scores,
                                  merge_intervals)

FAMILIES = ("tilt_removal", "mirror_roll", "scale_0.5", "scale_1.5")
CONTROLS = ("yaw_control", "null_edit")
FRACTIONS = (0.10, 0.25, 0.50, 1.00)
CLIP_ROOT = Path("data/processed/track_d/clips")
PREREG = Path("configs/g04_path_demand_prereg.yaml")
GATE = yaml.safe_load(Path("configs/gate_frozen.yaml").read_text())
ALPHA = 0.05
SEED = 1
N_BOOT = 5000
MIN_USABLE = 6


# ------------------------------------------------------------------ population
def cfg() -> dict:
    return yaml.safe_load(PREREG.read_text())


def real_clip_ids(flights: list[str]) -> list[str]:
    out = []
    for f in flights:
        for d in sorted((CLIP_ROOT / f).iterdir()):
            if d.name.endswith("__real"):
                out.append(d.name[: -len("__real")])
    return out


def load_conditions(cid: str) -> dict[str, Clip]:
    flight = cid.split("#")[0]
    out = {}
    for cond in ("real",) + FAMILIES + CONTROLS:
        out[cond] = load_clip(CLIP_ROOT / flight / f"{cid}__{cond}")
    return out


def assert_composable(conds: dict[str, Clip]) -> None:
    base = conds["real"]
    for name, c in conds.items():
        if not (np.allclose(c.t, base.t) and np.allclose(c.pos, base.pos)
                and np.allclose(c.yaw, base.yaw)):
            raise SystemExit(f"{name}: path side differs from real")
        if c.lean_req is None or not np.allclose(c.lean_req, base.lean_req, equal_nan=True):
            raise SystemExit(f"{name}: lean_req differs from real")


# ------------------------------------------------------------------ windows
def window_table(clip: Clip, length: int, hop: int, demand_floor: float, quiet_ceiling: float,
                 demand_min_frames: int) -> list[dict]:
    """Every window with its path classification and its two statistics."""
    tilt = np.asarray(clip.tilt_req_deg, dtype=np.float64)
    resid = frame_residual_deg(clip.depicted_lean, clip.lean_req, clip.yaw)
    dep_tilt = np.degrees(np.arcsin(np.clip(
        np.linalg.norm(np.asarray(clip.depicted_lean, dtype=np.float64), axis=1), 0, 1)))
    usable = (np.isfinite(clip.depicted_lean).all(axis=1)
              & np.isfinite(clip.lean_req).all(axis=1))
    n = len(clip.t)
    starts = list(range(0, max(1, n - length + 1), hop))
    if starts[-1] + length < n:
        starts.append(n - length)
    out = []
    for lo in starts:
        hi = lo + length
        tw = tilt[lo:hi]
        u = usable[lo:hi]
        ok = int(u.sum()) >= MIN_USABLE
        # round 1's per-window signal abstention (>=2 frames with required tilt >= 10 deg) is
        # part of the "every window" rule and is therefore applied to that arm only; demand
        # windows satisfy it by construction (>=6 frames at >=15 deg) and quiet windows are
        # exempt because the absence of required tilt is what the quiet test measures.
        signal_ok = int(np.sum(tw >= 10.0)) >= 2
        demand = ok and int(np.sum(tw >= demand_floor)) >= demand_min_frames
        quiet = ok and bool(np.all(tw <= quiet_ceiling))
        out.append({
            "start": int(lo), "stop": int(hi), "n_usable": int(u.sum()),
            "signal_ok": bool(signal_ok),
            "demand": bool(demand), "quiet": bool(quiet),
            "residual": float(np.nanmedian(resid[lo:hi][u])) if ok else float("nan"),
            "dep_tilt": float(np.nanmedian(dep_tilt[lo:hi][u])) if ok else float("nan"),
            "req_tilt_med": float(np.nanmedian(tw)),
        })
    return out


def eligible(w: dict, select: str) -> bool:
    if select == "all":
        return bool(w["signal_ok"])
    return bool(w[select])


def clip_stat(win: list[dict], key: str, select: str, persistence: int) -> float:
    """Max over eligible windows (or over consecutive pairs' minima) of one statistic."""
    vals = [w[key] if eligible(w, select) and np.isfinite(w[key]) else np.nan for w in win]
    if persistence <= 1:
        fin = [v for v in vals if np.isfinite(v)]
        return float(max(fin)) if fin else float("nan")
    pairs = [min(vals[i], vals[i + 1]) for i in range(len(vals) - 1)
             if np.isfinite(vals[i]) and np.isfinite(vals[i + 1])]
    return float(max(pairs)) if pairs else float("nan")


def flagged_mask(win: list[dict], key: str, select: str, tau: float, persistence: int,
                 n_frames: int) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    vals = [w[key] if eligible(w, select) and np.isfinite(w[key]) else np.nan for w in win]
    if persistence <= 1:
        for w, v in zip(win, vals):
            if np.isfinite(v) and v > tau:
                mask[w["start"]:w["stop"]] = True
        return mask
    for i in range(len(vals) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and min(vals[i], vals[i + 1]) > tau:
            mask[win[i]["start"]:win[i + 1]["stop"]] = True
    return mask


# ------------------------------------------------------------------ injections
def inject(real: Clip, edit: Clip, frac: float) -> tuple[Clip, np.ndarray]:
    n = real.n_frames
    if frac >= 1.0:
        return edit, np.ones(n, dtype=bool)
    m = max(1, int(round(frac * n)))
    lo = (n - m) // 2
    mask = np.zeros(n, dtype=bool)
    mask[lo:lo + m] = True
    dep = real.depicted_lean.copy()
    dep[mask] = edit.depicted_lean[mask]
    return replace(real, depicted_lean=dep, meta=dict(real.meta)), mask


def inject_unrequited_tilt(real: Clip, win: list[dict], delta_deg: float) -> tuple[Clip, np.ndarray]:
    """Add ``tan(delta) * u_hat`` to the depicted lean inside quiet windows (pre-registered)."""
    dep = np.asarray(real.depicted_lean, dtype=np.float64).copy()
    u = np.nanmean(dep, axis=0)
    if not np.isfinite(u).all() or np.linalg.norm(u) < 1e-9:
        u = np.array([1.0, 0.0])
    u = u / np.linalg.norm(u)
    mask = np.zeros(real.n_frames, dtype=bool)
    for w in win:
        if w["quiet"]:
            mask[w["start"]:w["stop"]] = True
    add = np.tan(np.deg2rad(delta_deg)) * u
    dep[mask] = dep[mask] + add
    norm = np.linalg.norm(dep, axis=1)
    cap = np.sin(np.deg2rad(80.0))
    over = norm > cap
    dep[over] = dep[over] / norm[over, None] * cap
    return replace(real, depicted_lean=dep, meta=dict(real.meta)), mask


# ------------------------------------------------------------------ stats helpers
def stat_of(r: dict, key: str) -> float:
    return float(r["whole"]) if key == "whole" else float(r["stats"].get(key, float("nan")))


def fold_thresholds(rows: list[dict], flights: list[str], key: str, alpha: float) -> dict:
    out = {}
    for f in flights:
        fit = [stat_of(r, key) for r in rows if r["flight"] != f and np.isfinite(stat_of(r, key))]
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


def clustered_rate(flags_by_flight: dict[str, list[bool]]) -> dict:
    vals = {f: np.asarray(v, float) for f, v in flags_by_flight.items() if v}
    allv = np.concatenate(list(vals.values())) if vals else np.array([])
    k, n = int(np.nansum(allv)), int(len(allv))
    out = {"k": k, "n": n, "rate": float(k / n) if n else float("nan"),
           "wilson": wilson(k, n), "n_clusters": len(vals)}
    if len(vals) < 2:
        out["boot"] = (float("nan"), float("nan"))
        out["clusters_few_warning"] = True
        return out
    rng = np.random.default_rng(SEED)
    keys = list(vals)
    draws = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        draws.append(float(np.mean(np.concatenate([vals[keys[i]] for i in pick]))))
    out["boot"] = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
    out["clusters_few_warning"] = len(vals) < 10
    return out


def rate_by_flight(rows: list[dict], key: str, flights: list[str]) -> dict:
    """Per-arm denominators: a clip with no finite statistic is not testable for that arm."""
    return clustered_rate({f: [bool(r["flags"].get(key)) for r in rows
                               if r["flight"] == f and np.isfinite(stat_of(r, key))]
                           for f in flights})


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/g04_path_demand")
    ap.add_argument("--length", type=int, default=WINDOW_FRAMES)
    ap.add_argument("--hop", type=int, default=HOP_FRAMES)
    args = ap.parse_args()

    c = cfg()
    sel = c["window_selection_from_the_path_only"]
    demand_floor = float(sel["demand_tilt_floor_deg"])
    quiet_ceiling = float(sel["quiet_tilt_ceiling_deg"])
    demand_min = 6
    dev = list(c["population"]["dev_flights"])
    clip_ids = real_clip_ids(dev)
    print(f"dev block: {len(clip_ids)} real clips over {len(dev)} flights; "
          f"demand>={demand_floor:g} deg in {demand_min}/8 frames, quiet<={quiet_ceiling:g} deg")

    real_rows, var_rows = [], []
    for cid in clip_ids:
        conds = load_conditions(cid)
        assert_composable(conds)
        flight = conds["real"].flight
        win = window_table(conds["real"], args.length, args.hop, demand_floor, quiet_ceiling,
                           demand_min)
        rec = {"clip": cid, "flight": flight, "n_frames": conds["real"].n_frames,
               "n_windows": len(win),
               "n_demand": sum(w["demand"] for w in win),
               "n_quiet": sum(w["quiet"] for w in win),
               "whole": clip_median_whole(conds["real"]),
               "gate_kept": gate_kept_from_conf(conds["real"].depicted_conf, GATE),
               "stats": {}}
        for select, key in (("all", "residual"), ("demand", "residual"), ("quiet", "dep_tilt")):
            for p in (1, 2):
                rec["stats"][f"{select}_{key}_p{p}"] = clip_stat(win, key, select, p)
        real_rows.append(rec)

        variants = [(f, fr) for f in FAMILIES + CONTROLS for fr in FRACTIONS]
        for fam, frac in variants:
            clip, truth = inject(conds["real"], conds[fam], frac)
            w2 = window_table(clip, args.length, args.hop, demand_floor, quiet_ceiling, demand_min)
            v = {"clip": cid, "flight": flight, "family": fam, "fraction": frac, "truth": truth,
                 "kind": "pixel edit", "stats": {}, "win": w2}
            for select, key in (("all", "residual"), ("demand", "residual"), ("quiet", "dep_tilt")):
                for p in (1, 2):
                    v["stats"][f"{select}_{key}_p{p}"] = clip_stat(w2, key, select, p)
            var_rows.append(v)
        for delta in c["injections"]["unrequited_tilt_in_quiet_windows"]["delta_deg"]:
            if not any(w["quiet"] for w in win):
                continue
            clip, truth = inject_unrequited_tilt(conds["real"], win, float(delta))
            w2 = window_table(clip, args.length, args.hop, demand_floor, quiet_ceiling, demand_min)
            v = {"clip": cid, "flight": flight, "family": f"unrequited_{delta:g}deg",
                 "fraction": 1.0, "truth": truth, "kind": "synthetic depicted-side injection",
                 "stats": {}, "win": w2}
            for select, key in (("all", "residual"), ("demand", "residual"), ("quiet", "dep_tilt")):
                for p in (1, 2):
                    v["stats"][f"{select}_{key}_p{p}"] = clip_stat(w2, key, select, p)
            var_rows.append(v)

    out = {"dev_flights": dev, "n_real_clips": len(clip_ids), "window_frames": args.length,
           "hop_frames": args.hop, "alpha": ALPHA, "n_boot": N_BOOT, "seed": SEED,
           "demand_floor_deg": demand_floor, "quiet_ceiling_deg": quiet_ceiling,
           "demand_min_frames": demand_min, "blocks": {},
           "population": {
               "windows": int(sum(r["n_windows"] for r in real_rows)),
               "demand_windows": int(sum(r["n_demand"] for r in real_rows)),
               "quiet_windows": int(sum(r["n_quiet"] for r in real_rows)),
               "clips_with_demand": int(sum(1 for r in real_rows if r["n_demand"] > 0)),
               "clips_with_quiet": int(sum(1 for r in real_rows if r["n_quiet"] > 0)),
               "gate_kept_clips": int(sum(1 for r in real_rows if r["gate_kept"])),
               "gate_kept_with_quiet": int(sum(1 for r in real_rows
                                               if r["gate_kept"] and r["n_quiet"] > 0)),
           }}

    for block in ("ungated", "gated"):
        pool = ([r for r in real_rows if r["gate_kept"]] if block == "gated" else real_rows)
        keys = [f"{s}_{k}_p{p}" for s, k in (("all", "residual"), ("demand", "residual"),
                                             ("quiet", "dep_tilt")) for p in (1, 2)]
        taus = {k: fold_thresholds(pool, dev, k, ALPHA) for k in keys}
        taus["whole"] = fold_thresholds(pool, dev, "whole", ALPHA)
        eval_ids = {r["clip"] for r in pool}

        for r in real_rows:
            r["flags"] = r.get("flags", {})
            for k in keys:
                r["flags"][k] = bool(np.isfinite(r["stats"][k]) and r["stats"][k] > taus[k][r["flight"]])
            r["flags"]["whole"] = bool(np.isfinite(r["whole"]) and r["whole"] > taus["whole"][r["flight"]])

        for v in var_rows:
            v.setdefault("flags", {})
            for k in keys:
                t = taus[k][v["flight"]]
                v["flags"][k] = bool(np.isfinite(t) and np.isfinite(v["stats"][k])
                                     and v["stats"][k] > t)
        scored_real = [r for r in real_rows if r["clip"] in eval_ids]
        blk = {"taus": taus,
               "real_false_reject": {k: rate_by_flight(scored_real, k, dev) for k in keys + ["whole"]},
               "n_testable": {k: int(sum(1 for r in scored_real if np.isfinite(r["stats"][k])))
                              for k in keys},
               "false_segments_per_minute": {}, "rates": {}, "localisation": {}}

        minutes, segs = 0.0, {k: 0 for k in keys}
        for r in scored_real:
            conds_real = load_clip(CLIP_ROOT / r["flight"] / f"{r['clip']}__real")
            win = window_table(conds_real, args.length, args.hop, demand_floor, quiet_ceiling,
                               demand_min)
            minutes += float(conds_real.t[-1] - conds_real.t[0])
            for k in keys:
                select, key = k.split("_")[0], "residual" if k.startswith(("all", "demand")) else "dep_tilt"
                p = int(k[-1])
                t = np.nan if not np.isfinite(taus[k][r["flight"]]) else taus[k][r["flight"]]
                if np.isfinite(t):
                    mask = flagged_mask(win, key, select, t, p, conds_real.n_frames)
                    segs[k] += len(merge_intervals(mask))
        blk["false_segments_per_minute"] = {k: (segs[k] / (minutes / 60.0) if minutes else None)
                                            for k in keys}
        blk["real_minutes"] = minutes / 60.0
        # supporting diagnostic (not pre-registered): the matched demand comparison on the clip
        # set that is testable under BOTH the every-window and the demand rule
        demand_keys = [k for k in keys if k.startswith(("all", "demand"))]
        common = {r["clip"] for r in scored_real
                  if all(np.isfinite(r["stats"][k]) for k in demand_keys)}
        blk["common_testable"] = {
            "n": len(common), "combines": [f"{a} vs {b}" for a, b in (("all", "demand"),)],
            "false_reject": {k: clustered_rate(
                {f: [bool(r["flags"][k]) for r in scored_real
                     if r["clip"] in common and r["flight"] == f] for f in dev})
                for k in demand_keys}}

        # interval metrics + detection
        for v in var_rows:
            v["interval"] = {}
            for k in keys:
                select, key = k.split("_")[0], "residual" if k.startswith(("all", "demand")) else "dep_tilt"
                p = int(k[-1])
                t = taus[k][v["flight"]]
                if not np.isfinite(t):
                    v["interval"][k] = {"iou": float("nan"), "f1": float("nan"),
                                        "onset_err_frames": float("nan"), "detected": False}
                    continue
                mask = flagged_mask(v["win"], key, select, t, p, len(v["truth"]))
                sc = interval_scores(mask, v["truth"])
                sc["severity"] = v["stats"][k]
                v["interval"][k] = sc

        def entry(fam: str, frac: float) -> dict:
            recs = [v for v in var_rows if v["family"] == fam and v["fraction"] == frac
                    and v["clip"] in eval_ids]
            e = {"n_clips": len(recs)}
            for k in keys:
                e[k] = clustered_rate({f: [bool(v["flags"][k]) for v in recs
                                           if v["flight"] == f and np.isfinite(v["stats"][k])]
                                       for f in dev})
                e[k + "_iou"] = (float(np.nanmean([v["interval"][k]["iou"] for v in recs]))
                                 if recs else float("nan"))
                e[k + "_f1"] = (float(np.nanmean([v["interval"][k]["f1"] for v in recs]))
                                if recs else float("nan"))
                onsets = [v["interval"][k]["onset_err_frames"] for v in recs if v["flags"][k]]
                e[k + "_onset"] = float(np.nanmedian(onsets)) if onsets else None
            return e

        for fam in FAMILIES + CONTROLS:
            for frac in FRACTIONS:
                blk["rates"][f"{fam}@{frac:g}"] = entry(fam, frac)
                recs = [v for v in var_rows if v["family"] == fam and v["fraction"] == frac
                        and v["clip"] in eval_ids]
                blk["rates"][f"{fam}@{frac:g}"]["common_n"] = sum(
                    1 for v in recs if all(np.isfinite(v["stats"][x]) for x in demand_keys))
                for k in demand_keys:
                    sub = [v for v in recs if all(np.isfinite(v["stats"][x]) for x in demand_keys)]
                    blk["rates"][f"{fam}@{frac:g}"][k + "_common"] = clustered_rate(
                        {f: [bool(v["flags"][k]) for v in sub if v["flight"] == f] for f in dev})
        for delta in c["injections"]["unrequited_tilt_in_quiet_windows"]["delta_deg"]:
            blk["rates"][f"unrequited_{delta:g}deg@1"] = entry(f"unrequited_{delta:g}deg", 1.0)
        out["blocks"][block] = blk

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    lines = ["clip,flight,family,fraction,kind,block,stat,flag,iou,f1,onset_err_frames,severity"]
    for block in ("ungated", "gated"):
        for v in var_rows:
            eval_ok = True
            for k, sc in v["interval"].items():
                lines.append(",".join(str(x) for x in [
                    v["clip"], v["flight"], v["family"], v["fraction"], v["kind"], block, k,
                    bool(v["flags"][k]), round(float(sc["iou"]), 4), round(float(sc["f1"]), 4),
                    sc["onset_err_frames"], round(float(v["stats"][k]), 3)]))
    (out_dir / "variants.csv").write_text("\n".join(lines) + "\n")

    # ---- print
    def fmt(b: dict) -> str:
        lo, hi = b["boot"]
        star = "*" if b.get("clusters_few_warning") else ""
        return f"{b['rate']:.3f} ({b['k']}/{b['n']}) [{lo:.2f},{hi:.2f}]{star}"

    print(f"\npopulation: {out['population']}")
    keys = [f"{s}_{k}_p{p}" for s, k in (("all", "residual"), ("demand", "residual"),
                                         ("quiet", "dep_tilt")) for p in (1, 2)]
    for block in ("ungated", "gated"):
        b = out["blocks"][block]
        print(f"\n=== {block} ===  testable clips per statistic: {b['n_testable']}")
        for k in keys + ["whole"]:
            print(f"  FRR {k:22s} {fmt(b['real_false_reject'][k])}")
        print("  false segments/min: " + ", ".join(f"{k}={v}" for k, v in
                                                   b["false_segments_per_minute"].items() if v is not None))
        for fam in FAMILIES + CONTROLS:
            for frac in FRACTIONS:
                e = b["rates"][f"{fam}@{frac:g}"]
                print(f"  {fam:13s} f={frac:4.2f} all {fmt(e['all_residual_p1'])} "
                      f"demand {fmt(e['demand_residual_p1'])} | p2: all {fmt(e['all_residual_p2'])} "
                      f"demand {fmt(e['demand_residual_p2'])} | IoU d1 {e['demand_residual_p1_iou']:.2f}")
        for dl in c["injections"]["unrequited_tilt_in_quiet_windows"]["delta_deg"]:
            e = b["rates"][f"unrequited_{float(dl):g}deg@1"]
            print(f"  unrequited {dl:g} deg: quiet {fmt(e['quiet_dep_tilt_p1'])} "
                  f"({e['n_clips']} clips) IoU {e['quiet_dep_tilt_p1_iou']:.2f}")
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
