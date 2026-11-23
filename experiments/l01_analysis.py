"""Track L, L01 analysis: scene-stratified reader error on the simulated feasible dev clips.

Reads only stored per-frame arrays produced by ``remote/l_score_sim.py`` (no GPU, no pixels) and
the frozen configs. Everything it computes is listed in ``configs/l01_scene_stratified_prereg.yaml``:

* per-clip error and signed bias in the verdict's lean basis;
* the 3 families x 4 mounts grid, family rows and the family spread per mount;
* the frozen S3 confidence gate, applied unchanged, with keep/drop error;
* J01's constant / slow / fast decomposition per clip, at the simulated 30 Hz sampling.

    PYTHONPATH=. python experiments/l01_analysis.py
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.depicted.anatomy import decompose

ROOT = Path(".")
MANIFEST = ROOT / "results/i02/manifest.csv"
GRID = ROOT / "results/i02/reader_feasible_grid.csv"
GATE = ROOT / "configs/gate_frozen.yaml"
SCORES = ROOT / "data/l01/scores"
OUT = ROOT / "results/l01"
FAMILIES = ["urban_checker", "forest_ridge", "industrial"]
MOUNTS = [0.0, 15.0, 30.0, 45.0]


def gate_tau() -> float:
    text = GATE.read_text()
    for line in text.splitlines():
        if '"tau"' in line:
            return float(line.split(":")[1].strip().rstrip(","))
    raise RuntimeError("tau not found in gate_frozen.yaml")


def summarise(x) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "median": float(np.median(x)),
            "mean": float(np.mean(x)), "p90": float(np.percentile(x, 90)),
            "min": float(x.min()), "max": float(x.max())}


def anatomy(err: np.ndarray, t: np.ndarray) -> dict:
    """J01's decomposition (copied byte-identically from track-j-reader) on one lean axis."""
    err = np.asarray(err, dtype=float)
    if err.size < 8 or not np.isfinite(err).all():
        return {}
    return decompose(err, np.asarray(t, dtype=float), [(0, int(err.size))])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(SCORES))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    scores = Path(args.scores)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [r for r in csv.DictReader(MANIFEST.open())
            if r["split"] == "dev" and r["label"] == "feasible"]
    grid = {r["clip_id"]: r for r in csv.DictReader(GRID.open())}
    tau = gate_tau()

    per_clip = []
    repro = []
    missing = []
    for row in rows:
        cid = row["clip_id"]
        path = scores / f"{cid}.npz"
        if not path.exists():
            missing.append(cid)
            continue
        z = np.load(path)
        err = np.asarray(z["up_error_deg"], dtype=float)
        lean_est = np.asarray(z["lean_est"], dtype=float)
        lean_dep = np.asarray(z["lean_dep"], dtype=float)
        e = lean_est - lean_dep
        s3 = -float(np.nanmedian(z["latitude_confidence_mean"]))
        dt = float(z["t"][1] - z["t"][0]) if len(z["t"]) > 1 else np.nan

        # reproduction on the exact frames I02 scored: its stored score JSON carries the
        # per-frame rows, so the frame-enumeration ambiguity of the summary alone is removed
        g = grid[cid]
        stored = ROOT / "results/i02/scores" / f"{cid}_score.json"
        if stored.exists():
            frames_i02 = [int(r["frame"]) for r in json.loads(stored.read_text())["rows"]]
            pos = {int(f): k for k, f in enumerate(np.asarray(z["frame"]))}
            pairs = [(float(r["up_error_deg"]), float(err[pos[int(r["frame"])]]))
                     for r in json.loads(stored.read_text())["rows"] if int(r["frame"]) in pos]
            sub = np.array([b for a, b in pairs])
            st = np.array([a for a, b in pairs])
            diff = np.abs(sub - st)
        else:
            sub, diff = err, np.array([np.nan])
        repro.append({"clip_id": cid, "n_i02": int(sub.size),
                      "i02_median": float(g["up_med"]),
                      "rescore_median_subset": float(np.median(sub)),
                      "delta": float(np.median(sub) - float(g["up_med"])),
                      "frame_median_abs_diff": float(np.nanmedian(diff)),
                      "frame_p90_abs_diff": float(np.nanpercentile(diff, 90))})

        an = {}
        tt = np.asarray(z["t"], dtype=float)
        for k, axis in enumerate(("forward", "left")):
            an[axis] = anatomy(e[:, k], tt)
        per_clip.append({
            "clip_id": cid, "family": row["family"], "uptilt_deg": float(row["uptilt_deg"]),
            "scene_seed": int(row["scene_seed"]), "maneuver": row["maneuver"],
            "n_frames": int(err.size), "dt": dt,
            "err_median": float(np.median(err)),
            "err_mean": float(np.mean(err)),
            "err_p90": float(np.percentile(err, 90)),
            "frac_gt10": float(np.mean(err > 10.0)),
            "frac_gt20": float(np.mean(err > 20.0)),
            "bias_forward": float(e[:, 0].mean()),
            "bias_left": float(e[:, 1].mean()),
            "bias_forward_se": float(e[:, 0].std(ddof=1) / np.sqrt(len(e))),
            "bias_left_se": float(e[:, 1].std(ddof=1) / np.sqrt(len(e))),
            "s3_latitude_confidence": s3, "kept": bool(np.isfinite(s3) and s3 <= tau),
            "up_confidence_mean": float(np.nanmedian(z["up_confidence_mean"])),
            "gravity_uncertainty_median": float(np.nanmedian(z["gravity_uncertainty_raw"])),
            "lean_dep_eq_req_max_abs": float(np.abs(lean_dep -
                                                    np.asarray(z["lean_req"], dtype=float)).max()),
            "anatomy_forward": an["forward"], "anatomy_left": an["left"],
        })

    if missing:
        print(f"MISSING {len(missing)} clips: {missing}")

    # ---------------- cells and families -------------------------------------
    def agg(group, keep_only=None):
        g = [r for r in group if keep_only is None or r["kept"] == keep_only]
        if not g:
            return {"clips": 0, "frames": 0}
        med = [r["err_median"] for r in g]
        return {
            "clips": len(g), "frames": int(sum(r["n_frames"] for r in g)),
            "median_of_clip_medians": float(np.median(med)),
            "min_clip_median": float(np.min(med)), "max_clip_median": float(np.max(med)),
            "mean_bias_forward": float(np.mean([r["bias_forward"] for r in g])),
            "se_bias_forward": float(np.std([r["bias_forward"] for r in g], ddof=1) /
                                     np.sqrt(len(g))) if len(g) > 1 else float("nan"),
            "mean_bias_left": float(np.mean([r["bias_left"] for r in g])),
            "se_bias_left": float(np.std([r["bias_left"] for r in g], ddof=1) /
                                  np.sqrt(len(g))) if len(g) > 1 else float("nan"),
            "frac_gt10_frameweighted": float(np.average([r["frac_gt10"] for r in g],
                                                        weights=[r["n_frames"] for r in g])),
            "frac_gt20_frameweighted": float(np.average([r["frac_gt20"] for r in g],
                                                        weights=[r["n_frames"] for r in g])),
            "gate_keep": int(sum(r["kept"] for r in g)), "gate_n": len(g),
        }

    cells = []
    for fam in FAMILIES:
        for m in MOUNTS:
            g = [r for r in per_clip if r["family"] == fam and r["uptilt_deg"] == m]
            row = {"family": fam, "uptilt_deg": m}
            row.update(agg(g))
            row["gate_keep"] = int(sum(r["kept"] for r in g))
            cells.append(row)

    families = []
    for fam in FAMILIES:
        g = [r for r in per_clip if r["family"] == fam]
        row = {"family": fam}
        row.update(agg(g))
        row["gate_keep"] = int(sum(r["kept"] for r in g))
        families.append(row)
    pooled = {"family": "all"}
    pooled.update(agg(per_clip))
    pooled["gate_keep"] = int(sum(r["kept"] for r in per_clip))

    spread = []
    for m in MOUNTS:
        vals = [c["median_of_clip_medians"] for c in cells if c["uptilt_deg"] == m]
        spread.append({"uptilt_deg": m, "max_minus_min": float(max(vals) - min(vals)),
                       "values": vals})

    # ---------------- gate ---------------------------------------------------
    gate_rows = []
    for fam in FAMILIES + ["all"]:
        g = per_clip if fam == "all" else [r for r in per_clip if r["family"] == fam]
        kept = [r for r in g if r["kept"]]
        dropped = [r for r in g if not r["kept"]]
        gate_rows.append({
            "family": fam, "n": len(g), "kept": len(kept), "dropped": len(dropped),
            "keep_rate": len(kept) / len(g) if g else float("nan"),
            "kept_median_of_clip_medians": float(np.median([r["err_median"] for r in kept]))
            if kept else float("nan"),
            "dropped_median_of_clip_medians": float(np.median([r["err_median"] for r in dropped]))
            if dropped else float("nan"),
            "kept_frac_gt10": float(np.average([r["frac_gt10"] for r in kept],
                                               weights=[r["n_frames"] for r in kept]))
            if kept else float("nan"),
        })
    forest_keep = [x for x in gate_rows if x["family"] == "forest_ridge"][0]["keep_rate"]
    open_clips = [r for r in per_clip if r["family"] in ("urban_checker", "industrial")]
    open_keep = sum(r["kept"] for r in open_clips) / len(open_clips) if open_clips else float("nan")

    # ---------------- anatomy ------------------------------------------------
    an_rows = []
    for fam in FAMILIES + ["all"]:
        g = per_clip if fam == "all" else [r for r in per_clip if r["family"] == fam]
        an_rows.append({"family": fam, "pooled": _pool_anatomy(g), "per_clip": _clip_anatomy(g)})

    # ---------------- write --------------------------------------------------
    scenes = []
    for fam in FAMILIES:
        seeds = sorted({r["scene_seed"] for r in per_clip if r["family"] == fam})
        for sd in seeds:
            g = [r for r in per_clip if r["family"] == fam and r["scene_seed"] == sd]
            row = {"family": fam, "scene_seed": sd}
            row.update(agg(g))
            row["gate_keep"] = int(sum(r["kept"] for r in g))
            scenes.append(row)

    (out / "per_clip.csv").write_text(_csv(
        [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in per_clip]))
    (out / "cells.csv").write_text(_csv(cells))
    (out / "families.csv").write_text(_csv(families + [pooled]))
    (out / "scenes.csv").write_text(_csv(scenes))
    (out / "anatomy.json").write_text(json.dumps(an_rows, indent=1))
    (out / "reader_rescore.json").write_text(json.dumps({
        "clips": len(per_clip), "frames": int(sum(r["n_frames"] for r in per_clip)),
        "gate_tau": tau,
        "reproduction": {
            "max_abs_delta": float(np.max([abs(r["delta"]) for r in repro])) if repro else None,
            "p90_abs_delta": float(np.percentile([abs(r["delta"]) for r in repro], 90)) if repro else None,
            "pooled_median_delta": float(np.median([r["delta"] for r in repro])) if repro else None,
            "n_over_0.05_original": int(sum(abs(r["delta"]) > 0.05 for r in repro)),
            "n_over_0.25_amended": int(sum(abs(r["delta"]) > 0.25 for r in repro)),
            "frame_median_abs_diff_max": float(np.max([r["frame_median_abs_diff"] for r in repro])) if repro else None,
            "frame_p90_abs_diff_max": float(np.max([r["frame_p90_abs_diff"] for r in repro])) if repro else None,
            "n_frame_median_over_0.10": int(sum(r["frame_median_abs_diff"] > 0.10 for r in repro)),
            "n_frame_p90_over_0.50": int(sum(r["frame_p90_abs_diff"] > 0.50 for r in repro)),
            "rows": repro,
        },
        "gate": gate_rows, "gate_forest_vs_open": {
            "forest_keep_rate": forest_keep, "open_keep_rate": open_keep},
        "spread": spread,
    }, indent=1))
    print(json.dumps(json.loads((out / "reader_rescore.json").read_text())["reproduction"]
                     | {"gate": gate_rows}, indent=1)[:2000])
    return 0


def _pool_anatomy(rows) -> dict:
    """Energy-weighted pool of the per-clip decomposition (frames, both axes)."""
    tot = {"bias": 0.0, "slow": 0.0, "fast": 0.0, "n": 0}
    for r in rows:
        for axis in ("forward", "left"):
            a = r.get(f"anatomy_{axis}")
            if not a:
                continue
            n = a["n"]
            tot["bias"] += a["ms_bias"] * n
            tot["slow"] += a["ms_slow"] * n
            tot["fast"] += a["ms_fast"] * n
            tot["n"] += n
    if not tot["n"]:
        return {}
    s = tot["bias"] + tot["slow"] + tot["fast"]
    return {"frames_x_axes": tot["n"], "bias": tot["bias"] / s, "slow": tot["slow"] / s,
            "fast": tot["fast"] / s, "mean_squared_error_deg2": s / tot["n"]}


def _clip_anatomy(rows) -> dict:
    shares = []
    for r in rows:
        b = s = f = 0.0
        for axis in ("forward", "left"):
            a = r.get(f"anatomy_{axis}")
            if not a:
                continue
            b += a["ms_bias"]
            s += a["ms_slow"]
            f += a["ms_fast"]
        tot = b + s + f
        if tot > 1e-12:
            shares.append([b / tot, s / tot, f / tot])
    if not shares:
        return {}
    arr = np.asarray(shares)
    return {"clips": len(shares), "bias_median": float(np.median(arr[:, 0])),
            "slow_median": float(np.median(arr[:, 1])),
            "fast_median": float(np.median(arr[:, 2]))}


def _csv(rows: list[dict]) -> str:
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    lines = [",".join(keys)]
    for r in rows:
        vals = []
        for k in keys:
            v = r.get(k, "")
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append(",".join(vals))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
