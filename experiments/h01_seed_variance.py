"""H01 Part 1: ANWM seed variance and the oracle best-of-N curve.

Inputs (stored, never recomputed here):
  * depicted tilt — the existing reader (GeoCalib 224 px, camera_to_body_pitched mount):
      - seed-0 and the older seed-repeat clips: results/gate_a2/anwm/depicted.jsonl
      - new seeds generated for H01: results/h01_seed_depicted/depicted.jsonl
  * the commanded paths — results/anwm_clips/specs.json and results/h01_seed_specs/specs.json
  * the 20 pre-registered base paths — results/h01_seed_specs/paths.json

Statistics are fixed in configs/h01_pilot_prereg.yaml (part1_seed_variance). The primary
residual is the per-step required tilt at dt = 1 s with k_d = 0.295, joined frame by frame
to depicted tilt, clip statistic = median |required - depicted|; feasible when <= 6.46 deg.

    pixi run python experiments/h01_seed_variance.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.a2_anwm_required import frame_required, summarise
from experiments.a2_anwm_score import depicted_tilt_deg, frame_arrays, gate_of
from src.perception.gate import clip_signal, kept

ROOT = Path(__file__).resolve().parents[1]
THRESH_DEG = 6.46
K_PRIMARY = 0.295
DT_PRIMARY = 1.0
N_BOOT = 5000
RNG_SEED = 1


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["clip_id"], []).append(r)
    return out


def visual_of(clip_dir: Path | None) -> dict:
    if clip_dir is None or not (clip_dir / "manifest.json").exists():
        return {}
    return json.loads((clip_dir / "manifest.json").read_text()).get("visual_integrity", {})


def clip_stats(rows: list[dict], spec: dict, cfg: dict, visual: dict,
               base: str | None = None) -> dict:
    arrs = frame_arrays(rows)
    tilt = depicted_tilt_deg(arrs["up"])
    val, ok = gate_of(cfg, rows)
    med_std = float(np.median(visual["frame_std"])) if visual.get("frame_std") else np.nan
    med_td = float(np.median(visual["temporal_delta"])) if visual.get("temporal_delta") else np.nan
    collapsed = bool((np.isfinite(med_std) and med_std < 0.05)
                     or (np.isfinite(med_td) and med_td < 0.02))
    pos = np.asarray(spec["positions"], float)
    req = frame_required(pos, DT_PRIMARY, K_PRIMARY, "step")
    n = min(len(req), len(tilt))
    req, dep = req[:n], tilt[:n]
    m = np.isfinite(req) & np.isfinite(dep)
    resid = np.abs(req[m] - dep[m])
    req_k0 = frame_required(pos, DT_PRIMARY, 0.0, "step")[:n]
    resid_k0 = np.abs(req_k0[m] - dep[m])
    req_sp = frame_required(pos, DT_PRIMARY, K_PRIMARY, "spline")[:n]
    resid_sp = np.abs(req_sp[m] - dep[m])
    return {
        "clip_id": spec["id"], "base_path": base or spec["meta"].get("base_path", spec["id"]),
        "seed": int(spec.get("seed", 0)),
        "maneuver": spec["meta"].get("maneuver"),
        "policy": spec["meta"].get("policy"),
        "designed_tilt_deg": spec.get("designed_tilt_deg"),
        "start_traj": spec["start_traj"], "start_time": int(spec["start_time"]),
        "depicted_median_deg": float(np.nanmedian(dep)),
        "depicted_p90_deg": float(np.nanpercentile(dep, 90)),
        "depicted_per_frame": [float(x) for x in dep],
        "required_median_deg": float(np.nanmedian(req[m])),
        "required_frac_ge10": float(np.mean(req[m] >= 10.0)),
        "residual_median_deg": float(np.median(resid)),
        "residual_median_k0_deg": float(np.median(resid_k0)),
        "residual_median_spline_deg": float(np.median(resid_sp)),
        "gate_signal": float(val) if np.isfinite(val) else None,
        "gate_kept": bool(ok), "collapsed": collapsed,
        "testable": bool(ok and not collapsed and np.mean(req[m] >= 10.0) >= 12 / 16),
        "feasible": bool(np.median(resid) <= THRESH_DEG),
        "n_frames": int(len(dep)),
    }


def cluster_boot(values: np.ndarray, clusters: list[str], fn, n_boot=N_BOOT) -> dict:
    """Cluster bootstrap (resample the units, not the rows); descriptive under 10 units."""
    rng = np.random.default_rng(RNG_SEED)
    u = sorted(set(clusters))
    by = {c: np.array([v for v, cc in zip(values, clusters) if cc == c], float) for c in u}
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(u, size=len(u), replace=True)
        draws.append(fn(np.concatenate([by[c] for c in pick])))
    return {"point": float(fn(np.asarray(values, float))),
            "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)),
            "n_clusters": len(u),
            "descriptive": bool(len(u) < 10)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-specs", default="results/anwm_clips/specs.json")
    ap.add_argument("--base-depicted", default="results/gate_a2/anwm/depicted.jsonl")
    ap.add_argument("--new-specs", default="results/h01_seed_specs/specs.json")
    ap.add_argument("--new-depicted", default="results/h01_seed_depicted/depicted.jsonl")
    ap.add_argument("--paths", default="results/h01_seed_specs/paths.json")
    ap.add_argument("--clip-root", default="data/processed/h01_seed_clips")
    ap.add_argument("--gate", default="configs/gate_frozen.yaml")
    ap.add_argument("--out", default="results/h01_seed_variance")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load((ROOT / args.gate).read_text())
    paths = [p["base_path"] for p in json.loads((ROOT / args.paths).read_text())]
    base_specs = {s["id"]: s for s in json.loads((ROOT / args.base_specs).read_text())}
    new_specs = ({s["id"]: s for s in json.loads((ROOT / args.new_specs).read_text())}
                 if (ROOT / args.new_specs).exists() else {})
    base_rows = group_rows(load_jsonl(ROOT / args.base_depicted))
    new_rows = (group_rows(load_jsonl(ROOT / args.new_depicted))
                if (ROOT / args.new_depicted).exists() else {})
    clip_root = ROOT / args.clip_root

    existing_by_base_seed = {}
    for cid, spec in base_specs.items():
        if spec["regime"] != "C":
            continue
        b = spec["meta"].get("base_path", cid)
        if cid.endswith(f"_s{spec.get('seed', 0)}") and int(spec.get("seed", 0)) != 0:
            b = cid.rsplit("_s", 1)[0]
        existing_by_base_seed[(b, int(spec.get("seed", 0)))] = spec
    new_by_base_seed = {(s["meta"]["base_path"], int(s["seed"])): s for s in new_specs.values()}

    per_clip = []
    repeats = {}
    for base in paths:
        for seed in range(5):
            # deterministic choice: the existing clip wins for seeds that already exist
            spec = existing_by_base_seed.get((base, seed)) or new_by_base_seed.get((base, seed))
            if spec is None:
                continue
            rows = base_rows.get(spec["id"]) or new_rows.get(spec["id"])
            if not rows:
                continue
            vis = visual_of(clip_root / spec["regime"] / spec["id"])
            per_clip.append(clip_stats(rows, spec, cfg, vis, base=base))
        # same-seed repeatability: existing and new clips at the same seed, same path
        for seed in (1, 2):
            ex = existing_by_base_seed.get((base, seed))
            nw = new_by_base_seed.get((base, seed))
            if ex and nw and base_rows.get(ex["id"]) and new_rows.get(nw["id"]):
                a, b = clip_stats(base_rows[ex["id"]], ex, cfg, {}, base=base), \
                    clip_stats(new_rows[nw["id"]], nw, cfg, {}, base=base)
                repeats[base] = {"seed": seed,
                                 "depicted_existing_deg": a["depicted_median_deg"],
                                 "depicted_new_deg": b["depicted_median_deg"],
                                 "depicted_abs_diff_deg": abs(a["depicted_median_deg"]
                                                              - b["depicted_median_deg"]),
                                 "residual_abs_diff_deg": abs(a["residual_median_deg"]
                                                              - b["residual_median_deg"])}

    df = pd.DataFrame(per_clip).sort_values(["base_path", "seed"])
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_clip.csv", index=False)

    # ---- per-path spread
    spread = []
    for base, g in df.groupby("base_path"):
        g = g.sort_values("seed")
        spread.append({
            "base_path": base, "maneuver": g.maneuver.iloc[0], "policy": g.policy.iloc[0],
            "designed_tilt_deg": g.designed_tilt_deg.iloc[0], "n_seeds": len(g),
            "depicted_median_deg": float(g.depicted_median_deg.mean()),
            "depicted_seed_sd_deg": float(g.depicted_median_deg.std(ddof=1)),
            "depicted_seed_range_deg": float(g.depicted_median_deg.max()
                                             - g.depicted_median_deg.min()),
            "residual_median_deg": float(g.residual_median_deg.mean()),
            "residual_seed_sd_deg": float(g.residual_median_deg.std(ddof=1)),
            "residual_seed_range_deg": float(g.residual_median_deg.max()
                                             - g.residual_median_deg.min()),
            "all_testable": bool(g.testable.all()),
            "n_feasible_seeds": int(g.feasible.sum()),
            "feasible_any": bool(g.feasible.any()),
        })
    sp = pd.DataFrame(spread).sort_values("base_path")
    sp.to_csv(out / "per_path_spread.csv", index=False)

    # ---- best-of-N (oracle): fixed seed order 0..4, min residual over the first N
    rows_bn = []
    max_n = int(df.groupby("base_path").size().max())
    for n in range(1, max_n + 1):
        vals, feasible, clusters = [], [], []
        for base, g in df.groupby("base_path"):
            g = g.sort_values("seed").head(n)
            if len(g) < n:
                continue
            r = float(g.residual_median_deg.min())
            vals.append(r)
            feasible.append(r <= THRESH_DEG)
            clusters.append(base)
        vals_arr = np.asarray(vals, float)
        rows_bn.append({"N": n, "n_paths": len(vals),
                        "mean_oracle_residual_deg": float(vals_arr.mean()),
                        "median_oracle_residual_deg": float(np.median(vals_arr)),
                        "feasible_k": int(np.sum(feasible)),
                        "feasible_n": len(feasible),
                        "feasible_rate": float(np.mean(feasible))})
    bn = pd.DataFrame(rows_bn)
    bn.to_csv(out / "best_of_n.csv", index=False)

    # ---- oracle reduction r_1 - r_5 with a path-clustered bootstrap
    piv = {b: g.sort_values("seed") for b, g in df.groupby("base_path")}
    complete = [b for b, g in piv.items() if len(g) >= 5]
    if complete:
        reduction = np.array([piv[b].residual_median_deg.iloc[:1].values[0]
                              - piv[b].residual_median_deg.iloc[:5].min() for b in complete])
        red_ci = cluster_boot(reduction, complete, np.median)
        red_mean = cluster_boot(reduction, complete, np.mean)
        n_flip = int(sum((piv[b].residual_median_deg.iloc[0] > THRESH_DEG)
                         and (piv[b].residual_median_deg.iloc[:5].min() <= THRESH_DEG)
                         for b in complete))
    else:
        reduction, red_ci, red_mean, n_flip = np.array([]), {}, {}, 0

    # ---- design contrast: depicted tilt vs designed tilt (5/10/20), per-path means
    des = sp.dropna(subset=["designed_tilt_deg"])
    slope = {}
    if len(des) >= 3:
        x, y = des.designed_tilt_deg.to_numpy(float), des.depicted_median_deg.to_numpy(float)
        A = np.vstack([x, np.ones_like(x)]).T
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        slope = {"deg_per_deg": float(coef[0]), "intercept_deg": float(coef[1]),
                 "n_paths": int(len(x))}
        by = des.groupby("designed_tilt_deg").depicted_median_deg.agg(["mean", "count"])
        slope["by_design"] = {str(k): {"mean_depicted_deg": float(v["mean"]),
                                       "n_paths": int(v["count"])}
                              for k, v in by.iterrows()}

    summary = {
        "git": git_hash(), "prereg": "configs/h01_pilot_prereg.yaml",
        "reader": "results/gate_a2/anwm/depicted.jsonl + results/h01_seed_depicted/depicted.jsonl",
        "threshold_deg": THRESH_DEG, "k_d": K_PRIMARY, "dt_s": DT_PRIMARY,
        "n_paths": int(sp.shape[0]), "n_clips": int(df.shape[0]),
        "seeds_per_path": {str(k): int(v) for k, v in
                           df.groupby("base_path").size().items()},
        "depicted_spread": {
            "median_seed_sd_deg": float(sp.depicted_seed_sd_deg.median()),
            "median_seed_range_deg": float(sp.depicted_seed_range_deg.median()),
            "max_seed_range_deg": float(sp.depicted_seed_range_deg.max())},
        "residual_spread": {
            "median_seed_sd_deg": float(sp.residual_seed_sd_deg.median()),
            "median_seed_range_deg": float(sp.residual_seed_range_deg.median()),
            "max_seed_range_deg": float(sp.residual_seed_range_deg.max())},
        "best_of_n": bn.to_dict(orient="records"),
        "oracle_reduction_1_to_5": {
            "per_path": [float(v) for v in reduction],
            "median_deg": red_ci, "mean_deg": red_mean,
            "n_paths_flipped_infeasible_to_feasible": n_flip,
            "n_paths": len(complete)},
        "design_contrast": slope,
        "same_seed_repeat": repeats,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("best_of_n",)}, indent=1,
                     default=float)[:4000])
    print(bn.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
