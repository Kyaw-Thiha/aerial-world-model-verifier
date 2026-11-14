"""Task 1b: diagnose the depicted-vs-truth shrinkage from stored per-frame arrays.

For every frame of the 57 testable real clips (frozen d02 set), pair the depicted lean
with the truth-attitude lean at the same frame, after the gain fit's lag.  Then ask:

* is the relation a **linear shrinkage** (constant slope below 1),
* a **saturation** (slope falling with tilt), or
* **noise / errors-in-variables** (low R^2, residual variance dominating)?

Outputs per-flight-type and per-flight slopes, slopes by truth-tilt bin, the share of
variance that is residual noise, and plots.  The calibration test (if the compression is
monotone) is in ``experiments/d_calibrate.py``.

    PYTHONPATH=. pixi run python experiments/d_shrinkage.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.verdict import load_clip
from src.verdict.dense import truth_lean
from src.verdict.fit import _shift_series

ROOT = Path(__file__).resolve().parents[1]
FLIGHTS = ["flight-01a-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
           "flight-13a-trackRATM", "flight-01p-ellipse", "flight-07p-lemniscate",
           "flight-13p-trackRATM", "flight-16p-trackRATM"]
TRACK = {"flight-13a-trackRATM", "flight-13p-trackRATM", "flight-16p-trackRATM"}
BINS = [0, 5, 10, 20, 30, 45, 90]


def flight_type(flight: str) -> str:
    if flight in TRACK:
        return "track"
    return "piloted" if "p" in flight.split("-")[1] else "autonomous"


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def fit_slope(x: np.ndarray, y: np.ndarray, intercept: bool = True) -> dict:
    """Per-axis OLS slope(s) and R^2 for y on x; x is (N, 2), y is (N, 2)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = {"n": int(len(x))}
    for axis, name in ((0, "x"), (1, "y")):
        A = np.column_stack([x[:, axis], np.ones(len(x))]) if intercept else x[:, axis:axis + 1]
        beta = np.linalg.lstsq(A, y[:, axis], rcond=None)[0]
        resid = y[:, axis] - A @ beta
        sst = float(np.sum((y[:, axis] - y[:, axis].mean()) ** 2))
        out[f"slope_{name}"] = float(beta[0])
        out[f"intercept_{name}"] = float(beta[1]) if intercept else 0.0
        out[f"r2_{name}"] = float(1.0 - np.sum(resid ** 2) / sst) if sst > 0 else float("nan")
        out[f"resid_std_{name}"] = float(np.std(resid))
    # shared 2-D slope
    n = len(x)
    A = np.zeros((2 * n, 3))
    A[0::2, 0] = x[:, 0]
    A[1::2, 0] = x[:, 1]
    A[0::2, 1] = 1.0
    A[1::2, 2] = 1.0
    b = np.linalg.lstsq(A, np.column_stack([y[:, 0], y[:, 1]]).ravel(), rcond=None)[0]
    yv = np.column_stack([y[:, 0], y[:, 1]]).ravel()
    r = yv - A @ b
    sst = float(np.sum((yv - yv.mean()) ** 2))
    out["slope_scalar"] = float(b[0])
    out["r2_scalar"] = float(1.0 - np.sum(r ** 2) / sst) if sst > 0 else float("nan")
    out["noise_share"] = float(np.sum(r ** 2) / sst) if sst > 0 else float("nan")
    out["var_ratio_y_minus_x"] = float(np.var(y - x) / np.var(y)) if np.var(y) > 0 else float("nan")
    return out


def cluster_bootstrap_slope(df: pd.DataFrame, n_boot=2000, seed=20261114) -> dict:
    """Cluster bootstrap (whole flights) of the per-axis slope."""
    flights = sorted(df.flight.unique())
    rng = np.random.default_rng(seed)
    boots = {"x": [], "y": []}
    for _ in range(n_boot):
        pick = rng.choice(len(flights), size=len(flights), replace=True)
        sub = pd.concat([df[df.flight == flights[i]] for i in pick])
        s = fit_slope(sub[["truth_x", "truth_y"]].to_numpy(),
                      sub[["dep_x", "dep_y"]].to_numpy())
        boots["x"].append(s["slope_x"])
        boots["y"].append(s["slope_y"])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in boots.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/d_eval/d02")
    ap.add_argument("--out", default="results/d_shrinkage")
    ap.add_argument("--tag", default="d01")
    args = ap.parse_args()

    from experiments.d_eval import load_results, real_testable
    prereg = yaml.safe_load((ROOT / "configs/gain_prereg.yaml").read_text())
    gate = yaml.safe_load((ROOT / "configs/gate_frozen.yaml").read_text())
    results = load_results(ROOT / args.run / "primary_results.json")
    clips = {}
    for path in sorted((ROOT / "data/processed/track_d/clips").glob("*/*/verdict_clip.npz")):
        c = load_clip(path)
        clips[(c.flight, c.clip_id, c.condition)] = c
    testable = real_testable(clips, results, prereg["abstention"], gate)

    rows = []
    per_clip = []
    for flight in FLIGHTS:
        spec = np.load(ROOT / f"data/processed/p3a/p3a_v1/{flight}.npz", allow_pickle=False)
        from scipy.spatial.transform import Rotation
        yaw = Rotation.from_matrix(
            Rotation.from_quat(spec["quat"]).as_matrix()).as_euler("ZYX")[:, 0]
        truth, truth_tilt = truth_lean(spec["quat"], yaw)
        clip_table = json.loads(
            (ROOT / "data/processed/p3a/p3a_v1/clips.json").read_text())["clips"]
        for meta in [c for c in clip_table if c["flight"] == flight]:
            key = (flight, meta["clip_id"], "real")
            if key not in clips or not testable.get((flight, meta["clip_id"]),
                                                    {}).get("testable"):
                continue
            clip = clips[key]
            lo, hi = meta["index"]
            sl = slice(int(lo), int(hi))
            lag = results[key].lag_ms
            tr = _shift_series(truth[sl], clip.t, lag / 1000.0)
            dep = clip.depicted_lean
            tt = _shift_series(truth_tilt[sl], clip.t, lag / 1000.0)
            ok = (np.isfinite(dep).all(axis=1) & np.isfinite(tr).all(axis=1)
                  & np.isfinite(tt))
            for i in np.flatnonzero(ok):
                rows.append({"flight": flight, "type": flight_type(flight),
                             "clip_id": meta["clip_id"],
                             "truth_x": tr[i, 0], "truth_y": tr[i, 1],
                             "dep_x": dep[i, 0], "dep_y": dep[i, 1],
                             "truth_tilt_deg": tt[i]})
            s = fit_slope(tr[ok], dep[ok])
            s.update({"flight": flight, "type": flight_type(flight),
                      "clip_id": meta["clip_id"], "lag_ms": lag})
            per_clip.append(s)

    out = ROOT / args.out / args.tag
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "frames.csv", index=False)
    pc = pd.DataFrame(per_clip)
    pc.to_csv(out / "per_clip_slopes.csv", index=False)

    summary = {"run": args.run, "git": git_hash(), "n_frames": int(len(df)),
               "n_clips": int(pc.clip_id.nunique()), "by_type": {}, "by_flight": {},
               "bins": {}, "per_clip_median": {}}
    for typ in ("piloted", "autonomous", "track"):
        sub = df[df.type == typ]
        if not len(sub):
            continue
        s = fit_slope(sub[["truth_x", "truth_y"]].to_numpy(),
                      sub[["dep_x", "dep_y"]].to_numpy())
        s["slope_ci"] = cluster_bootstrap_slope(sub)
        summary["by_type"][typ] = s
    for flight in FLIGHTS:
        sub = df[df.flight == flight]
        if len(sub):
            summary["by_flight"][flight] = fit_slope(
                sub[["truth_x", "truth_y"]].to_numpy(),
                sub[["dep_x", "dep_y"]].to_numpy())
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sub = df[(df.truth_tilt_deg >= lo) & (df.truth_tilt_deg < hi)]
        if len(sub) < 30:
            summary["bins"][f"{lo}-{hi}"] = {"n": int(len(sub))}
            continue
        summary["bins"][f"{lo}-{hi}"] = fit_slope(
            sub[["truth_x", "truth_y"]].to_numpy(),
            sub[["dep_x", "dep_y"]].to_numpy())
    summary["bins_by_type"] = {}
    for typ in ("piloted", "autonomous", "track"):
        summary["bins_by_type"][typ] = {}
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            sub = df[(df.type == typ) & (df.truth_tilt_deg >= lo)
                     & (df.truth_tilt_deg < hi)]
            if len(sub) < 30:
                summary["bins_by_type"][typ][f"{lo}-{hi}"] = {"n": int(len(sub))}
                continue
            summary["bins_by_type"][typ][f"{lo}-{hi}"] = fit_slope(
                sub[["truth_x", "truth_y"]].to_numpy(),
                sub[["dep_x", "dep_y"]].to_numpy())
    for typ in ("piloted", "autonomous", "track"):
        sub = pc[pc.type == typ]
        if len(sub):
            summary["per_clip_median"][typ] = {
                "slope_x": float(sub.slope_x.median()),
                "slope_y": float(sub.slope_y.median()),
                "noise_share": float(sub.noise_share.median()),
                "n": int(len(sub)),
            }
    (out / "shrinkage.json").write_text(json.dumps(summary, indent=1, default=float))

    # plots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True, sharey=True)
    for ax, typ in zip(axes, ("piloted", "autonomous", "track")):
        sub = df[df.type == typ]
        if not len(sub):
            ax.set_title(f"{typ} (none)")
            continue
        ax.hexbin(np.abs(sub.truth_x), np.abs(sub.dep_x), gridsize=40,
                  bins="log", cmap="viridis", mincnt=1)
        lim = max(np.percentile(np.abs(sub.truth_x), 99.5),
                  np.percentile(np.abs(sub.dep_x), 99.5)) * 1.05
        ax.plot([0, lim], [0, lim], "w--", lw=1, label="identity")
        s = summary["by_type"][typ]
        ax.plot([0, lim], [s["intercept_x"], s["intercept_x"] + s["slope_x"] * lim],
                "r-", lw=1.2, label=f"fit x: {s['slope_x']:.2f}")
        ax.set_title(f"{typ}: slope x {s['slope_x']:.2f}, y {s['slope_y']:.2f}, "
                     f"R2 {s['r2_scalar']:.2f}")
        ax.set_xlabel("|truth lean| (x axis)")
        ax.set_ylabel("|depicted lean| (x axis)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "shrinkage_scatter.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    keys = list(summary["bins"])
    for ax, axis_name in zip(axes, ("x", "y")):
        vals = [summary["bins"][k].get(f"slope_{axis_name}", np.nan) for k in keys]
        ns = [summary["bins"][k].get("n", 0) for k in keys]
        ax.plot(range(len(keys)), vals, "-o")
        for i, (v, n) in enumerate(zip(vals, ns)):
            if np.isfinite(v):
                ax.annotate(f"n={n}", (i, v), fontsize=6, xytext=(0, 5),
                            textcoords="offset points")
        ax.axhline(1.0, color="k", ls=":")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=30)
        ax.set_ylabel(f"slope {axis_name}")
        ax.set_title(f"depicted {axis_name} on truth, by truth-tilt bin")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "slope_by_tilt_bin.png", dpi=150)
    plt.close(fig)

    # plain-language table
    lines = ["# Shrinkage diagnosis", "",
             f"run `{args.run}` · {len(df)} frames · {pc.clip_id.nunique()} clips", "",
             "| type | n frames | slope x | slope y | scalar R2 | noise share |",
             "| --- | --- | --- | --- | --- | --- |"]
    for typ, s in summary["by_type"].items():
        lines.append(f"| {typ} | {s['n']} | {s['slope_x']:.3f} | {s['slope_y']:.3f} | "
                     f"{s['r2_scalar']:.3f} | {s['noise_share']:.3f} |")
    lines += ["", "| truth-tilt bin (deg) | n | slope x | slope y | R2 |",
              "| --- | --- | --- | --- | --- |"]
    for k, s in summary["bins"].items():
        lines.append(f"| {k} | {s.get('n', 0)} | {s.get('slope_x', float('nan')):.3f} | "
                     f"{s.get('slope_y', float('nan')):.3f} | "
                     f"{s.get('r2_scalar', float('nan')):.3f} |")
    for typ in ("piloted", "autonomous", "track"):
        lines += ["", f"| {typ} bin (deg) | n | slope x | slope y | R2 |",
                  "| --- | --- | --- | --- | --- |"]
        for k, s in summary["bins_by_type"][typ].items():
            lines.append(f"| {k} | {s.get('n', 0)} | {s.get('slope_x', float('nan')):.3f} | "
                         f"{s.get('slope_y', float('nan')):.3f} | "
                         f"{s.get('r2_scalar', float('nan')):.3f} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
