"""Flatness floor: how well does the thrust axis implied by mocap motion match the
thrust axis the mocap attitude actually shows?

No pixels. Ground-truth position is differentiated twice (Savitzky-Golay), the
flatness equation turns acceleration into an implied body z-axis, and the angle
to the true body z-axis from mocap attitude is the residual. That residual is
the floor: no pixel-based verifier can beat it, because both sides are already
exact here.

Protocol
--------
* Split by flight. Leave-one-flight-out: the drag coefficient and the
  Savitzky-Golay window are chosen on the other flights only, then applied to the
  held-out flight. The gate number is the pooled median over held-out frames.
* One scalar world-frame drag coefficient k_d (1/s) per fold, fitted by
  minimising the median residual.
* Diagnostic, not part of the gate: body-frame drag with separate x/y
  coefficients, to see whether the high-speed residual is a drag-model mismatch.

Outputs land in ``results/phase0_floor/<run_id>/``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.datasets.tii import load_tii
from src.physics.diff import edge_slice, savgol_derivatives
from src.physics.flatness import (
    G0,
    angle_between_deg,
    fit_body_drag,
    fit_kd,
    implied_thrust_axis,
    implied_thrust_axis_body_drag,
    thrust_axis_from_quat,
    tilt_deg,
    valid_thrust_mask,
)

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "experiment": "phase0_floor",
    "dataset_root": "data/raw/tii",
    "flights": [
        "flight-01a-ellipse",
        "flight-04a-ellipse",
        "flight-07a-lemniscate",
        "flight-13a-trackRATM",
        "flight-01p-ellipse",
        "flight-07p-lemniscate",
        "flight-13p-trackRATM",
        "flight-16p-trackRATM",
    ],
    "sg_polyorder": 3,
    "sg_windows": [11, 21, 31, 51, 71, 101, 151, 201, 301, 401],
    "kd_bounds": [0.0, 3.0],
    "tilt_signal_deg": 10.0,
    "speed_bins_ms": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25],
    "g": G0,
    "seed": 0,  # no stochastic step; recorded for run metadata completeness
    "notes": "mocap-only floor; split by flight; leave-one-flight-out",
}


def git_info() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    return {"commit": commit, "dirty": bool(dirty), "dirty_files": dirty.splitlines()[:20]}


def find_mocap(root: Path, flight: str) -> Path:
    hits = list(root.glob(f"*/mocap_{flight}.csv"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one mocap file for {flight} under {root}, found {hits}")
    return hits[0]


def derive(clip, window: int, polyorder: int, g: float) -> dict:
    """Differentiate one clip and return trimmed, aligned per-frame quantities."""
    vel, acc = savgol_derivatives(clip.pos, clip.dt, window, polyorder)
    sl = edge_slice(clip.n, window)
    true_axis = thrust_axis_from_quat(clip.quat, clip.quat_order)[sl]
    v, a = vel[sl], acc[sl]
    mask = valid_thrust_mask(a, v, 0.0, g)
    return {
        "t": clip.t[sl],
        "vel": v,
        "acc": a,
        "true_axis": true_axis,
        "mask": mask,
        "speed": np.linalg.norm(v, axis=1),
        "acc_norm": np.linalg.norm(a, axis=1),
        "acc_xy_norm": np.linalg.norm(a[:, :2], axis=1),
    }


def rotation_offset_deg(z: np.ndarray, w: np.ndarray) -> float:
    """Angle of the constant rotation that best maps z onto w (frame-offset probe)."""
    H = w.T @ z
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def summarize(r: np.ndarray) -> dict:
    return {
        "median": float(np.median(r)),
        "p90": float(np.percentile(r, 90)),
        "p95": float(np.percentile(r, 95)),
        "mean": float(np.mean(r)),
        "n": int(len(r)),
    }


def pooled_median(derived: dict, flights: list[str], k_d: float, g: float) -> float:
    vals = [
        angle_between_deg(
            implied_thrust_axis(derived[f]["acc"], derived[f]["vel"], k_d, g),
            derived[f]["true_axis"],
        )
        for f in flights
    ]
    return float(np.median(np.concatenate(vals)))


def fit_kd_pooled(derived: dict, flights: list[str], g: float, bounds, xatol=1e-3) -> float:
    """Fit one scalar k_d on several flights by minimising the pooled median residual."""
    from scipy.optimize import minimize_scalar

    def loss(k: float) -> float:
        return pooled_median(derived, flights, float(k), g)

    res = minimize_scalar(loss, bounds=tuple(bounds), method="bounded",
                          options={"xatol": xatol})
    return float(res.x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None, help="override the run id (default: UTC time + git hash)")
    ap.add_argument("--quick", action="store_true", help="single window, no sweeps (smoke test)")
    args = ap.parse_args()

    g = CONFIG["g"]
    windows = CONFIG["sg_windows"][:1] if args.quick else CONFIG["sg_windows"]
    poly = CONFIG["sg_polyorder"]
    flights = CONFIG["flights"]
    root = ROOT / CONFIG["dataset_root"]

    git = git_info()
    run_id = args.run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{git['commit'][:8]}"
    out = ROOT / "results" / CONFIG["experiment"] / run_id
    (out / "frames").mkdir(parents=True, exist_ok=True)
    print(f"run {run_id} -> {out.relative_to(ROOT)}")

    clips = {f: load_tii(find_mocap(root, f)) for f in flights}
    print(f"loaded {len(clips)} flights, rates "
          f"{sorted({round(c.fps, 2) for c in clips.values()})} Hz")

    # --- per-flight facts and derivatives for every candidate window ----------
    derived: dict[int, dict[str, dict]] = {}
    flight_facts: dict[str, dict] = {}
    for f, c in clips.items():
        facts = {
            "source": str(Path(c.meta["source"]).relative_to(ROOT)),
            "track": c.meta.get("track"),
            "control": c.meta.get("control"),
            "camera_uptilt_deg": c.cam_uptilt_deg,
            "rate_hz": c.fps,
            "n_mocap": c.n,
            "duration_s": float(c.t[-1] - c.t[0]),
            "mocap_dt_ms": [c.meta["dt_min_s"] * 1e3, c.meta["dt_max_s"] * 1e3],
            "quat_order": c.quat_order,
            "rot_layout": c.meta["rot_layout"],
        }
        flight_facts[f] = facts
        for w in windows:
            derived.setdefault(w, {})[f] = derive(c, w, poly, g)
        d = derived[windows[0]][f]
        sp = d["speed"]
        facts.update({
            "speed_max_ms": float(sp.max()),
            "speed_p95_ms": float(np.percentile(sp, 95)),
            "speed_median_ms": float(np.median(sp)),
            "frac_frames_speed_gt_10ms": float(np.mean(sp > 10.0)),
            "true_tilt_median_deg": float(np.median(tilt_deg(d["true_axis"]))),
            "acc_p95_ms2": float(np.percentile(d["acc_norm"], 95)),
        })

    # --- leave-one-flight-out ------------------------------------------------
    folds = {}
    for held in flights:
        fit_set = [f for f in flights if f != held]
        sweep = {}
        for w in windows:
            k = fit_kd_pooled(derived[w], fit_set, g, CONFIG["kd_bounds"])
            sweep[w] = {"k_d": k, "fit_median_deg": pooled_median(derived[w], fit_set, k, g)}
        w_star = min(windows, key=lambda w: sweep[w]["fit_median_deg"])
        k_star = sweep[w_star]["k_d"]
        d = derived[w_star][held]
        r_free = angle_between_deg(
            implied_thrust_axis(d["acc"], d["vel"], 0.0, g), d["true_axis"])
        r_fit = angle_between_deg(
            implied_thrust_axis(d["acc"], d["vel"], k_star, g), d["true_axis"])
        implied = implied_thrust_axis(d["acc"], d["vel"], k_star, g)
        tilt_imp = tilt_deg(implied)

        # body-frame drag diagnostic, fitted on the other flights only
        bd_fit_axes = np.concatenate([derived[w_star][f]["acc"] for f in fit_set])
        bd_fit_vel = np.concatenate([derived[w_star][f]["vel"] for f in fit_set])
        bd_fit_true = np.concatenate([derived[w_star][f]["true_axis"] for f in fit_set])
        bd_fit_quat = np.concatenate([clips[f].quat[edge_slice(clips[f].n, w_star)] for f in fit_set])
        bd_fit_mask = np.concatenate([derived[w_star][f]["mask"] for f in fit_set])
        bd = fit_body_drag(bd_fit_axes, bd_fit_vel, bd_fit_true, bd_fit_quat,
                           clips[fit_set[0]].quat_order, g=g, mask=bd_fit_mask)
        z_bd = implied_thrust_axis_body_drag(d["acc"], d["vel"], clips[held].quat[edge_slice(clips[held].n, w_star)],
                                             clips[held].quat_order, (bd["k_x"], bd["k_y"]), g)
        r_bd = angle_between_deg(z_bd, d["true_axis"])

        sel = d["acc_xy_norm"] > 2.0
        folds[held] = {
            "window": w_star,
            "sg_polyorder": poly,
            "k_d": k_star,
            "n_heldout_frames": int(len(r_fit)),
            "drag_free": summarize(r_free),
            "fitted_drag": summarize(r_fit),
            "body_drag": summarize(r_bd),
            "body_drag_coeffs": {"k_x": bd["k_x"], "k_y": bd["k_y"]},
            "frac_implied_tilt_gt_10deg": float(np.mean(tilt_imp > CONFIG["tilt_signal_deg"])),
            "high_speed_p95": {
                "drag_free": float(np.percentile(r_free[d["speed"] > 12.0], 95)) if (d["speed"] > 12.0).sum() > 10 else None,
                "fitted_drag": float(np.percentile(r_fit[d["speed"] > 12.0], 95)) if (d["speed"] > 12.0).sum() > 10 else None,
                "body_drag": float(np.percentile(r_bd[d["speed"] > 12.0], 95)) if (d["speed"] > 12.0).sum() > 10 else None,
            },
            "window_sweep_on_fit_flights": {str(k): v for k, v in sweep.items()},
            "frame_offset_deg": rotation_offset_deg(implied[sel], d["true_axis"][sel]) if sel.sum() > 50 else None,
        }

        np.savez_compressed(
            out / "frames" / f"{held}.npz",
            t=d["t"], implied_axis=implied, true_axis=d["true_axis"],
            residual_fitted_deg=r_fit, residual_drag_free_deg=r_free, residual_body_drag_deg=r_bd,
            speed_ms=d["speed"], acc_ms2=d["acc"], acc_norm_ms2=d["acc_norm"],
            tilt_implied_deg=tilt_imp, tilt_true_deg=tilt_deg(d["true_axis"]),
            window=w_star, k_d=k_star, fps=clips[held].fps,
        )
        print(f"  fold held={held:24s} w={w_star:3d} k_d={k_star:.3f} "
              f"free={folds[held]['drag_free']['median']:.2f} "
              f"fit={folds[held]['fitted_drag']['median']:.2f} "
              f"body={folds[held]['body_drag']['median']:.2f} deg")

    # --- pooled held-out gate numbers ---------------------------------------
    all_fit = np.concatenate([
        np.load(out / "frames" / f"{f}.npz")["residual_fitted_deg"] for f in flights])
    all_free = np.concatenate([
        np.load(out / "frames" / f"{f}.npz")["residual_drag_free_deg"] for f in flights])
    all_body = np.concatenate([
        np.load(out / "frames" / f"{f}.npz")["residual_body_drag_deg"] for f in flights])
    gate = {
        "definition": "pooled median residual over held-out frames, leave-one-flight-out",
        "fitted_drag": summarize(all_fit),
        "drag_free": summarize(all_free),
        "body_drag_diagnostic": summarize(all_body),
        "per_flight_spread_median_deg": {
            "min": float(min(folds[f]["fitted_drag"]["median"] for f in flights)),
            "max": float(max(folds[f]["fitted_drag"]["median"] for f in flights)),
            "iqr": float(np.percentile([folds[f]["fitted_drag"]["median"] for f in flights], 75)
                         - np.percentile([folds[f]["fitted_drag"]["median"] for f in flights], 25)),
        },
        "median_improvement_from_drag_deg": float(np.median(all_free) - np.median(all_fit)),
    }

    # --- residual vs speed and vs implied tilt -------------------------------
    speed_bins = np.array(CONFIG["speed_bins_ms"], dtype=float)
    tilt_bins = np.array([0, 5, 10, 20, 30, 45, 60, 90, 135, 181], dtype=float)
    curves = {}
    for f in flights:
        z = np.load(out / "frames" / f"{f}.npz")
        sp, ti = z["speed_ms"], z["tilt_implied_deg"]
        row = {"speed": {}, "tilt": {}}
        for lo, hi in zip(speed_bins[:-1], speed_bins[1:]):
            m = (sp >= lo) & (sp < hi)
            if m.sum() >= 20:
                row["speed"][f"{lo:g}-{hi:g}"] = {
                    "n": int(m.sum()),
                    "median_free": float(np.median(z["residual_drag_free_deg"][m])),
                    "median_fit": float(np.median(z["residual_fitted_deg"][m])),
                    "median_body": float(np.median(z["residual_body_drag_deg"][m])),
                }
        for lo, hi in zip(tilt_bins[:-1], tilt_bins[1:]):
            m = (ti >= lo) & (ti < hi)
            if m.sum() >= 20:
                row["tilt"][f"{lo:g}-{hi:g}"] = {
                    "n": int(m.sum()),
                    "median_free": float(np.median(z["residual_drag_free_deg"][m])),
                    "median_fit": float(np.median(z["residual_fitted_deg"][m])),
                }
        curves[f] = row

    # --- plots ---------------------------------------------------------------
    kinds = {"autonomous": "o", "pilot": "s"}
    plt.figure(figsize=(8, 4.5))
    for f in flights:
        z = np.load(out / "frames" / f"{f}.npz")
        plt.scatter(np.log10(np.clip(z["speed_ms"], 0.1, None)), z["residual_fitted_deg"],
                    s=2, alpha=0.25, label=f)
    plt.xlabel("log10 speed (m/s)"); plt.ylabel("residual, fitted drag (deg)")
    plt.title("Angular residual vs speed, held-out per flight")
    plt.legend(fontsize=6, markerscale=3, ncol=2); plt.tight_layout()
    plt.savefig(out / "residual_vs_speed.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 4.5))
    labels = [f.replace("flight-", "") for f in flights]
    free_med = [folds[f]["drag_free"]["median"] for f in flights]
    fit_med = [folds[f]["fitted_drag"]["median"] for f in flights]
    body_med = [folds[f]["body_drag"]["median"] for f in flights]
    x = np.arange(len(flights)); wdt = 0.27
    plt.bar(x - wdt, free_med, wdt, label="drag-free")
    plt.bar(x, fit_med, wdt, label="fitted k_d (gate)")
    plt.bar(x + wdt, body_med, wdt, label="body x/y drag (diagnostic)")
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=7)
    plt.ylabel("median residual (deg)"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "per_flight_median.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 4.5))
    for w in windows:
        ys = [folds[f]["window_sweep_on_fit_flights"][str(w)]["fit_median_deg"] for f in flights]
        plt.plot([w] * len(ys), ys, "o", alpha=0.5)
    plt.xlabel("Savitzky-Golay window (samples)"); plt.ylabel("fit-set median residual (deg)")
    plt.title("Window sweep, one point per fold (chosen window shown per fold)")
    plt.tight_layout(); plt.savefig(out / "window_sweep.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 4.5))
    for f in flights:
        z = np.load(out / "frames" / f"{f}.npz")
        plt.scatter(z["tilt_implied_deg"], z["residual_fitted_deg"], s=2, alpha=0.2, label=f)
    plt.xscale("symlog", linthresh=5)
    plt.xlabel("implied tilt (deg)"); plt.ylabel("residual, fitted drag (deg)")
    plt.title("Residual vs implied tilt"); plt.legend(fontsize=6, markerscale=3, ncol=2)
    plt.tight_layout(); plt.savefig(out / "residual_vs_tilt.png", dpi=150); plt.close()

    metrics = {
        "run_id": run_id,
        "git": git,
        "config": CONFIG,
        "flights": flight_facts,
        "loo_folds": folds,
        "gate": gate,
        "residual_curves": curves,
        "artifacts": {
            "frames": sorted(p.name for p in (out / "frames").glob("*.npz")),
            "plots": ["residual_vs_speed.png", "per_flight_median.png",
                      "window_sweep.png", "residual_vs_tilt.png"],
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    import yaml
    (out / "config.yaml").write_text(yaml.safe_dump({"config": CONFIG, "run_id": run_id,
                                                     "git": git}, sort_keys=False))
    (out / "git_hash.txt").write_text(git["commit"] + (" (dirty)" if git["dirty"] else "") + "\n")

    print("\ngate: pooled held-out median residual")
    print(f"  drag-free    {gate['drag_free']['median']:.2f} deg "
          f"(p90 {gate['drag_free']['p90']:.2f}, p95 {gate['drag_free']['p95']:.2f})")
    print(f"  fitted k_d   {gate['fitted_drag']['median']:.2f} deg "
          f"(p90 {gate['fitted_drag']['p90']:.2f}, p95 {gate['fitted_drag']['p95']:.2f})")
    print(f"  body x/y     {gate['body_drag_diagnostic']['median']:.2f} deg "
          f"(diagnostic only)")
    spread = gate["per_flight_spread_median_deg"]
    print(f"  per-flight median range {spread['min']:.2f} to {spread['max']:.2f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
