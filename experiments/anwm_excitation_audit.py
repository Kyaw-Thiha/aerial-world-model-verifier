"""Excitation audit on released ANWM trajectories: how much tilt does the data imply?

The release fixes metric *distance* per step (3.30 m mean spacing; AerialVLN's own
action set is 5 m forward, 2 m up/down, 15 deg yaw) but never fixes a duration.
Since `tan(theta) = |a_xy| / (a_z + g)` and `a` scales as 1/dt^2, the implied tilt
is undefined until a dt is chosen. This audit therefore never reports a single
number: it sweeps dt, and separately measures which dt best explains the roll and
pitch the released frames actually carry.

Input: `data/raw/anwm/*.npz` from `scripts/fetch_anwm.py` (pose keys only, no images).
Output: `results/anwm_excitation/audit.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.physics.flatness import G0, implied_tilt_from_positions, tilt_from_roll_pitch

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data/raw/anwm"
OUT_DIR = ROOT / "results/anwm_excitation"
SPLIT_DIR = ROOT / "third_party/anwm/data/splits/airvln_16"

K_D_FLOOR = 0.297          # from results/phase0_floor (mocap fit, 1/s)
TILT_THRESHOLD_DEG = 10.0  # the verifier's signal threshold
CANDIDATE_DT = {"paper_1s": 1.0, "spacing_3p30m_at_5ms": 3.30 / 5.0, "code_input_fps4": 0.25}
DT_GRID = np.geomspace(0.05, 2.0, 48)
SG_WINDOW, SG_POLY = 11, 3


def load_split() -> tuple[set[str], set[str]]:
    def names(split: str) -> set[str]:
        p = SPLIT_DIR / split / "traj_names.txt"
        return set(p.read_text().split()) if p.exists() else set()
    return names("train"), names("test")


def implied_tilt(pts: np.ndarray, dt: float, k_d: float = 0.0) -> np.ndarray:
    tilt, _ = implied_tilt_from_positions(pts, dt, SG_WINDOW, SG_POLY, k_d)
    return tilt


def per_step_tilt(pts: np.ndarray, dt: float, k_d: float = 0.0) -> np.ndarray:
    """Tilt demanded by the *commanded* per-step velocity change.

    These trajectories are teleported: each frame is a new pose with a piecewise
    constant velocity, so the acceleration is a jump between frames rather than a
    smooth curve. A Savitzky-Golay window smears those jumps over ~10 steps and
    understates them; this estimator computes a_k = (v_k - v_{k-1}) / dt directly,
    which is the honest reading of what the data commands.
    """
    v = np.diff(pts, axis=0) / dt                      # (T-1, 3) velocity per step
    dv = np.diff(v, axis=0) / dt                       # (T-2, 3) acceleration at boundaries
    v_mid = 0.5 * (v[1:] + v[:-1])
    denom = dv[:, 2] + G0
    ok = np.linalg.norm(dv + np.array([0.0, 0.0, G0]), axis=1) > 1.0
    out = np.full(len(dv), np.nan)
    if ok.any():
        out[ok] = np.degrees(np.arctan2(
            np.linalg.norm((dv + k_d * v_mid)[ok][:, :2], axis=1), denom[ok]))
    return out


def audit_trajectory(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    pts = np.asarray(z["point"], dtype=np.float64) if "point" in z.files else \
        np.column_stack([np.asarray(z["position"], dtype=np.float64), np.zeros(len(z["position"]))])
    roll = np.asarray(z["roll"], dtype=np.float64).ravel() if "roll" in z.files else None
    pitch = np.asarray(z["pitch"], dtype=np.float64).ravel() if "pitch" in z.files else None
    dep = tilt_from_roll_pitch(roll, pitch, degrees=False) \
        if roll is not None and pitch is not None else None

    row = {"file": path.name, "n_frames": int(len(pts)), "dt": {}}
    for tag, dt in CANDIDATE_DT.items():
        entry = {"dt_s": dt}
        for label, k_d in (("drag_free", 0.0), ("k_d_floor", K_D_FLOOR)):
            t = implied_tilt(pts, dt, k_d)
            finite = np.isfinite(t)
            entry[label] = {
                "median_tilt_deg": float(np.nanmedian(t)),
                "p90_tilt_deg": float(np.nanpercentile(t, 90)),
                "frac_frames_gt_10deg": float(np.nanmean(t > TILT_THRESHOLD_DEG)),
                "n_valid": int(finite.sum()),
            }
        for label, k_d in (("drag_free", 0.0), ("k_d_floor", K_D_FLOOR)):
            t = per_step_tilt(pts, dt, k_d)
            entry[f"per_step_{label}"] = {
                "median_tilt_deg": float(np.nanmedian(t)),
                "p90_tilt_deg": float(np.nanpercentile(t, 90)),
                "frac_frames_gt_10deg": float(np.nanmean(t > TILT_THRESHOLD_DEG)),
            }
        if dep is not None:
            t = implied_tilt(pts, dt, K_D_FLOOR)
            m = np.isfinite(t)
            entry["median_abs_diff_vs_depicted_deg"] = float(np.median(np.abs(t[m] - dep[m])))
        row["dt"][tag] = entry

    if dep is not None:
        row["depicted_tilt_deg"] = {
            "median": float(np.median(dep)),
            "p90": float(np.percentile(dep, 90)),
            "frac_gt_10deg": float(np.mean(dep > TILT_THRESHOLD_DEG)),
        }
        best = None
        for dt in DT_GRID:
            try:
                t = implied_tilt(pts, dt, K_D_FLOOR)
            except ValueError:
                continue
            m = np.isfinite(t)
            if m.sum() < 20:
                continue
            score = float(np.median(np.abs(t[m] - dep[m])))
            if best is None or score < best[1]:
                best = (float(dt), score)
        row["best_dt_by_attitude"] = {"dt_s": best[0], "median_abs_diff_deg": best[1]} if best else None
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    files = sorted(IN_DIR.glob("*.npz"))[: args.limit]
    if not files:
        raise SystemExit(f"no npz in {IN_DIR}; run scripts/fetch_anwm.py --fetch-test")
    train, test = load_split()

    rows = []
    att = []
    for p in files:
        r = audit_trajectory(p)
        z = np.load(p, allow_pickle=False)
        if "roll" in z.files and "pitch" in z.files:
            att.append((float(np.abs(z["roll"]).max()), float(np.abs(z["pitch"]).max()),
                        float(np.abs(z["roll"]).mean()), float(np.abs(z["pitch"]).mean())))
        stem = p.stem
        r["split"] = "test" if stem in test else "train" if stem in train else "unknown"
        rows.append(r)

    def pooled(metric_path: str) -> dict | None:
        vals = []
        for r in rows:
            node = r
            for k in metric_path.split("."):
                node = node[k]
            vals.append(node)
        vals = [v for v in vals if v is not None]
        return {"mean": float(np.mean(vals)), "min": float(np.min(vals)),
                "max": float(np.max(vals))} if vals else None

    summary = {
        "source": "EmbodiedCity/ANWM-Dataset (apache-2.0), pose keys only",
        "n_trajectories": len(rows),
        "n_test_split": sum(r["split"] == "test" for r in rows),
        "n_train_split": sum(r["split"] == "train" for r in rows),
        "k_d_used": K_D_FLOOR,
        "tilt_threshold_deg": TILT_THRESHOLD_DEG,
        "sg": {"window": SG_WINDOW, "polyorder": SG_POLY},
        "dt_is_undefined_in_release": True,
        "by_dt": {
            tag: {
                "dt_s": dt,
                "implied_median_tilt_deg_drag_free": pooled(f"dt.{tag}.drag_free.median_tilt_deg"),
                "implied_median_tilt_deg_k_d": pooled(f"dt.{tag}.k_d_floor.median_tilt_deg"),
                "frac_gt_10deg_k_d": pooled(f"dt.{tag}.k_d_floor.frac_frames_gt_10deg"),
                "per_step_median_tilt_deg_drag_free": pooled(f"dt.{tag}.per_step_drag_free.median_tilt_deg"),
                "per_step_p90_tilt_deg_drag_free": pooled(f"dt.{tag}.per_step_drag_free.p90_tilt_deg"),
                "per_step_frac_gt_10deg_drag_free": pooled(f"dt.{tag}.per_step_drag_free.frac_frames_gt_10deg"),
                "median_abs_diff_vs_depicted": pooled(f"dt.{tag}.median_abs_diff_vs_depicted_deg"),
            }
            for tag, dt in CANDIDATE_DT.items()
        },
        "depicted_attitude": {
            "max_abs_roll_rad": max(a[0] for a in att) if att else None,
            "max_abs_pitch_rad": max(a[1] for a in att) if att else None,
            "mean_abs_roll_rad": float(np.mean([a[2] for a in att])) if att else None,
            "mean_abs_pitch_rad": float(np.mean([a[3] for a in att])) if att else None,
            "note": "roll and pitch are the recorded reference-path attitude; FFP uses them, "
                    "so these are the attitudes the released frames depict",
        },
        "best_dt_by_attitude": {
            "median": float(np.median([r["best_dt_by_attitude"]["dt_s"] for r in rows
                                       if r.get("best_dt_by_attitude")])),
            "n": sum(bool(r.get("best_dt_by_attitude")) for r in rows),
            "grid_max_s": float(DT_GRID[-1]),
            "degenerate": bool(
                any(r.get("best_dt_by_attitude") for r in rows)
                and abs(float(np.median([r["best_dt_by_attitude"]["dt_s"] for r in rows
                                         if r.get("best_dt_by_attitude")])) - DT_GRID[-1]) < 1e-9),
            "why": "depicted tilt is identically zero, so the objective decreases with dt and "
                   "runs to the grid edge; use the per-dt discrepancy table instead",
        } if any(r.get("best_dt_by_attitude") for r in rows) else None,
        "per_trajectory": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "audit.json").write_text(json.dumps(summary, indent=1))

    print(f"trajectories {len(rows)} (test {summary['n_test_split']}, train {summary['n_train_split']})")
    print(f"{'dt':26s} {'dt s':>6} {'sg(0)':>7} {'sg(k_d)':>8} {'sg>10':>6} "
          f"{'step(0)':>8} {'step p90':>9} {'step>10':>8}")
    for tag, v in summary["by_dt"].items():
        m0 = v["implied_median_tilt_deg_drag_free"]
        mk = v["implied_median_tilt_deg_k_d"]
        f10 = v["frac_gt_10deg_k_d"]
        dd = v["median_abs_diff_vs_depicted"]
        ps = v["per_step_median_tilt_deg_drag_free"]
        pp = v["per_step_p90_tilt_deg_drag_free"]
        pf = v["per_step_frac_gt_10deg_drag_free"]
        print(f"{tag:26s} {v['dt_s']:6.3f} "
              f"{m0['mean'] if m0 else float('nan'):7.2f} {mk['mean'] if mk else float('nan'):8.2f} "
              f"{f10['mean'] if f10 else float('nan'):6.3f} "
              f"{ps['mean'] if ps else float('nan'):8.2f} {pp['mean'] if pp else float('nan'):9.2f} "
              f"{pf['mean'] if pf else float('nan'):8.3f}")
    if summary["best_dt_by_attitude"]:
        print("best dt by attitude agreement:", summary["best_dt_by_attitude"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
