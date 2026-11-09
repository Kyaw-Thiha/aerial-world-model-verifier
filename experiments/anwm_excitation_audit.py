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

from src.generation.audit import per_step_tilt as _per_step_tilt
from src.generation.audit import pooled_stats
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
    """Delegates to src.generation.audit (one implementation, tested against the old body).

    These trajectories are teleported: each frame is a new pose with piecewise constant
    velocity, so acceleration is a jump between frames. A Savitzky-Golay window smears the
    jumps over ~10 steps and understates them; this estimator computes
    a_k = (v_k - v_{k-1}) / dt directly, which is the honest reading of what is commanded.
    """
    return _per_step_tilt(pts, dt, k_d)


def audit_trajectory(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    pts = np.asarray(z["point"], dtype=np.float64) if "point" in z.files else \
        np.column_stack([np.asarray(z["position"], dtype=np.float64), np.zeros(len(z["position"]))])
    roll = np.asarray(z["roll"], dtype=np.float64).ravel() if "roll" in z.files else None
    pitch = np.asarray(z["pitch"], dtype=np.float64).ravel() if "pitch" in z.files else None
    dep = tilt_from_roll_pitch(roll, pitch, degrees=False) \
        if roll is not None and pitch is not None else None

    row = {"file": path.name, "n_frames": int(len(pts)), "dt": {}}
    arrays: dict[str, np.ndarray] = {}
    for tag, dt in CANDIDATE_DT.items():
        entry = {"dt_s": dt}
        for label, k_d in (("drag_free", 0.0), ("k_d_floor", K_D_FLOOR)):
            t = implied_tilt(pts, dt, k_d)
            arrays[f"{tag}.sg.{label}"] = t
            finite = np.isfinite(t)
            entry[label] = {
                "median_tilt_deg": float(np.nanmedian(t)),
                "p90_tilt_deg": float(np.nanpercentile(t, 90)),
                "frac_frames_gt_10deg": float(np.nanmean(t > TILT_THRESHOLD_DEG)),
                "n_valid": int(finite.sum()),
            }
        for label, k_d in (("drag_free", 0.0), ("k_d_floor", K_D_FLOOR)):
            t = per_step_tilt(pts, dt, k_d)
            arrays[f"{tag}.per_step.{label}"] = t
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
    return row, arrays


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
    pooled_arrays: dict[str, list[np.ndarray]] = {}
    for p in files:
        r, arrays = audit_trajectory(p)
        for k, v in arrays.items():
            pooled_arrays.setdefault(k, []).append(v)
        z = np.load(p, allow_pickle=False)
        if "roll" in z.files and "pitch" in z.files:
            att.append((float(np.abs(z["roll"]).max()), float(np.abs(z["pitch"]).max()),
                        float(np.abs(z["roll"]).mean()), float(np.abs(z["pitch"]).mean())))
        stem = p.stem
        r["split"] = "test" if stem in test else "train" if stem in train else "unknown"
        rows.append(r)

    def pooled(key: str) -> dict | None:
        """Median / p90 / fraction above threshold over ONE pooled array per cell.

        The Task 0 fix: the previous version averaged per-trajectory medians (bimodal: 0 deg
        for trajectories with fewer than half their boundaries at transitions), which is not
        the median of any step population and can exceed the threshold while the pooled
        fraction stays below 0.5.
        """
        arr = pooled_arrays.get(key)
        if not arr:
            return None
        st = pooled_stats(arr, TILT_THRESHOLD_DEG)
        return {**st, "per_trajectory_median_mean": float(np.mean([np.nanmedian(a) for a in arr]))}

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
                "sg_drag_free": pooled(f"{tag}.sg.drag_free"),
                "sg_k_d": pooled(f"{tag}.sg.k_d_floor"),
                "per_step_drag_free": pooled(f"{tag}.per_step.drag_free"),
                "per_step_k_d": pooled(f"{tag}.per_step.k_d_floor"),
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
    print("one pooled population per cell (median | p90 | frac>10deg):")
    for tag, v in summary["by_dt"].items():
        def fmt(n):
            return "   n/a   " if not n else f"{n['median']:6.2f}|{n['p90']:6.1f}|{n['frac_gt']:5.3f}"
        print(f"  {tag:24s} dt={v['dt_s']:.3f}  step(0) {fmt(v['per_step_drag_free'])}"
              f"  step(k_d) {fmt(v['per_step_k_d'])}  sg(0) {fmt(v['sg_drag_free'])}"
              f"  sg(k_d) {fmt(v['sg_k_d'])}  n={v['per_step_drag_free']['n'] if v['per_step_drag_free'] else 0}")
    if summary["best_dt_by_attitude"]:
        print("best dt by attitude agreement:", summary["best_dt_by_attitude"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
