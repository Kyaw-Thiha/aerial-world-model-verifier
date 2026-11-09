"""Track A2, Task 1: evaluate the amendment-2 confidence gate against the depicted-tilt error.

Question. Which pixel-only signal best says "this clip's depicted tilt cannot be trusted?",
and what threshold should freeze the decision before any generated clip is scored?

Method. Every clip of the Gate 3 clip set is scored with six candidate signals
(``src/perception/gate.py``), and labelled by the error of the depicted gravity against
mocap (label = median error > 10 deg). Signals never see the label, the path, the implied
tilt or any verdict outcome. Selection and threshold follow the committed amendment 2:
highest AUROC among single-frame signals, flight-level bootstrap interval, tau maximising
TPR - FPR subject to keeping at least 60 % of fit clips, leave-one-flight-out.

Nothing here excludes a flight. A dropped clip is "not testable".

    PYTHONPATH=. pixi run -e gpu python experiments/a2_gate_validate.py \\
        --gate-rows results/a2_gate/tii --out results/a2_gate --freeze
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.perception.frames import body_to_world
from src.perception.gate import (SINGLE_FRAME, bootstrap_auroc_ci, clip_label,
                                 clip_signal, kept, lofo_tau, pick_tau, up_truth_in_camera)
from src.fusion.attitude import relative_rotations
from src.score.verdict import clip_index

ROOT = Path(__file__).resolve().parents[1]
LABEL_THR = 10.0
RESULTS = ["S1_gravity_uncertainty", "S2_up_confidence", "S3_latitude_confidence",
           "S4_temporal_jitter", "S5_vo_agreement", "S6_uncertainty_instability"]


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def load_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["frame"]] = r
    return out


def array_of(rows: dict[str, dict], names: list[str], key: str) -> np.ndarray:
    vals = [rows[n].get(key, np.nan) if n in rows else np.nan for n in names]
    return np.array([np.nan if v is None else v for v in vals], dtype=np.float64)


def dpvo_rotations(flight: str, dpvo_root: Path, names: list[str]) -> np.ndarray | None:
    """Camera-to-camera relative rotations for consecutive kept frames, NaN where missing."""
    d = dpvo_root / f"{flight}_stride10_fov100_s0.5" / "poses.npz"
    if not d.exists():
        return None
    z = np.load(d)
    poses, ts = z["poses"], z["frame_ts_us"]
    quat = poses[:, 3:7]                       # tx ty tz qx qy qz qw
    R_wc = Rotation.from_quat(quat).as_matrix()
    idx = {int(t): i for i, t in enumerate(ts)}
    want = [int(Path(n).stem.split("_")[1]) for n in names]
    sel = np.array([idx.get(t, -1) for t in want])
    have = sel >= 0
    R_rel = np.full((max(0, len(names) - 1), 3, 3), np.nan)
    pair_ok = np.zeros(max(0, len(names) - 1), dtype=bool)
    if have.sum() >= 2:
        Rs = np.full((len(names), 3, 3), np.nan)
        Rs[have] = R_wc[sel[have]]
        rr = relative_rotations(Rs)
        pair_ok = have[:-1] & have[1:]
        R_rel[pair_ok] = rr[pair_ok]
    cov = float(pair_ok.mean()) if pair_ok.size else 0.0
    return {"R_rel": R_rel, "pair_ok": pair_ok, "coverage": cov, "n_rows": int(have.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-rows", default="results/a2_gate/tii")
    ap.add_argument("--specs", default="data/processed/p3a/p3a_v1")
    ap.add_argument("--dpvo", default="data/raw/dpvo")
    ap.add_argument("--out", default="results/a2_gate")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + git_hash()
    out = ROOT / args.out / run
    out.mkdir(parents=True, exist_ok=True)
    spec_dir = ROOT / args.specs
    gate_root = ROOT / args.gate_rows
    clips = json.loads((spec_dir / "clips.json").read_text())["clips"]

    # ---- 1. per-clip signals and labels
    table: list[dict] = []
    for p in sorted(spec_dir.glob("flight-*.npz")):
        flight = p.name.replace(".npz", "")
        spec = np.load(p, allow_pickle=False)
        names = [str(n) for n in spec["names"]]
        dt = float(np.median(np.diff(spec["t"])))
        rows_by_cond = {k.replace(".jsonl", "").split("__")[1]: load_jsonl(f)
                        for f in sorted(gate_root.glob(f"{flight}__*.jsonl"))
                        for k in [f.name]}
        if not rows_by_cond:
            print(f"{flight}: no gate rows, skipped", flush=True)
            continue
        truth = up_truth_in_camera(spec["quat"], float(spec["uptilt_deg"]))
        dp = dpvo_rotations(flight, ROOT / args.dpvo, names)
        for clip in clips:
            if clip["flight"] != flight:
                continue
            idx = clip_index(clip)
            rec = {"clip_id": clip["clip_id"], "flight": flight,
                   "track": clip["track"], "control": clip["control"],
                   "speed_median": clip["speed_median"], "dt_s": dt}
            # label: identity-condition pixels against mocap, no nuisances
            ident = rows_by_cond.get("identity")
            if ident is not None:
                up_all = np.stack([np.array(ident[n]["gravity_up_cam"]) if n in ident
                                   else np.full(3, np.nan) for n in names])
                lab = clip_label(up_all[idx], truth[idx], LABEL_THR)
                rec.update({"median_error_deg": lab["median_error_deg"],
                            "label": lab["label"], "n_label_frames": lab["n_frames"]})
            for cond, rows in rows_by_cond.items():
                up_all = np.stack([np.array(rows[n]["gravity_up_cam"]) if n in rows
                                   else np.full(3, np.nan) for n in names])
                arrs = {
                    "up": up_all[idx],
                    "gravity_uncertainty_raw": array_of(rows, names,
                                                        "gravity_uncertainty_raw")[idx],
                    "up_confidence_mean": array_of(rows, names, "up_confidence_mean")[idx],
                    "latitude_confidence_mean": array_of(rows, names,
                                                         "latitude_confidence_mean")[idx],
                }
                for sig in RESULTS:
                    val = np.nan
                    if sig in ("S1_gravity_uncertainty", "S2_up_confidence",
                               "S3_latitude_confidence", "S6_uncertainty_instability"):
                        val = clip_signal(sig, arrs, dt_s=dt)
                    elif sig == "S4_temporal_jitter":
                        val = clip_signal(sig, arrs, dt_s=dt)
                    elif sig == "S5_vo_agreement":
                        if dp is not None and dp["coverage"] >= 0.5:
                            pair_idx = np.zeros(len(names) - 1, dtype=bool)
                            pair_idx[idx[:-1]] = dp["pair_ok"][idx[:-1]]
                            R = dp["R_rel"][idx[:-1]]
                            R = np.where(pair_idx[:, None, None], R, np.nan)
                            val = clip_signal(sig, arrs, R_rel=R)
                    rec[f"{cond}::{sig}"] = float(val) if np.isfinite(val) else None
            if dp is not None:
                rec["dpvo_coverage"] = dp["coverage"]
            table.append(rec)
        print(f"{flight}: {sum(1 for r in table if r['flight'] == flight)} clips", flush=True)

    # ---- 2. validation on the identity condition
    ident_tab = [r for r in table if r.get("label") is not None]
    labels = np.array([r["label"] for r in ident_tab], dtype=np.float64)
    flights = np.array([r["flight"] for r in ident_tab])
    sig_vals: dict[str, np.ndarray] = {}
    for sig in RESULTS:
        sig_vals[sig] = np.array([np.nan if r.get(f"identity::{sig}") is None
                                  else r[f"identity::{sig}"] for r in ident_tab],
                                 dtype=np.float64)

    n_pos = int(labels.sum())
    validation = {"n_clips": len(ident_tab), "n_positive": n_pos,
                  "label_deg": LABEL_THR, "signals": {}}
    for sig in RESULTS:
        v = sig_vals[sig]
        ci = bootstrap_auroc_ci(v, labels, flights, n_boot=args.n_boot, seed=0)
        loo = lofo_tau(v, labels, flights)
        hold = {}
        kept_err, drop_err = [], []
        for u in np.unique(flights):
            m = flights == u
            tau_f = loo[str(u)]["tau"]
            k = np.array([kept(vv, tau_f) for vv in v[m]])
            hold[str(u)] = {
                "n": int(m.sum()), "tau": tau_f,
                "coverage": float(np.mean(k)) if m.sum() else float("nan"),
                "n_dropped_pos": int(np.sum(~k & (labels[m] == 1))),
                "n_dropped_neg": int(np.sum(~k & (labels[m] == 0))),
            }
            errs = np.array([r["median_error_deg"] for r, mm in zip(ident_tab, m) if mm])
            kept_err += list(errs[k & np.isfinite(errs)])
            drop_err += list(errs[(~k) & np.isfinite(errs)])
        validation["signals"][sig] = {
            "eligible_to_freeze": bool(SINGLE_FRAME[sig]),
            "single_frame": SINGLE_FRAME[sig],
            "n_finite": int(np.isfinite(v).sum()),
            "auroc": ci, "lofo": hold,
            "lofo_coverage_overall": float(np.mean([h["coverage"] for h in hold.values()])),
            "kept_median_error_deg": float(np.median(kept_err)) if kept_err else None,
            "dropped_median_error_deg": float(np.median(drop_err)) if drop_err else None,
            "kept_n": len(kept_err), "dropped_n": len(drop_err),
        }
        print(f"{sig}: AUROC {ci['auroc']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}] "
              f"coverage {validation['signals'][sig]['lofo_coverage_overall']:.2f}",
              flush=True)

    # ---- 3. selection (amendment 2)
    eligible = [s for s in RESULTS if SINGLE_FRAME[s] and np.isfinite(
        validation["signals"][s]["auroc"]["auroc"])]
    best = max(eligible, key=lambda s: validation["signals"][s]["auroc"]["auroc"])
    bi = validation["signals"][best]["auroc"]
    validated = bool(n_pos >= 5 and np.isfinite(bi["lo"]) and bi["lo"] > 0.5)
    tau_all = pick_tau(sig_vals[best], labels)
    validation["selection"] = {
        "rule": "highest AUROC among single-frame signals (amendment_2.selection_criterion)",
        "best_signal": best, "best_auroc": bi, "validated": validated,
        "tau_all_flights": float(tau_all),
        "kept_fraction_all": float(np.mean([kept(vv, tau_all) for vv in sig_vals[best]])),
        "eligible": eligible,
    }
    (out / "gate_validation.json").write_text(json.dumps(
        {"run_id": run, "git": git_hash(), "gate_rows": args.gate_rows,
         "validation": validation, "clips": table}, indent=1, default=float))
    # CSV for eyeballing the per-clip table
    cols = ["clip_id", "flight", "track", "control", "speed_median", "dt_s",
            "median_error_deg", "label"]
    cols += [f"identity::{s}" for s in RESULTS]
    with (out / "clip_table.csv").open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in table:
            fh.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")

    print(json.dumps(validation["selection"], indent=2, default=float))

    # ---- 4. freeze (committed before any generated clip is scored)
    if args.freeze:
        if not validated:
            print("NOT VALIDATED: no configs/gate_frozen.yaml written")
            return 3
        cfg = {
            "prereg_amendment": 2,
            "signal": best,
            "tau": float(tau_all),
            "direction": "drop when signal > tau",
            "label_deg": LABEL_THR,
            "min_keep": 0.60,
            "validation": {"auroc": bi["auroc"], "lo": bi["lo"], "hi": bi["hi"],
                           "n_clips": len(ident_tab), "n_positive": n_pos,
                           "lofo_coverage": validation["signals"][best]["lofo_coverage_overall"],
                           "kept_median_error_deg":
                               validation["signals"][best]["kept_median_error_deg"],
                           "dropped_median_error_deg":
                               validation["signals"][best]["dropped_median_error_deg"]},
            "fit_run": run,
            "commit": "filled by the freezing commit",
        }
        (ROOT / "configs/gate_frozen.yaml").write_text(
            "# FROZEN confidence gate (pre-registration amendment 2).\n"
            "# Written by experiments/a2_gate_validate.py; committed BEFORE any generated\n"
            "# clip is scored. Do not edit: a change is a new amendment.\n"
            + json.dumps({k: v for k, v in cfg.items() if k != "commit"}, indent=2) + "\n")
        print("wrote configs/gate_frozen.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
