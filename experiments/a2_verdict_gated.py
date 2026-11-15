"""Track A2, Task 2: the verdict test again, with the frozen confidence gate and drag as axes.

Same protocol as Gate 3 (``experiments/phase3a_verdict.py``) — same clips, same edits,
same nuisances, same abstention rule, same conformal threshold — with three changes:

* the confidence gate (amendment 2) is applied per condition, from the frozen
  ``configs/gate_frozen.yaml``;
* the conformal threshold is fitted over **all eight flights** (amendment 2), with no
  flight excluded by identity;
* the drag constant becomes an axis: ``k_d`` in {0, 0.15, 0.295}. ``k_d`` = 0 is the
  conservative claim (a flight must justify its tilt with no drag at all).

The gate-off run at ``k_d`` = 0.295 with the old ``non_trackratm`` policy is computed in the
same pass and must reproduce Gate 3 (pooled AUROC 0.8214, acceptance 0.7108); if it does
not, the extension changed the baseline and the run is void.

    PYTHONPATH=. pixi run -e gpu python experiments/a2_verdict_gated.py \\
        --gate-rows results/a2_gate/tii --out results/a2_verdict_gated
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from src.datasets.tii import load_tii
from src.perception.gate import SINGLE_FRAME, clip_signal, kept
from src.physics.flatness import G0, implied_thrust_axis, tilt_deg
from src.score.prereg import verify_prereg
from src.score.verdict import (auroc_detect, clip_index, depicted_axis_world, fit_nuisances,
                               score_clip, shift_series, speed_bin)
from experiments.phase3a_verdict import evaluation, is_trackratm, threshold_lookup

ROOT = Path(__file__).resolve().parents[1]
FLIGHTS = ["flight-01a-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
           "flight-13a-trackRATM", "flight-01p-ellipse", "flight-07p-lemniscate",
           "flight-13p-trackRATM", "flight-16p-trackRATM"]
NEGATIVES = ["tilt_k0", "tilt_k05", "tilt_k15", "mirror_roll", "yaw_control"]
K_DS = [0.295, 0.15, 0.0]
SG_WINDOW_S = 0.37
SG_POLY = 3


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def verify_gate_config(path: Path) -> dict:
    """The frozen gate must be committed and unmodified, for the same reason as the prereg."""
    rel = str(path.relative_to(ROOT))
    subprocess.check_output(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT)
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel],
                           cwd=ROOT).returncode != 0
    if dirty:
        raise SystemExit(f"{rel} has uncommitted changes; commit the freeze first")
    import yaml
    return yaml.safe_load(path.read_text())


def load_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["frame"]] = r
    return out


def mocap_derivs(mocap_path: Path) -> dict:
    """Velocity and acceleration at the native rate, once per flight (k_d-free)."""
    clip = load_tii(mocap_path, name=mocap_path.stem)
    dt = 1.0 / clip.fps
    w = min(101, len(clip.pos) - 1)
    if w % 2 == 0:
        w -= 1
    vel = savgol_filter(clip.pos, w, SG_POLY, deriv=1, delta=dt, axis=0, mode="interp")
    acc = savgol_filter(clip.pos, w, SG_POLY, deriv=2, delta=dt, axis=0, mode="interp")
    return {"clip": clip, "vel": vel, "acc": acc, "w": w}


def axis_at(d: dict, t_frame: np.ndarray, k_d: float) -> np.ndarray:
    """Implied thrust axis sampled at ``t_frame`` for a given drag constant."""
    clip, w = d["clip"], d["w"]
    axis = implied_thrust_axis(d["acc"], d["vel"], k_d, G0)
    axis[: w // 2] = np.nan
    axis[-(w // 2):] = np.nan
    tt = np.clip(t_frame, clip.t[0], clip.t[-1])
    return np.column_stack([np.interp(tt, clip.t, axis[:, k]) for k in range(3)])


def gate_value(cfg: dict, arrs: dict, dt_s: float) -> float:
    return clip_signal(cfg["signal"], arrs, dt_s=dt_s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-rows", default="results/gate_a2/tii")
    ap.add_argument("--specs", default="data/processed/p3a/p3a_v1")
    ap.add_argument("--mocap-root", default="data/raw/tii")
    ap.add_argument("--gate-config", default="configs/gate_frozen.yaml")
    ap.add_argument("--gate", default="on", choices=["on", "off"])
    ap.add_argument("--gate-apply", default="per_condition",
                    choices=["per_condition", "identity"],
                    help="'identity' gates each clip on its REAL-condition pixels and applies "
                         "that decision to every condition (analysis only, not pre-registered)")
    ap.add_argument("--policy", default="all_flights")
    ap.add_argument("--k-ds", default=",".join(str(k) for k in K_DS))
    ap.add_argument("--out", default="results/a2_verdict_gated")
    args = ap.parse_args()

    info = verify_prereg()
    prereg = info["doc"]
    cfg = verify_gate_config(ROOT / args.gate_config) if args.gate == "on" else None
    if cfg is not None and not SINGLE_FRAME[cfg["signal"]]:
        raise SystemExit(f"{cfg['signal']} is not a single-frame signal; amendment 2 forbids it")

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + git_hash()
    out = ROOT / args.out / run
    out.mkdir(parents=True, exist_ok=True)

    spec_dir = ROOT / args.specs
    gate_root = ROOT / args.gate_rows
    clips = json.loads((spec_dir / "clips.json").read_text())["clips"]
    k_ds = [float(k) for k in args.k_ds.split(",")]

    # per-flight inputs, built once
    flights: dict[str, dict] = {}
    for flight in FLIGHTS:
        spec = np.load(spec_dir / f"{flight}.npz", allow_pickle=False)
        control, _ = flight_parts_cached(flight)
        mocap = ROOT / args.mocap_root / control / f"mocap_{flight}.csv"
        R_bw = Rotation.from_quat(spec["quat"]).as_matrix()
        yaw = Rotation.from_matrix(R_bw).as_euler("ZYX")[:, 0]
        rows_by_cond = {f.name.split("__")[1].replace(".jsonl", ""): load_jsonl(f)
                        for f in sorted(gate_root.glob(f"{flight}__*.jsonl"))}
        flights[flight] = {"spec": spec, "d": mocap_derivs(mocap), "yaw": yaw,
                           "rows": rows_by_cond,
                           "names": [str(n) for n in spec["names"]],
                           "uptilt": float(spec["uptilt_deg"]),
                           "dt": float(np.median(np.diff(spec["t"])))}

    results: dict[str, dict] = {}
    for k_d in k_ds:
        # pair-level decisions, computed once per (flight, clip) when requested
        pairs: dict[tuple[str, str], bool] = {}
        records: list[dict] = []
        max_axis_diff = 0.0
        for flight, F in flights.items():
            spec = F["spec"]
            axis = axis_at(F["d"], spec["t"], k_d)
            if abs(k_d - 0.295) < 1e-12:
                # the recomputation must reproduce the stored Gate 3 axis exactly
                a, b = axis, spec["implied_axis"]
                m = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
                an = a[m] / np.linalg.norm(a[m], axis=1, keepdims=True)
                bn = b[m] / np.linalg.norm(b[m], axis=1, keepdims=True)
                max_axis_diff = max(max_axis_diff, float(
                    np.degrees(np.arccos(np.clip(np.sum(an * bn, axis=1), -1, 1))).max()))
            tilt = tilt_deg(axis)
            for clip in clips:
                if clip["flight"] != flight:
                    continue
                idx = clip_index(clip)
                if cfg is not None and args.gate_apply == "identity":
                    irows = F["rows"].get("identity")
                    nm = F["names"]
                    if irows is not None:
                        arrs_i = {"up": np.stack([np.array(irows[n]["gravity_up_cam"])
                                                  if n in irows else np.full(3, np.nan)
                                                  for n in nm])[idx],
                                  "gravity_uncertainty_raw": np.array(
                                      [irows[n].get("gravity_uncertainty_raw", np.nan)
                                       if n in irows else np.nan for n in nm])[idx],
                                  "up_confidence_mean": np.array(
                                      [irows[n].get("up_confidence_mean", np.nan)
                                       if n in irows else np.nan for n in nm])[idx],
                                  "latitude_confidence_mean": np.array(
                                      [irows[n].get("latitude_confidence_mean", np.nan)
                                       if n in irows else np.nan for n in nm])[idx]}
                        pairs[(flight, clip["clip_id"])] = bool(
                            kept(clip_signal(cfg["signal"], arrs_i, dt_s=F["dt"]),
                                 float(cfg["tau"])))
                for cond, rows in F["rows"].items():
                    names = F["names"]
                    up_all = np.stack([np.array(rows[n]["gravity_up_cam"]) if n in rows
                                       else np.full(3, np.nan) for n in names])
                    valid_all = np.array([float(rows[n].get("valid_frac", 1.0)) >= 0.5
                                          if n in rows else False for n in names])
                    rec = {"clip_id": clip["clip_id"], "flight": flight,
                           "track": clip["track"], "control": clip["control"],
                           "condition": cond, "speed": clip["speed_median"],
                           "bin": speed_bin(clip["speed_median"],
                                            prereg["thresholds"]["speed_bins_mps"]),
                           "tilt_implied_median": float(np.nanmedian(tilt[idx])),
                           "valid_frac_mean": float(np.mean(valid_all[idx])),
                           "edit_angle_deg": float(np.median(spec[f"angle_{cond}"][idx]))
                           if f"angle_{cond}" in spec.files else 0.0,
                           "k_d": k_d}
                    sub_up, sub_t, sub_tilt = up_all[idx], spec["t"][idx], tilt[idx]
                    sub_yaw, sub_valid = F["yaw"][idx], valid_all[idx]
                    r = score_clip(sub_up, axis[idx], sub_yaw, sub_t, F["uptilt"], sub_tilt,
                                   sub_valid, prereg)
                    rec.update(r)
                    rec["abstain_reasons_excitation"] = list(r["abstain_reasons"])
                    if cfg is not None:
                        arrs = {"up": sub_up,
                                "gravity_uncertainty_raw": np.array(
                                    [rows[n].get("gravity_uncertainty_raw", np.nan)
                                     if n in rows else np.nan for n in names])[idx],
                                "up_confidence_mean": np.array(
                                    [rows[n].get("up_confidence_mean", np.nan)
                                     if n in rows else np.nan for n in names])[idx],
                                "latitude_confidence_mean": np.array(
                                    [rows[n].get("latitude_confidence_mean", np.nan)
                                     if n in rows else np.nan for n in names])[idx]}
                        g = gate_value(cfg, arrs, F["dt"])
                        rec["gate_signal"] = float(g) if np.isfinite(g) else None
                        if args.gate_apply == "identity":
                            rec["gate_kept"] = pairs.get((flight, clip["clip_id"]), False)
                        else:
                            rec["gate_kept"] = bool(kept(g, float(cfg["tau"])))
                        if not rec["gate_kept"]:
                            rec["testable"] = False
                            rec["abstain_reasons"] = list(r["abstain_reasons"]) + [
                                f"confidence_gate {cfg['signal']}={g} > tau={cfg['tau']}"]
                    records.append(rec)

        ev = evaluation(records, prereg, policy=args.policy)
        ev["unmatched_auroc"] = unmatched(records)
        ev["abstention_with_gate"] = abstention(records)
        ev["acceptance_per_flight"] = acceptance_per_flight(records, ev)
        ev["k_d"] = k_d
        ev["gate"] = {"mode": args.gate, "apply": args.gate_apply, "config": cfg,
                      "pre_registered": args.gate_apply == "per_condition"}
        ev["policy"] = args.policy
        ev["max_axis_reproduction_deg"] = max_axis_diff if abs(k_d - 0.295) < 1e-12 else None
        ev["_records"] = records      # per-clip rows, so a table can be traced to a run
        results[f"kd_{k_d:g}"] = ev
        def f(x, nd=3):
            return "n/a" if x is None else f"{x:.{nd}f}"
        n_test = sum(1 for r in records if r["testable"])
        print(f"k_d={k_d}: records {len(records)} testable {n_test} "
              f"pooled {f(ev['family_auroc_pooled'])} "
              f"acceptance {f(ev['heldout_acceptance'])} (n={ev.get('heldout_n')}) "
              f"abstain {f(ev['abstention_with_gate']['overall'])}", flush=True)

    (out / "metrics.json").write_text(json.dumps(
        {"run_id": run, "git": git_hash(),
         "prereg": {k: info[k] for k in ("path", "sha256", "git_hash")},
         "gate_config": cfg, "config": vars(args), "results": results},
        indent=1, default=float))
    print(json.dumps({k: {"records": len(v.get("_records", [])), "pooled": v["family_auroc_pooled"],
                          "acceptance": v["heldout_acceptance"],
                          "yaw_auroc": (v["families"].get("yaw_control") or {}).get("auroc"),
                          "mdv80": v.get("mdv_smallest_rejected80_deg")}
                      for k, v in results.items()}, indent=2, default=float))
    return 0


def flight_parts_cached(flight: str) -> tuple[str, str]:
    from src.datasets.tii_video import flight_parts
    return flight_parts(flight)


def unmatched(records: list[dict]) -> dict:
    """Pooled AUROC with every testable real clip as one positive, all edits as negatives.

    This is the harder reading of the same numbers: Gate 3's per-family AUROC pairs each
    edit with its own original clip, which repeats each real clip up to three times.
    """
    real = {(r["flight"], r["clip_id"]): r["score"] for r in records
            if r["condition"] == "identity" and r["testable"]}
    real_scores = np.array(list(real.values()), dtype=np.float64)
    out: dict = {"n_real_unique": len(real), "per_family": {}}
    pooled = []
    for cond in NEGATIVES:
        neg = np.array([r["score"] for r in records
                        if r["condition"] == cond and r["testable"]], dtype=np.float64)
        out["per_family"][cond] = {
            "n": int(len(neg)),
            "auroc": auroc_detect(real_scores, neg) if len(neg) else None}
        if cond.startswith("tilt_k"):
            pooled.extend(neg.tolist())
    out["pooled_tilt_families"] = auroc_detect(real_scores, np.array(pooled)) if pooled else None
    return out


def abstention(records: list[dict]) -> dict:
    """Not-testable fraction overall, by flight type, and by cause (excitation vs gate)."""
    ident = [r for r in records if r["condition"] == "identity"]

    def frac(sel, pred):
        sub = [r for r in ident if sel(r)]
        return float(np.mean([pred(r) for r in sub])) if sub else None

    return {
        "overall": frac(lambda r: True, lambda r: not r["testable"]),
        "excitation": frac(lambda r: True, lambda r: not r["testable"]
                           and r.get("gate_kept") is not False),
        "gate": frac(lambda r: True, lambda r: r.get("gate_kept") is False),
        "trackratm": frac(is_trackratm, lambda r: not r["testable"]),
        "autonomous": frac(lambda r: r["control"] == "autonomous",
                           lambda r: not r["testable"]),
        "piloted": frac(lambda r: r["control"] == "piloted", lambda r: not r["testable"]),
        "per_flight": {f: frac(lambda r, f=f: r["flight"] == f, lambda r: not r["testable"])
                       for f in sorted({r["flight"] for r in ident})},
        "gate_dropped_per_flight": {
            f: frac(lambda r, f=f: r["flight"] == f, lambda r: r.get("gate_kept") is False)
            for f in sorted({r["flight"] for r in ident})},
    }


def acceptance_per_flight(records: list[dict], ev: dict) -> dict:
    """Acceptance per flight at the held-out thresholds of the primary policy."""
    out = {}
    fold = {f["flight"]: f["thresholds"] for f in ev.get("fold_thresholds", [])}
    for r in records:
        if r["condition"] != "identity":
            continue
        f = r["flight"]
        d = out.setdefault(f, {"n_testable": 0, "accepted": 0, "n_clips": 0})
        d["n_clips"] += 1
        if not r["testable"]:
            continue
        t = threshold_lookup(fold.get(f, {"pooled": {"threshold": np.nan}}), r["bin"])
        if not np.isfinite(t):
            continue
        d["n_testable"] += 1
        d["accepted"] += int(r["score"] <= t)
    for d in out.values():
        d["acceptance"] = d["accepted"] / d["n_testable"] if d["n_testable"] else None
    return out


if __name__ == "__main__":
    raise SystemExit(main())
