"""Track P, Part 2: thrust-envelope admissibility of the commanded path (Check C).

Pre-registration `configs/p2_envelope_prereg.yaml` (committed before any envelope number).
Path only: no video, no rotation estimate, no reader.

    PYTHONPATH=. pixi run python experiments/p2_envelope_gate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from experiments.p_units import H04_PATHS, iter_units, sim_manifest
from src.physics.flatness import G0
from src.verdict.anchor import first_sustained_motion, last_sustained_motion
from src.verdict.rate import K_D, filter_gain, thrust_axis, uniform_path, axis_rate

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/p/p2"
PHYS = {"tilt": 60.0, "twr": 2.0, "rate": 720.0}
Q = 0.995


def path_quantities(t: np.ndarray, pos: np.ndarray) -> dict:
    """Peak tilt, thrust ratio, acceleration, required body rate and jerk for one path."""
    tg, vel, acc, interior = uniform_path(t, pos)
    b3 = thrust_axis(acc, vel)
    u = acc + np.array([0.0, 0.0, G0]) + K_D * vel
    tilt = np.degrees(np.arccos(np.clip(b3[:, 2], -1, 1)))
    twr = np.linalg.norm(u, axis=1) / G0
    amag = np.linalg.norm(acc, axis=1)
    rate = np.degrees(np.linalg.norm(axis_rate(tg, b3), axis=1))
    dt = float(np.median(np.diff(tg)))
    jerk = np.linalg.norm(np.column_stack([
        savgol_filter(pos_g[:, k], 31, 3, deriv=3, delta=dt) for k in range(3)]), axis=1) \
        if (pos_g := np.column_stack([np.interp(tg, t, pos[:, k]) for k in range(3)])) is not None \
        else np.zeros(len(tg))
    keep = interior & np.isfinite(tilt) & np.isfinite(rate)
    if keep.sum() < 10:
        keep = np.ones(len(tg), bool)
    # exploratory amendment A: the airborne interior only (the N3 rule), because the ends of a
    # real recording are take-off, landing or post-landing handling, not flight
    speed = np.linalg.norm(vel, axis=1)
    t0 = first_sustained_motion(tg, speed)
    t1 = last_sustained_motion(tg, speed)
    air = keep.copy()
    if np.isfinite(t0) and np.isfinite(t1) and t1 > t0:
        air &= (tg >= t0) & (tg <= t1)
    if air.sum() < 10:
        air = keep
    out = {"peak_tilt_deg": float(np.max(tilt[keep])),
            "q995_tilt_deg": float(np.quantile(tilt[keep], Q)),
            "peak_twr": float(np.max(twr[keep])),
            "q995_twr": float(np.quantile(twr[keep], Q)),
            "peak_acc_ms2": float(np.max(amag[keep])),
            "q995_acc_ms2": float(np.quantile(amag[keep], Q)),
            "peak_rate_deg_s": float(np.nanmax(rate[keep])),
            "q995_rate_deg_s": float(np.nanquantile(rate[keep], Q)),
            "peak_jerk_ms3": float(np.nanmax(jerk[keep])),
            "q995_jerk_ms3": float(np.nanquantile(jerk[keep], Q)),
            "duration_s": float(t[-1] - t[0]), "n_frames": int(len(tg)),
            "air_t0_s": float(t0) if np.isfinite(t0) else None,
            "air_t1_s": float(t1) if np.isfinite(t1) else None,
            "air_peak_tilt_deg": float(np.max(tilt[air])),
            "air_peak_twr": float(np.max(twr[air])),
            "air_peak_rate_deg_s": float(np.nanmax(rate[air])),
            "air_peak_jerk_ms3": float(np.nanmax(jerk[air])),
            "air_q995_tilt_deg": float(np.quantile(tilt[air], Q)),
            "air_q995_twr": float(np.quantile(twr[air], Q)),
            "air_q995_rate_deg_s": float(np.nanquantile(rate[air], Q)),
            "air_q995_jerk_ms3": float(np.nanquantile(jerk[air], Q))}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    real = []
    for kind in ("uzh", "tii"):
        for u in iter_units(kind):
            q = path_quantities(u["t"], u["pos"])
            real.append({"kind": kind, "unit": u["unit"], **q})
    real = pd.DataFrame(real)
    sim = []
    man = sim_manifest("dev")
    for u in iter_units("sim"):
        q = path_quantities(u["t"], u["pos"])
        m = u["row"]
        sim.append({"kind": "sim", "unit": u["unit"], "label": m["label"],
                    "defect_family": m["defect_family"], "family": m["family"],
                    "mount": float(m["uptilt_deg"]), "maneuver": m["maneuver"],
                    "peak_tilt_req_manifest": float(m["peak_tilt_req_deg"]), **q})
    sim = pd.DataFrame(sim)
    # H04 generated path set (the commanded paths given to SolarWM), 1 kHz
    d = np.load(H04_PATHS, allow_pickle=True)
    keys = sorted({k.split("__")[0] for k in d.keys()})
    gen = []
    for key in keys:
        t = np.asarray(d[f"{key}__t_1khz"], float)
        pos = np.asarray(d[f"{key}__pos_1khz"], float)
        gen.append({"kind": "gen", "unit": key, **path_quantities(t, pos)})
    gen = pd.DataFrame(gen)

    real.to_csv(OUT / "real_flights.csv", index=False)
    pd.concat([sim, gen], ignore_index=True).to_csv(OUT / "paths.csv", index=False)

    # ---------------- data envelope, leave one flight out
    quantities = {"tilt": ("q995_tilt_deg", "peak_tilt_deg"),
                  "twr": ("q995_twr", "peak_twr"),
                  "rate": ("q995_rate_deg_s", "peak_rate_deg_s"),
                  "jerk": ("q995_jerk_ms3", "peak_jerk_ms3"),
                  "acc": ("q995_acc_ms2", "peak_acc_ms2")}
    env, loo = {}, {}
    for q, (col, _) in quantities.items():
        vals = real[col].to_numpy(float)
        env[q] = float(np.nanmax(vals))
        loo[q] = {}
        for i, row in real.iterrows():
            others = np.delete(vals, i)
            thr = float(np.nanmax(others))
            loo[q][row.unit] = {"threshold": thr, "value": float(vals[i]),
                                "rejected": bool(vals[i] > thr)}
    summary = {"prereg": "configs/p2_envelope_prereg.yaml", "physics": PHYS,
               "filter": filter_gain(), "real_envelope_max": env,
               "real_loo": {q: {"rejected": f"{sum(v['rejected'] for v in loo[q].values())}"
                                f"/{len(loo[q])}",
                                "failing": [k for k, v in loo[q].items() if v["rejected"]],
                                "max_threshold": max(v["threshold"] for v in loo[q].values())}
                            for q in loo},
               "real_per_flight": {r.unit: {q: float(r[quantities[q][0]]) for q in quantities}
                                   for _, r in real.iterrows()}}

    # physics gate on real flights
    phys_rej = {}
    for _, r in real.iterrows():
        phys_rej[r.unit] = {"tilt": bool(r.peak_tilt_deg > PHYS["tilt"]),
                            "twr": bool(r.peak_twr > PHYS["twr"]),
                            "rate": bool(r.peak_rate_deg_s > PHYS["rate"])}
    summary["physics_gate_real"] = {
        "rejected": f"{sum(any(v.values()) for v in phys_rej.values())}/{len(phys_rej)}",
        "by_flight": phys_rej,
        "tilt_rejected": f"{sum(v['tilt'] for v in phys_rej.values())}/{len(phys_rej)}",
        "twr_rejected": f"{sum(v['twr'] for v in phys_rej.values())}/{len(phys_rej)}",
        "rate_rejected": f"{sum(v['rate'] for v in phys_rej.values())}/{len(phys_rej)}",
    }
    # conjunction of physics and data envelopes on real flights, leave one out
    conj = {}
    for _, r in real.iterrows():
        d_thr = {q: max(v["threshold"] for k, v in loo[q].items() if k != r.unit) for q in loo}
        d_ok = all(r[quantities[q][0]] <= d_thr[q] for q in ("tilt", "twr", "rate", "jerk"))
        p_ok = not any(phys_rej[r.unit].values())
        conj[r.unit] = bool(d_ok and p_ok)
    summary["conjunction_real"] = {"pass": f"{sum(conj.values())}/{len(conj)}",
                                   "failing": [k for k, v in conj.items() if not v]}

    # ---------------- sim paths and generated paths
    def gate(df, phys=True):
        return (df.peak_tilt_deg <= PHYS["tilt"]) & (df.peak_twr <= PHYS["twr"]) \
            & (df.peak_rate_deg_s <= PHYS["rate"]) if phys else pd.Series(True, index=df.index)

    feas = sim[sim.label == "feasible"]
    thrust = sim[sim.defect_family == "thrust_limit"]
    det = int((~gate(thrust)).sum())
    summary["path_defects"] = {
        "n": int(len(thrust)), "detected_by_physics_gate": f"{det}/{len(thrust)}",
        "failing_clips": thrust[~gate(thrust)].unit.tolist(),
        "feasible_flagged": f"{int((~gate(feas)).sum())}/{len(feas)}",
        "peak_tilt_range_feasible": [float(feas.peak_tilt_deg.min()),
                                     float(feas.peak_tilt_deg.max())],
        "peak_twr_range_feasible": [float(feas.peak_twr.min()), float(feas.peak_twr.max())],
        "manifest_vs_recomputed_peak_tilt_max_abs_diff": float(
            np.max(np.abs(feas.peak_tilt_deg - feas.peak_tilt_req_manifest))),
    }
    for name, df in (("sim_dev", sim), ("sim_feasible", feas), ("gen_h04", gen)):
        row = {}
        for q in ("tilt", "twr", "rate", "jerk"):
            thr = env[q]
            row[q] = f"{int((df[quantities[q][1]] > thr).sum())}/{len(df)}"
        row["physics_tilt"] = f"{int((df.peak_tilt_deg > PHYS['tilt']).sum())}/{len(df)}"
        row["physics_twr"] = f"{int((df.peak_twr > PHYS['twr']).sum())}/{len(df)}"
        summary[f"outside_envelope_{name}"] = row
    # in/out split for the sim set: do the tilt verdicts differ?
    outside = (sim.peak_tilt_deg > PHYS["tilt"]) | (sim.peak_twr > PHYS["twr"]) \
        | (sim.peak_rate_deg_s > PHYS["rate"])
    summary["sim_in_out"] = {
        "outside": f"{int(outside.sum())}/{len(sim)}",
        "outside_by_family": {k: f"{int(g.sum())}/{len(g)}"
                              for k, g in outside.groupby(sim.defect_family)},
        "peak_tilt_median_outside": float(sim[outside].peak_tilt_deg.median())
        if outside.any() else None,
        "peak_tilt_median_inside": float(sim[~outside].peak_tilt_deg.median()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: summary[k] for k in
                      ("real_envelope_max", "real_loo", "physics_gate_real", "conjunction_real",
                       "path_defects", "outside_envelope_sim_dev", "outside_envelope_sim_feasible",
                       "outside_envelope_gen_h04", "sim_in_out")}, indent=1, default=float))


if __name__ == "__main__":
    main()
