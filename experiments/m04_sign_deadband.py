"""Track M follow-up: the sign sub-tier with a reader-noise-tied dead-band.

Pre-registered in ``configs/m04_sign_deadband_prereg.yaml`` (committed before this script was
run).  CPU only, zero GPU spend.

Two statistics, computed on the same frames and windowed identically:

* **frozen** -- the published rule (``src.verdict.coherence._sign_agreement``, dead-band 1e-4);
* **dead-band** -- the same rule with the dead-band at ``sin(sigma_n)`` (and 0.5x / 2x as a
  sensitivity), so a lean component smaller than the reader's own noise contributes no sign
  evidence.

Populations: the real development clips (fitted threshold, all edit families), the real
held-out flights (descriptive), the simulated development clips (transferred threshold) and
the H04 arm-A clips (post-hoc illustration only).

    PYTHONPATH=. pixi run python experiments/m04_sign_deadband.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.airframe.kinematics import TII_FLIGHTS, load_tii_kinematics
from src.airframe.schema_v3 import required_lean_at
from src.verdict.gate_b import gravity_gc, lean_from_gravity
from src.verdict.selective import pooled_rate_interval

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "data/processed/track_d/clips"
OUT = ROOT / "results/m04"
DEV = ["flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse",
       "flight-07p-lemniscate", "flight-13a-trackRATM", "flight-16p-trackRATM"]
HELD = ["flight-13p-trackRATM", "flight-07a-lemniscate"]
CONDS = ("real", "tilt_removal", "mirror_roll", "scale_0.5", "scale_1.5",
         "yaw_control", "null_edit")
WIN, HOP = 8, 4
SIGMA_N = 1.910587818373582
VARIANTS = {"frozen": 1e-4, "db0.5": float(np.sin(np.deg2rad(0.5 * SIGMA_N))),
            "db1.0": float(np.sin(np.deg2rad(1.0 * SIGMA_N))),
            "db2.0": float(np.sin(np.deg2rad(2.0 * SIGMA_N)))}
PRIMARY = "db1.0"
MIN_USABLE, MIN_SIGN = 6, 2


def windows(n: int) -> list[tuple[int, int]]:
    if n <= WIN:
        return [(0, n)]
    return [(s, s + WIN) for s in range(0, n - WIN + 1, HOP)]


def sign_agreement(a: np.ndarray, b: np.ndarray, eps: float) -> tuple[float, int]:
    keep = (np.abs(a) > eps) & (np.abs(b) > eps)
    if keep.sum() == 0:
        return 0.5, 0
    return float(np.mean(np.sign(a[keep]) == np.sign(b[keep]))), int(keep.sum())


def sign_score(dep: np.ndarray, req: np.ndarray, eps: float) -> float:
    """Max over the published windows of 1 - min(s_x, s_y)."""
    vals = []
    for lo, hi in windows(len(dep)):
        d, q = dep[lo:hi], req[lo:hi]
        ok = np.isfinite(d).all(1) & np.isfinite(q).all(1)
        d, q = d[ok], q[ok]
        if len(d) < MIN_USABLE:
            continue
        ss, ns = [], []
        for j in range(2):
            s, n = sign_agreement(d[:, j], q[:, j], eps)
            ss.append(s)
            ns.append(n)
        if min(ns) < MIN_SIGN:
            continue
        vals.append(1.0 - min(ss))
    return float(max(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- real clips
def real_table() -> pd.DataFrame:
    kin = {f: load_tii_kinematics(f) for f in TII_FLIGHTS}
    keep = pd.read_csv(ROOT / "results/m01/signal_by_clip.csv")[
        ["clip_id", "kept_S3"]]
    keepmap = dict(zip(keep.clip_id, keep.kept_S3.astype(bool)))
    rows = []
    for f in TII_FLIGHTS:
        cache = {}
        for d in sorted((CLIPS / f).iterdir()):
            cid, cond = d.name.split("__")
            if cond not in CONDS:
                continue
            z = np.load(d / "verdict_clip.npz")
            if cid not in cache:
                cache[cid] = required_lean_at(kin[f], "M0", {}, np.asarray(z["t"], float),
                                              np.asarray(z["yaw"], float))["lean"]
            dep = np.asarray(z["depicted_lean"], float)
            rec = {"clip_id": cid, "flight": f, "cond": cond,
                   "in_dev": f in DEV, "kept_S3": keepmap.get(cid, False)}
            for name, eps in VARIANTS.items():
                rec[name] = sign_score(dep, cache[cid], eps)
            rows.append(rec)
    return pd.DataFrame(rows)


def _tau(vals, flights, heldout=None):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) < 5:
        return float("nan")
    return float(np.quantile(v, 0.95))


def evaluate_real(df: pd.DataFrame) -> dict:
    dev = df[df.in_dev & df.kept_S3]
    real = dev[dev.cond == "real"]
    out = {}
    for name in VARIANTS:
        hit = tot = 0
        per = {}
        for f in DEV:
            r = real[real.flight == f][name].values
            fit = real[real.flight != f][name].values
            fit = fit[np.isfinite(fit)]
            tau = float(np.quantile(fit, 0.95)) if len(fit) >= 5 else float("nan")
            ok = np.isfinite(r)
            h = int(np.sum(r[ok] > tau)) if np.isfinite(tau) else 0
            per[f] = (h, int(ok.sum()))
            hit += h
            tot += int(ok.sum())
        ent = {"false_reject": f"{hit}/{tot}"}
        for cond in CONDS[1:]:
            h = t = 0
            for _, row in dev[dev.cond == cond].iterrows():
                fit = real[real.flight != row.flight][name].values
                fit = fit[np.isfinite(fit)]
                if len(fit) < 5:
                    continue
                t += 1
                h += int(np.isfinite(row[name]) and row[name] > np.quantile(fit, 0.95))
            ent[cond] = f"{h}/{t}"
        ent["per_flight_false_reject"] = {k: list(v) for k, v in per.items()}
        out[name] = ent
    return out


# --------------------------------------------------------------------------- simulated
def sim_table() -> pd.DataFrame:
    frames = []
    for p in sorted((ROOT / "data/m02_scores/v2").glob("sim_shard*.jsonl")):
        frames += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    d = pd.DataFrame(frames)
    man = {r["clip_id"]: r for r in
           pd.read_csv(ROOT / "results/m02/i02_manifest_copy.csv").to_dict("records")}
    rows = []
    for cid, g in d.groupby("clip_id"):
        g = g.sort_values("frame_index")
        gt = np.load(ROOT / "data/m02_inputs/sim/gt" / f"{cid}.npz")
        R_cb = np.asarray(gt["R_cb"], float)
        dep = lean_from_gravity(np.vstack([gravity_gc(r) for _, r in g.iterrows()]), R_cb)
        t = np.asarray(g.frame_index.values, float) / 30.0
        tg = np.asarray(gt["t"], float)
        req = np.column_stack([np.interp(t, tg, np.asarray(gt["lean_req"])[:, j])
                               for j in range(2)])
        rec = {"clip_id": cid, "family": man[cid]["family"],
               "defect": man[cid]["defect_family"]}
        for name, eps in VARIANTS.items():
            rec[name] = sign_score(dep, req, eps)
        rows.append(rec)
    return pd.DataFrame(rows)


def evaluate_sim(df: pd.DataFrame, taus: dict) -> dict:
    out = {}
    for name in VARIANTS:
        tau = taus[name]
        feas = df[df.defect == "feasible"]
        det = df[df.defect != "feasible"]
        out[name] = {"tau_transferred": tau,
                     "false_reject": f"{int(np.sum(feas[name] > tau))}/{len(feas)}"
                     if np.isfinite(tau) else "n/a",
                     "detected": f"{int(np.sum(det[name] > tau))}/{len(det)}"
                     if np.isfinite(tau) else "n/a"}
    return out


# --------------------------------------------------------------------------- H04
def h04_table() -> pd.DataFrame:
    paths = np.load(ROOT / "results/m03/inputs/h04_paths.npz")
    rows = []
    frames = []
    for p in sorted((ROOT / "data/m02_scores/v2").glob("h04_shard*.jsonl")):
        frames += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    d = pd.DataFrame(frames)
    for cid, g in d.groupby("clip_id"):
        path = cid[5:] if cid.startswith("main_") else cid
        path = re.sub(r"_g\d+$", "", path)
        for suffix in ("_A_flatness", "_B_level", "_C_mirrored_roll"):
            path = path.replace(suffix, "")
        key = f"{path}__lean_req_rule"
        if key not in paths.files:
            continue
        g = g.sort_values("frame_index")
        dep = lean_from_gravity(np.vstack([gravity_gc(r) for _, r in g.iterrows()]),
                                np.eye(3))
        t = np.asarray(g.frame_index.values, float) / 30.0
        tt = np.asarray(paths[f"{path}__t"], float)
        lean = np.asarray(paths[key], float)
        req = np.column_stack([np.interp(t, tt, lean[:, j]) for j in range(2)])
        arm = "A_flatness" if "_A_" in cid else ("B_level" if "_B_" in cid
                                                 else "C_mirrored")
        rec = {"clip_id": cid, "arm": arm, "path": path}
        for name, eps in VARIANTS.items():
            rec[name] = sign_score(dep, req, eps)
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    real = real_table()
    real.to_csv(OUT / "real_sign.csv", index=False)
    res = {"label": "new pre-registered variant: configs/m04_sign_deadband_prereg.yaml",
           "sigma_n_deg": SIGMA_N, "dead_bands": VARIANTS, "primary": PRIMARY}
    res["real_dev"] = evaluate_real(real)

    sim = sim_table()
    sim.to_csv(OUT / "sim_sign.csv", index=False)
    taus = {}
    for name in VARIANTS:
        ent = res["real_dev"][name]["false_reject"].split("/")
        r = real[real.in_dev & real.kept_S3 & (real.cond == "real")][name].values
        # the median of the leave-one-flight-out thresholds, as M01 quotes it
        taus[name] = float(np.median([
            np.quantile(real[real.in_dev & real.kept_S3 & (real.cond == "real")
                             & (real.flight != f)][name].dropna().values, 0.95)
            for f in DEV
            if len(real[real.in_dev & real.kept_S3 & (real.cond == "real")
                        & (real.flight != f)][name].dropna()) >= 5]))
    res["lofo_tau_median"] = taus
    res["sim_dev"] = evaluate_sim(sim, taus)

    h4 = h04_table()
    h4.to_csv(OUT / "h04_sign.csv", index=False)
    res["h04_arm_a_posthoc"] = {
        arm: {name: {"median": float(g[name].median()),
                     "n": int(g[name].notna().sum())}
              for name in VARIANTS}
        for arm, g in h4.groupby("arm")}
    res["sim_by_family"] = {
        fam: {name: float(g[name].median()) for name in VARIANTS}
        for fam, g in sim[sim.defect == "feasible"].groupby("family")}
    (OUT / "sign_deadband.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
