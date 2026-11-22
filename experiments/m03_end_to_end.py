"""Track M, Stage 3: end-to-end abstention across the four gate configurations and three tiers.

Pre-registered in ``configs/m03_endtoend_prereg.yaml``; thresholds come from
``configs/m01_gate_a_frozen.yaml`` and ``configs/m02_gate_b_frozen.yaml``.  CPU only.

Configurations: G0 (S3 only), GA (S3 + A), GB (S3 + B), GAB (both).
Tiers: R (scale-free coherence, window max), F0 (fixed calibration, clip-median axis angle),
C2 (per-clip Huber offset + drag, clip-median axis angle).

Populations: real development clips, real held-out (13p / 07a, descriptive), the simulated
development clips, and the published H04 / H01 generator verdicts.

    PYTHONPATH=. pixi run python experiments/m03_end_to_end.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from src.verdict.abstain import gate_kept_from_conf
from src.verdict.lean import depicted_lean_from_gravity
from src.verdict.selective import (cluster_interval, conformal_quantile,
                                   pooled_rate_interval, rate_interval)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/m03"
CLIPS = ROOT / "data/processed/track_d/clips"
P3A = ROOT / "data/processed/p3a/p3a_v1"
GATE_TABLE = ROOT / "results/a2_gate/20261109T201449Z_298ff20/clip_table.csv"
GATE_CONFIG = ROOT / "configs/gate_frozen.yaml"
MANIFEST = ROOT / "results/m02/i02_manifest_copy.csv"

DEV = ["flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse",
       "flight-07p-lemniscate", "flight-13a-trackRATM", "flight-16p-trackRATM"]
HELD = ["flight-13p-trackRATM", "flight-07a-lemniscate"]
FAMILIES = ("tilt_removal", "mirror_roll", "scale_0.5", "scale_1.5")
CONTROLS = ("yaw_control", "null_edit")
CONDS = ("real",) + FAMILIES + CONTROLS
TIERS = ("R", "F0", "C2")
CONFIGS = ("G0_S3_only", "GA", "GB", "GAB")
WIN, HOP = 8, 4
VAR_FLOOR = 0.01
HUBER_DEG = 3.0
HUBER_DELTA = float(np.sin(np.deg2rad(HUBER_DEG)))
KD_GRID = np.unique(np.round(np.concatenate([np.arange(0.0, 0.401, 0.05),
                                             np.array([0.295])]), 3))
TAU_ALPHA = 0.05


# --------------------------------------------------------------------------- frozen thresholds
def frozen() -> dict:
    import yaml
    a = yaml.safe_load((ROOT / "configs/m01_gate_a_frozen.yaml").read_text())
    b = yaml.safe_load((ROOT / "configs/m02_gate_b_frozen.yaml").read_text())
    return {"lam_a": float(a["gate_a"]["lam_deg"]),
            "near_vertical": float(a["gate_a"]["near_vertical_deg"]),
            "tau_b": float(b["gate_b"]["tau_deg"]),
            "f0": float(a["coherence_tier"]["tau_lofo_median_accepted_dev"]),
            "sigma_n": float(a["reader_noise"]["sigma_n_deg"])}


def s3_gate() -> dict:
    g = json.loads("\n".join(l for l in GATE_CONFIG.read_text().splitlines()
                             if not l.strip().startswith("#")))
    tau = float(g["tau"])
    return {r["clip_id"]: float(r["identity::S3_latitude_confidence"]) <= tau
            for r in csv.DictReader(open(GATE_TABLE))}


def s3_per_condition() -> dict:
    """A2's own convention: the frozen S3 signal applied to each condition's own confidences."""
    g = json.loads("\n".join(l for l in GATE_CONFIG.read_text().splitlines()
                             if not l.strip().startswith("#")))
    out = {}
    for d in sorted(CLIPS.glob("*/*")):
        cid, cond = d.name.split("__")
        z = np.load(d / "verdict_clip.npz")
        out[(cid, cond)] = bool(gate_kept_from_conf(np.asarray(z["depicted_conf"], float), g))
    return out


# --------------------------------------------------------------------------- tiers for TII
def coherence_series(dep: np.ndarray, req: np.ndarray) -> float:
    vals = []
    n = len(dep)
    spans = [(0, n)] if n <= WIN else [(s, s + WIN) for s in range(0, n - WIN + 1, HOP)]
    for lo, hi in spans:
        d, q = dep[lo:hi], req[lo:hi]
        ok = np.isfinite(d).all(1) & np.isfinite(q).all(1)
        d, q = d[ok], q[ok]
        if len(d) < 6:
            continue
        rs = []
        for j in range(2):
            if np.std(d[:, j]) < VAR_FLOOR or np.std(q[:, j]) < VAR_FLOOR:
                rs = []
                break
            rs.append(float(np.corrcoef(d[:, j], q[:, j])[0, 1]))
        if rs:
            vals.append(1.0 - min(rs))
    return float(max(vals)) if vals else float("nan")


def axis_angle_deg(dep_lean, req_lean, yaw) -> np.ndarray:
    from src.depicted.clips import up_body_from_lean
    from src.depicted.metrics import angle_deg
    from src.depicted.stat_a2 import depicted_axis_from_up_body
    a = depicted_axis_from_up_body(up_body_from_lean(np.asarray(dep_lean, float)),
                                   np.asarray(yaw, float))
    b = depicted_axis_from_up_body(up_body_from_lean(np.asarray(req_lean, float)),
                                   np.asarray(yaw, float))
    return angle_deg(a, b)


def huber_offset(dep, req):
    r = dep - req
    ok = np.isfinite(r).all(1)
    if ok.sum() < 4:
        return np.zeros(2)
    r = r[ok]
    o = np.median(r, axis=0)
    for _ in range(6):
        e = np.linalg.norm(r - o, axis=1)
        w = np.where(e <= HUBER_DELTA, 1.0, HUBER_DELTA / np.maximum(e, 1e-12))
        o = np.sum(w[:, None] * r, axis=0) / np.sum(w)
    return o


def huber_loss(dep, req):
    o = huber_offset(dep, req)
    e = np.linalg.norm(dep - req - o, axis=1)
    e = e[np.isfinite(e)]
    h = np.where(e <= HUBER_DELTA, 0.5 * e ** 2, HUBER_DELTA * (e - 0.5 * HUBER_DELTA))
    return float(np.mean(h)) if len(h) else np.inf


def tiers_tii(dep, lean_by_kd, yaw) -> dict:
    """R, F0 and C2 statistics for one depicted side of one clip."""
    f0 = float(np.nanmedian(axis_angle_deg(dep, lean_by_kd[0.295], yaw)))
    k2 = min(KD_GRID, key=lambda k: huber_loss(dep, lean_by_kd[float(k)]))
    off = huber_offset(dep, lean_by_kd[float(k2)])
    c2 = float(np.nanmedian(axis_angle_deg(dep, lean_by_kd[float(k2)] + off, yaw)))
    r = coherence_series(dep, lean_by_kd[0.295])
    return {"R": r, "F0": f0, "C2": c2, "k_d_C2": float(k2)}


# --------------------------------------------------------------------------- real table
def build_real(fr: dict) -> pd.DataFrame:
    from src.airframe.kinematics import TII_FLIGHTS, load_tii_kinematics
    from src.airframe.models import unit
    from src.airframe.schema_v3 import sample_axis
    from src.physics.flatness import G0

    sig = pd.read_csv(ROOT / "results/m01/signal_by_clip.csv")
    sigmap = {r.clip_id: r for r in sig.itertuples()}
    dpath = ROOT / "results/m02/real_by_clip.csv"
    if dpath.exists():
        dis = pd.read_csv(dpath)
        dismap = {(r.clip_id, r.cond): r for r in dis.itertuples()}
    else:
        dismap = {}
    s3 = s3_per_condition()
    kin = {f: load_tii_kinematics(f) for f in TII_FLIGHTS}
    rows = []
    for f in TII_FLIGHTS:
        k = kin[f]
        native = {kd: unit(k.a + np.array([0.0, 0.0, G0]) + kd * k.v) for kd in
                  (0.0, 0.295, 0.35, 0.40)}
        for d in sorted((CLIPS / f).iterdir()):
            cid, cond = d.name.split("__")
            if cond not in CONDS:
                continue
            z = np.load(d / "verdict_clip.npz")
            yaw = np.asarray(z["yaw"], float)
            t = np.asarray(z["t"], float)
            lean_by_kd = {kd: sample_axis(k, native[kd], t, yaw)[1]
                          for kd in native}
            # a k_d grid fit needs the full grid
            for kd in KD_GRID:
                if float(kd) in lean_by_kd:
                    continue
                lean_by_kd[float(kd)] = sample_axis(
                    k, unit(k.a + np.array([0.0, 0.0, G0]) + kd * k.v), t, yaw)[1]
            st = tiers_tii(np.asarray(z["depicted_lean"], float), lean_by_kd, yaw)
            s = sigmap.get(cid)
            dm = dismap.get((cid, cond))
            rows.append({"clip_id": cid, "flight": f, "cond": cond,
                         "in_dev": f in DEV, "kept_S3_clip": bool(fr["s3"].get(cid, True)),
                         "kept_S3_cond": bool(s3.get((cid, cond), True)),
                         "sigma_min": float(s.sigma_min) if s is not None else np.nan,
                         "tilt_max": float(s.tilt_max) if s is not None else np.nan,
                         "dis_med": float(dm.dis_med) if dm is not None else np.nan,
                         **st})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- sim table
def build_sim(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-clip tiers on the simulated set, with the GC224 reader as the depicted side."""
    man = {r["clip_id"]: r for r in csv.DictReader(open(MANIFEST))}
    out = []
    for cid, g in rows[rows.source == "sim"].groupby("clip_id"):
        g = g.sort_values("frame_index")
        gt = np.load(ROOT / "data/m02_inputs/sim/gt" / f"{cid}.npz")
        R_cb = np.asarray(gt["R_cb"], float)
        dep = depicted_lean_from_gravity(np.vstack(g.gc_gravity.values), R_cb)
        t = np.asarray(g.frame_index.values, float) / 30.0
        tg = np.asarray(gt["t"], float)
        yaw_a = np.interp(t, tg, np.asarray(gt["yaw"], float))
        acc = np.column_stack([np.interp(t, tg, np.asarray(gt["acc"])[:, j])
                               for j in range(3)])
        vel = np.column_stack([np.interp(t, tg, np.asarray(gt["vel"])[:, j])
                               for j in range(3)])
        truth = np.column_stack([np.interp(t, tg, np.asarray(gt["lean_dep"])[:, j])
                                 for j in range(2)])
        lean_by_kd = {}
        for kd in KD_GRID:
            ax = acc + np.array([0.0, 0.0, 9.81]) + float(kd) * vel
            ax = ax / np.linalg.norm(ax, axis=1, keepdims=True)
            lean_by_kd[float(kd)] = world_to_yaw_frame_np(ax[:, :2], yaw_a)
        req = lean_by_kd[0.295]
        st = tiers_tii(dep, lean_by_kd, yaw_a)
        out.append({"clip_id": cid, "family": man[cid]["family"],
                    "defect": man[cid]["defect_family"],
                    "mount": float(man[cid]["uptilt_deg"]), "n": int(len(dep)),
                    "sigma_min": float(np.degrees(req.std(0).min())),
                    "tilt_max": float(np.degrees(np.nanmax(np.arcsin(
                        np.clip(np.linalg.norm(req, axis=1), 0, 1))))),
                    "err_reader": float(np.degrees(np.median(
                        np.linalg.norm(dep - truth, axis=1)))),
                    **st})
    return pd.DataFrame(out)


def load_score_rows(tag: str) -> pd.DataFrame:
    """One or more comma-separated score tags."""
    frames = []
    for t in [x.strip() for x in tag.split(",") if x.strip()]:
        for p in sorted((ROOT / "data/m02_scores" / t).glob("*.jsonl")):
            frames += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if not frames:
        raise SystemExit("no score rows; pull them from the volume first")
    df = pd.DataFrame(frames)
    if "cond" not in df:
        df["cond"] = "real"
    return df


def world_to_yaw_frame_np(vec, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.column_stack([c * vec[:, 0] + s * vec[:, 1], -s * vec[:, 0] + c * vec[:, 1]])


# --------------------------------------------------------------------------- evaluation
def gate_masks(real: pd.DataFrame, fr: dict) -> dict:
    out = {}
    weak = (real.sigma_min >= fr["lam_a"]) & (real.tilt_max <= fr["near_vertical"])
    # Gate B, like S3, decides ONCE per clip, on the clip's real video.  Applying it to each
    # condition's own pixels is A2's per-condition convention and is reported separately: it
    # drops 68 of the 69 development tilt-removal clips, because removing the tilt changes
    # what the readers say about the same scene.
    real_dis = real[real.cond == "real"].set_index("clip_id").dis_med
    dis = real.clip_id.map(real_dis).le(fr["tau_b"])
    dis = dis.fillna(False)
    # The S3 decision is taken ONCE per clip, on the clip's real video: in operation the gate
    # sees one video, and the edited conditions are counterfactual probes.  A2's published
    # convention gates each condition on its own pixels; that is kept as a separate row
    # (G0_S3_per_condition) because it changes the denominators by condition.
    s3 = real.kept_S3_clip.fillna(False)
    out["G0_S3_only"] = s3
    out["GA"] = s3 & weak
    out["GB"] = s3 & dis
    out["GAB"] = s3 & weak & dis
    out["G0_S3_per_condition"] = real.kept_S3_cond.fillna(False)
    out["GB_per_condition"] = s3 & (real.dis_med <= fr["tau_b"]).fillna(False)
    return out


def _tau_pooled(acc: pd.DataFrame, tier: str) -> float:
    """Conformal 95th percentile over every accepted real development clip (pooled cut).

    Used as the primary comparison because for GB/GAB the accepted set can hold fewer than
    five real clips on a flight, which makes a leave-one-flight-out fold undefined.  The LOFO
    variant is reported beside it wherever it is computable.
    """
    vals = acc[(acc.cond == "real") & acc.in_dev][tier].values
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return float("nan")
    return float(np.quantile(vals, 1.0 - TAU_ALPHA))


def _tau_lofo(acc: pd.DataFrame, tier: str, flight: str) -> float:
    """Conformal threshold from the accepted real clips of the other development flights."""
    fit = acc[(acc.cond == "real") & (acc.flight != flight) & acc.in_dev]
    vals = fit[tier].values
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return float("nan")
    return float(np.quantile(vals, 1.0 - TAU_ALPHA))


def evaluate(real: pd.DataFrame, masks: dict, tier: str, dev: bool = True) -> dict:
    """Rates for one tier under each gate configuration, thresholds leave-one-flight-out."""
    base = real[(real.in_dev == dev)]
    out = {}
    for cfg, mask in masks.items():
        sub = base.copy()
        sub["acc"] = mask.reindex(base.index).fillna(False)
        acc = sub[sub.acc]
        ent = {"accepted_clips": int(acc.clip_id.nunique()),
               "total_clips": int(base.clip_id.nunique()),
               "coverage": acc.clip_id.nunique() / max(base.clip_id.nunique(), 1)}
        tau_pool = _tau_pooled(acc if dev else real[real.in_dev], tier)
        taus = {f: tau_pool for f in (DEV if dev else HELD)}
        ent["tau_pooled"] = tau_pool
        ent["tau_lofo_median"] = float(np.nanmedian(
            [_tau_lofo(acc, tier, f) for f in (DEV if dev else HELD)]))
        for cond in CONDS:
            hit = tot = 0
            for _, r in acc[acc.cond == cond].iterrows():
                tau = taus.get(r.flight, np.nan)
                if not np.isfinite(tau):
                    continue
                tot += 1
                hit += int(np.isfinite(r[tier]) and r[tier] > tau)
            ent[cond] = f"{hit}/{tot}"
        out[cfg] = ent
    return out


def reason_codes(real: pd.DataFrame, fr: dict) -> pd.DataFrame:
    """Per-clip gate decisions and reason codes, one row per clip and condition."""
    out = []
    for _, r in real.iterrows():
        reasons = []
        if not bool(r.kept_S3_clip):
            reasons.append("low_confidence_S3")
        if not (np.isfinite(r.sigma_min) and r.sigma_min >= fr["lam_a"]):
            reasons.append("weak_signal")
        if not (np.isfinite(r.tilt_max) and r.tilt_max <= fr["near_vertical"]):
            reasons.append("near_vertical")
        if not (np.isfinite(r.dis_med) and r.dis_med <= fr["tau_b"]):
            reasons.append("readers_disagree")
        out.append({"clip_id": r.clip_id, "flight": r.flight, "cond": r.cond,
                    "in_dev": bool(r.in_dev),
                    "sigma_min": r.sigma_min, "tilt_max": r.tilt_max,
                    "dis_med": r.dis_med, "R": r.R, "F0": r.F0, "C2": r.C2,
                    "k_d_C2": r.k_d_C2,
                    "reason_codes": ";".join(reasons) if reasons else "none"})
    return pd.DataFrame(out)


def clustered_rates(real: pd.DataFrame, masks: dict, tier: str) -> dict:
    """Per-condition hit rates with flight-clustered (descriptive) intervals."""
    base = real[real.in_dev]
    out = {}
    for cfg, mask in masks.items():
        sub = base.copy()
        sub["acc"] = mask.reindex(base.index).fillna(False)
        acc = sub[sub.acc]
        tau_pool = _tau_pooled(acc, tier)
        taus = {f: tau_pool for f in DEV}
        ent = {"tau_pooled": tau_pool}
        ent["tau_lofo_median"] = float(np.nanmedian(
            [_tau_lofo(acc, tier, f) for f in DEV]))
        for cond in CONDS:
            hits, dens, flags = [], [], []
            for f in DEV:
                rows = acc[(acc.cond == cond) & (acc.flight == f)]
                tau = taus[f]
                if not np.isfinite(tau):
                    continue
                h = int(np.sum(rows[tier].values > tau))
                hits.append(h)
                dens.append(int(len(rows)))
                flags.append(h / max(len(rows), 1) if len(rows) else np.nan)
            ok = np.isfinite(flags)
            if ok.sum() >= 2:
                ci = pooled_rate_interval(np.array(hits)[ok], np.array(dens)[ok],
                                          np.array(DEV)[ok])
            else:
                ci = {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                      "n_clusters": int(ok.sum()), "note": "no computable interval"}
            ent[cond] = {"hit": int(np.sum(hits)), "n": int(np.sum(dens)), **ci}
        out[cfg] = ent
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fr = frozen()
    fr["s3"] = s3_gate()
    print("frozen:", {k: v for k, v in fr.items() if k != "s3"})

    real = build_real(fr)
    real.to_csv(OUT / "real_tiers.csv", index=False)
    masks = gate_masks(real, fr)
    for cfg, m in masks.items():
        real[cfg] = m
    real.to_csv(OUT / "real_tiers.csv", index=False)

    rc = reason_codes(real, fr)
    rc.to_csv(OUT / "reason_codes_real.csv", index=False)
    print("reason-code composition (clip level, real condition):",
          rc[(rc.cond == "real")].reason_codes.value_counts().to_dict())

    res = {}
    for tier in TIERS:
        res[f"real_dev_{tier}"] = evaluate(real, masks, tier, dev=True)
        res[f"real_heldout_{tier}"] = evaluate(real, masks, tier, dev=False)
    (OUT / "real_by_config.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print("\n==", k)
        print(pd.DataFrame(v).T.to_string())

    # ---- sim
    sc = pd.read_csv(ROOT / "results/m02/sim_by_clip.csv")  # for disagreement
    dismap = dict(zip(sc.clip_id, sc.dis_med))
    sim = build_sim(load_score_rows(args.tag))
    sim.to_csv(OUT / "sim_tiers.csv", index=False)
    sim["dis_med"] = sim.clip_id.map(dismap)
    sim_masks = {
        "GA": (sim.sigma_min >= fr["lam_a"]) & (sim.tilt_max <= fr["near_vertical"]),
        "GB": sim.dis_med <= fr["tau_b"],
        "GAB": (sim.sigma_min >= fr["lam_a"]) & (sim.tilt_max <= fr["near_vertical"])
        & (sim.dis_med <= fr["tau_b"]),
    }
    sim_masks["G0_S3_only"] = pd.Series(True, index=sim.index)
    rc = []
    for _, r in sim.iterrows():
        reasons = []
        if not (np.isfinite(r.sigma_min) and r.sigma_min >= fr["lam_a"]):
            reasons.append("weak_signal")
        if not (np.isfinite(r.tilt_max) and r.tilt_max <= fr["near_vertical"]):
            reasons.append("near_vertical")
        if not (np.isfinite(r.dis_med) and r.dis_med <= fr["tau_b"]):
            reasons.append("readers_disagree")
        rc.append({**{k: r[k] for k in ("clip_id", "family", "defect", "mount",
                                        "sigma_min", "tilt_max", "dis_med", "R",
                                        "F0", "C2", "err_reader")},
                   "reason_codes": ";".join(reasons) if reasons else "none"})
    pd.DataFrame(rc).to_csv(OUT / "reason_codes_sim.csv", index=False)
    simres = {}
    for tier in TIERS:
        ent = {}
        thr = res[f"real_dev_{tier}"]["G0_S3_only"]["tau_pooled"]
        for cfg, m in sim_masks.items():
            acc = sim[m.fillna(False)]
            feas = acc[acc.defect == "feasible"]
            det = acc[acc.defect != "feasible"]
            ent[cfg] = {"accepted": int(len(acc)), "total": int(len(sim)),
                        "coverage": len(acc) / max(len(sim), 1),
                        "false_reject": f"{int(np.sum(feas[tier] > thr))}/{len(feas)}"
                        if np.isfinite(thr) else "n/a",
                        "detected": f"{int(np.sum(det[tier] > thr))}/{len(det)}"
                        if np.isfinite(thr) else "n/a",
                        "tau": thr}
        simres[tier] = ent
    (OUT / "sim_by_config.json").write_text(json.dumps(simres, indent=1))
    print("\n== sim")
    print(json.dumps(simres, indent=1))


if __name__ == "__main__":
    main()
