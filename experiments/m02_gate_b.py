"""Track M, Stage 2: Gate B (reader agreement) -- development analysis.

Pre-registered in ``configs/m02_gate_b_prereg.yaml``.  Reads the reader score JSONLs produced
by ``remote/m_readers.py`` (pulled to ``data/m02_scores/<tag>/``) plus the stored frozen
reader and truth.  CPU only.

Reports, in this order:
1. the sign-convention probe (a pure image rotation is a camera roll);
2. reader-versus-truth error per source, per reader, and for the ensemble;
3. whether disagreement predicts reader error -- AUROC, by family and mount, against the
   frozen S3 gate and against S3 combined with disagreement;
4. Gate B as selective prediction: risk-coverage and conformal risk control on the real dev
   flights, both risks (coherence false-reject and yaw-control flag rate);
5. the forest check -- the 12 forest clips S3 keeps;
6. the ensemble arm -- trimmed RMSE and per-family bias, and the coherence tier at a matched
   false-reject.

    PYTHONPATH=. pixi run python experiments/m02_gate_b.py --tag v1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.perception.frames import camera_to_body_pitched
from src.verdict.gate_b import (disagreement, ensemble_lean, gate_b, gravity_gc,
                                gravity_pf, lean_difference_deg, lean_from_gravity,
                                roll_of)
from src.verdict.lean import world_to_yaw_frame
from src.verdict.selective import auroc, cluster_auroc, crc_from_counts, rate_interval

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "data/m02_scores"
OUT = ROOT / "results/m02"
CLIPS = ROOT / "data/processed/track_d/clips"
P3A = ROOT / "data/processed/p3a/p3a_v1"
GATE_TABLE = ROOT / "results/a2_gate/20261109T201449Z_298ff20/clip_table.csv"
GATE_CONFIG = ROOT / "configs/gate_frozen.yaml"
L01 = ROOT / "data/m02_inputs/l01_per_frame"
MANIFEST = ROOT / "results/m02/i02_manifest_copy.csv"

DEV = ["flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse",
       "flight-07p-lemniscate", "flight-13a-trackRATM", "flight-16p-trackRATM"]
HELD = ["flight-13p-trackRATM", "flight-07a-lemniscate"]
COND_MAP = {"identity": "real", "tilt_k0": "tilt_removal", "tilt_k05": "scale_0.5",
            "tilt_k15": "scale_1.5", "mirror_roll": "mirror_roll",
            "yaw_control": "yaw_control"}
WIN, HOP = 8, 4
VAR_FLOOR = 0.01
ALPHA = 0.05
TAU_GRID = np.round(np.arange(0.0, 30.01, 0.25), 2)
LABEL_DEG = 5.0


# --------------------------------------------------------------------------- loading
def load_scores(tag: str) -> pd.DataFrame:
    frames = []
    for p in sorted((SCORES / tag).glob("*.jsonl")):
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        frames += rows
    if not frames:
        raise SystemExit(f"no score rows under {SCORES / tag}")
    df = pd.DataFrame(frames)
    df["cond"] = df.get("cond", pd.Series(["real"] * len(df))).replace(COND_MAP)
    return df


def gate_tables():
    g = json.loads("\n".join(l for l in GATE_CONFIG.read_text().splitlines()
                             if not l.strip().startswith("#")))
    tau = float(g["tau"])
    rows = list(csv.DictReader(open(GATE_TABLE)))
    keep = {r["clip_id"]: float(r["identity::S3_latitude_confidence"]) <= tau for r in rows}
    s3 = {r["clip_id"]: float(r["identity::S3_latitude_confidence"]) for r in rows}
    return keep, s3


def tii_truth() -> dict:
    """Per (clip_id, frame index): truth lean from mocap through the recorded mount."""
    out = {}
    for flight in {c.split("#")[0] for c in
                   [d.name.split("__")[0] for d in CLIPS.glob("*/*")]}:
        spec = np.load(P3A / f"{flight}.npz")
        R = np.array([np.eye(3)] * len(spec["quat"]))
        from scipy.spatial.transform import Rotation
        R = Rotation.from_quat(spec["quat"]).as_matrix()
        z = R[:, :, 2]
        yaw_all = Rotation.from_matrix(R).as_euler("ZYX")[:, 0]
        for d in sorted((CLIPS / flight).iterdir()):
            cid = d.name.split("__")[0]
            if cid in out:
                continue
            z0 = np.load(d / "verdict_clip.npz")
            yaw = np.asarray(z0["yaw"], float)
            i = np.clip(np.searchsorted(spec["t"], np.asarray(z0["t"], float)), 0,
                        len(spec["t"]) - 1)
            out[cid] = world_to_yaw_frame(z[i][:, :2], yaw)
    return out


def tii_clip_meta() -> dict:
    out = {}
    for d in sorted(CLIPS.glob("*/*__real")):
        c = json.loads((d / "meta.json").read_text())
        out[c["clip_id"]] = {"flight": c["flight"], "uptilt_deg": float(c["uptilt_deg"])}
    return out


# --------------------------------------------------------------------------- statistics
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


def trimmed_rmse(err: np.ndarray, frac: float = 0.9) -> float:
    e = np.asarray(err, dtype=np.float64)
    e = e[np.isfinite(e)]
    if not len(e):
        return float("nan")
    n = max(1, int(round(frac * len(e))))
    keep = np.sort(e)[:n]
    return float(np.sqrt(np.mean(keep ** 2)))


# --------------------------------------------------------------------------- probe
def _fit_pitch_sign(df: pd.DataFrame) -> dict:
    """Pick PF's pitch sign by minimising the median lean disagreement with R_gc224."""
    out = {}
    for sign in (1.0, -1.0):
        ang = []
        for _, r in df.iterrows():
            u_pf = gravity_pf(r.to_dict(), pitch_sign=sign)
            u_gc = np.asarray(r["gc_gravity"], float)
            ang.append(float(np.degrees(np.arccos(np.clip(np.dot(u_pf, u_gc), -1, 1)))))
        out[sign] = float(np.median(ang))
    best = min(out, key=out.get)
    return {"median_deg_by_sign": {str(k): v for k, v in out.items()},
            "chosen": float(best), "spread_deg": float(abs(out[1.0] - out[-1.0]))}


def probe_check() -> dict:
    p = SCORES.parent / "m02_probe.json"
    if not p.exists():
        return {"available": False}
    rows = json.loads(p.read_text())
    df = pd.DataFrame(rows)
    df["gc_roll"] = df["gc_gravity"].apply(lambda g: float(roll_of(g)[0]))
    per = []
    for deg, g in df.groupby("rotation_deg"):
        per.append({"rotation_deg": float(deg),
                    "gc_roll_median": float(g.gc_roll.median()),
                    "pf_roll_median": float(g.pf_roll_deg.median())})
    pf_fit = _fit_pitch_sign(df)
    t = pd.DataFrame(per).sort_values("rotation_deg")
    d_gc = float(t.gc_roll_median.iloc[-1] - t.gc_roll_median.iloc[0])
    d_pf = float(t.pf_roll_median.iloc[-1] - t.pf_roll_median.iloc[0])
    return {"available": True, "table": per, "gc_roll_span_deg": d_gc,
            "pf_roll_span_deg": d_pf, "expected_span_deg": 20.0,
            "gc_follows_rotation": bool(abs(d_gc - 20.0) < 3.0),
            "pf_follows_rotation": bool(abs(d_pf - 20.0) < 3.0),
            "same_sign": bool(np.sign(d_gc) == np.sign(d_pf)),
            "pitch_sign_fit": pf_fit}


# --------------------------------------------------------------------------- per-clip
def build_real_table(scores: pd.DataFrame) -> pd.DataFrame:
    keep, s3 = gate_tables()
    meta = tii_clip_meta()
    truth = tii_truth()
    rows = []
    tii = scores[scores.source == "tii"]
    for (cid, cond), g in tii.groupby(["clip_id", "cond"]):
        g = g.sort_values("frame_index")
        m = meta[cid]
        R_cb = camera_to_body_pitched(np.deg2rad(m["uptilt_deg"]))
        l_pf = lean_from_gravity(np.vstack(g.pf_gravity.values), R_cb)
        l_gc = lean_from_gravity(np.vstack(g.gc_gravity.values), R_cb)
        c = np.load(CLIPS / m["flight"] / f"{cid}__{cond}" / "verdict_clip.npz")
        dep = np.asarray(c["depicted_lean"], float)
        req = np.asarray(c["lean_req"], float)      # v2 side, for the archive only
        n = min(len(l_pf), len(dep))
        l_pf, l_gc, dep = l_pf[:n], l_gc[:n], dep[:n]
        d_stats = disagreement(lean_difference_deg(l_pf, l_gc))
        ens = ensemble_lean(dep, l_pf)
        tru = truth[cid][:n]
        rows.append({"source": "tii", "clip_id": cid, "flight": m["flight"], "cond": cond,
                     "in_dev": m["flight"] in DEV, "kept_S3": bool(keep.get(cid, False)),
                     "s3": s3.get(cid, np.nan), "n": n,
                     "dis_med": d_stats.median, "dis_p90": d_stats.p90,
                     "dis_wmax": d_stats.window_max,
                     "err_frozen": float(np.nanmedian(lean_difference_deg(dep, tru))),
                     "err_pf": float(np.nanmedian(lean_difference_deg(l_pf, tru))),
                     "err_gc224": float(np.nanmedian(lean_difference_deg(l_gc, tru))),
                     "err_ens_frozen_pf": float(np.nanmedian(
                         lean_difference_deg(ens, tru))),
                     "coh_frozen": coherence_series(dep, req),
                     "coh_pf": coherence_series(l_pf, req),
                     "coh_ens": coherence_series(ens, req)})
    df = pd.DataFrame(rows)
    # frame-level table for AUROC
    frames = []
    for (cid, cond), g in tii.groupby(["clip_id", "cond"]):
        if cond != "real":
            continue
        g = g.sort_values("frame_index")
        m = meta[cid]
        R_cb = camera_to_body_pitched(np.deg2rad(m["uptilt_deg"]))
        l_pf = lean_from_gravity(np.vstack(g.pf_gravity.values), R_cb)
        l_gc = lean_from_gravity(np.vstack(g.gc_gravity.values), R_cb)
        c = np.load(CLIPS / m["flight"] / f"{cid}__real" / "verdict_clip.npz")
        dep = np.asarray(c["depicted_lean"], float)
        n = min(len(l_pf), len(dep))
        e = lean_difference_deg(dep[:n], truth[cid][:n])
        d = lean_difference_deg(l_pf[:n], l_gc[:n])
        for i in range(n):
            frames.append({"clip_id": cid, "flight": m["flight"], "frame": i,
                           "dis": float(d[i]), "err_frozen": float(e[i])})
    return df, pd.DataFrame(frames)


def build_sim_table(scores: pd.DataFrame) -> pd.DataFrame:
    man = {r["clip_id"]: r for r in csv.DictReader(open(MANIFEST))}
    keep, s3 = gate_tables()   # S3 is a TII-fitted gate; sim uses L01's own confidence
    rows = []
    sim = scores[scores.source == "sim"]
    for cid, g in sim.groupby("clip_id"):
        g = g.sort_values("frame_index")
        gt = np.load(ROOT / "data/m02_inputs/sim/gt" / f"{cid}.npz")
        R_cb = np.asarray(gt["R_cb"], float)
        l_pf = lean_from_gravity(np.vstack(g.pf_gravity.values), R_cb)
        l_gc = lean_from_gravity(np.vstack(g.gc_gravity.values), R_cb)
        # truth and the frozen reader at the same frame times (30 Hz grid)
        t = np.asarray(g.frame_index.values, float) / 30.0
        tg = np.asarray(gt["t"], float)
        lean_dep = np.column_stack([np.interp(t, tg, gt["lean_dep"][:, j]) for j in range(2)])
        idx = np.clip(np.round(t * 30).astype(int), 0, len(tg) - 1)
        lean_req = np.column_stack([np.interp(t, tg, gt["lean_req"][:, j]) for j in range(2)])
        l01p = L01 / f"{cid}.npz"
        err_l01 = np.nan
        s3v = np.nan
        if l01p.exists():
            z = np.load(l01p)
            fr = np.asarray(z["frame"], int)
            idx01 = np.clip(idx, 0, len(fr) - 1)
            l_est = np.asarray(z["lean_est"], float)[idx01]
            err_l01 = float(np.nanmedian(lean_difference_deg(l_est, lean_dep)))
            s3v = float(-np.nanmedian(np.asarray(z["latitude_confidence_mean"], float)[idx01]))
        d_stats = disagreement(lean_difference_deg(l_pf, l_gc))
        rows.append({"source": "sim", "clip_id": cid,
                     "family": man[cid]["family"],
                     "defect": man[cid]["defect_family"],
                     "mount": float(man[cid]["uptilt_deg"]),
                     "n": int(len(l_pf)),
                     "dis_med": d_stats.median, "dis_p90": d_stats.p90,
                     "dis_wmax": d_stats.window_max,
                     "err_frozen": err_l01,
                     "s3": s3v,
                     "err_pf": float(np.nanmedian(lean_difference_deg(l_pf, lean_dep))),
                     "err_gc224": float(np.nanmedian(
                         lean_difference_deg(l_gc, lean_dep))),
                     "err_ens": float(np.nanmedian(lean_difference_deg(
                         ensemble_lean(l_pf, l_gc), lean_dep))),
                     "bias_fwd": float(np.degrees(np.median(
                         (lean_dep - l_pf)[:, 0]))),
                     "coh_frozen": np.nan,
                     "coh_pf": np.nan, "coh_ens": np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- gate B
def gate_b_select(real: pd.DataFrame, tau_grid=TAU_GRID) -> dict:
    """CRC on the real dev clips: accept iff disagreement <= tau."""
    base = real[real.in_dev & real.kept_S3]
    n_total = base.clip_id.nunique()
    curve = []
    for tau in tau_grid:
        acc = base.dis_med <= tau
        n_clips = base[acc].clip_id.nunique()
        res = {"tau": float(tau), "accepted": int(n_clips),
               "coverage": n_clips / max(n_total, 1)}
        for cond in ("real", "yaw_control", "null_edit"):
            hit = 0
            sub = base[(base.cond == cond) & acc]
            for _, r in sub.iterrows():
                fit = base[(base.cond == "real") & acc & (base.flight != r.flight)]
                fit = fit.coh_frozen.values
                fit = fit[np.isfinite(fit)]
                if len(fit) < 5:
                    continue
                hit += int(np.isfinite(r.coh_frozen) and
                           r.coh_frozen > float(np.quantile(fit, 0.95)))
            res[cond] = hit
        curve.append(res)
    curve = pd.DataFrame(curve)
    fr = crc_from_counts(curve.tau, curve.accepted, curve.real, ALPHA)
    yw = crc_from_counts(curve.tau, curve.accepted, curve.yaw_control, ALPHA)
    lams = [x.lam for x in (fr, yw) if np.isfinite(x.lam)]
    joint = max(lams) if len(lams) == 2 else float("nan")
    return {"curve": curve, "false_reject": fr.as_dict(), "yaw_control": yw.as_dict(),
            "tau_joint": float(joint), "n_clips": int(n_total)}


def at_tau(real: pd.DataFrame, tau: float) -> dict:
    base = real[real.in_dev & real.kept_S3]
    acc = base.dis_med <= tau
    out = {"tau": float(tau), "n_total": int(base.clip_id.nunique()),
           "n_accepted": int(base[acc].clip_id.nunique())}
    out["coverage"] = out["n_accepted"] / max(out["n_total"], 1)
    rates = {}
    for cond in ("real", "tilt_removal", "mirror_roll", "scale_0.5", "scale_1.5",
                 "yaw_control", "null_edit"):
        sub = base[(base.cond == cond) & acc]
        hit = abst = 0
        for _, r in sub.iterrows():
            fit = base[(base.cond == "real") & acc & (base.flight != r.flight)]
            fit = fit.coh_frozen.values
            fit = fit[np.isfinite(fit)]
            if len(fit) < 5:
                abst += 1
                continue
            if not np.isfinite(r.coh_frozen):
                abst += 1
            elif r.coh_frozen > float(np.quantile(fit, 0.95)):
                hit += 1
        rates[cond] = {"hit": int(hit), "n": int(len(sub)), "skipped": int(abst)}
    out["rates"] = rates
    return out


def _rank_pct(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    if ok.sum() > 1:
        order = np.argsort(np.argsort(x[ok]))
        out[ok] = order / (ok.sum() - 1)
    return out


def combined_score(dis, neg_s3) -> np.ndarray:
    """Max of the two leave-one-out-invariant rank percentiles (both oriented = risky)."""
    a = _rank_pct(dis)
    b = _rank_pct(neg_s3)
    with np.errstate(invalid="ignore"):
        return np.fmax(a, b)


def auroc_table(real: pd.DataFrame, frames: pd.DataFrame, sim: pd.DataFrame) -> dict:
    out = {}
    r = real[(real.cond == "real") & real.in_dev & real.kept_S3]
    out["real_clip"] = {
        "n": int(len(r)),
        "positives": int((r.err_frozen > LABEL_DEG).sum()),
        "dis": cluster_auroc(r.dis_med, (r.err_frozen > LABEL_DEG).astype(int),
                             r.flight, draws=2000),
        "s3": cluster_auroc(-r.s3, (r.err_frozen > LABEL_DEG).astype(int),
                            r.flight, draws=2000),
    }
    f = frames[frames.flight.isin(DEV)]
    out["real_frame"] = {
        "n": int(len(f)),
        "dis": cluster_auroc(f.dis, (f.err_frozen > LABEL_DEG).astype(int),
                             f.flight, draws=2000),
    }
    s = sim[(sim.defect == "feasible") & sim.err_frozen.notna()]
    lab = (s.err_frozen > LABEL_DEG).astype(int)
    out["sim_clip"] = {
        "n": int(len(s)),
        "positives": int(lab.sum()),
        "dis": cluster_auroc(s.dis_med, lab, s.family, draws=2000),
        "s3": cluster_auroc(-s.s3, lab, s.family, draws=2000),
        "combined": cluster_auroc(combined_score(s.dis_med.values, -s.s3.values), lab,
                                  s.family, draws=2000),
    }
    r0 = real[(real.cond == "real") & real.in_dev & real.kept_S3]
    lab0 = (r0.err_frozen > LABEL_DEG).astype(int)
    out["real_clip"]["combined"] = cluster_auroc(
        combined_score(r0.dis_med.values, -r0.s3.values), lab0, r0.flight, draws=2000)
    for fam in ("urban_checker", "forest_ridge", "industrial"):
        sub = s[s.family == fam]
        out[f"sim_clip_{fam}"] = {
            "n": int(len(sub)), "positives": int((sub.err_frozen > LABEL_DEG).sum()),
            "auroc_dis": float(auroc(sub.dis_med, (sub.err_frozen > LABEL_DEG).astype(int))),
            "auroc_s3": float(auroc(-sub.s3, (sub.err_frozen > LABEL_DEG).astype(int))),
        }
    return out


def forest_check(sim: pd.DataFrame, tau_b: float) -> dict:
    """The forest clips the frozen S3 gate keeps, and what Gate B would do with them."""
    g = json.loads("\n".join(l for l in GATE_CONFIG.read_text().splitlines()
                             if not l.strip().startswith("#")))
    s3_tau = float(g["tau"])
    forest = sim[(sim.family == "forest_ridge") & (sim.defect == "feasible")].copy()
    forest["kept_S3"] = forest.s3 <= s3_tau
    kept = forest[forest.kept_S3]
    dropped = forest[~forest.kept_S3]
    out = {"n_forest": int(len(forest)),
           "n_kept_S3": int(len(kept)), "n_dropped_S3": int(len(dropped)),
           "kept_median_error": float(kept.err_frozen.median()) if len(kept) else None,
           "dropped_median_error": float(dropped.err_frozen.median()) if len(dropped)
           else None,
           "kept_median_disagreement": float(kept.dis_med.median()) if len(kept) else None,
           "forest_median_disagreement": float(forest.dis_med.median()),
           "open_median_disagreement": float(
               sim[(sim.family != "forest_ridge") & (sim.defect == "feasible")]
               .dis_med.median()),
           "gate_b_abstains_on_kept_forest": int(np.sum(kept.dis_med > tau_b))
           if np.isfinite(tau_b) else None,
           "gate_b_abstains_on_all_forest": int(np.sum(forest.dis_med > tau_b))
           if np.isfinite(tau_b) else None}
    return out


def ensemble_table(sim: pd.DataFrame) -> dict:
    s = sim[sim.defect == "feasible"]
    rows = {}
    for fam, g in s.groupby("family"):
        rows[fam] = {k: {"trimmed_rmse": trimmed_rmse(g[k].values),
                         "bias_fwd": float(np.nanmedian(g.bias_fwd.values)),
                         "median": float(np.nanmedian(g[k].values))}
                     for k in ("err_frozen", "err_pf", "err_gc224", "err_ens")}
    rows["all"] = {k: {"trimmed_rmse": trimmed_rmse(s[k].values),
                       "median": float(np.nanmedian(s[k].values))}
                   for k in ("err_frozen", "err_pf", "err_gc224", "err_ens")}
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    pr = probe_check()
    (OUT / "probe.json").write_text(json.dumps(pr, indent=1))
    print("probe:", {k: v for k, v in pr.items() if k != "table"})

    scores = load_scores(args.tag)
    print("score rows:", len(scores), scores.source.value_counts().to_dict())
    real, frames = build_real_table(scores)
    sim = build_sim_table(scores)
    real.to_csv(OUT / "real_by_clip.csv", index=False)
    sim.to_csv(OUT / "sim_by_clip.csv", index=False)
    frames.to_csv(OUT / "real_frames.csv", index=False)

    au = auroc_table(real, frames, sim)
    (OUT / "auroc.json").write_text(json.dumps(au, indent=1))
    print(json.dumps(au, indent=1))

    sel = gate_b_select(real)
    sel["curve"].to_csv(OUT / "gate_b_curve.csv", index=False)
    (OUT / "gate_b_selection.json").write_text(
        json.dumps({k: v for k, v in sel.items() if k != "curve"}, indent=1))
    print("gate B selection:", {k: v for k, v in sel.items() if k != "curve"})

    tau = sel["tau_joint"]
    if np.isfinite(tau):
        at = at_tau(real, tau)
        (OUT / "gate_b_at_tau.json").write_text(json.dumps(at, indent=1))
        print("at tau:", json.dumps(at, indent=1))

    (OUT / "forest.json").write_text(json.dumps(forest_check(sim, tau), indent=1))
    ens = ensemble_table(sim)
    (OUT / "ensemble.json").write_text(json.dumps(ens, indent=1))
    print("ensemble:", json.dumps(ens, indent=1))


if __name__ == "__main__":
    main()
