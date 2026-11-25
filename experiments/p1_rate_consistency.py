"""Track P, Part 1: the body-rate consistency check on video rotation (Check B).

Pre-registration `configs/p1_rate_consistency_prereg.yaml` (committed before any video-derived
rate was scored). Development data only: 200 simulated dev clips, 4 outdoor UZH sequences and the
6 TII development flights; the 234 SolarWM generated clips are scored under Part 3 of the same
pre-registration and never set a threshold.

    PYTHONPATH=. pixi run python experiments/p1_rate_consistency.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from experiments.p_common import (WINDOWS, clip_score, literal_side, path_side, video_side,
                                  window_stats)
from experiments.p_units import (H04_PATHS, TII_DEV, UZH_OUTDOOR, iter_units, sim_manifest)
from src.perception.frames import camera_to_body_pitched
from src.verdict.rate import K_D, residuals, rotation_rates, central_axis_rate, gauge_fit

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/p/p1"
PRIMARY = 0.5
BUDGETS = (2, 4, 6, 8, 10)
# testability floor: 3 x the source's P0 null window residual at 0.5 s (pre-registered rule)
FLOOR = {"sim": 3 * 2.20, "uzh": 3 * 1.07, "tii": 3 * 25.15, "gen": 3 * 2.20}
BOOT = 5000
RNG = np.random.default_rng(1)


# ------------------------------------------------------------------ scoring
def score(u: dict, kind: str, attitude=None, draw: int = 1) -> dict:
    path = path_side(u)
    v = video_side(u, path, attitude=attitude)
    rows = {w: window_stats(v["res"], w, min_req_deg_s=FLOOR[kind]) for w in WINDOWS}
    out = {"kind": kind, "unit": u["unit"], "n_windows": len(rows[PRIMARY]),
           "n_frames": len(v["res"]["t"]), "gauge_fit_deg": v["gauge"]["fit_deg"]}
    for w in WINDOWS:
        out[f"abs_w{w}"] = clip_score(rows[w], "abs_deg_s")
        out[f"rel_w{w}"] = clip_score(rows[w], "rel")
        out[f"cos_w{w}"] = clip_score(rows[w], "cos")
        out[f"req_w{w}"] = clip_score(rows[w], "req_deg_s")
    # lag that maximises the cross-correlation of the required and depicted rate magnitudes
    rr = v["res"]
    out["lag_s"] = lag_of(u, rr)
    lit = literal_side(u, path, v["R_wc"])
    out["literal_abs"] = clip_score(window_stats(lit, PRIMARY, min_req_deg_s=FLOOR[kind]))
    # exploratory post-hoc tier: remove a constant 3-DoF rate offset over the clip
    ev = v["res"]["err_vec"]
    good = np.isfinite(ev).all(1)
    out["off_w0.5"] = float("nan")
    if good.sum() >= 8:
        res = v["res"].copy()
        res["err_deg_s"] = np.full(len(ev), np.nan)
        res["err_deg_s"][good] = np.degrees(np.linalg.norm(ev[good] - ev[good].mean(0), axis=1))
        out["off_w0.5"] = clip_score(window_stats(res, PRIMARY, min_req_deg_s=FLOOR[kind]))
    out["_rows"] = rows
    out["_res"] = rr
    return out


def lag_of(u: dict, res: dict, max_lag: float = 0.5) -> float:
    """Exploratory: lag in seconds maximising corr(|rb|, |rd|) over +/-0.5 s."""
    t = np.asarray(res["t"], float)
    rb = np.asarray(res["req_deg_s"], float)
    rd = np.asarray(res["dep_deg_s"], float)
    ok = np.isfinite(rb) & np.isfinite(rd)
    if ok.sum() < 8:
        return float("nan")
    t, rb, rd = t[ok], rb[ok], rd[ok]
    dt = float(np.median(np.diff(t)))
    best, best_lag = -np.inf, float("nan")
    for lag in np.arange(-max_lag, max_lag + 1e-9, dt):
        r = np.interp(t, t + lag, rd, left=np.nan, right=np.nan)
        m = np.isfinite(r)
        if m.sum() < 8 or np.std(rb[m]) < 1e-9 or np.std(r[m]) < 1e-9:
            continue
        c = float(np.corrcoef(rb[m], r[m])[0, 1])
        if c > best:
            best, best_lag = c, float(lag)
    return best_lag


def thresholds(feasible: pd.Series, budgets=BUDGETS, low_is_bad: bool = False) -> dict:
    vals = np.sort(np.asarray(feasible.dropna(), float))
    if not low_is_bad:
        vals = vals[::-1]
    return {k: float(vals[k - 1]) if len(vals) >= k else float("nan") for k in budgets}


def cluster_bootstrap(flags: np.ndarray, clusters: np.ndarray, n: int = BOOT) -> dict:
    """Cluster bootstrap of a rate; fewer than 10 clusters is descriptive."""
    flags = np.asarray(flags, float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    if len(uniq) < 2:
        return {"rate": float(np.nanmean(flags)) if len(flags) else float("nan"),
                "lo": float("nan"), "hi": float("nan"), "clusters": int(len(uniq)),
                "descriptive": True}
    vals = []
    for _ in range(n):
        pick = RNG.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(clusters == c) for c in pick])
        vals.append(np.nanmean(flags[idx]))
    return {"rate": float(np.nanmean(flags)), "lo": float(np.nanpercentile(vals, 2.5)),
            "hi": float(np.nanpercentile(vals, 97.5)), "clusters": int(len(uniq)),
            "descriptive": bool(len(uniq) < 10)}


def yaw_warp(u: dict, R_est: np.ndarray, amplitude_deg: float = 20.0, period_s: float = 2.0,
             axis: str = "thrust") -> np.ndarray:
    """Rotate the camera attitude about the depicted thrust axis (or the optical axis)."""
    t = np.asarray(u["t_video"], float)
    psi = np.deg2rad(amplitude_deg) * np.sin(2 * np.pi * t / period_s)
    R_cb = u["R_cb"]
    if axis == "thrust":
        b = np.einsum("nij,j->ni", R_est, R_cb.T @ np.array([0.0, 0.0, 1.0]))
        axis_vec = b
    else:
        axis_vec = np.einsum("nij,j->ni", R_est, np.array([0.0, 0.0, 1.0]))
    out = np.empty_like(R_est)
    for k in range(len(R_est)):
        a = axis_vec[k] / max(np.linalg.norm(axis_vec[k]), 1e-12)
        out[k] = Rotation.from_rotvec(a * psi[k]).as_matrix() @ R_est[k]
    return out


# ------------------------------------------------------------------ generated clips
def generated_units() -> list[dict]:
    d = np.load(H04_PATHS, allow_pickle=True)
    poses_dir = ROOT / "data/n04/poses/gen/fov90"
    R_cb = camera_to_body_pitched(0.0)
    keys = sorted({k.split("__")[0] for k in d.keys()}, key=len, reverse=True)
    out = []
    for p in sorted(poses_dir.glob("*.npz")):
        clip = p.stem
        key = next((k for k in keys if f"_{k}_" in clip), None)
        if key is None or f"{key}__t_1khz" not in d:
            continue
        t = np.asarray(d[f"{key}__t_1khz"], float)
        pos = np.asarray(d[f"{key}__pos_1khz"], float)
        arm = {"A": "A_flatness", "B": "B_level", "C": "C_mirrored_roll"}.get(
            clip.split("_")[-3], clip.split("_")[-3])
        P = np.load(p)
        R_est = np.asarray(P["c2w"], float)[:, :3, :3]
        tq = np.asarray(P["t"], float) if "t" in P else np.arange(len(R_est)) / 16.0
        yaw = np.asarray(d[f"{key}__yaw"], float)
        out.append({"kind": "gen", "unit": clip, "row": {"arm": arm, "path": key},
                    "t": t, "pos": pos, "vel": None, "acc": None,
                    "yaw": np.interp(tq, np.asarray(d[f"{key}__t"], float), yaw),
                    "R_cb": R_cb, "t_video": tq,
                    "fps_video": float(1.0 / np.median(np.diff(tq))), "R_est": R_est,
                    "duration_s": float(t[-1])})
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sim = iter_units("sim")
    uzh = iter_units("uzh")
    tii = iter_units("tii")
    gen = generated_units()
    print(f"units sim={len(sim)} uzh={len(uzh)} tii={len(tii)} gen={len(gen)}")

    # ---------------- score every unit (draw 1 for sim, the only draw elsewhere)
    scored, windows = [], []
    for group, kind in ((sim, "sim"), (uzh, "uzh"), (tii, "tii"), (gen, "gen")):
        for u in group:
            s = score(u, kind)
            rows = s.pop("_rows")
            res = s.pop("_res")
            for w, rr in rows.items():
                for x in rr:
                    windows.append({"kind": kind, "unit": u["unit"], "window_s": w,
                                    "abs_deg_s": x["abs_deg_s"], "rel": x["rel"],
                                    "cos": x["cos"], "req_deg_s": x["req_deg_s"],
                                    "t_mid_s": x["lo"] / u["fps_video"] + w / 2})
            rec = {"kind": kind, "unit": u["unit"], **s}
            if kind == "sim":
                m = u["row"]
                rec.update({"label": m["label"], "violation": m["violation"],
                            "defect_family": m["defect_family"], "family": m["family"],
                            "maneuver": m["maneuver"], "mount": float(m["uptilt_deg"]),
                            "peak_tilt_req_deg": float(m["peak_tilt_req_deg"]),
                            "interval_start_s": m["interval_start_s"],
                            "interval_end_s": m["interval_end_s"],
                            "interval_fraction": float(m["interval_fraction"]),
                            "magnitude_unit": m["magnitude_unit"],
                            "magnitude_deg": float(m["magnitude_deg"] or 0.0)})
            elif kind == "gen":
                rec.update({"arm": u["row"]["arm"], "path": u["row"]["path"]})
            else:
                rec.update({"family": "outdoor" if u["unit"] in UZH_OUTDOOR else "indoor",
                            "maneuver": "", "mount": float("nan"),
                            "label": "real", "violation": "", "defect_family": "feasible",
                            "peak_tilt_req_deg": float("nan"),
                            "interval_start_s": "", "interval_end_s": "",
                            "interval_fraction": 1.0, "magnitude_unit": "",
                            "magnitude_deg": 0.0})
            scored.append(rec)
    df = pd.DataFrame(scored)
    win = pd.DataFrame(windows)
    df.to_csv(OUT / "per_clip.csv", index=False)
    win.to_csv(OUT / "windows.csv", index=False)

    summary: dict = {"prereg": "configs/p1_rate_consistency_prereg.yaml",
                     "floor_deg_s": FLOOR, "primary_window_s": PRIMARY}

    # ---------------- sim operating points
    thr = {tier: thresholds(df[(df.kind == "sim") & (df.label == "feasible")]
                            [f"{tier}_w{PRIMARY}"], low_is_bad=(tier == "cos"))
           for tier in ("abs", "rel", "cos")}
    summary["thresholds_sim"] = thr
    for tier in ("abs", "rel", "cos"):
        df[f"flag_{tier}"] = np.where(
            df[f"{tier}_w{PRIMARY}"].isna(), np.nan,
            (df[f"{tier}_w{PRIMARY}"] <= thr[tier][4]) if tier == "cos"
            else (df[f"{tier}_w{PRIMARY}"] >= thr[tier][4]))

    feas = df[(df.kind == "sim") & (df.label == "feasible")]
    dyn = df[(df.kind == "sim") & (df.peak_tilt_req_deg >= 15)]
    whole = df[(df.kind == "sim") & (df.interval_fraction == 1.0)]
    blocks = {}
    for name, sub in (("feasible", feas), ("feasible_dynamic", feas[feas.peak_tilt_req_deg >= 15]),
                      ("all_defect", whole[whole.label == "infeasible"]),
                      ("removal", whole[whole.defect_family == "tilt_removed"]),
                      ("mirror", whole[whole.defect_family == "mirrored_roll"]),
                      ("scale", whole[whole.defect_family == "tilt_scaled"]),
                      ("scale_0.5", whole[whole.magnitude_unit.fillna("").str.contains("x0.5")]),
                      ("scale_1.5", whole[whole.magnitude_unit.fillna("").str.contains("x1.5")]),
                      ("thrust_limit", whole[whole.defect_family == "thrust_limit"]),
                      ("partial", df[(df.kind == "sim") & (df.interval_fraction > 0)
                                     & (df.interval_fraction < 1.0)])):
        entry = {"n": int(len(sub)),
                 "testable": int(sub[f"abs_w{PRIMARY}"].notna().sum())}
        for tier in ("abs", "rel", "cos"):
            v = sub[f"flag_{tier}"].to_numpy(float)
            ok = ~np.isnan(v)
            entry[tier] = (f"{int(np.nansum(v))}/{int(ok.sum())}" if ok.sum() else "0/0")
        entry["abs_rate_dynamic"] = None
        if len(sub):
            f = sub[sub.peak_tilt_req_deg >= 15] if sub.peak_tilt_req_deg.notna().any() else sub
            if len(f):
                entry["abs_rate_dynamic"] = (
                    f"{int(np.nansum(f['flag_abs']))}/{int(f['flag_abs'].notna().sum())}")
        blocks[name] = entry
    summary["sim_detection_at_4of80"] = blocks

    # curve over budgets
    curve = {}
    for k in BUDGETS:
        row = {"feasible": f"{int((feas[f'abs_w{PRIMARY}'] >= thr['abs'][k]).sum())}/{len(feas)}"}
        for name, fam in (("removal", "tilt_removed"), ("mirror", "mirrored_roll"),
                          ("scale", "tilt_scaled"), ("partial", "partial")):
            s = df[(df.kind == "sim") & (df.defect_family == fam)]
            if name == "partial":
                s = df[(df.kind == "sim") & (df.interval_fraction > 0)
                       & (df.interval_fraction < 1.0)]
            row[name] = f"{int((s[f'abs_w{PRIMARY}'] >= thr['abs'][k]).sum())}/{int(s[f'abs_w{PRIMARY}'].notna().sum())}"
        curve[str(k)] = row
    summary["sim_budget_curve"] = curve

    # strata by family and mount for the whole-clip camera defects
    strata = {}
    for fam in sorted(df[df.kind == "sim"].family.dropna().unique()):
        for mk in (0.0, 15.0, 30.0, 45.0):
            s = df[(df.kind == "sim") & (df.family == fam) & (df.mount == mk)
                   & (df.interval_fraction == 1.0) & df.defect_family.isin(
                       ["tilt_removed", "mirrored_roll", "tilt_scaled"])]
            f = feas[(feas.family == fam) & (feas.mount == mk)]
            if len(s) or len(f):
                strata[f"{fam}_{mk:g}"] = {
                    "defect": f"{int(np.nansum(s['flag_abs']))}/{int(s['flag_abs'].notna().sum())}"
                    if len(s) else "0/0",
                    "feasible": f"{int(np.nansum(f['flag_abs']))}/{int(f['flag_abs'].notna().sum())}"
                    if len(f) else "0/0"}
    summary["sim_strata"] = strata

    # ---------------- controls
    yaw_sc = []
    for u in sim:
        if u["row"]["label"] != "feasible":
            continue
        R = yaw_warp(u, u["R_est"])
        s = score(u, "sim", attitude=R)
        s.pop("_rows"); s.pop("_res")
        yaw_sc.append({"unit": u["unit"], "abs": s[f"abs_w{PRIMARY}"]})
    ydf = pd.DataFrame(yaw_sc)
    summary["yaw_control"] = {
        "n": int(len(ydf)),
        "testable": int(ydf["abs"].notna().sum()),
        "flagged": f"{int((ydf['abs'] >= thr['abs'][4]).sum())}/{int(ydf['abs'].notna().sum())}",
    }
    # camera-roll variant (exploratory, mount dependent)
    roll_flagged = {}
    for mk in (0.0, 15.0, 30.0, 45.0):
        vals = []
        for u in sim:
            if u["row"]["label"] != "feasible" or float(u["row"]["uptilt_deg"]) != mk:
                continue
            R = yaw_warp(u, u["R_est"], axis="optical")
            s = score(u, "sim", attitude=R)
            if np.isfinite(s[f"abs_w{PRIMARY}"]):
                vals.append(s[f"abs_w{PRIMARY}"] >= thr["abs"][4])
        roll_flagged[f"{mk:g}"] = f"{int(np.sum(vals))}/{len(vals)}"
    summary["camera_roll_control"] = roll_flagged

    # null edit: draw 2 poses on the feasible clips
    from experiments.p_units import SIM_POSES, SIM_GT
    null_scores = []
    for u in sim:
        if u["row"]["label"] != "feasible":
            continue
        p2 = ROOT / "data/n05/poses/sim200/scale1_rep2" / f"{u['unit']}.npz"
        if not p2.exists():
            continue
        R2 = np.asarray(np.load(p2)["c2w"], float)[:, :3, :3]
        s = score({**u, "R_est": R2}, "sim")
        null_scores.append({"unit": u["unit"], "abs": s[f"abs_w{PRIMARY}"]})
    ndf = pd.DataFrame(null_scores)
    summary["null_edit"] = {
        "n": int(len(ndf)),
        "flagged": f"{int((ndf['abs'] >= thr['abs'][4]).sum())}/{int(ndf['abs'].notna().sum())}"}

    # partial localisation
    part = df[(df.kind == "sim") & (df.interval_fraction > 0)
              & (df.interval_fraction < 1.0)]
    io = []
    for _, r in part.iterrows():
        wrows = win[(win.kind == "sim") & (win.unit == r.unit) & (win.window_s == PRIMARY)]
        if not len(wrows):
            io.append({"unit": r.unit, "fraction": r.interval_fraction, "hit": 0, "iou": 0.0,
                       "n_flagged": 0})
            continue
        ws = PRIMARY
        flagged = wrows[(wrows.abs_deg_s >= thr["abs"][4]) & wrows.req_deg_s.notna()]
        lo, hi = float(r.interval_start_s), float(r.interval_end_s)
        on = flagged[(flagged.t_mid_s + ws / 2 >= lo) & (flagged.t_mid_s - ws / 2 <= hi)]
        hit = int(len(on) > 0)
        iou = 0.0
        if len(flagged):
            a = np.clip(np.minimum(flagged.t_mid_s + ws / 2, hi)
                        - np.maximum(flagged.t_mid_s - ws / 2, lo), 0, None).sum()
            u_ = ((flagged.t_mid_s + ws / 2) - (flagged.t_mid_s - ws / 2)).sum() + (hi - lo) - a
            iou = float(a / u_) if u_ > 0 else 0.0
        io.append({"unit": r.unit, "fraction": r.interval_fraction, "hit": hit, "iou": iou,
                   "n_flagged": int(len(flagged))})
    iodf = pd.DataFrame(io)
    summary["partial"] = {
        str(f): {"n": int(len(g)), "hit": f"{int(g['hit'].sum())}/{len(g)}",
                 "median_iou": float(g["iou"].median())}
        for f, g in iodf.groupby("fraction")}

    # ---------------- real UZH, leave one sequence out
    uz = df[(df.kind == "uzh") & df.unit.isin(UZH_OUTDOOR)]
    uw = win[(win.kind == "uzh") & (win.window_s == PRIMARY) & win.unit.isin(UZH_OUTDOOR)]
    loo = {}
    for seq in UZH_OUTDOOR:
        others = [s for s in UZH_OUTDOOR if s != seq]
        w_thr = float(np.nanpercentile(uw[uw.unit.isin(others)].abs_deg_s.dropna(), 95))
        c_thr = float(np.nanpercentile(uz[uz.unit.isin(others)]["abs_w0.5"].dropna(), 95))
        mine = uw[uw.unit == seq]
        fr = float((mine.abs_deg_s >= w_thr).mean()) if len(mine) else float("nan")
        clip = uz[uz.unit == seq].iloc[0]
        loo[seq] = {"window_threshold_deg_s": w_thr, "window_false_reject": fr,
                    "n_windows": int(len(mine)),
                    "clip_threshold_deg_s": c_thr,
                    "clip_score_deg_s": float(clip["abs_w0.5"])
                    if np.isfinite(clip["abs_w0.5"]) else None,
                    "clip_rejected": bool(np.isfinite(clip["abs_w0.5"]) and clip["abs_w0.5"] >= c_thr),
                    "n_testable_windows": int(clip.n_windows)}
    summary["uzh_leave_one_out"] = loo
    summary["uzh_clip_scores"] = {r.unit: (float(r["abs_w0.5"]) if np.isfinite(r["abs_w0.5"]) else None)
                                  for _, r in uz.iterrows()}
    # real-calibrated threshold applied to the simulated clips (deployment column)
    tau = float(np.nanpercentile(uw.abs_deg_s.dropna(), 95))
    summary["real_calibrated"] = {
        "tau_window_deg_s": tau,
        "sim_feasible_flagged": f"{int((feas['abs_w0.5'] >= tau).sum())}/{int(feas['abs_w0.5'].notna().sum())}",
        "sim_removal": f"{int((df[(df.kind=='sim') & (df.defect_family=='tilt_removed')]['abs_w0.5'] >= tau).sum())}"
                       f"/{int(df[(df.kind=='sim') & (df.defect_family=='tilt_removed')]['abs_w0.5'].notna().sum())}",
        "sim_mirror": f"{int((df[(df.kind=='sim') & (df.defect_family=='mirrored_roll')]['abs_w0.5'] >= tau).sum())}"
                      f"/{int(df[(df.kind=='sim') & (df.defect_family=='mirrored_roll')]['abs_w0.5'].notna().sum())}",
    }
    # JSON-safe nested dict (windows.csv and per_clip.csv hold the per-unit detail)
    summary["uzh_leave_one_out"] = {k: {kk: (float(vv) if isinstance(vv, (np.floating,))
                                           else vv) for kk, vv in v.items()}
                                    for k, v in summary["uzh_leave_one_out"].items()}

    # ---------------- TII descriptive
    summary["tii"] = {r.unit: {"clip_score_deg_s": (float(r["abs_w0.5"])
                                                    if np.isfinite(r["abs_w0.5"]) else None),
                               "n_testable_windows": int(r.n_windows),
                               "n_windows": int(r.n_windows)} for _, r in df[df.kind == "tii"].iterrows()}

    # ---------------- generated (SolarWM, exploratory)
    g = df[df.kind == "gen"]
    summary["generated_solarwm"] = {
        arm: {"n": int(len(sub)),
              "testable": int(sub["abs_w0.5"].notna().sum()),
              "median_clip_score": (float(sub["abs_w0.5"].median())
                                    if sub["abs_w0.5"].notna().any() else None),
              "flagged_at_sim_4of80": f"{int((sub['abs_w0.5'] >= thr['abs'][4]).sum())}"
                                      f"/{int(sub['abs_w0.5'].notna().sum())}"}
        for arm, sub in g.groupby("arm")}
    summary["generated_by_path"] = {
        p: {"arm": sub.arm.iloc[0], "n": int(len(sub)),
            "median_clip_score": (float(sub["abs_w0.5"].median())
                                  if sub["abs_w0.5"].notna().any() else None)}
        for p, sub in g.groupby("path")}

    # ---------------- lags (exploratory)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: summary[k] for k in
                      ("sim_detection_at_4of80", "yaw_control", "null_edit", "partial",
                       "uzh_leave_one_out", "generated_solarwm")}, indent=1, default=float))


if __name__ == "__main__":
    main()
