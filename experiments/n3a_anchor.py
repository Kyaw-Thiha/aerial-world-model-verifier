"""Track N3, Part A: anchor availability and accuracy from ground truth, no video.

Pre-registration `configs/n3a_anchor_prereg.yaml` + amendment A, both committed before any count
here was computed. This script never opens a reader score or a pose rotation: it uses each clip's
own truth trajectory (and, for the UZH/TII rows, the truth attitude to *score* the anchor).

Run:  PYTHONPATH=. pixi run python experiments/n3a_anchor.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from src.datasets.tii import load_tii
from src.verdict.anchor import (G_ANCHOR, K_D_ANCHOR, angle_between_deg, find_windows,
                                first_sustained_motion, implied_thrust_axis, implied_tilt_deg,
                                last_sustained_motion, mount_rotated, select_anchor,
                                select_anchor_first, snap_to_grid)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/n3/n3a"
GT = ROOT / "data/n05/gt"
POSES = ROOT / "data/n05/poses/sim200/scale1"
SCORES = ROOT / "data/n05/scores"
MANIFEST = ROOT / "results/n2/inputs/manifest.csv"
UZH = ROOT / "data/uzh"
TII = ROOT / "data/raw/tii"

E3 = np.array([0.0, 0.0, 1.0])
TIERS = {"P1": (2.0, 1.0), "P2": (4.0, 1.0), "P3": (6.0, 1.0)}
MIN_DUR = 1.0
# the airborne filter is a REAL-footage rule: the simulated clips are rendered in mid-air and never
# take off or land, so applying it there would throw away valid anchors for no physical reason
AIRBORNE_FILTER_KINDS = ("uzh", "tii")
SG = {"fs_hz": 100.0, "window": 31, "poly": 3}


# ------------------------------------------------------------------ derivative pipeline
def uniform_derivatives(t: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position -> uniform grid -> (t, vel, acc) with one Savitzky-Golay filter (amendment A)."""
    t = np.asarray(t, float)
    pos = np.asarray(pos, float)
    tg = np.arange(t[0], t[-1], 1.0 / SG["fs_hz"])
    pg = np.column_stack([np.interp(tg, t, pos[:, k]) for k in range(3)])
    vel = np.column_stack([savgol_filter(pg[:, k], SG["window"], SG["poly"], deriv=1,
                                         delta=1.0 / SG["fs_hz"]) for k in range(3)])
    acc = np.column_stack([savgol_filter(pg[:, k], SG["window"], SG["poly"], deriv=2,
                                         delta=1.0 / SG["fs_hz"]) for k in range(3)])
    return tg, vel, acc


# ------------------------------------------------------------------ one unit
def analyse_unit(kind: str, unit: str, t: np.ndarray, acc: np.ndarray, vel: np.ndarray,
                 R_bw: np.ndarray, t_lo: float, t_hi: float, R_cb: np.ndarray | None,
                 meta: dict) -> list[dict]:
    """Anchor rows for one unit and every tier.

    Amendment B: a quiet window is only used as an anchor if the unit is airborne there, i.e. it
    lies strictly between the first and the last sustained motion. A unit whose path never reaches
    the motion threshold (a pure hover) is treated as airborne throughout. The reason: while the
    drone rests on a support the flatness equation does not hold (a normal force replaces thrust),
    so a pre-takeoff or post-landing rest cannot pin the start -- the UZH and TII footage both
    begin and end that way.
    """
    t = np.asarray(t, float)
    theta = implied_tilt_deg(acc, vel)
    speed = np.linalg.norm(np.asarray(vel, float), axis=1)
    t_motion = first_sustained_motion(t, speed)
    t_motion_last = last_sustained_motion(t, speed)
    hover_only = not np.isfinite(t_motion)
    use_filter = kind in AIRBORNE_FILTER_KINDS
    axis = implied_thrust_axis(acc, vel)
    inside = (t >= t_lo - 1e-9) & (t <= t_hi + 1e-9)
    # the true total tilt at a frame is angle(R_bw^T e3, e3) -- mount-free by construction
    true_tilt = angle_between_deg(np.einsum("nij,j->ni", np.asarray(R_bw, float), E3)
                                  if R_bw is not None else np.full((len(t), 3), np.nan),
                                  np.broadcast_to(E3, (len(t), 3)))
    z = meta.get("_alt_m")
    model = (angle_between_deg(axis, np.einsum("nij,j->ni", np.asarray(R_bw, float), E3))
             if R_bw is not None else np.full(len(t), np.nan))
    rows = []
    for tier, (thr, dur) in TIERS.items():
        wins_raw = find_windows(theta, t, thr, dur, valid=inside)
        wins = wins_raw if (hover_only or not use_filter) else [
            w for w in wins_raw if t[w.start] >= t_motion and t[w.stop - 1] <= t_motion_last]
        best = select_anchor(wins)
        first = select_anchor_first(wins)
        rec = {"kind": kind, "unit": unit, "tier": tier, "thr_deg": thr, "min_dur_s": dur,
               "n_windows": len(wins), "n_windows_raw": len(wins_raw),
               "has_anchor": int(best is not None),
               "has_anchor_raw": int(bool(wins_raw)),
               "hover_only": int(hover_only),
               "t_first_motion_s": (None if not np.isfinite(t_motion) else float(t_motion)),
               "t_last_motion_s": (None if not np.isfinite(t_motion_last) else float(t_motion_last)),
               "frame_frac_quiet": float(np.mean(inside & np.isfinite(theta)
                                                 & (theta <= thr))),
               "t_lo_s": float(t_lo), "t_hi_s": float(t_hi),
               **{f"meta_{k}": v for k, v in meta.items() if not k.startswith("_")}}
        raw_best = select_anchor(wins_raw)
        if use_filter and raw_best is not None:
            # the window the filter rejected: kept so the reason for the rejection is a stored
            # number and not a claim (pre-takeoff / post-landing rests, amendment B)
            ra = raw_best.argmin
            rec["raw_anchor_t_s"] = float(t[ra])
            rec["raw_anchor_err_deg"] = float(true_tilt[ra])
            rec["raw_theta_imp_deg"] = float(theta[ra])
            rec["raw_anchor_alt_above_min_m"] = (None if z is None else float(
                np.median(z[raw_best.start:raw_best.stop]) - np.nanmin(z)))
            rec["raw_anchor_before_first_motion"] = int(np.isfinite(t_motion)
                                                        and t[ra] < t_motion)
            rec["raw_anchor_after_last_motion"] = int(np.isfinite(t_motion_last)
                                                     and t[ra] > t_motion_last)
        if best is not None:
            a = best.argmin
            rec.update({
                "anchor_t_s": float(t[a]),
                "anchor_frac": float((t[a] - t_lo) / max(t_hi - t_lo, 1e-9)),
                "window_dur_s": float(t[best.stop - 1] - t[best.start]),
                "window_mean_tilt_deg": best.mean_tilt_deg,
                "window_max_tilt_deg": float(np.nanmax(theta[best.start:best.stop])),
                "theta_imp_at_anchor_deg": float(theta[a]),
                "anchor_err_deg": float(true_tilt[a]),
                "model_check_deg": float(model[a]),
                "first_window_anchor_t_s": float(first.t_argmin_s),
                "anchor_alt_above_min_m": (None if z is None else float(
                    np.median(z[best.start:best.stop]) - np.nanmin(z))),
                "n_windows_sep_s": float(len(wins)),
            })
            if kind in ("sim", "sim_diffpos"):
                fidx = np.asarray(np.load(SCORES / f"{unit}.npz")["frame"], int)
                tf = fidx / 30.0
                rec["lever_median_s"] = float(np.median(np.abs(tf - t[a])))
                if kind == "sim":
                    d = np.load(POSES / f"{unit}.npz")
                    rec["anchor_pose_frame"] = snap_to_grid(t[a], d["t"])
                if R_cb is not None:
                    for deg, tag in ((2.0, "2"), (5.0, "5")):
                        rec[f"mount_err_{tag}deg"] = float(angle_between_deg(
                            (R_cb.T @ E3)[None], (mount_rotated(R_cb, deg).T @ E3)[None])[0])
        rows.append(rec)
    return rows


def iter_units():
    """Yield every analysis unit with its trajectory, attitude and metadata."""
    man = {r["clip_id"]: r for r in csv.DictReader(open(MANIFEST)) if r["split"] == "dev"}
    for cid in sorted(man):
        m = man[cid]
        g = np.load(GT / f"{cid}.npz")
        t = np.asarray(g["t"], float)
        R_bw = np.asarray(g["R_bw"], float)
        fidx = np.asarray(np.load(SCORES / f"{cid}.npz")["frame"], int)
        meta = {"family": m["family"], "mount": float(m["uptilt_deg"]),
                "maneuver": m["maneuver"], "label": m["label"],
                "dynamic": int(float(m["peak_tilt_req_deg"]) >= 15.0),
                "peak_tilt_req_deg": float(m["peak_tilt_req_deg"])}
        yield dict(kind="sim", unit=cid, t=t, acc=np.asarray(g["acc"], float),
                   vel=np.asarray(g["vel"], float), R_bw=R_bw, t_lo=fidx[0] / 30.0,
                   t_hi=fidx[-1] / 30.0, R_cb=np.asarray(g["R_cb"], float), meta=meta)
        tg, vg, ag = uniform_derivatives(t, np.asarray(g["pos"], float))
        Rg = np.stack([R_bw[snap_to_grid(x, t)] for x in tg])
        yield dict(kind="sim_diffpos", unit=cid, t=tg, acc=ag, vel=vg, R_bw=Rg,
                   t_lo=fidx[0] / 30.0, t_hi=fidx[-1] / 30.0,
                   R_cb=np.asarray(g["R_cb"], float), meta=meta)
    for d in sorted(p for p in UZH.iterdir() if (p / "truth.npz").exists()):
        tr = np.load(d / "truth.npz")
        gt = np.asarray(tr["gt"], float)
        t0 = float(tr["gt_t0"])
        pos = gt[:, 1:4]
        R_bw = Rotation.from_quat(gt[:, 4:8]).as_matrix()
        tg, vg, ag = uniform_derivatives(gt[:, 0] - t0, pos)
        t_unix = np.asarray(tr["t_unix"], float) - t0
        Rg = R_bw[np.searchsorted(gt[:, 0] - t0, tg)]
        meta = {"rig": d.name.split("_")[0], "name": d.name,
                "dynamic": int(np.nanmax(implied_tilt_deg(ag, vg)) >= 15.0),
                "_alt_m": np.interp(tg, gt[:, 0] - t0, gt[:, 2])}
        yield dict(kind="uzh", unit=d.name, t=tg, acc=ag, vel=vg, R_bw=Rg,
                   t_lo=float(t_unix[0]), t_hi=float(t_unix[-1]), R_cb=None, meta=meta)
    for kind in ("autonomous", "piloted"):
        for f in sorted((TII / kind).glob("mocap_*.csv")):
            clip = load_tii(f)
            t = np.asarray(clip.t, float)
            R_bw = Rotation.from_quat(np.asarray(clip.quat, float)).as_matrix()
            tg, vg, ag = uniform_derivatives(t, np.asarray(clip.pos, float))
            Rg = np.stack([R_bw[snap_to_grid(x, t)] for x in tg])
            meta = {"kind_name": kind, "name": f.stem.replace("mocap_", ""),
                    "dynamic": int(np.nanmax(implied_tilt_deg(ag, vg)) >= 15.0),
                    "_alt_m": np.interp(tg, t, np.asarray(clip.pos, float)[:, 2])}
            yield dict(kind="tii", unit=f"{kind}/{meta['name']}", t=tg, acc=ag, vel=vg,
                       R_bw=Rg, t_lo=float(tg[0]), t_hi=float(tg[-1]), R_cb=None, meta=meta)


SURFACE_THR = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0)
SURFACE_DUR = (0.25, 0.5, 1.0)


def availability_surface(edge_u: dict) -> list[dict]:
    u = edge_u
    """Descriptive: coverage and quietest-window quality over a threshold x duration grid."""
    t = np.asarray(u["t"], float)
    theta = implied_tilt_deg(u["acc"], u["vel"])
    speed = np.linalg.norm(np.asarray(u["vel"], float), axis=1)
    t_motion = first_sustained_motion(t, speed)
    t_motion_last = last_sustained_motion(t, speed)
    hover_only = not np.isfinite(t_motion)
    use_filter = edge_u["kind"] in AIRBORNE_FILTER_KINDS
    inside = (t >= u["t_lo"] - 1e-9) & (t <= u["t_hi"] + 1e-9)
    rows = []
    for thr in SURFACE_THR:
        for dur in SURFACE_DUR:
            wins = find_windows(theta, t, thr, dur, valid=inside)
            air = wins if (hover_only or not use_filter) else [
                w for w in wins if t[w.start] >= t_motion and t[w.stop - 1] <= t_motion_last]
            rows.append({"kind": u["kind"], "unit": u["unit"], "thr_deg": thr,
                         "min_dur_s": dur, "n_windows": len(wins), "n_airborne": len(air),
                         "dynamic": u["meta"]["dynamic"]})
    # the quietest window of each duration, whatever the threshold: the availability curve
    for dur in SURFACE_DUR:
        best = np.nan
        n_need = max(int(round(dur / float(np.median(np.diff(t))))), 1)
        idx = np.flatnonzero(inside)
        if len(idx) >= n_need:
            maxt = np.array([np.nanmax(theta[idx[i:i + n_need]]) for i in
                             range(0, len(idx) - n_need + 1)])
            best = float(np.nanmin(maxt)) if len(maxt) else np.nan
        rows.append({"kind": u["kind"], "unit": u["unit"], "thr_deg": np.nan,
                     "min_dur_s": dur, "quietest_window_max_tilt_deg": best,
                     "dynamic": u["meta"]["dynamic"]})
    return rows


# ------------------------------------------------------------------ statistics
def summarize(rows: list[dict]) -> dict:
    out = {}
    for kind in ("sim", "sim_diffpos", "uzh", "tii"):
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        units = sorted({r["unit"] for r in sub})
        out[kind] = {"n_units": len(units), "tiers": {}}
        for tier in TIERS:
            tsub = [r for r in sub if r["tier"] == tier]
            have = [r for r in tsub if r["has_anchor"]]
            dyn = [r for r in tsub if r["meta_dynamic"]]
            dyn_have = [r for r in dyn if r["has_anchor"]]
            block = {
                "n": len(tsub), "n_anchor": len(have),
                "n_anchor_raw": int(sum(r["has_anchor_raw"] for r in tsub)),
                "coverage": f"{len(have)}/{len(tsub)}",
                "coverage_raw": f"{sum(r['has_anchor_raw'] for r in tsub)}/{len(tsub)}",
                "n_dynamic": len(dyn), "n_dynamic_anchor": len(dyn_have),
                "coverage_dynamic": f"{len(dyn_have)}/{len(dyn)}",
            }
            if have:
                err = np.asarray([r["anchor_err_deg"] for r in have], float)
                mod = np.asarray([r["model_check_deg"] for r in have], float)
                block["anchor_err_deg"] = {
                    "median": float(np.nanmedian(err)), "p90": float(np.nanquantile(err, 0.9)),
                    "max": float(np.nanmax(err))}
                block["model_check_deg"] = {
                    "median": float(np.nanmedian(mod)), "p90": float(np.nanquantile(mod, 0.9)),
                    "max": float(np.nanmax(mod))}
                block["theta_imp_at_anchor_deg"] = {
                    "median": float(np.nanmedian([r["theta_imp_at_anchor_deg"] for r in have])),
                    "p90": float(np.nanquantile([r["theta_imp_at_anchor_deg"] for r in have], .9))}
                block["anchor_frac"] = {
                    "median": float(np.nanmedian([r["anchor_frac"] for r in have])),
                    "p10": float(np.nanquantile([r["anchor_frac"] for r in have], .1)),
                    "p90": float(np.nanquantile([r["anchor_frac"] for r in have], .9))}
                block["n_windows"] = {"median": float(np.nanmedian([r["n_windows"] for r in have])),
                                      "max": int(np.nanmax([r["n_windows"] for r in have]))}
                block["frame_frac_quiet"] = float(np.mean([r["frame_frac_quiet"] for r in tsub]))
                if kind in ("sim", "sim_diffpos"):
                    block["lever_median_s"] = {
                        "median": float(np.nanmedian([r["lever_median_s"] for r in have])),
                        "p90": float(np.nanquantile([r["lever_median_s"] for r in have], .9)),
                        "max": float(np.nanmax([r["lever_median_s"] for r in have]))}
            out[kind]["tiers"][tier] = block
    return out


def surface_summary(surface: list[dict]) -> dict:
    """Coverage fraction over the descriptive threshold x duration grid, per dataset."""
    out = {}
    for kind in ("sim", "sim_diffpos", "uzh", "tii"):
        sub = [r for r in surface if r["kind"] == kind and np.isfinite(r.get("thr_deg", np.nan))]
        if not sub:
            continue
        out[kind] = {"all": {}, "dynamic": {}}
        for thr in SURFACE_THR:
            for dur in SURFACE_DUR:
                cell = [r for r in sub if r["thr_deg"] == thr and r["min_dur_s"] == dur]
                dcell = [r for r in cell if r["dynamic"]]
                out[kind]["all"][f"{thr:g}deg_{dur:g}s"] = (
                    f"{sum(r['n_airborne'] > 0 for r in cell)}/{len(cell)}")
                out[kind]["dynamic"][f"{thr:g}deg_{dur:g}s"] = (
                    f"{sum(r['n_airborne'] > 0 for r in dcell)}/{len(dcell)}")
        q = [(r["min_dur_s"], r["quietest_window_max_tilt_deg"])
             for r in surface if r["kind"] == kind
             and "quietest_window_max_tilt_deg" in r]
        out[kind]["quietest_window_max_tilt_deg"] = {
            f"{dur:g}s_{p}": float(np.nanquantile([v for d, v in q if d == dur], p))
            for dur in SURFACE_DUR for p in (0.1, 0.25, 0.5, 0.75, 0.9)}
    return out


def strata(rows: list[dict], tier: str) -> dict:
    """Coverage by family / mount / maneuver for the simulated set."""
    sub = [r for r in rows if r["kind"] == "sim" and r["tier"] == tier]
    out = {"family": {}, "mount": {}, "maneuver": {}}
    for key, field in (("family", "meta_family"), ("mount", "meta_mount"), ("maneuver",
                                                                          "meta_maneuver")):
        for val in sorted({r[field] for r in sub}):
            cell = [r for r in sub if r[field] == val]
            out[key][str(val)] = f"{sum(r['has_anchor'] for r in cell)}/{len(cell)}"
    feas_forest = [r for r in sub if r["meta_family"] == "forest_ridge" and r["meta_label"] ==
                   "feasible"]
    out["forest_feasible"] = f"{sum(r['has_anchor'] for r in feas_forest)}/{len(feas_forest)}"
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    units = list(iter_units())
    rows = []
    surface = []
    for u in units:
        rows += analyse_unit(u["kind"], u["unit"], u["t"], u["acc"], u["vel"], u["R_bw"],
                             u["t_lo"], u["t_hi"], u["R_cb"], u["meta"])
        surface += availability_surface(u)
    fields = sorted({k for r in rows for k in r})
    with open(OUT / "windows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    sfields = sorted({k for r in surface for k in r})
    with open(OUT / "surface.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields, extrasaction="ignore")
        w.writeheader()
        w.writerows(surface)
    summary = {"config": {"g": G_ANCHOR, "k_d": K_D_ANCHOR, "tiers": TIERS,
                          "selection": "min mean theta_imp, ties earliest",
                          "sg": SG, "scored_interval": "inside the scored frames only"},
               "overall": summarize(rows),
               "sim_strata": {t: strata(rows, t) for t in TIERS},
               "surface": surface_summary(surface),
               "criteria": {}}
    o = summary["overall"]
    p1 = o["sim"]["tiers"]["P1"]
    p1d = o["sim_diffpos"]["tiers"]["P1"]
    summary["criteria"]["availability_P1"] = {
        "criterion": ">=70% of 200 clips, >=60% of 131 dynamic, >=60% of 26 feasible forest",
        "all": p1["coverage"], "dynamic": p1["coverage_dynamic"],
        "forest_feasible": summary["sim_strata"]["P1"]["forest_feasible"],
        "pass": bool(p1["n_anchor"] / p1["n"] >= 0.70
                     and p1["n_dynamic_anchor"] / p1["n_dynamic"] >= 0.60
                     and (lambda s: int(s.split("/")[0]) / int(s.split("/")[1]) >= 0.60)(
                         summary["sim_strata"]["P1"]["forest_feasible"])),
    }
    if "anchor_err_deg" in p1:
        summary["criteria"]["accuracy_P1"] = {
            "criterion": "median <= 2.0 deg, p90 <= 3.0 deg, model check median <= 0.5 deg",
            "median": p1["anchor_err_deg"]["median"], "p90": p1["anchor_err_deg"]["p90"],
            "model_median": p1["model_check_deg"]["median"],
            "pass": bool(p1["anchor_err_deg"]["median"] <= 2.0
                         and p1["anchor_err_deg"]["p90"] <= 3.0
                         and p1["model_check_deg"]["median"] <= 0.5)}
    summary["criteria"]["diffpos_agreement_P1"] = {
        "coverage_stored_derivatives": p1["coverage"],
        "coverage_differentiated_positions": p1d["coverage"],
        "note": "pipeline independence of the simulated availability number"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary["criteria"], indent=1))
    for kind, block in summary["overall"].items():
        for tier, b in block["tiers"].items():
            print(kind, tier, b["coverage"], "dyn", b["coverage_dynamic"],
                  "err_med", b.get("anchor_err_deg", {}).get("median"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
