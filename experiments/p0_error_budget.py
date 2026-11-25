"""Track P, Part 0: error budget for the body-rate check and the pre-registered stop rule.

Pre-registration `configs/p0_error_budget_prereg.yaml` (committed before this run). Everything
here is computed from ground-truth paths and ground-truth attitudes plus synthetic noise matched
to the measured DPVO step noise; no video rotation is scored.

    PYTHONPATH=. pixi run python experiments/p0_error_budget.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.p_common import (WINDOWS, clip_score, literal_side, null_attitude, path_side,
                                  quantile, video_side, window_stats)
from experiments.p_units import scale_lean, sim_manifest, iter_units
from src.verdict.rate import (filter_gain, match_sigma, noisy_sequence, step_errors)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/p/p0"
DRAWS = 8
RNG = np.random.default_rng(1)
PRIMARY = 0.5


def measured_sigma(units: list[dict]) -> dict:
    """Per source: measured per-step increment error of the extractor and the matched sigma."""
    out = {}
    for u in units:
        tq = np.asarray(u["t_video"], float)
        R_true = np.einsum("nij,jk->nik", u["R_bw_cam_v"], u["R_cb"])
        e = step_errors(u["R_est"][: len(tq)], R_true, tq)
        out.setdefault(u["kind"], []).extend(e.tolist())
    res = {}
    for k, v in out.items():
        v = np.asarray(v)
        res[k] = {"step_err_median_deg": float(np.median(v)),
                  "step_err_p90_deg": float(np.quantile(v, 0.9)),
                  "sigma_deg": match_sigma(v), "n": int(len(v))}
    return res


def analyse(u: dict, sigma_deg: float) -> tuple[dict, list[dict], list[dict]]:
    path = path_side(u)
    row = {"kind": u["kind"], "unit": u["unit"], "sigma_deg": sigma_deg}
    # (i) required rate distribution from the 100 Hz Savitzky-Golay derivative
    r = np.linalg.norm(path["rb"][path["interior"]], axis=1)
    row.update({"req_med_deg_s": np.degrees(np.median(r)),
                "req_p90_deg_s": np.degrees(np.quantile(r, 0.9)),
                "req_p99_deg_s": np.degrees(np.quantile(r, 0.99))})
    # tighter filter, for the filter-effect measurement on real mocap
    tight = path_side(u, filter_window_s=0.11)
    rt = np.linalg.norm(tight["rb"][tight["interior"]], axis=1)
    row["req_med_tight_deg_s"] = np.degrees(np.median(rt))
    # noiseless null: self-consistency of the pipeline
    R_null = null_attitude(u, path)
    nul = video_side(u, path, attitude=R_null)
    ns = nul["res"]
    row["self_consistency_med_deg_s"] = float(np.nanmedian(ns["err_deg_s"]))
    row["gauge_fit_deg"] = nul["gauge"]["fit_deg"]
    row["gauge_spread_deg"] = nul["gauge"]["spread_deg"]
    # the handoff-literal form on the same noiseless feasible attitude
    lit = literal_side(u, path, R_null)
    row["literal_med_deg_s"] = float(np.nanmedian(lit["err_deg_s"]))
    # eight noisy realisations of the feasible attitude, reused by every window
    null_noisy = [video_side(u, path, attitude=noisy_sequence(nul["R_wc"], sigma_deg, RNG))["res"]
                  for _ in range(DRAWS)]
    clip_rows: dict[float, dict] = {}
    for w in WINDOWS:
        pooled = [x["abs_deg_s"] for res in null_noisy
                  for x in window_stats(res, w, min_req_deg_s=0.0)]
        clip_rows[w] = {"mdr_window_deg_s": quantile(pooled, 0.95),
                        "null_window_med_deg_s": quantile(pooled, 0.5),
                        "n_null_windows": len(pooled)}
        row[f"floor_w{w}"] = 3.0 * clip_rows[w]["null_window_med_deg_s"]
    floors = {w: row[f"floor_w{w}"] for w in WINDOWS}
    for w in WINDOWS:
        rows = window_stats(ns, w, min_req_deg_s=floors[w])
        clip_rows[w]["null_noiseless_med_deg_s"] = clip_score(rows)
        clip_rows[w]["req_med_window_deg_s"] = clip_score(rows, "req_deg_s")
        clip_rows[w]["n_windows"] = len(rows)
    # (iii) the labelled defect on this clip, from its rendered attitude
    d_rows = []
    if u["kind"] == "sim":
        tq = np.asarray(u["t_video"], float)
        R_def = np.einsum("nij,jk->nik", u["R_bw_cam_v"], u["R_cb"])
        d_res = video_side(u, path, attitude=R_def)["res"]
        row["defect"] = u["row"]["defect_family"]
        row["label"] = u["row"]["label"]
        row["violation"] = u["row"]["violation"]
        row["family"] = u["row"]["family"]
        row["mount"] = float(u["row"]["uptilt_deg"])
        row["peak_tilt_req_deg"] = float(u["row"]["peak_tilt_req_deg"])
        row["dynamic"] = int(row["peak_tilt_req_deg"] >= 15.0)
        row["partial"] = int(u["row"]["interval_start_s"] != "")
        lo = u["row"]["interval_start_s"]
        hi = u["row"]["interval_end_s"]
        def in_interval(rows, w):
            if lo == "":
                return rows
            return [x for x in rows if (x["lo"] / u["fps_video"] + w) >= float(lo)
                    and (x["lo"] / u["fps_video"]) <= float(hi)]
        def_noisy = [video_side(u, path,
                                attitude=noisy_sequence(R_def, sigma_deg, RNG))["res"]
                     for _ in range(DRAWS)]
        for w in WINDOWS:
            rows_def = in_interval(window_stats(d_res, w, min_req_deg_s=floors[w]), w)
            delta = clip_score(rows_def)
            det = [clip_score(in_interval(window_stats(res, w, min_req_deg_s=floors[w]), w))
                   > clip_rows[w]["mdr_window_deg_s"] for res in def_noisy]
            clip_rows[w].update({"defect_delta_deg_s": delta,
                                 "defect_detect_frac": float(np.mean(det))})
            d_rows.append({"kind": u["kind"], "unit": u["unit"], "window_s": w,
                           "mdr_window_deg_s": clip_rows[w]["mdr_window_deg_s"],
                           "defect_delta_deg_s": delta,
                           "defect_detect_frac": float(np.mean(det)),
                           "detectable": int(np.isfinite(delta) and
                                             delta >= clip_rows[w]["mdr_window_deg_s"])})
    else:
        d_rows.append({"kind": u["kind"], "unit": u["unit"], "window_s": PRIMARY,
                       "mdr_window_deg_s": clip_rows[PRIMARY]["mdr_window_deg_s"],
                       "defect_delta_deg_s": float("nan"), "defect_detect_frac": float("nan"),
                       "detectable": -1})
    rows = []
    for w in WINDOWS:
        rows.append({"kind": u["kind"], "unit": u["unit"], "window_s": w, **row,
                     **clip_rows[w]})
    return row, rows, d_rows


def scale_sweep(feasible: list[dict], sigmas: dict) -> pd.DataFrame:
    """Detection curve for the sim tilt-scale rule, per clip and scale, at the primary window."""
    rows = []
    grid = np.round(np.arange(0.5, 1.51, 0.1), 2)
    for u in feasible:
        path = path_side(u)
        R_null = null_attitude(u, path)
        pooled = []
        for _ in range(DRAWS):
            R = noisy_sequence(R_null, sigmas["sim"]["sigma_deg"], RNG)
            res = video_side(u, path, attitude=R)["res"]
            pooled.extend([x["abs_deg_s"] for x in window_stats(res, PRIMARY, 0.0)])
        mdr = quantile(pooled, 0.95)
        R_bw = u["R_bw_v"]
        for s in grid:
            R_def = np.einsum("nij,jk->nik", scale_lean(R_bw, float(s)), u["R_cb"])
            res = video_side(u, path, attitude=R_def)["res"]
            delta = clip_score(window_stats(res, PRIMARY, 0.0))
            rows.append({"clip_id": u["unit"], "scale": float(s), "abs_scale_error": abs(1.0 - float(s)),
                         "mdr_deg_s": mdr, "defect_delta_deg_s": delta,
                         "detectable": int(np.isfinite(delta) and delta >= mdr),
                         "peak_tilt_req_deg": float(u["row"]["peak_tilt_req_deg"]),
                         "family": u["row"]["family"], "mount": float(u["row"]["uptilt_deg"])})
    return pd.DataFrame(rows)


def summarise() -> dict:
    budget = pd.read_csv(OUT / "budget.csv")
    stop = pd.read_csv(OUT / "stop_rule.csv")
    sweep = pd.read_csv(OUT / "scale_sweep.csv")
    sim = iter_units("sim")
    man = sim_manifest("dev")
    dyn = stop[(stop.kind == "sim") & (stop.window_s == PRIMARY) & stop["detectable"].isin([0, 1])].copy()
    dyn["defect_family"] = [man[c]["defect_family"] for c in dyn["unit"]]
    dyn["dynamic"] = [float(man[c]["peak_tilt_req_deg"]) >= 15.0 for c in dyn["unit"]]
    pop = dyn[(dyn["defect_family"].isin(["tilt_removed", "mirrored_roll", "tilt_scaled"]))
              & (dyn["dynamic"])]
    stop_verdict = {"population": int(len(pop)),
                    "not_detectable": int((pop["detectable"] == 0).sum()),
                    "by_family": {k: f"{int(g['detectable'].sum())}/{len(g)}"
                                  for k, g in pop.groupby("defect_family")},
                    "fraction_not_detectable": float((pop["detectable"] == 0).mean())
                    if len(pop) else float("nan"),
                    "stop": bool(len(pop) and (pop["detectable"] == 0).sum() > len(pop) / 2)}
    summary = {
        "prereg": "configs/p0_error_budget_prereg.yaml",
        "filter": filter_gain(),
        "noise": measured_sigma(iter_units("sim") + iter_units("uzh") + iter_units("tii")),
        "stop_rule": stop_verdict,
        "budget_by_kind_window": {
            f"{k}_w{w}": {
                "n_units": int(((budget.kind == k) & (budget.window_s == w)).sum()),
                "req_med_deg_s": quantile(budget[(budget.kind == k) & (budget.window_s == w)]
                                          ["req_med_deg_s"], 0.5),
                "req_p90_deg_s": quantile(budget[(budget.kind == k) & (budget.window_s == w)]
                                          ["req_p90_deg_s"], 0.5),
                "null_window_med_deg_s": quantile(budget[(budget.kind == k) & (budget.window_s == w)]
                                                  ["null_window_med_deg_s"], 0.5),
                "mdr_window_deg_s": quantile(budget[(budget.kind == k) & (budget.window_s == w)]
                                             ["mdr_window_deg_s"], 0.5),
                "self_consistency_med_deg_s": quantile(budget[budget.kind == k]
                                                       ["self_consistency_med_deg_s"], 0.5),
                "literal_med_deg_s": quantile(budget[budget.kind == k]["literal_med_deg_s"], 0.5),
                "gauge_fit_deg": quantile(budget[budget.kind == k]["gauge_fit_deg"], 0.5),
            } for k in ("sim", "uzh", "tii") for w in WINDOWS
        },
        "req_by_family_w0.5": {
            k: {"n": int(len(g)),
                "req_med_deg_s": quantile(g["req_med_deg_s"], 0.5),
                "mdr_med_deg_s": quantile(g["mdr_window_deg_s"], 0.5)}
            for k, g in budget[(budget.kind == "sim") & (budget.window_s == PRIMARY)]
            .groupby("family")},
        "defect_delta_by_family": {},
        "scale_sweep": {
            "n_clips": int(sweep["clip_id"].nunique()),
            "detectable_by_abs_scale_error": {
                f"{round(float(e), 1):.1f}": f"{int(g['detectable'].sum())}/{len(g)}"
                for e, g in sweep.groupby(sweep["abs_scale_error"].round(1))},
        },
    }
    for (fam, dyn_), g in dyn.groupby(["defect_family", "dynamic"]):
        key = f"{fam}{'_dynamic' if dyn_ else '_static'}"
        summary["defect_delta_by_family"][key] = {
            "n": int(len(g)),
            "delta_med_deg_s": quantile(g["defect_delta_deg_s"], 0.5),
            "mdr_med_deg_s": quantile(g["mdr_window_deg_s"], 0.5),
            "detectable": f"{int(g['detectable'].sum())}/{len(g)}",
            "detect_frac_med": float(np.nanmedian(g["defect_detect_frac"])),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(run: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if run:
        run_all()
    print(json.dumps(summarise(), indent=1))


def run_all() -> None:
    sim = iter_units("sim")
    uzh = iter_units("uzh")
    tii = iter_units("tii")
    sig = measured_sigma(sim + uzh + tii)
    print("sigma", json.dumps(sig, indent=1))

    budget, stop, extra = [], [], []
    for group in (sim, uzh, tii):
        for u in group:
            _, rows, d_rows = analyse(u, sig[u["kind"]]["sigma_deg"])
            budget.extend(rows)
            stop.extend(d_rows)
    pd.DataFrame(budget).to_csv(OUT / "budget.csv", index=False)
    pd.DataFrame(stop).to_csv(OUT / "stop_rule.csv", index=False)
    sweep = scale_sweep([u for u in sim if u["row"]["label"] == "feasible"
                         and float(u["row"]["peak_tilt_req_deg"]) >= 15.0], sig)
    sweep.to_csv(OUT / "scale_sweep.csv", index=False)


if __name__ == "__main__":
    import sys
    main(run="--summarise" not in sys.argv)
