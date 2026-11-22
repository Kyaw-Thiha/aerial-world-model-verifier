"""Track K, Part 1: reconcile the calibration-mode labels of every published floor.

Re-runs A2's own implied-axis construction (``experiments/a2_modes.py``) in all four
modes, re-derives the schema-v2 lean semantics from the files, and recomputes J's F0
threshold under both readings.  Writes ``results/k01/``.

No frozen config or published report is edited.  CPU only.

    PYTHONPATH=. python experiments/k01_calibration_modes.py
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.physics.flatness import angle_between_deg, G0
from src.verdict.lean import world_to_yaw_frame

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/k01"
FLIGHTS = ["flight-01a-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
           "flight-13a-trackRATM", "flight-01p-ellipse", "flight-07p-lemniscate",
           "flight-13p-trackRATM", "flight-16p-trackRATM"]
DEV = {"flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse",
       "flight-07p-lemniscate", "flight-13a-trackRATM", "flight-16p-trackRATM"}
K0 = 0.295


def jload(p):
    return json.loads(Path(p).read_text())


def load_gate() -> dict[str, bool]:
    raw = (ROOT / "configs/gate_frozen.yaml").read_text()
    cfg = json.loads("\n".join(l for l in raw.splitlines() if not l.strip().startswith("#")))
    rows = list(csv.DictReader(open(ROOT / "results/a2_gate/20261109T201449Z_298ff20/"
                                    "clip_table.csv")))
    return {r["clip_id"]: float(r["identity::S3_latitude_confidence"]) <= float(cfg["tau"])
            for r in rows}


def clip_records() -> list[dict]:
    """Every TII clip with its stored schema-v2 arrays, truth attitude and metadata."""
    recs = []
    from scipy.spatial.transform import Rotation
    from src.depicted.clips import up_body_from_lean
    from src.depicted.stat_a2 import depicted_axis_from_up_body
    for f in FLIGHTS:
        spec = np.load(ROOT / f"data/processed/p3a/p3a_v1/{f}.npz")
        R = Rotation.from_quat(spec["quat"]).as_matrix()
        z_all = R[:, :, 2]
        for d in sorted((ROOT / f"data/processed/track_d/clips/{f}").iterdir()):
            if not d.is_dir():
                continue
            z = np.load(d / "verdict_clip.npz")
            cond = d.name.split("__")[-1]
            i = np.clip(np.searchsorted(spec["t"], z["t"]), 0, len(spec["t"]) - 1)
            yaw = z["yaw"]
            lean_act = world_to_yaw_frame(z_all[i][:, :2], yaw)
            ax_act = depicted_axis_from_up_body(up_body_from_lean(lean_act), yaw)
            ax_dep = depicted_axis_from_up_body(up_body_from_lean(z["depicted_lean"]), yaw)
            recs.append({"flight": f, "clip_id": d.name.split("__")[0], "cond": cond,
                         "depicted_lean": z["depicted_lean"],
                         "lean_req": z["lean_req"], "vel_req": z["vel_req"],
                         "yaw": yaw, "act_axis": ax_act, "dep_axis": ax_dep,
                         "tilt_req_deg": z["tilt_req_deg"]})
    return recs


def required_axes(rec: dict, k_d: float) -> np.ndarray:
    from src.depicted.clips import up_body_from_lean
    from src.depicted.stat_a2 import depicted_axis_from_up_body
    lean = rec["lean_req"] + k_d * rec["vel_req"]
    return depicted_axis_from_up_body(up_body_from_lean(lean), rec["yaw"])


def lag_apply(axis: np.ndarray, t: np.ndarray, lag: float) -> np.ndarray:
    if abs(lag) < 1e-12:
        return axis
    out = np.column_stack([np.interp(t + lag, t, axis[:, j], left=np.nan, right=np.nan)
                           for j in range(3)])
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(np.isfinite(n) & (n > 1e-9), n, np.nan)


def clip_score(rec: dict, k_d: float, fit_lag: bool, t: np.ndarray) -> float:
    ax_req = required_axes(rec, k_d)
    ax_dep = rec["dep_axis"]
    best = None
    lags = np.arange(-0.150, 0.1501, 0.025) if fit_lag else [0.0]
    for lag in lags:
        r = angle_between_deg(ax_dep, lag_apply(ax_req, t, lag))
        v = float(np.nanmedian(r))
        if best is None or (np.isfinite(v) and v < best):
            best = v
    return best


def exact_clip_scores(recs: list[dict], k_d: float, fit_lag: bool) -> np.ndarray:
    """A2's exact nonlinear construction: normalise(a + g e3 + k_d v) from native mocap."""
    from scipy.signal import savgol_filter
    from scipy.spatial.transform import Rotation
    from src.datasets.tii import load_tii
    from src.datasets.tii_video import flight_parts
    from src.depicted.clips import up_body_from_lean
    from src.depicted.stat_a2 import depicted_axis_from_up_body
    from src.verdict.lean import world_to_yaw_frame

    out = []
    for f in FLIGHTS:
        control, _ = flight_parts(f)
        clip = load_tii(ROOT / f"data/raw/tii/{control}/mocap_{f}.csv", name=f)
        dt = 1.0 / clip.fps
        v = savgol_filter(clip.pos, 101, 3, deriv=1, delta=dt, axis=0, mode="interp")
        a = savgol_filter(clip.pos, 101, 3, deriv=2, delta=dt, axis=0, mode="interp")
        for r in [x for x in recs if x["flight"] == f]:
            z = np.load(ROOT / f"data/processed/track_d/clips/{f}/"
                               f"{r['clip_id']}__real/verdict_clip.npz")
            tq = np.clip(z["t"], clip.t[0], clip.t[-1])
            vv = np.column_stack([np.interp(tq, clip.t, v[:, j]) for j in range(3)])
            aa = np.column_stack([np.interp(tq, clip.t, a[:, j]) for j in range(3)])
            u = aa + np.array([0.0, 0.0, G0]) + k_d * vv
            u = u / np.linalg.norm(u, axis=1, keepdims=True)
            ax_req = depicted_axis_from_up_body(up_body_from_lean(
                world_to_yaw_frame(u[:, :2], r["yaw"])), r["yaw"])
            best = None
            lags = np.arange(-0.150, 0.1501, 0.025) if fit_lag else [0.0]
            for lag in lags:
                rr = angle_between_deg(r["dep_axis"], lag_apply(ax_req, z["t"], lag))
                val = float(np.nanmedian(rr))
                if best is None or (np.isfinite(val) and val < best):
                    best = val
            out.append(best)
    return np.array(out, float)


def lofo_median_threshold(values: np.ndarray, flights: np.ndarray,
                          heldout: set[str]) -> tuple[float, list[float]]:
    """J03's LOFO threshold: one 95th percentile per real clip from the other flights,
    then the median over clips (clip-frequency weighted, which is how J03 reports it)."""
    taus = []
    dev = np.array([f not in heldout for f in flights])
    for i, f in enumerate(flights):
        fit = values[dev & (flights != f)]
        fit = fit[np.isfinite(fit)]
        if len(fit) >= 5:
            taus.append(float(np.quantile(fit, 0.95)))
    return (float(np.median(taus)) if taus else float("nan")), taus


def schema_semantics(recs: list[dict]) -> dict:
    """Is the stored lean_req drag-free or k_d=0.295?  Compared with a direct recompute."""
    from scipy.signal import savgol_filter
    from scipy.spatial.transform import Rotation
    from src.datasets.tii import load_tii
    from src.datasets.tii_video import flight_parts
    from src.verdict.lean import world_to_yaw_frame

    diffs = {}
    for f in FLIGHTS:
        control, _ = flight_parts(f)
        clip = load_tii(ROOT / f"data/raw/tii/{control}/mocap_{f}.csv", name=f)
        dt = 1.0 / clip.fps
        w = 101
        v = savgol_filter(clip.pos, w, 3, deriv=1, delta=dt, axis=0, mode="interp")
        a = savgol_filter(clip.pos, w, 3, deriv=2, delta=dt, axis=0, mode="interp")
        for d in sorted((ROOT / f"data/processed/track_d/clips/{f}").iterdir()):
            if not d.is_dir() or not d.name.endswith("__real"):
                continue
            z = np.load(d / "verdict_clip.npz")
            tq = np.clip(z["t"], clip.t[0], clip.t[-1])
            vv = np.column_stack([np.interp(tq, clip.t, v[:, j]) for j in range(3)])
            aa = np.column_stack([np.interp(tq, clip.t, a[:, j]) for j in range(3)])
            yaw = z["yaw"]
            u0 = aa + np.array([0.0, 0.0, G0])
            u0 /= np.linalg.norm(u0, axis=1, keepdims=True)
            u1 = aa + np.array([0.0, 0.0, G0]) + K0 * vv
            u1 /= np.linalg.norm(u1, axis=1, keepdims=True)
            l0 = world_to_yaw_frame(u0[:, :2], yaw)
            l1 = world_to_yaw_frame(u1[:, :2], yaw)
            m = np.isfinite(z["lean_req"]).all(1) & np.isfinite(l0).all(1)
            diffs.setdefault("stored_vs_dragfree", []).extend(
                np.degrees(np.linalg.norm(z["lean_req"][m] - l0[m], axis=1)).tolist())
            diffs.setdefault("stored_vs_kd295", []).extend(
                np.degrees(np.linalg.norm(z["lean_req"][m] - l1[m], axis=1)).tolist())
            m2 = m & np.isfinite(z["vel_req"]).all(1)
            diffs.setdefault("stored_plus_kd295vel_vs_kd295", []).extend(
                np.degrees(np.linalg.norm(
                    z["lean_req"][m2] + K0 * z["vel_req"][m2] - l1[m2], axis=1)).tolist())
            # the schema's linear reparameterisation against the exact flatness lean
            gap = np.degrees(np.linalg.norm(
                z["lean_req"][m2] + K0 * z["vel_req"][m2] - l1[m2], axis=1))
            diffs.setdefault("linear_vs_exact_all", []).extend(gap.tolist())
            test = m2 & (np.asarray(z["tilt_req_deg"], float) >= 10.0)
            qnorm = np.linalg.norm(aa + np.array([0.0, 0.0, G0]), axis=1)
            test = test & (qnorm >= 2.0)
            diffs.setdefault("linear_vs_exact_testable", []).extend(
                np.degrees(np.linalg.norm(
                    z["lean_req"][test] + K0 * z["vel_req"][test] - l1[test],
                    axis=1)).tolist())
    out = {}
    for k, vv in diffs.items():
        vv = np.asarray(vv)
        out[k] = {"n": int(len(vv)), "median_deg": float(np.median(vv)),
                  "p95_deg": float(np.percentile(vv, 95)),
                  "p99_deg": float(np.percentile(vv, 99)), "max_deg": float(vv.max())}
    out["conclusion"] = ("lean_req is the DRAG-FREE required lean; the required side at "
                         "k_d=0.295 is lean_req + 0.295*vel_req")
    return out


def _huber_offset(dep: np.ndarray, req: np.ndarray, delta: float, iters: int = 6) -> np.ndarray:
    r = dep - req
    ok = np.isfinite(r).all(1)
    if ok.sum() < 4:
        return np.zeros(2)
    r = r[ok]
    o = np.median(r, axis=0)
    for _ in range(iters):
        e = np.linalg.norm(r - o, axis=1)
        w = np.where(e <= delta, 1.0, delta / np.maximum(e, 1e-12))
        o = np.sum(w[:, None] * r, axis=0) / np.sum(w)
    return o


def _huber_loss(dep: np.ndarray, req: np.ndarray, delta_deg: float) -> float:
    o = _huber_offset(dep, req, np.sin(np.deg2rad(delta_deg)))
    e = np.linalg.norm(dep - req - o, axis=1)
    e = e[np.isfinite(e)]
    d = np.deg2rad(delta_deg)
    h = np.where(e <= d, 0.5 * e ** 2, d * (e - 0.5 * d))
    return float(np.mean(h)) if len(h) else np.inf


def j03_drag_diagnostics(recs: list[dict], gate: dict[str, bool]) -> dict:
    """J03's per-clip Huber drag fit, with the schema read physically.

    J03 fitted ``lean_req + (k - 0.295) vel_req`` on k in [0, 0.40]. Here the same fit is
    run on ``lean_req + k vel_req`` with k on the same physical grid, so the fitted value
    is the drag coefficient, not an offset from 0.295.
    """
    grid = np.round(np.arange(0.0, 0.401, 0.05), 3)
    ks = []
    for r in recs:
        if r["cond"] != "real" or not gate.get(r["clip_id"], False):
            continue
        dep, req, vel = r["depicted_lean"], r["lean_req"], r["vel_req"]
        ok = np.isfinite(dep).all(1) & np.isfinite(req).all(1) & np.isfinite(vel).all(1)
        loss = [(_huber_loss(dep[ok], (req[ok] + k * vel[ok]), 3.0), k) for k in grid]
        ks.append(min(loss)[1])
    hist = {f"{k:.2f}": int(sum(1 for x in ks if abs(x - k) < 1e-9)) for k in grid}
    return {"n": len(ks), "grid": [float(k) for k in grid], "histogram": hist,
            "median": float(np.median(ks)) if ks else float("nan"),
            "p10": float(np.percentile(ks, 10)) if ks else float("nan"),
            "p90": float(np.percentile(ks, 90)) if ks else float("nan"),
            "note": "physical k_d grid; compare J03's shifted grid [0,0.40] = "
                    "k_d,eff [-0.295,+0.105]"}


def main() -> int:
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT / run
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. A2's own construction in all four modes
    import experiments.a2_modes as modes
    prereg = yaml.safe_load((ROOT / "configs/verdict_preregistration.yaml").read_text())
    cfg = yaml.safe_load((ROOT / "configs/gate_frozen.yaml").read_text())
    a2 = {}
    for k_d in modes.K_DS:
        for mode in modes.MODES:
            recs = modes.tii_records(ROOT / "results/gate_a2/tii", k_d, mode, prereg, cfg)
            pooled = modes.thresholds_for(recs, prereg, holdout=None, policy="all_flights")
            folds = {f: modes.thresholds_for(recs, prereg, holdout=f, policy="all_flights")
                     ["pooled"]["threshold"] for f in modes.FLIGHTS}
            real = [r for r in recs if r["condition"] == "identity" and r["testable"]]
            a2[f"kd{k_d}_{mode}"] = {
                "pooled_threshold": float(pooled["pooled"]["threshold"]),
                "pooled_n": int(pooled["pooled"]["n"]),
                "fold_spread": [float(min(folds.values())), float(max(folds.values()))],
                "real_median_residual": float(np.median([r["score"] for r in real])),
                "real_n": len(real),
            }
    # ANWM nulls in the same modes (copy of a2_modes' own construction)
    by_clip, man, floor_rows = modes.anwm_frames()
    cruise = [c for c in by_clip if (man[c].get("meta") or {}).get("maneuver") == "cruise"]
    from src.perception.frames import camera_to_body_pitched
    from src.score.verdict import depicted_axis_world
    from src.physics.flatness import tilt_deg
    gt_up = np.asarray([r["gravity_up_cam"] for r in floor_rows])
    gt_tilt = tilt_deg(depicted_axis_world(gt_up, camera_to_body_pitched(0.0),
                                           np.zeros(len(gt_up))))
    nulls = {}
    for k_d in modes.K_DS:
        vals = []
        for c in cruise:
            rc = by_clip[c]
            dep = tilt_deg(depicted_axis_world(
                np.asarray([r["gravity_up_cam"] for r in rc]),
                camera_to_body_pitched(0.0), np.zeros(len(rc))))
            pos = np.asarray(man[c]["commanded_path"]["positions"], float)
            try:
                req = modes.frame_required(pos, 1.0, k_d, "spline")
            except ValueError:
                continue
            n = min(len(req), len(dep))
            r = np.abs(dep[:n] - req[:n])
            vals.append(r[np.isfinite(r)])
        v = np.concatenate(vals) if vals else np.array([])
        nulls[f"kd{k_d}"] = {"n": int(v.size), "median": float(np.median(v)),
                             "p95": float(np.percentile(v, 95)),
                             "p99": float(np.percentile(v, 99))}
    for k, v in a2.items():
        k_d = 0.0 if "_lag_only" in k and k.startswith("kd0.0") else (
            0.295 if k.startswith("kd0.295") else None)
        mode_kd = float(k.split("_")[0][2:])
        nl = nulls[f"kd{mode_kd}"]
        v["anwm_cruise_p99"] = nl["p99"]
        v["primary_threshold"] = float(max(v["pooled_threshold"], nl["p99"],
                                           float(np.percentile(gt_tilt, 99))))
        v["detection_floor"] = float(v["primary_threshold"] + nl["median"])
    gt_level = {"n": int(len(gt_tilt)), "median": float(np.median(gt_tilt)),
                "p99": float(np.percentile(gt_tilt, 99))}

    # ---- 2. schema semantics
    recs = clip_records()
    sem = schema_semantics(recs)
    gate = load_gate()

    # ---- 3. J03's F0 under both readings, gate-kept dev, no lag and with the lag nuisance
    real = [r for r in recs if r["cond"] == "real" and r["clip_id"] in gate
            and gate[r["clip_id"]] and r["flight"] in DEV]
    f0 = {}
    flights_arr = np.array([r["flight"] for r in real])
    for tag, k_d in (("drag_free", 0.0), ("kd_0.295", 0.295)):
        # J03's F0 has no lag; the frozen verdict fits it. Report both.
        t = np.arange(len(real[0]["yaw"])) * 0.083  # only length matters for the no-lag arm
        vals_nolag = np.array([clip_score(r, k_d, False, t) for r in real])
        vals_lag = []
        for r in real:
            z = np.load(ROOT / f"data/processed/track_d/clips/{r['flight']}/"
                               f"{r['clip_id']}__real/verdict_clip.npz")
            vals_lag.append(clip_score(r, k_d, True, z["t"]))
        vals_lag = np.array(vals_lag)
        thr_n, taus_n = lofo_median_threshold(vals_nolag, flights_arr, {"flight-13p-trackRATM",
                                                                        "flight-07a-lemniscate"})
        thr_l, taus_l = lofo_median_threshold(vals_lag, flights_arr, {"flight-13p-trackRATM",
                                                                      "flight-07a-lemniscate"})
        f0[tag] = {"k_d": k_d, "median_residual": float(np.nanmedian(vals_nolag)),
                   "threshold_no_lag": thr_n, "threshold_with_lag": thr_l,
                   "n": len(real), "fold_taus_no_lag": taus_n}

    exact = {}
    for tag, k_d in (("drag_free", 0.0), ("kd_0.295", 0.295)):
        vals_n = exact_clip_scores(real, k_d, False)
        vals_l = exact_clip_scores(real, k_d, True)
        thr_n, _ = lofo_median_threshold(vals_n, flights_arr,
                                         {"flight-13p-trackRATM", "flight-07a-lemniscate"})
        thr_l, _ = lofo_median_threshold(vals_l, flights_arr,
                                         {"flight-13p-trackRATM", "flight-07a-lemniscate"})
        exact[tag] = {"k_d": k_d, "median_residual_no_lag": float(np.nanmedian(vals_n)),
                      "threshold_no_lag": thr_n, "threshold_with_lag": thr_l, "n": len(real)}

    j03_drag = j03_drag_diagnostics(recs, gate)
    doc = {"run": run, "a2_modes": a2, "exact_f0": exact, "j03_drag_corrected": j03_drag, "anwm_nulls": nulls, "gt_level": gt_level,
           "schema_semantics": sem, "j03_f0": f0}
    (out / "k01.json").write_text(json.dumps(doc, indent=1, default=_jsonable))
    _mapping(out, doc)
    print(json.dumps(doc["a2_modes"], indent=1))
    print(json.dumps(doc["schema_semantics"], indent=1))
    print(json.dumps(doc["j03_f0"], indent=1))
    print(json.dumps(doc["j03_drag_corrected"], indent=1))
    print("wrote", out)
    return 0


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def _mapping(out: Path, doc: dict) -> None:
    a = doc["a2_modes"]
    ex = doc["exact_f0"]
    j = doc["j03_f0"]
    rows = [("value_deg", "quantity", "construction", "mode_k_d", "nuisance",
             "population", "source", "label_status", "note")]
    rows += [
        (f"{a['kd0.0_lag_only']['pooled_threshold']:.2f}", "conformal threshold",
         "A2 implied-axis, per-clip median residual", "0 (drag-free)",
         "lag only (uptilt offset fixed 0)", "8 flights, gate per clip, 58 testable",
         "a2_modes 20261110T165343Z (A2 branch 71fb651) / reproduced 20261122T071523Z (e7e3d84)", "correct",
         "report 07 amendment 9"),
        (f"{a['kd0.0_full_parity']['pooled_threshold']:.2f}", "conformal threshold",
         "A2 implied-axis", "0 (drag-free)", "full parity (uptilt + lag)",
         "8 flights, gate per clip", "a2_modes (A2 branch 71fb651)", "correct", ""),
        (f"{a['kd0.295_lag_only']['pooled_threshold']:.2f}", "conformal threshold",
         "A2 implied-axis", "0.295", "lag only", "8 flights, gate per clip",
         "a2_modes (A2 branch 71fb651)", "correct", ""),
        (f"{a['kd0.295_full_parity']['pooled_threshold']:.2f}", "conformal threshold (TII)",
         "A2 implied-axis", "0.295", "full parity", "8 flights, gate per clip",
         "a2_modes", "correct",
         f"binding primary is the ANWM cruise p99 {a['kd0.295_full_parity']['anwm_cruise_p99']:.2f}"),
        (f"{a['kd0.0_lag_only']['detection_floor']:.1f}", "detection floor",
         "threshold + ANWM cruise median", "0 (drag-free)", "lag only",
         "8 flights + ANWM cruise", "a2_modes (A2 branch 71fb651)", "correct", ""),
        (f"{a['kd0.0_full_parity']['detection_floor']:.1f}", "detection floor",
         "threshold + ANWM cruise median", "0 (drag-free)", "full parity",
         "8 flights + ANWM cruise", "a2_modes (A2 branch 71fb651)", "correct", ""),
        (f"{a['kd0.295_lag_only']['detection_floor']:.1f}", "detection floor",
         "threshold + ANWM cruise median", "0.295", "lag only",
         "8 flights + ANWM cruise", "a2_modes (A2 branch 71fb651)", "correct", ""),
        (f"{a['kd0.295_full_parity']['detection_floor']:.1f}", "detection floor",
         "threshold + ANWM cruise median", "0.295", "full parity",
         "8 flights + ANWM cruise", "a2_modes (A2 branch 71fb651)", "correct", ""),
        ("1.24", "pooled held-out median residual (physics floor)",
         "E1 flatness vs mocap attitude", "fitted per fold (~0.297)", "none",
         "8 flights, 114,059 frames",
         "phase0_floor 20261108T044153Z (f3b5d2bf) / reproduced 20261122T072349Z (41d1dad)", "correct", ""),
        ("5.46", "pooled held-out median residual (drag-free physics floor)",
         "E1 flatness vs mocap attitude", "0", "none", "8 flights, 114,059 frames",
         "phase0_floor (f3b5d2bf)", "correct", ""),
        ("1.22", "body-frame x/y drag diagnostic (median residual)",
         "E1 body drag vs mocap attitude", "body k_xy fitted", "none", "8 flights",
         "phase0_floor", "correct", "diagnostic only"),
        ("14.46", "LOFO threshold, mocap-attitude anchor",
         "J02 arm (c): mocap attitude + synthetic error (pre-f7ba108 code, required=lean_req)",
         "0 (drag-free)", "none", "100 TII clips, 8 flights",
         "track-j <f7ba108", "MODE LABEL WRONG",
         "J02 calls this k_d=0.295; lean_req is drag-free, so this is the drag-free arm"),
        ("31.11", "LOFO threshold, mocap-attitude anchor",
         "J02 arm (c) with ARM_B_KD=0.0", "-0.295 (negative drag)",
         "none", "100 TII clips, 8 flights", "track-j f7ba108 (ARM_B_KD=0.0)",
         "MODE LABEL WRONG", "J02 calls this drag-free; it is k_d=-0.295"),
        ("6.46", "fixed cut used as an operating point", "A2 inferred-axis score",
         "0.295", "full parity", "8 flights, testable pairs",
         "report 04 / report 07 section 3", "correct (as a fixed cut)",
         "withdrawn as a cross-k_d comparison in report 07 amendment 8"),
        ("5.76-12.95", "held-out per-fold threshold range", "A2 implied-axis",
         "0.295", "varies by fold", "8 flights, gate per clip", "report 07 section 3",
         "correct", "the range each pair inherits at k_d .295"),
        ("10.52", "MDV at 8 frames, synthetic centre point", "J02 window sweep",
         "0 (drag-free)", "none", "100 TII clips", "track-j f7ba108",
         "MODE LABEL WRONG", "J02's table is headed k_d = 0.295; it is drag-free"),
        ("9.04", "LOFO threshold at the synthetic centre point", "J02 grid (sigma 4.8, bias 2)",
         "0 (drag-free)", "none", "100 TII clips", "track-j f7ba108",
         "MODE LABEL WRONG", "labelled k_d = 0.295"),
        ("7.66", "LOFO threshold, measured-error replay", "J02 replay",
         "0 (drag-free)", "none", "100 TII clips", "track-j f7ba108",
         "MODE LABEL WRONG", "labelled k_d = 0.295"),
        ("15.91", "LOFO median threshold, C1 (offset only)", "J03 C1", "0 (drag-free)",
         "per-clip Huber offset", "gate-kept dev 69", "track-j f7ba108",
         "MODE LABEL WRONG", "the offset arm also used the drag-free required side"),
        ("13.84", "LOFO median threshold, C2 (offset + k_d)", "J03 C2",
         "linear grid shifted by -0.295", "offset + bounded k_d",
         "gate-kept dev 69", "track-j f7ba108", "MODE LABEL WRONG",
         "k=0.40 means k_d,eff=+0.105; k=0 means -0.295; and the term is linearised"),
        ("14.05", "LOFO median threshold, F0", "J03 F0", "0 (drag-free)", "none",
         "gate-kept six dev flights, 69 clips", "track-j f7ba108", "MODE LABEL WRONG",
         f"our exact reconstruction gives {ex['drag_free']['threshold_no_lag']:.2f}"),
        (f"{j['kd_0.295']['threshold_no_lag']:.2f}",
         "LOFO median threshold, F0, schema linear reparameterisation",
         "clean lean_req + 0.295*vel_req (J03's intended reading)", "0.295 linearised",
         "none", "gate-kept six dev flights, 69 clips", "this run (K01 script)",
         "proposed correction (linearised)",
         "not the frozen construction; the linearisation overstates the threshold"),
        (f"{ex['kd_0.295']['threshold_no_lag']:.2f}",
         "LOFO median threshold, F0, exact construction",
         "normalise(a + g e3 + 0.295 v) from native mocap", "0.295 exact",
         "none", "gate-kept six dev flights, 69 clips", "this run (K01 script)",
         "proposed correction (exact)", "the construction A2's 8.96 uses"),
        (f"{ex['kd_0.295']['threshold_with_lag']:.2f}",
         "LOFO median threshold, F0, exact + lag",
         "normalise(a + g e3 + 0.295 v), per-clip lag nuisance", "0.295 exact",
         "lag", "gate-kept six dev flights, 69 clips", "this run (K01 script)",
         "proposed correction (exact, lag)", ""),
        ("6.9 / 4.9", "whole-clip MDV, perfect path + synthetic error",
         "J02 calibration_mode arms", "6.9: drag-free; 4.9: k_d=-0.295", "none",
         "100 TII clips", "track-j f7ba108", "MODE LABEL WRONG",
         "J02 labels 6.9 as k_d=0.295 and 4.9 as drag-free"),
        ("10.65", "whole-clip MDV, measured reader error replay", "J02 replay",
         "0 (drag-free)", "none", "100 TII clips", "track-j f7ba108", "MODE LABEL WRONG",
         "labelled k_d=0.295"),
        ("6.88 / 8.35", "perfect-path floor, whole clip / 8-frame window", "J02",
         "0 (drag-free)", "none", "100 TII clips", "track-j f7ba108", "MODE LABEL WRONG",
         "labelled k_d=0.295"),
    ]
    with open(out / "mapping.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
