"""Track N, Part 0: the rotation-propagation pre-test.

Pre-registration: ``configs/n01_pretest_prereg.yaml`` (committed before this ran). This script
computes only what that file defines:

* per-unit propagated-lean error at 1, 5, 10 and 20 s with an ORACLE initial gravity,
  ``u_prop = (R_est[j]^T R_est[i]) u_true[i]`` against ``u_true[j]`` (angle in degrees);
* the same-unit reader error (sim: L01 frozen reader; UZH: E01 variant a; TII: frozen reader);
* the extractor failure reasons and rates;
* the pre-registered decision rule, applied as written.

Usage
-----
PYTHONPATH=. python experiments/n01_rotation_pretest.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from src.depicted.uzh import published_mount
from src.verdict.lean import depicted_lean_from_gravity, lean_to_roll_pitch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/n01_pretest"
POSES = ROOT / "data/n01/poses"
UZH_READER = ROOT / "data/n01/uzh_reader"
L01 = Path("/home/kevin/Documents/Projects/awmv-l/data/l01")
E3 = np.array([0.0, 0.0, 1.0])

HORIZONS = (1, 5, 10, 20)
UZH_OUTDOOR = ["outdoor_forward_1", "outdoor_forward_3", "outdoor_forward_5"]
UZH_INDOOR = ["indoor_forward_3", "indoor_forward_5", "indoor_forward_6",
              "indoor_forward_7", "indoor_forward_9", "indoor_forward_10"]
UZH_45 = ["indoor_45_2", "indoor_45_4", "indoor_45_9", "indoor_45_13", "outdoor_45_1"]
TII = ["flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
       "flight-07p-lemniscate", "flight-13a-trackRATM", "flight-13p-trackRATM",
       "flight-16p-trackRATM"]
TII_PILOTED = {"flight-01p-ellipse", "flight-07p-lemniscate", "flight-13p-trackRATM",
               "flight-16p-trackRATM"}
# E01's published per-sequence medians, variant a (pinhole + Kalibr focal prior), for the
# pipeline check. Transcribed from reports/E01_uzh_audit_and_perframe.md, "Per sequence".
E01_VARIANT_A = {"indoor_forward_3": 0.99, "indoor_forward_5": 0.71, "indoor_forward_6": 1.19,
                 "indoor_forward_7": 1.82, "indoor_forward_9": 0.94,
                 "indoor_forward_10": 0.89, "outdoor_forward_1": 8.36, "outdoor_forward_3": 7.90,
                 "outdoor_forward_5": 9.59}


# --------------------------------------------------------------------------- geometry
def up_of(R_cw: np.ndarray) -> np.ndarray:
    """World-up in the camera frame for camera-to-world rotations (n,3,3)."""
    return np.einsum("nji,j->ni", np.asarray(R_cw, float), E3)


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0)))


def horizon_errors(R_est: np.ndarray, R_true: np.ndarray, t: np.ndarray, h: float,
                   fps: float, R_cb: np.ndarray | None) -> dict:
    """Propagate the true up from every start with the estimated rotations; error at h."""
    n = min(len(t), len(R_est), len(R_true))
    t = np.asarray(t, float)[:n]
    R_est, R_true = np.asarray(R_est, float)[:n], np.asarray(R_true, float)[:n]
    up_t = up_of(R_true)
    idx = np.searchsorted(t, t + h)
    j = np.minimum(idx, n - 1)
    ok = (idx < n) & (j > np.arange(n)) & (np.abs(t[j] - t - h) <= 1.5 / fps)
    if not ok.any():
        return {"n": 0, "median_deg": float("nan"), "p90_deg": float("nan"),
                "median_pitch_deg": float("nan"), "median_roll_deg": float("nan"),
                "errors_deg": np.array([])}
    i = np.arange(n)[ok]
    jj = j[ok]
    rel_est = np.einsum("nji,njk->nik", R_est[jj], R_est[i])
    u_i = up_t[i]
    pred = np.einsum("nij,nj->ni", rel_est, u_i)
    u_j = up_t[jj]
    err = angle_deg(pred, u_j)
    lean_p = lean_rp(pred, R_cb)
    lean_r = lean_rp(u_j, R_cb)
    d = lean_p - lean_r
    return {"n": int(len(err)), "median_deg": float(np.median(err)),
            "p90_deg": float(np.percentile(err, 90)),
            "median_pitch_deg": float(np.median(np.abs(d[:, 0]))),
            "median_roll_deg": float(np.median(np.abs(d[:, 1]))),
            "errors_deg": err}


def lean_rp(up_cam: np.ndarray, R_cb: np.ndarray | None) -> np.ndarray:
    """Lean (pitch, roll) in degrees from up in the camera frame; identity mount if None."""
    R = np.eye(3) if R_cb is None else np.asarray(R_cb, float)
    lean = depicted_lean_from_gravity(np.asarray(up_cam, float), R)
    pitch, roll = lean_to_roll_pitch(lean)
    return np.column_stack([np.degrees(np.asarray(pitch, float)),
                            np.degrees(np.asarray(roll, float))])


def steps(R_est: np.ndarray, R_true: np.ndarray) -> tuple[float, float]:
    """Median per-step relative rotation of the estimate and of the truth (degrees)."""
    if len(R_est) < 2 or len(R_true) < 2:
        return float("nan"), float("nan")
    re = np.einsum("nji,njk->nik", R_est[:-1], R_est[1:])
    rt = np.einsum("nji,njk->nik", R_true[:-1], R_true[1:])
    a = np.degrees(np.linalg.norm(Rotation.from_matrix(re).as_rotvec(), axis=1))
    b = np.degrees(np.linalg.norm(Rotation.from_matrix(rt).as_rotvec(), axis=1))
    return float(np.median(a)), float(np.median(b))


# --------------------------------------------------------------------------- units
def load_poses(kind: str, unit: str) -> dict | None:
    p = POSES / kind / f"{unit}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {"c2w": np.asarray(d["c2w"], float), "n_input": int(d["n_input"]),
            "n_pose": int(d["n_pose"]), "fps": float(d["fps"])}


def tii_mocap(flight: str, ts_us: np.ndarray) -> np.ndarray:
    import pandas as pd

    kind = "piloted" if flight in TII_PILOTED else "autonomous"
    path = ROOT / f"data/raw/tii/{kind}/mocap_{flight}.csv"
    df = pd.read_csv(path, usecols=["timestamp"] + [f"drone_rot[{i}]" for i in range(9)])
    t_us = df["timestamp"].to_numpy(np.int64)
    rot = df[[f"drone_rot[{i}]" for i in range(9)]].to_numpy(float)
    R = np.empty((len(rot), 3, 3))
    for c in range(3):
        R[:, :, c] = rot[:, 3 * c:3 * c + 3]
    order = np.argsort(t_us)
    q = Rotation.from_matrix(R[order]).as_quat()
    sl = Slerp((t_us[order] - t_us[order][0]) * 1e-6, Rotation.from_quat(q))
    return sl((np.asarray(ts_us, float) - t_us[order][0]) * 1e-6).as_matrix()


def uzh_truth(seq: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    meta = json.loads((ROOT / f"data/uzh/{seq}/meta.json").read_text())
    truth = np.load(ROOT / f"data/uzh/{seq}/truth.npz")
    R_cb, _ = published_mount(meta)
    R_bw = Rotation.from_quat(np.asarray(truth["quat"], float)).as_matrix()
    return np.einsum("nij,jk->nik", R_bw, R_cb), R_cb, np.asarray(truth["quat"], float), \
        float(meta["fps_measured"])


def sim_truth_tables(clip_id: str, t_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(L01 / "sim_dev/gt" / f"{clip_id}.npz")
    t = np.asarray(d["t"], float)
    R_cb = np.asarray(d["R_cb"], float)
    # R_bw_cam is the BODY attitude the camera is rendered on (src/sim/labels.py); the
    # camera-to-world is R_bw_cam @ R_cb (src/sim/dataset.py, R_wc).
    R_cw = np.einsum("nij,jk->nik", np.asarray(d["R_bw_cam"], float), R_cb)
    q = Rotation.from_matrix(R_cw).as_quat()
    sl = Slerp(t, Rotation.from_quat(q))
    R_at = sl(np.clip(t_pose, t[0], t[-1])).as_matrix()
    return R_at, R_cb


def sim_duration(clip_id: str) -> float:
    d = np.load(L01 / "sim_dev/gt" / f"{clip_id}.npz")
    return float(np.asarray(d["t"], float)[-1])


def reader_median_sim(clip_id: str) -> float:
    d = np.load(L01 / "scores" / f"{clip_id}.npz")
    return float(np.median(np.asarray(d["up_error_deg"], float)))


def reader_median_uzh(seq: str, variant: str = "a") -> float:
    meta = json.loads((ROOT / f"data/uzh/{seq}/meta.json").read_text())
    truth = np.load(ROOT / f"data/uzh/{seq}/truth.npz")
    R_cb, _ = published_mount(meta)
    R_cw = np.einsum("nij,jk->nik", Rotation.from_quat(np.asarray(truth["quat"], float)).as_matrix(), R_cb)
    up_true = up_of(R_cw)
    rows = [json.loads(l) for l in (UZH_READER / f"{seq}__{variant}.jsonl").read_text().splitlines() if l.strip()]
    idx = np.array([r["index"] for r in rows])
    u = np.array([r["gravity_up_cam"] for r in rows], float)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    return float(np.median(angle_deg(u, up_true[idx])))


def reader_median_tii(flight: str) -> float:
    p = ROOT / f"results/gate_a2/tii/{flight}__identity.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return float(np.median([r["angle_deg"] for r in rows]))


# --------------------------------------------------------------------------- evaluation
def evaluate_unit(stratum: str, kind: str, unit: str, R_est, R_true, t, fps, R_cb,
                  duration: float, reader: float | None) -> tuple[dict, list[dict]]:
    est_step, true_step = steps(R_est, R_true)
    reasons = []
    if len(R_est) == 0:
        reasons.append("crash")
    if len(R_est) and len(R_est) < 0.9 * len(R_true):
        reasons.append("partial_poses")
    if np.isfinite(est_step) and np.isfinite(true_step) and true_step > 0.02 \
            and est_step < 0.10 * true_step:
        reasons.append("frozen_estimate")
    recs, per_h = [], {}
    for h in HORIZONS:
        e = horizon_errors(R_est, R_true, t, h, fps, R_cb)
        per_h[h] = e
        recs.append({"stratum": stratum, "kind": kind, "unit": unit, "horizon_s": h,
                     "n": e["n"], "median_deg": e["median_deg"], "p90_deg": e["p90_deg"],
                     "median_pitch_deg": e["median_pitch_deg"],
                     "median_roll_deg": e["median_roll_deg"], "duration_s": duration,
                     "reader_median_deg": reader, "est_step_median_deg": est_step,
                     "true_step_median_deg": true_step})
    if duration > 10.2 and per_h[10]["n"] == 0:
        reasons.append("no_horizon_pair")
    short = duration < 10.2
    return {"stratum": stratum, "kind": kind, "unit": unit, "duration_s": duration,
            "reader_median_deg": reader, "reasons": reasons, "short": short,
            "est_step_median_deg": est_step, "true_step_median_deg": true_step,
            "n_input": len(R_true), "n_pose": len(R_est), "per_h": per_h}, recs


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    units, rows = [], []
    checks = {}

    # ---- simulated blocks (three families)
    pop = list(csv.DictReader(open(OUT / "sim_population.csv")))
    fam_stratum = {"forest_ridge": "forest_sim", "urban_checker": "urban_sim",
                   "industrial": "industrial_sim"}
    for r in pop:
        cid = r["clip_id"]
        poses = load_poses("sim", cid)
        dur = sim_duration(cid)
        R_cb = np.load(L01 / "sim_dev/gt" / f"{cid}.npz")["R_cb"]
        if poses is None or poses["n_pose"] == 0:
            units.append({"stratum": fam_stratum[r["family"]], "kind": "sim", "unit": cid,
                          "duration_s": dur, "reader_median_deg": reader_median_sim(cid),
                          "reasons": ["crash"], "short": dur < 10.2, "per_h": {},
                          "n_input": int(r["n_frames"]), "n_pose": 0,
                          "est_step_median_deg": float("nan"),
                          "true_step_median_deg": float("nan")})
            continue
        R_est = poses["c2w"][:, :3, :3]
        t = np.arange(len(poses["c2w"])) / poses["fps"]
        R_true, R_cb = sim_truth_tables(cid, t)
        rec, rr = evaluate_unit(fam_stratum[r["family"]], "sim", cid, R_est, R_true, t,
                                poses["fps"], R_cb, dur, reader_median_sim(cid))
        rec["n_input"] = poses["n_input"]
        units.append(rec)
        rows += rr

    # ---- UZH
    uzh_groups = {"real_outdoor_uzh": UZH_OUTDOOR, "real_indoor_uzh": UZH_INDOOR,
                  "real_45down_uzh": UZH_45}
    readers = {v: {} for v in "abc"}
    for stratum, seqs in uzh_groups.items():
        for seq in seqs:
            R_true, R_cb, _q, fps = uzh_truth(seq)
            for v in "abc":
                readers[v][seq] = reader_median_uzh(seq, v)
            dur = len(R_true) / fps
            poses = load_poses("uzh", seq)
            if poses is None or poses["n_pose"] == 0:
                units.append({"stratum": stratum, "kind": "uzh", "unit": seq, "duration_s": dur,
                              "reader_median_deg": readers["a"][seq], "reasons": ["crash"],
                              "short": dur < 10.2, "per_h": {}, "n_input": len(R_true),
                              "n_pose": 0, "est_step_median_deg": float("nan"),
                              "true_step_median_deg": float("nan")})
                continue
            n = min(len(poses["c2w"]), len(R_true))
            R_est = poses["c2w"][:n, :3, :3]
            t = np.arange(n) / fps
            rec, rr = evaluate_unit(stratum, "uzh", seq, R_est, R_true[:n], t, fps, R_cb,
                                    dur, readers["a"][seq])
            rec["n_input"] = poses["n_input"]
            units.append(rec)
            rows += rr

    # ---- TII from stored stride-8 poses
    for flight in TII:
        d = np.load(ROOT / f"data/raw/dpvo/{flight}_stride8_fov100_s0.5/poses.npz")
        poses, ts = np.asarray(d["poses"], float), np.asarray(d["frame_ts_us"], float)
        R_est = Rotation.from_quat(poses[:, 3:7] / np.linalg.norm(poses[:, 3:7], axis=1,
                                                                 keepdims=True)).as_matrix()
        from src.perception.frames import camera_to_body_pitched
        uptilt = {"flight-01a-ellipse": 50, "flight-04a-ellipse": 50, "flight-07a-lemniscate": 40,
                  "flight-13a-trackRATM": 50, "flight-01p-ellipse": 30,
                  "flight-07p-lemniscate": 30, "flight-13p-trackRATM": 50,
                  "flight-16p-trackRATM": 50}[flight]
        R_cb = camera_to_body_pitched(np.deg2rad(uptilt))
        R_true = np.einsum("nij,jk->nik", tii_mocap(flight, ts), R_cb)
        t = (ts - ts[0]) * 1e-6
        rec, rr = evaluate_unit("tii", "tii", flight, R_est, R_true, t, 1.0 / np.median(np.diff(t)),
                                R_cb, float(t[-1] - t[0]), reader_median_tii(flight))
        rec["n_input"] = len(poses)
        units.append(rec)
        rows += rr

    # ---- validation checks before the rule
    R = Rotation.from_euler("zyx", np.array([[10, -20, 30], [0, 0, 0], [-5, 40, 10]]),
                            degrees=True).as_matrix()
    ident = horizon_errors(R, R, np.arange(30) / 15.0, 1.0, 15.0, None)
    checks["identity_true_rotations"] = {"n": ident["n"], "max_deg": float(ident["errors_deg"].max())}
    checks["e01_variant_a_reproduction"] = {
        k: {"computed": readers["a"][k], "published": E01_VARIANT_A[k],
            "abs_diff": abs(readers["a"][k] - E01_VARIANT_A[k])}
        for k in E01_VARIANT_A if k in readers["a"]}
    # sim geometry: camera up from R_bw_cam vs the render's lean_dep
    geom = []
    for r in pop[:20]:
        d = np.load(L01 / "sim_dev/gt" / f"{r['clip_id']}.npz")
        u = up_of(np.asarray(d["R_bw_cam"], float))
        lean = lean_rp(u, np.asarray(d["R_cb"], float))
        ref = np.degrees(np.asarray(d["lean_dep"], float))
        geom.append(float(np.nanmax(np.abs(lean - ref))))
    checks["sim_geometry_lean_dep_max_abs_diff_deg"] = float(np.max(geom))

    # ---- aggregate
    strata = defaultdict(list)
    for u in units:
        strata[u["stratum"]].append(u)
    summary = {"units": len(units), "strata": {}, "readers": readers, "checks": checks}
    for name, us in strata.items():
        usable = [u for u in us if not u["short"] and not u["reasons"]]
        failed = [u for u in us if u["reasons"]]
        short = [u for u in us if u["short"]]
        rec = {"n_attempted": len(us), "n_failed": len(failed), "n_short": len(short),
               "failure_rate": len(failed) / max(1, len(us) - len(short)) if (len(us) - len(short)) else float("nan"),
               "failure_reasons": {r: sum(r in u["reasons"] for u in us)
                                   for r in ("crash", "partial_poses", "frozen_estimate",
                                             "no_horizon_pair")},
               "horizons": {}}
        for h in HORIZONS:
            meds = [u["per_h"][h]["median_deg"] for u in usable
                    if u["per_h"].get(h, {}).get("n", 0) > 0]
            allerr = np.concatenate([u["per_h"][h]["errors_deg"] for u in usable
                                     if u["per_h"].get(h, {}).get("n", 0) > 0]) \
                if any(u["per_h"].get(h, {}).get("n", 0) > 0 for u in usable) else np.array([])
            rec["horizons"][str(h)] = {
                "n_units": len(meds),
                "median_of_unit_medians_deg": float(np.median(meds)) if meds else float("nan"),
                "pooled_median_deg": float(np.median(allerr)) if len(allerr) else float("nan"),
                "pooled_p90_deg": float(np.percentile(allerr, 90)) if len(allerr) else float("nan"),
                "n_starts": int(len(allerr)),
            }
        rm = [u["reader_median_deg"] for u in usable if u["reader_median_deg"] is not None
              and np.isfinite(u["reader_median_deg"])]
        rec["reader_median_of_unit_medians_deg"] = float(np.median(rm)) if rm else float("nan")
        rec["ratio_at_10s"] = (rec["horizons"]["10"]["median_of_unit_medians_deg"]
                               / rec["reader_median_of_unit_medians_deg"]
                               if rm and np.isfinite(rec["horizons"]["10"]["median_of_unit_medians_deg"])
                               else float("nan"))
        rec["ratio_at_10s_pooled"] = (rec["horizons"]["10"]["pooled_median_deg"]
                                     / rec["reader_median_of_unit_medians_deg"]
                                     if rm else float("nan"))
        rec["pass_10s"] = bool(rec["ratio_at_10s"] <= 0.50) if np.isfinite(rec["ratio_at_10s"]) else None
        rec["pass_failures"] = bool(rec["failure_rate"] <= 0.20) if np.isfinite(rec["failure_rate"]) else None
        summary["strata"][name] = rec

    gate = ["forest_sim", "real_outdoor_uzh"]
    summary["decision"] = {
        "rule": "10 s median-of-unit-medians propagated error <= 0.50 x reader median, and "
                "failure rate <= 0.20, in BOTH forest_sim and real_outdoor_uzh",
        "per_stratum": {g: {"ratio": summary["strata"][g]["ratio_at_10s"],
                            "accuracy_ok": summary["strata"][g]["pass_10s"],
                            "failure_ok": summary["strata"][g]["pass_failures"]} for g in gate},
        "pass": bool(all(summary["strata"][g]["pass_10s"] for g in gate)
                     and all(summary["strata"][g]["pass_failures"] for g in gate)),
    }

    with open(OUT / "horizons.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "units.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stratum", "kind", "unit", "duration_s", "n_input", "n_pose",
                    "reader_median_deg", "est_step_median_deg", "true_step_median_deg",
                    "short", "reasons"])
        for u in units:
            w.writerow([u["stratum"], u["kind"], u["unit"], u["duration_s"], u["n_input"],
                        u["n_pose"], u["reader_median_deg"], u["est_step_median_deg"],
                        u["true_step_median_deg"], u["short"], "|".join(u["reasons"])])
    with open(OUT / "reader_same_units.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "variant_a_pinhole", "variant_b_raw_fisheye", "variant_c_est_focal"])
        for seq in readers["a"]:
            w.writerow([seq, readers["a"][seq], readers["b"][seq], readers["c"][seq]])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    for name, rec in summary["strata"].items():
        print(f"\n{name}: units {rec['n_attempted']} failed {rec['n_failed']} short {rec['n_short']} "
              f"reader {rec['reader_median_of_unit_medians_deg']:.2f}")
        for h in HORIZONS:
            v = rec["horizons"][str(h)]
            print(f"  {h:2d}s n={v['n_units']:2d} med={v['median_of_unit_medians_deg']:7.2f} "
                  f"pooled={v['pooled_median_deg']:7.2f} p90={v['pooled_p90_deg']:7.2f} "
                  f"starts={v['n_starts']}")
        print(f"  ratio@10s {rec['ratio_at_10s']:.2f} (pooled {rec['ratio_at_10s_pooled']:.2f}) "
              f"fail_rate {rec['failure_rate']:.2f} reasons {rec['failure_reasons']}")
    print("\nDECISION:", json.dumps(summary["decision"], indent=1))
    print("checks:", json.dumps(checks, indent=1)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
