"""Track N, Part 1 addendum: translation direction from an essential matrix.

Pre-registered in ``configs/n02_initial_gravity_prereg.yaml`` (commit ``8fbd8d6``) and amended
in ``configs/n02_amendment_a.yaml``, both before any number here was computed.

For frame pairs 0.1-0.5 s apart: SIFT matches -> RANSAC essential matrix -> recoverPose gives
the unit baseline direction t_est in the first camera frame. The truth direction is
``unit(R_cw_true(i)^T (pos_j - pos_i))``. ``delta = angle(t_est, t_true)`` is the observable
tilt component; rotation about the travel axis is not observable and is stated, not measured.

Stages
------
    --stage measure   pairs on the simulated feasible clips and the UZH outdoor sequences
    --stage generated track noise on the H04 / H01 generated clips (px)
    --stage analyse   per-second aggregation, speed/gap tables, the pre-registered rule

CPU only. The gpu pixi env is used for OpenCV: ``<main>/.pixi/envs/gpu/bin/python``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/n02"
L01 = Path("/home/kevin/Documents/Projects/awmv-l/data/l01")
H04 = Path("/home/kevin/Documents/Projects/awmv-m/data/m02_inputs/h04")
H01 = Path("/home/kevin/Documents/Projects/awmv-m/data/m02_inputs/h01")
GAPS = (0.1, 0.2, 0.3, 0.5)
START_STRIDE = 3
MIN_INLIERS = 20
SPEED_CUT = 8.0


# --------------------------------------------------------------------------- geometry helpers
def up_of(R_cw: np.ndarray) -> np.ndarray:
    return np.einsum("nji,j->ni", np.asarray(R_cw, float), np.array([0.0, 0.0, 1.0]))


def sampson_px(E: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Sampson distance in pixels for homogeneous pixel points (n,2)."""
    x1 = np.column_stack([p1, np.ones(len(p1))])
    x2 = np.column_stack([p2, np.ones(len(p2))])
    Ex1 = x1 @ E.T
    Etx2 = x2 @ E
    num = np.sum(x2 * Ex1, axis=1) ** 2
    den = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def pair_geometry(fa: np.ndarray, fb: np.ndarray, K: np.ndarray) -> dict:
    """Essential-matrix baseline direction between two grey frames."""
    import cv2

    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.01)  # amendment B
    ka, da = sift.detectAndCompute(fa, None)
    kb, db = sift.detectAndCompute(fb, None)
    if da is None or db is None or len(ka) < 30 or len(kb) < 30:
        return {"status": "few_features", "n_matches": 0, "n_inliers": 0,
                "delta_deg": np.nan, "sampson_median_px": np.nan, "t_est": None}
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(da, db, k=2)
    good = [m for m, n in (x for x in knn if len(x) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < MIN_INLIERS:
        return {"status": "few_matches", "n_matches": len(good), "n_inliers": 0,
                "delta_deg": np.nan, "sampson_median_px": np.nan, "t_est": None}
    p1 = np.float64([ka[m.queryIdx].pt for m in good])
    p2 = np.float64([kb[m.trainIdx].pt for m in good])
    E, mask = cv2.findEssentialMat(p1, p2, cameraMatrix=K, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    if E is None or mask is None:
        return {"status": "no_essential", "n_matches": len(good), "n_inliers": 0,
                "delta_deg": np.nan, "sampson_median_px": np.nan, "t_est": None}
    if E.shape[0] > 3:  # degenerate multi-solution return
        E = E[:3]
    mask = mask.ravel().astype(bool)
    n_in = int(mask.sum())
    if n_in < MIN_INLIERS:
        return {"status": "few_inliers", "n_matches": len(good), "n_inliers": n_in,
                "delta_deg": np.nan, "sampson_median_px": np.nan, "t_est": None}
    _, R, t, mask2 = cv2.recoverPose(E, p1[mask], p2[mask], cameraMatrix=K)
    t = np.asarray(t, float).ravel()
    t = t / np.linalg.norm(t)
    samp = sampson_px(E, p1[mask], p2[mask])
    return {"status": "ok", "n_matches": len(good), "n_inliers": n_in,
            "sampson_median_px": float(np.median(samp)), "t_est": t}


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
                                             -1.0, 1.0))))


# --------------------------------------------------------------------------- simulated clips
def _sim_truth(cid: str):
    d = np.load(L01 / "sim_dev/gt" / f"{cid}.npz")
    return (np.asarray(d["t"], float), np.asarray(d["pos"], float),
            np.asarray(d["R_bw_cam"], float), np.asarray(d["R_cb"], float))


def measure_sim(cid: str, family: str, mounts: str) -> list[dict]:
    import cv2

    video = L01 / "sim_dev/videos" / f"{cid}.mp4"
    cap = cv2.VideoCapture(str(video))
    frames, fps = [], float(cap.get(cv2.CAP_PROP_FPS))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        return []
    K = np.array([[320.0, 0, 320.0], [0, 320.0, 180.0], [0, 0, 1.0]])
    t_gt, pos, R_bw_cam, R_cb = _sim_truth(cid)
    ft = np.arange(len(frames)) / fps
    pos_f = np.column_stack([np.interp(ft, t_gt, pos[:, k]) for k in range(3)])
    from scipy.spatial.transform import Rotation, Slerp
    R_cw = np.einsum("nij,jk->nik", R_bw_cam, R_cb)
    R_cw_f = Slerp(t_gt, Rotation.from_quat(Rotation.from_matrix(R_cw).as_quat()))(ft).as_matrix()
    rows = []
    for i in range(0, len(frames) - 1, START_STRIDE):
        for gap in GAPS:
            step = int(round(gap * fps))
            j = i + step
            if j >= len(frames) or abs(step / fps - gap) > 0.02:
                continue
            g = pair_geometry(frames[i], frames[j], K)
            dp = pos_f[j] - pos_f[i]
            dt = ft[j] - ft[i]
            speed = float(np.linalg.norm(dp) / dt)
            # OpenCV recoverPose convention (amendment C): t = -R_cw(j)^T d
            t_true = -(R_cw_f[j].T @ dp)
            t_old = R_cw_f[i].T @ dp  # pre-amendment reference, kept as a diagnostic only
            delta = (angle_between(g["t_est"], t_true)
                     if g["t_est"] is not None else float("nan"))
            rows.append({"stratum": f"sim_{family}", "kind": "sim", "unit": cid,
                         "mount_deg": mounts, "fps": fps, "gap_s": gap, "i": i, "j": j, "dt_s": dt,
                         "speed_mps": speed, "delta_deg": delta,
                         "delta_old_deg": (angle_between(g["t_est"], t_old)
                                           if g["t_est"] is not None else float("nan")),
                         "delta_amb_deg": min(delta, 180.0 - delta) if np.isfinite(delta) else float("nan"),
                         "n_matches": g["n_matches"], "n_inliers": g["n_inliers"],
                         "sampson_median_px": g["sampson_median_px"], "status": g["status"],
                         "t_est": None if g["t_est"] is None else g["t_est"].tolist(),
                         "t_true": (t_true / np.linalg.norm(t_true)).tolist()})
    return rows


# --------------------------------------------------------------------------- UZH
def measure_uzh(seq: str) -> list[dict]:
    import cv2

    meta = json.loads((ROOT / f"data/uzh/{seq}/meta.json").read_text())
    truth = np.load(ROOT / f"data/uzh/{seq}/truth.npz")
    K = np.array([[meta["intrinsics_pinhole"]["fx"], 0, meta["intrinsics_pinhole"]["cx"]],
                  [0, meta["intrinsics_pinhole"]["fy"], meta["intrinsics_pinhole"]["cy"]],
                  [0, 0, 1.0]])
    from scipy.spatial.transform import Rotation
    from src.depicted.uzh import published_mount
    R_cb, _ = published_mount(meta)
    R_cw = np.einsum("nij,jk->nik", Rotation.from_quat(np.asarray(truth["quat"], float)).as_matrix(), R_cb)
    pos = np.asarray(truth["pos"], float)
    fps = float(meta["fps_measured"])
    frame_dir = ROOT / f"data/n01/uzh_frames/{seq}"
    paths = sorted(frame_dir.glob("*.png"))
    if not paths:
        return [{"status": "frames_missing", "unit": seq}]
    frames = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    n = min(len(frames), len(pos))
    rows = []
    for i in range(0, n - 1, 2):
        for gap in GAPS:
            step = int(round(gap * fps))
            j = i + step
            if j >= n or abs(step / fps - gap) > 0.02:
                continue
            g = pair_geometry(frames[i], frames[j], K)
            dp = pos[j] - pos[i]
            dt = (j - i) / fps
            speed = float(np.linalg.norm(dp) / dt)
            t_true = -(R_cw[j].T @ dp)  # amendment C
            t_old = R_cw[i].T @ dp      # pre-amendment reference, diagnostic only
            delta = (angle_between(g["t_est"], t_true)
                     if g["t_est"] is not None else float("nan"))
            rows.append({"stratum": "uzh_outdoor", "kind": "uzh", "unit": seq,
                         "mount_deg": "", "fps": fps, "gap_s": gap, "i": i, "j": j, "dt_s": dt,
                         "speed_mps": speed, "delta_deg": delta,
                         "delta_old_deg": (angle_between(g["t_est"], t_old)
                                           if g["t_est"] is not None else float("nan")),
                         "delta_amb_deg": min(delta, 180.0 - delta) if np.isfinite(delta) else float("nan"),
                         "n_matches": g["n_matches"], "n_inliers": g["n_inliers"],
                         "sampson_median_px": g["sampson_median_px"], "status": g["status"],
                         "t_est": None if g["t_est"] is None else g["t_est"].tolist(),
                         "t_true": (t_true / np.linalg.norm(t_true)).tolist()})
    return rows


# --------------------------------------------------------------------------- generated track noise
def pair_fundamental(fa: np.ndarray, fb: np.ndarray) -> dict:
    """Calibration-free two-view fit: fundamental matrix Sampson error (px) of RANSAC inliers."""
    import cv2

    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.01)
    ka, da = sift.detectAndCompute(fa, None)
    kb, db = sift.detectAndCompute(fb, None)
    if da is None or db is None or len(ka) < 30 or len(kb) < 30:
        return {"status": "few_features", "n_matches": 0, "n_inliers": 0,
                "sampson_median_px": float("nan")}
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for m, n in (x for x in knn if len(x) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < MIN_INLIERS:
        return {"status": "few_matches", "n_matches": len(good), "n_inliers": 0,
                "sampson_median_px": float("nan")}
    p1 = np.float64([ka[m.queryIdx].pt for m in good])
    p2 = np.float64([kb[m.trainIdx].pt for m in good])
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.RANSAC, 1.0, 0.999)
    if F is None or mask is None or F.shape != (3, 3):
        return {"status": "no_fundamental", "n_matches": len(good), "n_inliers": 0,
                "sampson_median_px": float("nan")}
    mask = mask.ravel().astype(bool)
    n_in = int(mask.sum())
    if n_in < MIN_INLIERS:
        return {"status": "few_inliers", "n_matches": len(good), "n_inliers": n_in,
                "sampson_median_px": float("nan")}
    samp = sampson_px(F, p1[mask], p2[mask])
    return {"status": "ok", "n_matches": len(good), "n_inliers": n_in,
            "sampson_median_px": float(np.median(samp))}


def measure_generated(path: str, tag: str) -> list[dict]:
    import cv2

    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 16.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(frames) < 4:
        return []
    rows = []
    for i in range(0, len(frames) - 1, 2):
        for gap in (0.2, 0.5):
            j = i + max(1, int(round(gap * fps)))
            if j >= len(frames):
                continue
            g = pair_fundamental(frames[i], frames[j])
            rows.append({"tag": tag, "clip": Path(path).stem, "i": i, "gap_s": gap,
                         "n_matches": g["n_matches"], "n_inliers": g["n_inliers"],
                         "sampson_median_px": g["sampson_median_px"], "status": g["status"]})
    return rows


# --------------------------------------------------------------------------- drivers
def _sim_task(a):
    return measure_sim(*a)

def _write(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run_measure() -> None:
    pop = list(csv.DictReader(open(ROOT / "results/n01_pretest/sim_population.csv")))
    tasks = [(r["clip_id"], r["family"], r["uptilt_deg"]) for r in pop]
    rows = []
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
        for k, part in enumerate(ex.map(_sim_task, tasks)):
            rows += part
            if k % 10 == 0:
                print(f"sim {k + 1}/{len(tasks)} clips, {len(rows)} pairs", flush=True)
    _write(rows, OUT / "pairs_sim.csv",
           ["stratum", "kind", "unit", "mount_deg", "fps", "gap_s", "i", "j", "dt_s", "speed_mps",
            "delta_deg", "delta_old_deg", "delta_amb_deg", "n_matches", "n_inliers", "sampson_median_px",
            "status", "t_est", "t_true"])
    print(f"wrote {len(rows)} sim pairs")

    uzh = ["outdoor_forward_1", "outdoor_forward_3", "outdoor_forward_5"]
    urows = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        for part in ex.map(measure_uzh, uzh):
            urows += part
    _write(urows, OUT / "pairs_uzh.csv",
           ["stratum", "kind", "unit", "mount_deg", "fps", "gap_s", "i", "j", "dt_s", "speed_mps",
            "delta_deg", "delta_old_deg", "delta_amb_deg", "n_matches", "n_inliers", "sampson_median_px",
            "status", "t_est", "t_true"])
    print(f"wrote {len(urows)} UZH pairs")


def _gen_task(a):
    return measure_generated(*a)


def run_generated() -> None:
    jobs = []
    vids = sorted(H04.glob("*.mp4"))
    for p in vids:
        jobs.append((str(p), "h04_solarwm"))
    # H01 ANWM clips are directories of frames; use the first frame sequence if present
    for d in sorted(H01.glob("*")):
        if d.is_dir():
            frames = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
            if frames:
                jobs.append((d, "h01_anwm"))
    jobs = [j for j in jobs if not Path(j[0]).is_dir()]
    print(f"generated jobs: {len(jobs)}")
    rows = []
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
        for k, part in enumerate(ex.map(_gen_task, jobs)):
            rows += part
            if k % 20 == 0:
                print(f"generated {k + 1}/{len(jobs)}", flush=True)
    _write(rows, OUT / "generated_track_noise.csv",
           ["tag", "clip", "i", "gap_s", "n_matches", "n_inliers", "sampson_median_px", "status"])
    print(f"wrote {len(rows)} generated pair rows")


def _parse_vec(x) -> np.ndarray:
    if not isinstance(x, str):
        return np.asarray(x, float)
    parts = x.strip().strip("[]").replace(",", " ").split()
    return np.asarray([float(v) for v in parts], float)


def run_analyse() -> None:
    import pandas as pd

    frames = []
    for name in ("pairs_sim.csv", "pairs_uzh.csv"):
        p = OUT / name
        if p.exists():
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    ok = df[df.status == "ok"].copy()
    out = {"n_pairs": int(len(df)), "n_ok": int(len(ok)),
           "fail_reasons": df[df.status != "ok"].status.value_counts().to_dict()}

    def stratum_table(g: pd.DataFrame) -> dict:
        hi = g[g.speed_mps >= SPEED_CUT]
        return {"n_pairs": int(len(g)), "n_ok": int(len(g)), "n_high": int(len(hi)),
                "n_units": int(g.unit.nunique()), "n_units_high": int(hi.unit.nunique()),
                "median_deg": float(np.nanmedian(g.delta_deg)),
                "median_high_deg": float(np.nanmedian(hi.delta_deg)) if len(hi) else None,
                "p90_high_deg": float(np.nanpercentile(hi.delta_deg, 90)) if len(hi) else None,
                "sampson_median_px": float(np.nanmedian(g.sampson_median_px))}

    out["strata"] = {k: stratum_table(g) for k, g in ok.groupby("stratum")}
    # per gap and speed bin, forest and outdoor
    out["by_gap"] = {k: {str(gp): float(np.nanmedian(gg.delta_deg)) for gp, gg in g.groupby("gap_s")}
                     for k, g in ok.groupby("stratum")}
    bins = [0, 2, 4, 6, 8, 12, 100]
    out["by_speed"] = {}
    for k, g in ok.groupby("stratum"):
        cut = pd.cut(g.speed_mps, bins)
        out["by_speed"][k] = {str(b): {"n": int(len(gg)),
                                       "median_deg": float(np.nanmedian(gg.delta_deg))}
                              for b, gg in g.groupby(cut, observed=True)}
    # per-second (1 s blocks, circular mean of directions)
    ps = []
    for (unit, gap), g in ok.groupby(["unit", "gap_s"]):
        fps_u = float(g.fps.iloc[0]) if "fps" in g else (30.0 if g.kind.iloc[0] == "sim" else 15.07)
        for sec, gg in g.groupby((g.i / fps_u).astype(int)):
            te = np.mean(np.array([_parse_vec(x) for x in gg.t_est]), axis=0)
            tt = np.mean(np.array([_parse_vec(x) for x in gg.t_true]), axis=0)
            ps.append({"unit": unit, "gap_s": gap, "second": int(sec),
                       "stratum": gg.stratum.iloc[0], "speed_mps": float(gg.speed_mps.median()),
                       "delta_deg": angle_between(te, tt) if np.linalg.norm(te) > 0 else np.nan})
    psd = pd.DataFrame(ps)
    psd.to_csv(OUT / "per_second.csv", index=False)
    out["per_second"] = {k: {"n": int(len(g)), "n_high": int((g.speed_mps >= SPEED_CUT).sum()),
                             "median_deg": float(np.nanmedian(g.delta_deg)),
                             "median_high_deg": float(np.nanmedian(g[g.speed_mps >= SPEED_CUT].delta_deg))
                             if (g.speed_mps >= SPEED_CUT).any() else None}
                         for k, g in psd.groupby("stratum")}

    # the pre-registered rule
    gate = {"forest": "sim_forest_ridge", "outdoor": "uzh_outdoor"}
    rule = {}
    for label, key in gate.items():
        s = out["strata"].get(key)
        if not s:
            rule[label] = {"evaluable": False, "reason": "no stratum"}
            continue
        evaluable = s["n_units_high"] >= 5 and s["n_high"] >= 100
        pair_ok = s["median_high_deg"] is not None and s["median_high_deg"] <= 3.0
        sec = out["per_second"].get(key, {})
        sec_ok = sec.get("median_high_deg") is not None and sec["median_high_deg"] <= 3.0
        rule[label] = {"evaluable": bool(evaluable), "median_pair_high_deg": s["median_high_deg"],
                       "median_second_high_deg": sec.get("median_high_deg"),
                       "pair_ok": bool(pair_ok), "second_ok": bool(sec_ok),
                       "pass": bool(evaluable and pair_ok and sec_ok)}
    out["decision"] = {"rule": "median delta <= 3.0 deg at speed >= 8 m/s in both strata, "
                               "per pair and per 1 s block, with >=5 clips and >=100 pairs",
                       "per_stratum": rule,
                       "pass": bool(all(v.get("pass") for v in rule.values()))}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["decision"], indent=1))
    for k, s in out["strata"].items():
        print(k, s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["measure", "generated", "analyse", "all"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("measure", "all"):
        run_measure()
    if a.stage in ("generated", "all"):
        run_generated()
    if a.stage in ("analyse", "all"):
        run_analyse()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
