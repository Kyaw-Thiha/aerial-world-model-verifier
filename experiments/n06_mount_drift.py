"""Track N, N06: the two-view relative-pose chain (measure) and the mount-drift diagnosis.

Pre-registration ``configs/n06_mount_drift_prereg.yaml`` and amendment A.

Stages
------
    --stage measure   two-view rotation chains for the 80 feasible sim clips, full and half res
    --stage analyse   the 10 s oracle-start drift by mount for four arms, and the UZH resolution
                      check

    PYTHONPATH=. python experiments/n06_mount_drift.py --stage measure
    PYTHONPATH=. python experiments/n06_mount_drift.py --stage analyse
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/n06"
VIDEOS = ROOT / "data/n05/videos"
GT = ROOT / "data/n05/gt"
POSES_FULL = ROOT / "data/n05/poses/sim200/scale1"
POSES_HALF = ROOT / "data/n05/poses/sim200/scale0.5"
CHAIN = ROOT / "data/n06/chain"
E3 = np.array([0.0, 0.0, 1.0])
GAP = 2  # frames (15 Hz)


def up_of(R):
    return np.einsum("...ji,j->...i", np.asarray(R, float), E3)


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b) /
                                                    (np.linalg.norm(a) * np.linalg.norm(b))), -1, 1))))


def chain_clip(cid: str, scale: float) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(VIDEOS / f"{cid}.mp4"))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if scale != 1.0:
            g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(g)
    cap.release()
    h, w = frames[0].shape[:2]
    f = (w / 2.0) / np.tan(np.deg2rad(45.0))
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    sift = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.01)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    kd = []
    for g in frames:
        k, d = sift.detectAndCompute(g, None)
        kd.append((np.float64([p.pt for p in k]), d))
    R_chain = [np.eye(3)]
    n_ok = 0
    for i in range(0, len(frames) - GAP, GAP):
        (p1, d1), (p2, d2) = kd[i], kd[i + GAP]
        if d1 is None or d2 is None or len(p1) < 20 or len(p2) < 20:
            R_chain.append(R_chain[-1])
            continue
        knn = bf.knnMatch(d1, d2, k=2)
        good = [m for m, n in (x for x in knn if len(x) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 20:
            R_chain.append(R_chain[-1])
            continue
        a1 = np.float64([p1[m.queryIdx] for m in good])
        a2 = np.float64([p2[m.trainIdx] for m in good])
        E, mask = cv2.findEssentialMat(a1, a2, cameraMatrix=K, method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)
        if E is None or mask is None or int(mask.sum()) < 20:
            R_chain.append(R_chain[-1])
            continue
        if E.shape[0] > 3:
            E = E[:3]
        mask = mask.ravel().astype(bool)
        _, Rrel, _, _ = cv2.recoverPose(E, a1[mask], a2[mask], cameraMatrix=K)
        R_chain.append(R_chain[-1] @ np.asarray(Rrel, float))
        n_ok += 1
    R = np.asarray(R_chain, float)
    t = np.arange(len(R)) * (GAP / 30.0)
    np.savez_compressed(CHAIN / f"scale{scale:g}" / f"{cid}.npz", R=R, t=t, n_ok=n_ok,
                        n_pairs=len(R) - 1)
    return {"clip_id": cid, "scale": scale, "n": len(R), "n_ok": n_ok,
            "frac_ok": n_ok / max(1, len(R) - 1), "fps": 30.0 / GAP}


def _task(a):
    return chain_clip(*a)


def run_measure() -> None:
    rows = [r for r in csv.DictReader(open(ROOT / "results/n05/clips.csv"))
            if r["condition"] == "feasible"]
    tasks = [(r["clip_id"], sc) for r in rows for sc in (1.0, 0.5)]
    for sc in (1.0, 0.5):
        (CHAIN / f"scale{sc:g}").mkdir(parents=True, exist_ok=True)
    out = []
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
        for i, r in enumerate(ex.map(_task, tasks)):
            out.append(r)
            if i % 20 == 0:
                print(f"chain {i + 1}/{len(tasks)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "chains.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print("wrote", len(out), "chains")


def drift(R_est, R_true, t, h=10.0):
    n = min(len(t), len(R_est), len(R_true))
    t = np.asarray(t, float)[:n]
    ut = up_of(np.asarray(R_true, float)[:n])
    idx = np.searchsorted(t, t + h)
    j = np.minimum(idx, n - 1)
    ok = (idx < n) & (j > np.arange(n)) & (np.abs(t[j] - t - h) <= 2.0 / 15.0)
    if not ok.any():
        return float("nan"), 0
    i = np.arange(n)[ok]
    rel = np.einsum("nji,njk->nik", np.asarray(R_est, float)[j[ok]], np.asarray(R_est, float)[i])
    pred = np.einsum("nij,nj->ni", rel, ut[i])
    err = np.degrees(np.arccos(np.clip(np.sum(pred * ut[j[ok]], axis=1), -1, 1)))
    return float(np.median(err)), int(len(err))


def true_attitude(cid: str, t: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Slerp
    g = np.load(GT / f"{cid}.npz")
    tgt = np.asarray(g["t"], float)
    R_cb = np.asarray(g["R_cb"], float)
    R_cw = np.einsum("nij,jk->nik", np.asarray(g["R_bw_cam"], float), R_cb)
    sl = Slerp(tgt, Rotation.from_quat(Rotation.from_matrix(R_cw).as_quat()))
    return sl(np.clip(t, tgt[0], tgt[-1])).as_matrix()


def run_analyse() -> None:
    rows = [r for r in csv.DictReader(open(ROOT / "results/n05/clips.csv"))
            if r["condition"] == "feasible"]
    arms = {}
    for r in rows:
        cid = r["clip_id"]
        d = np.load(POSES_FULL / f"{cid}.npz")
        t = np.arange(int(d["n_pose"])) / float(d["fps"])
        Rt = true_attitude(cid, t)
        arms.setdefault("dpvo_full", []).append((cid, r, drift(np.asarray(d["c2w"], float)[:, :3, :3], Rt, t)))
        if (POSES_HALF / f"{cid}.npz").exists():
            dh = np.load(POSES_HALF / f"{cid}.npz")
            th = np.arange(int(dh["n_pose"])) / float(dh["fps"])
            Rth = true_attitude(cid, th)
            arms.setdefault("dpvo_half", []).append((cid, r, drift(np.asarray(dh["c2w"], float)[:, :3, :3], Rth, th)))
        for sc in (1.0, 0.5):
            p = CHAIN / f"scale{sc:g}" / f"{cid}.npz"
            if p.exists():
                c = np.load(p)
                tt = np.asarray(c["t"], float)
                Rt2 = true_attitude(cid, tt)
                arms.setdefault(f"chain_full" if sc == 1.0 else "chain_half", []).append(
                    (cid, r, drift(np.asarray(c["R"], float), Rt2, tt)))
    summary = {"arms": {}, "uzh": {}}
    for arm, vals in arms.items():
        table = {}
        for mount in ("0.0", "15.0", "30.0", "45.0"):
            sub = [v for _, r, v in vals if r["mount"] == mount]
            forest = [v for _, r, v in vals if r["mount"] == mount and r["family"] == "forest_ridge" and v[1] > 0]
            table[mount] = {"n": len(sub),
                            "median": float(np.nanmedian([x[0] for x in sub])) if sub else None,
                            "forest_no_pairs": len(forest)}
        # forest-only medians
        fb = {}
        for mount in ("0.0", "15.0", "30.0", "45.0"):
            sub = [v[0] for _, r, v in vals if r["mount"] == mount and r["family"] == "forest_ridge"
                   and np.isfinite(v[0])]
            fb[mount] = {"n": len(sub), "median": float(np.median(sub)) if sub else None}
        summary["arms"][arm] = {"all": table, "forest": fb}
    summary["uzh"] = uzh_resolution()
    # DPVO run-to-run reproducibility on the forest 30/45 clips (a second full-resolution draw)
    rep_dir = ROOT / "data/n06/poses_repeat/scale1"
    repro = {}
    for mount in ("30.0", "45.0"):
        sub = [(cid, r) for cid, r, v in arms["dpvo_full"]
               if r["mount"] == mount and r["family"] == "forest_ridge"]
        diffs, a1, a2 = [], [], []
        for cid, r in sub:
            p2 = rep_dir / f"{cid}.npz"
            if not p2.exists():
                continue
            d1 = np.load(POSES_FULL / f"{cid}.npz")
            t1 = np.arange(int(d1["n_pose"])) / float(d1["fps"])
            e1, _ = drift(np.asarray(d1["c2w"], float)[:, :3, :3], true_attitude(cid, t1), t1)
            d2 = np.load(p2)
            t2 = np.arange(int(d2["n_pose"])) / float(d2["fps"])
            e2, _ = drift(np.asarray(d2["c2w"], float)[:, :3, :3], true_attitude(cid, t2), t2)
            if np.isfinite(e1) and np.isfinite(e2):
                a1.append(e1); a2.append(e2); diffs.append(abs(e1 - e2))
        repro[mount] = {"n": len(diffs),
                        "median_run1": float(np.median(a1)) if a1 else None,
                        "median_run2": float(np.median(a2)) if a2 else None,
                        "median_abs_diff": float(np.median(diffs)) if diffs else None,
                        "max_abs_diff": float(np.max(diffs)) if diffs else None}
    summary["reproducibility"] = repro
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def uzh_resolution() -> dict:
    rows = list(csv.DictReader(open(ROOT / "results/n02b/clips.csv")))
    out = {}
    for tag, d in (("crop_320x240", ROOT / "data/n01/poses/uzh_edits"),
                   ("full_640x480", ROOT / "data/n01/poses/uzh_edits_full")):
        ok = tot = 0
        for r in rows:
            p = d / f"{r['clip_id']}.npz"
            if not p.exists():
                continue
            tot += 1
            c = np.load(p)
            R = np.asarray(c["c2w"], float)[:, :3, :3]
            t = np.arange(len(R)) / float(c["fps"])
            Rt = true_attitude_uzh(r["seq"], t)
            e, n = drift(R, Rt, t, 2.0)
            if n and np.isfinite(e) and e <= 10.0:
                ok += 1
        out[tag] = {"valid": f"{ok}/{tot}"}
    return out


def true_attitude_uzh(seq: str, t: np.ndarray) -> np.ndarray:
    import json as _json
    from scipy.spatial.transform import Slerp
    from src.depicted.uzh import published_mount
    meta = _json.loads((ROOT / f"data/uzh/{seq}/meta.json").read_text())
    truth = np.load(ROOT / f"data/uzh/{seq}/truth.npz")
    R_cb, _ = published_mount(meta)
    R_cw = np.einsum("nij,jk->nik",
                     Rotation.from_quat(np.asarray(truth["quat"], float)).as_matrix(), R_cb)
    tt = np.arange(len(R_cw)) / float(meta["fps_measured"])
    sl = Slerp(tt, Rotation.from_quat(Rotation.from_matrix(R_cw).as_quat()))
    return sl(np.clip(t, tt[0], tt[-1])).as_matrix()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["measure", "analyse", "all"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("measure", "all"):
        run_measure()
    if a.stage in ("analyse", "all"):
        run_analyse()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
