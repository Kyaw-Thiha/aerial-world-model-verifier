"""Track N, Part 4: video rotation on generated clips (H04 SolarWM).

Pre-registration ``configs/n04_generated_rotation_prereg.yaml`` (before any DPVO run on the
clips). Read-only inputs: the H04 path table and reader JSONL from track-h3, the clip list and
video hashes in ``results/n04/clips.csv``. Poses from ``awmv-n-data/poses/gen/fov90``.

    PYTHONPATH=. python experiments/n04_generated_rotation.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.generation.attitude import body_rotation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/n04"
POSES = ROOT / "data/n04/poses/gen"
H3 = Path("/home/kevin/Documents/Projects/awmv-h3/results/h04")
E3 = np.array([0.0, 0.0, 1.0])
HORIZONS = (1, 2, 5)


def up_of(R):
    return np.einsum("...ji,j->...i", np.asarray(R, float), E3)


def ang(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def path_key(p: str) -> str:
    import re
    q = p[5:] if (p.startswith("main_") or p.startswith("swap_")) else p
    return re.sub(r"_F\d+$", "", q)


def true_attitudes(path: str, arm: str) -> np.ndarray:
    d = np.load(H3 / "paths/paths.npz", allow_pickle=True)
    key = path_key(path)
    yaw, pitch, roll = (np.asarray(d[f"{key}__{k}"], float) for k in ("yaw", "pitch", "roll"))
    if arm == "B_level":
        pitch = np.zeros_like(pitch)
        roll = np.zeros_like(roll)
    elif arm == "C_mirrored_roll":
        roll = -roll
    from src.perception.frames import camera_to_body_pitched
    R_bc = camera_to_body_pitched(0.0)
    return body_rotation(yaw, pitch, roll) @ R_bc[None]


def horizon_stats(R_est, R_true, t, h):
    n = min(len(t), len(R_est), len(R_true))
    t = np.asarray(t, float)[:n]
    up_t = up_of(np.asarray(R_true, float)[:n])
    idx = np.searchsorted(t, t + h)
    j = np.minimum(idx, n - 1)
    ok = (idx < n) & (j > np.arange(n)) & (np.abs(t[j] - t - h) <= 1.5 / 16.0)
    if not ok.any():
        return {"n": 0, "median_deg": float("nan")}
    i = np.arange(n)[ok]
    jj = j[ok]
    rel = np.einsum("nji,njk->nik", np.asarray(R_est, float)[jj], np.asarray(R_est, float)[i])
    pred = np.einsum("nij,nj->ni", rel, up_t[i])
    err = ang(pred, up_t[jj])
    return {"n": int(len(err)), "median_deg": float(np.median(err))}


def steps(R_est, R_true):
    if len(R_est) < 2:
        return float("nan"), float("nan")
    from scipy.spatial.transform import Rotation
    re = np.einsum("nji,njk->nik", R_est[:-1], R_est[1:])
    rt = np.einsum("nji,njk->nik", R_true[:-1], R_true[1:])
    return (float(np.median(np.degrees(np.linalg.norm(Rotation.from_matrix(re).as_rotvec(), axis=1)))),
            float(np.median(np.degrees(np.linalg.norm(Rotation.from_matrix(rt).as_rotvec(), axis=1)))))


def load_reader():
    rows = defaultdict(dict)
    with open(H3 / "depicted_main/depicted.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            u = np.asarray(r["gravity_up_cam"], float)
            rows[r["clip_id"]][int(Path(r["frame"]).stem)] = u / np.linalg.norm(u)
    return rows


def main() -> int:
    clips = list(csv.DictReader(open(OUT / "clips.csv")))
    reader = load_reader()
    tracking, horizons = [], []
    summary_strata = defaultdict(lambda: {"attempted": 0, "failed": 0, "reasons": defaultdict(int),
                                          "h": defaultdict(list), "reader": [],
                                          "per_clip": {}})
    for r in clips:
        cid, arm, path, draw = r["clip"], r["arm"], r["path"], r["draw"]
        stratum = f"{arm}"
        s = summary_strata[stratum]
        s["attempted"] += 1
        npz = POSES / "fov90" / f"{cid}.npz"
        R_true = true_attitudes(path, arm)
        n_true = len(R_true)
        if not npz.exists():
            tracking.append({"clip": cid, "arm": arm, "draw": draw, "status": "crash",
                             "n_input": 81, "n_pose": 0})
            s["failed"] += 1
            s["reasons"]["crash"] += 1
            continue
        d = np.load(npz)
        R_est = np.asarray(d["c2w"], float)[:, :3, :3]
        n_in, n_pose, fps = int(d["n_input"]), int(d["n_pose"]), float(d["fps"])
        est_step, true_step = steps(R_est, R_true[:len(R_est)])
        reasons = []
        if n_pose < 0.9 * n_in:
            reasons.append("partial_poses")
        if np.isfinite(est_step) and np.isfinite(true_step) and true_step > 0.02 \
                and est_step < 0.10 * true_step:
            reasons.append("frozen_estimate")
        tracking.append({"clip": cid, "arm": arm, "draw": draw,
                         "status": "ok" if not reasons else "|".join(reasons),
                         "n_input": n_in, "n_pose": n_pose,
                         "est_step_deg": est_step, "true_step_deg": true_step})
        if reasons:
            s["failed"] += 1
            for x in reasons:
                s["reasons"][x] += 1
            continue
        t = np.arange(n_pose) / fps
        rec = {"clip": cid, "arm": arm, "draw": draw}
        for h in HORIZONS:
            hs = horizon_stats(R_est, R_true, t, h)
            rec[f"n_{h}s"] = hs["n"]
            rec[f"err_{h}s"] = hs["median_deg"]
            horizons.append({"clip": cid, "arm": arm, "draw": draw, "horizon_s": h,
                             "n": hs["n"], "median_deg": hs["median_deg"]})
        # reader error on the same clip, against the commanded attitude
        ru = reader.get(cid, {})
        errs = []
        for k in range(min(len(R_true), n_pose)):
            if k in ru:
                errs.append(float(ang(ru[k], up_of(R_true[k]))))
        rec["reader_median_deg"] = float(np.median(errs)) if errs else float("nan")
        s["reader"].append(rec["reader_median_deg"])
        for h in HORIZONS:
            s["h"][h].append(rec[f"err_{h}s"])
        s["per_clip"][cid] = rec

    with open(OUT / "tracking.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tracking[0].keys()))
        w.writeheader()
        w.writerows(tracking)
    with open(OUT / "horizons.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(horizons[0].keys()))
        w.writeheader()
        w.writerows(horizons)
    out = {}
    for k, s in summary_strata.items():
        out[k] = {"attempted": s["attempted"], "failed": s["failed"],
                  "success_rate": (s["attempted"] - s["failed"]) / s["attempted"],
                  "reasons": dict(s["reasons"]),
                  "reader_median_deg": float(np.median(s["reader"])) if s["reader"] else None,
                  "horizons": {str(h): {"n_clips": int(np.isfinite(v).sum()),
                                        "median_deg": float(np.nanmedian(v)) if len(v) else None}
                               for h, v in s["h"].items()},
                  "per_clip": s["per_clip"]}
    out["assumption"] = "intrinsics 90 deg horizontal (f=432) for the generated video"
    out["clips"] = len(clips)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in out.items():
        if k in ("assumption", "clips"):
            continue
        print(k, {kk: v[kk] for kk in ("attempted", "failed", "success_rate", "reasons",
                                       "reader_median_deg", "horizons")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
