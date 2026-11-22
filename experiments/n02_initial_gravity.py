"""Track N, Part 1: how much does the initial gravity cost?

Pre-registration ``configs/n02_initial_gravity_prereg.yaml`` (commit ``8fbd8d6``). Arms:
oracle, first-frame reader, first-T reader (T = 1 s), and a two-parameter Huber fit of a constant
up vector to the reader readings carried into the start frame with the estimated rotations. The
constant fitted in the two lean axes IS the initial gravity plus the reader's constant offset, so
the "offset" of the pre-registration is not a separate free parameter.

Descriptive (Part 1 has no gate). Reports the start error and the propagated 10 s error per arm,
by scene family / sequence, with TII dev flights separated from the two held-out flights.

    PYTHONPATH=. python experiments/n02_initial_gravity.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/n02"
POSES = ROOT / "data/n01/poses"
UZH_READER = ROOT / "data/n01/uzh_reader"
L01 = Path("/home/kevin/Documents/Projects/awmv-l/data/l01")
E3 = np.array([0.0, 0.0, 1.0])
H = 10.0
T_FIRST = 1.0
T_FIT = 3.0
STRIDE_S = 1.0
TII_DEV = ["flight-01a-ellipse", "flight-01p-ellipse", "flight-04a-ellipse",
           "flight-07p-lemniscate", "flight-13a-trackRATM", "flight-16p-trackRATM"]
TII_HELD = ["flight-13p-trackRATM", "flight-07a-lemniscate"]


def up_of(R: np.ndarray) -> np.ndarray:
    return np.einsum("nji,j->ni", np.asarray(R, float), E3)


def ang(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def tangent_basis(u: np.ndarray):
    u = u / np.linalg.norm(u)
    e1 = np.cross(u, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(u, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    return e1, e2


def fit_constant(w: np.ndarray) -> tuple[np.ndarray, float]:
    """Huber fit of a constant unit vector to unit rows w (two tangent parameters)."""
    w = np.asarray(w, float)
    m = w.mean(axis=0)
    m /= np.linalg.norm(m)
    e1, e2 = tangent_basis(m)
    def residual(p):
        u = m + p[0] * e1 + p[1] * e2
        u = u / np.linalg.norm(u)
        return ang(u[None, :], w)
    r0 = residual(np.zeros(2))
    scale = 1.4826 * np.median(np.abs(r0 - np.median(r0))) + 1e-6
    kappa = max(1.345 * scale, 1e-3)
    res = least_squares(residual, np.zeros(2), loss="huber", f_scale=kappa)
    u = m + res.x[0] * e1 + res.x[1] * e2
    return u / np.linalg.norm(u), float(np.median(res.fun))


def sim_unit(cid: str):
    d = np.load(POSES / "sim" / f"{cid}.npz")
    R_est = np.asarray(d["c2w"], float)[:, :3, :3]
    fps = float(d["fps"])
    t = np.arange(len(R_est)) / fps
    g = np.load(L01 / "sim_dev/gt" / f"{cid}.npz")
    R_cb = np.asarray(g["R_cb"], float)
    R_cw = np.einsum("nij,jk->nik", np.asarray(g["R_bw_cam"], float), R_cb)
    R_true = Slerp(np.asarray(g["t"], float), Rotation.from_quat(Rotation.from_matrix(R_cw).as_quat()))(np.clip(t, 0, g["t"][-1])).as_matrix()
    r = np.load(L01 / "scores" / f"{cid}.npz")
    up_reader = np.asarray(r["up_est"], float)
    n = min(len(R_est), len(R_true), len(up_reader))
    return R_est[:n], R_true[:n], t[:n], up_reader[:n], fps


def uzh_unit(seq: str):
    d = np.load(POSES / "uzh" / f"{seq}.npz")
    R_est = np.asarray(d["c2w"], float)[:, :3, :3]
    meta = json.loads((ROOT / f"data/uzh/{seq}/meta.json").read_text())
    truth = np.load(ROOT / f"data/uzh/{seq}/truth.npz")
    from src.depicted.uzh import published_mount
    R_cb, _ = published_mount(meta)
    R_true = np.einsum("nij,jk->nik",
                       Rotation.from_quat(np.asarray(truth["quat"], float)).as_matrix(), R_cb)
    n = min(len(R_est), len(R_true))
    fps = float(meta["fps_measured"])
    rows = [json.loads(l) for l in (UZH_READER / f"{seq}__a.jsonl").read_text().splitlines() if l.strip()]
    up_reader = np.full((n, 3), np.nan)
    for r in rows:
        if r["index"] < n:
            u = np.asarray(r["gravity_up_cam"], float)
            up_reader[r["index"]] = u / np.linalg.norm(u)
    return R_est[:n], R_true[:n], np.arange(n) / fps, up_reader, fps


def tii_unit(flight: str):
    from src.depicted.anatomy import load_flight
    d = np.load(ROOT / f"data/raw/dpvo/{flight}_stride8_fov100_s0.5/poses.npz")
    poses, ts = np.asarray(d["poses"], float), np.asarray(d["frame_ts_us"], float)
    R_est = Rotation.from_quat(poses[:, 3:7] / np.linalg.norm(poses[:, 3:7], axis=1, keepdims=True)).as_matrix()
    t = (ts - ts[0]) * 1e-6
    tab = load_flight(flight)
    # truth up at poses comes from the Part 0 construction: mocap at pose times
    R_cb = tab.R_cb
    import pandas as pd
    kind = "piloted" if flight in ("flight-01p-ellipse", "flight-07p-lemniscate",
                                    "flight-13p-trackRATM", "flight-16p-trackRATM") else "autonomous"
    df = pd.read_csv(ROOT / f"data/raw/tii/{kind}/mocap_{flight}.csv",
                     usecols=["timestamp"] + [f"drone_rot[{i}]" for i in range(9)])
    tu = df["timestamp"].to_numpy(np.int64)
    rot = df[[f"drone_rot[{i}]" for i in range(9)]].to_numpy(float)
    R = np.empty((len(rot), 3, 3))
    for c in range(3):
        R[:, :, c] = rot[:, 3 * c:3 * c + 3]
    o = np.argsort(tu)
    sl = Slerp((tu[o] - tu[o][0]) * 1e-6,
               Rotation.from_quat(Rotation.from_matrix(R[o]).as_quat()))
    R_bw = sl((ts - ts[0]) * 1e-6).as_matrix()
    R_true = np.einsum("nij,jk->nik", R_bw, R_cb)
    up_reader = np.full((len(R_est), 3), np.nan)
    time_of = {nm: float(tab.t[k]) for k, nm in enumerate(tab.names)}
    for nm, u in zip(tab.names, tab.up_frozen):
        # join reader frame to the nearest pose by time (reader t is seconds)
        tt = time_of[nm]
        j = int(np.argmin(np.abs(t - tt)))
        if np.isfinite(u).all():
            up_reader[j] = u / np.linalg.norm(u)
    return R_est, R_true, t, up_reader, 1.0 / np.median(np.diff(t))


def evaluate(stratum: str, kind: str, unit: str, R_est, R_true, t, up_reader, fps) -> list[dict]:
    n = min(len(R_est), len(R_true), len(t))
    u_true = np.full((n, 3), np.nan)
    u_true = up_of(R_true[:n])
    stride = max(1, int(round(STRIDE_S * fps)))
    rows = []
    for i in range(0, n - 1, stride):
        if not np.isfinite(up_reader[i]).all():
            continue
        j = int(np.searchsorted(t, t[i] + H))
        if j >= n or abs(t[j] - t[i] - H) > 1.5 / fps:
            continue
        # reader readings carried into the start frame
        lo = i
        hi = min(n, int(np.searchsorted(t, t[i] + max(T_FIRST, T_FIT))))
        idx = np.arange(lo, hi)
        idx = idx[np.isfinite(up_reader[idx]).all(axis=1)]
        if len(idx) < 3:
            continue
        rel = np.einsum("nji,jk->nik", R_est[idx], R_est[i])  # R_est(k)^T R_est(i)
        w = np.einsum("nij,nj->ni", rel, up_reader[idx])
        arms = {"oracle": u_true[i],
                "first_frame_reader": up_reader[i],
                "first_T_reader": np.mean(w[t[idx] - t[i] <= T_FIRST], axis=0)
                if np.any(t[idx] - t[i] <= T_FIRST) else np.mean(w[:1], axis=0),
                "huber_3s": fit_constant(w[t[idx] - t[i] <= T_FIT])[0] if np.any(t[idx] - t[i] <= T_FIT) else np.mean(w, axis=0),
                "huber_all_10s": fit_constant(w)[0]}
        for arm, g0 in arms.items():
            g0 = g0 / np.linalg.norm(g0)
            rel_ij = R_est[j].T @ R_est[i]
            u_pred = rel_ij @ g0
            rows.append({"stratum": stratum, "kind": kind, "unit": unit, "arm": arm,
                         "i": i, "j": j, "start_error_deg": float(ang(g0, u_true[i])),
                         "prop10_deg": float(ang(u_pred, u_true[j]))})
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pop = list(csv.DictReader(open(ROOT / "results/n01_pretest/sim_population.csv")))
    rows = []
    for r in pop:
        cid = r["clip_id"]
        R_est, R_true, t, up_reader, fps = sim_unit(cid)
        rows += evaluate(f"sim_{r['family']}", "sim", cid, R_est, R_true, t, up_reader, fps)
    for stratum, seqs in (("real_outdoor_uzh", ["outdoor_forward_1", "outdoor_forward_3", "outdoor_forward_5"]),
                          ("real_indoor_uzh", ["indoor_forward_3", "indoor_forward_5", "indoor_forward_6",
                                               "indoor_forward_7", "indoor_forward_9", "indoor_forward_10"])):
        for seq in seqs:
            rows += evaluate(stratum, "uzh", seq, *uzh_unit(seq))
    for f in TII_DEV + TII_HELD:
        strip = "tii_dev" if f in TII_DEV else "tii_held"
        rows += evaluate(strip, "tii", f, *tii_unit(f))
    with open(OUT / "starts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # aggregate: per stratum x arm, median of per-unit medians
    agg = {}
    for key in sorted({r["stratum"] for r in rows}):
        agg[key] = {}
        for arm in ("oracle", "first_frame_reader", "first_T_reader", "huber_3s", "huber_all_10s"):
            sub = [r for r in rows if r["stratum"] == key and r["arm"] == arm]
            per = {}
            for r in sub:
                per.setdefault(r["unit"], []).append(r["prop10_deg"])
            unit_med = [float(np.median(v)) for v in per.values()]
            st = [r["start_error_deg"] for r in sub]
            agg[key][arm] = {"units": len(unit_med),
                             "start_error_median_deg": float(np.median(st)) if st else float("nan"),
                             "prop10_median_of_unit_medians_deg": float(np.median(unit_med)) if unit_med else float("nan")}
    (OUT / "starts_summary.json").write_text(json.dumps(agg, indent=1))
    for k, v in agg.items():
        print(k)
        for arm, s in v.items():
            print(f"   {arm:22s} start {s['start_error_median_deg']:7.2f}  10s {s['prop10_median_of_unit_medians_deg']:8.2f}  units {s['units']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
