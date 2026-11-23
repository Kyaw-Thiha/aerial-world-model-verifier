"""Track N addendum: does a turn make the tilt observable from translation directions?

Pre-registered in ``configs/n02_initial_gravity_prereg.yaml`` and clarified in
``configs/n02_amendment_a.yaml``, before any number here was computed.

The measurement is operational:
* start at window 1 with the FIRST-FRAME READER up vector (not the oracle), propagate to window 2
  with DPVO (the Part 0 construction);
* at each window the essential-matrix direction gives one signed scalar constraint on the camera's
  tilt (the component that moves the velocity direction up/down);
* a constant body-frame lean correction c = (roll, pitch) is fitted to satisfy those constraints
  (one constraint: minimum-norm; two: least squares), then the remaining lean error at window 2 is
  compared with the video-only error, by the angle theta between the two velocity directions.

Rotation about the velocity direction is not observable from directions at all; that is stated,
not measured.

    PYTHONPATH=. python experiments/n02_turn_test.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/n02"
E3 = np.array([0.0, 0.0, 1.0])
GAP = 0.2
TURN_MIN, TURN_MAX = 2.0, 5.0
BINS = [(0.0, 10.0), (10.0, 30.0), (30.0, 180.0)]


def parse_vec(x) -> np.ndarray:
    if not isinstance(x, str):
        return np.asarray(x, float)
    return np.asarray([float(v) for v in x.strip().strip("[]").replace(",", " ").split()], float)


def up_of(R):
    R = np.asarray(R, float)
    return np.einsum("...ji,j->...i", R, E3)


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def body_corr(c):
    return Rotation.from_rotvec(np.array([c[0], c[1], 0.0])).as_matrix()


def direction_mismatch(R_pred, v_world, t_est):
    """Signed angle about the up axis that moves the predicted velocity direction to t_est."""
    t_pred = -R_pred.T @ v_world
    t_pred = t_pred / np.linalg.norm(t_pred)
    up_cam = R_pred.T @ E3
    a = np.cross(t_pred, up_cam)
    n = np.linalg.norm(a)
    if n < 1e-8:
        return 0.0, t_pred
    a = a / n
    basis = np.cross(a, t_pred)
    nb = np.linalg.norm(basis)
    if nb < 1e-8:
        return 0.0, t_pred
    basis = basis / nb
    delta = float(np.arctan2(t_est @ basis, t_est @ t_pred))
    return delta, t_pred


def solve_correction(R_preds, v_worlds, t_ests, two: bool):
    """Fit c (2 params) so the direction mismatches vanish."""
    eps = 1e-3
    gs, ds = [], []
    for R, v, t in zip(R_preds, v_worlds, t_ests):
        d0, _ = direction_mismatch(R, v, t)
        g = []
        for k in range(2):
            dc = np.zeros(2)
            dc[k] = eps
            dk, _ = direction_mismatch(R @ body_corr(dc), v, t)
            g.append((dk - d0) / eps)
        gs.append(g)
        ds.append(-d0)
    G = np.asarray(gs)
    d = np.asarray(ds)
    if not (np.isfinite(G).all() and np.isfinite(d).all()):
        return None
    if two:
        c, *_ = np.linalg.lstsq(G, d, rcond=None)
    else:
        c = G[0] * (d[0] / (G[0] @ G[0]))
    return c


def unit_data(kind: str, unit: str):
    """(R_est, t, fps, R_true, pos) for a unit."""
    if kind == "sim":
        from experiments.n01_rotation_pretest import sim_truth_tables
        d = np.load(ROOT / f"data/n01/poses/sim/{unit}.npz")
        R_est = np.asarray(d["c2w"], float)[:, :3, :3]
        fps = float(d["fps"])
        t = np.arange(len(R_est)) / fps
        R_true, R_cb = sim_truth_tables(unit, t)
        g = np.load(Path("/home/kevin/Documents/Projects/awmv-l/data/l01/sim_dev/gt") / f"{unit}.npz")
        tgt = np.asarray(g["t"], float)
        pos = np.asarray(g["pos"], float)
        pos_i = np.column_stack([np.interp(t, tgt, pos[:, k]) for k in range(3)])
        return R_est, t, fps, R_true, pos_i, R_cb
    d = np.load(ROOT / f"data/n01/poses/uzh/{unit}.npz")
    R_est = np.asarray(d["c2w"], float)[:, :3, :3]
    meta = json.loads((ROOT / f"data/uzh/{unit}/meta.json").read_text())
    truth = np.load(ROOT / f"data/uzh/{unit}/truth.npz")
    from src.depicted.uzh import published_mount
    R_cb, _ = published_mount(meta)
    R_true = np.einsum("nij,jk->nik",
                       Rotation.from_quat(np.asarray(truth["quat"], float)).as_matrix(), R_cb)
    fps = float(meta["fps_measured"])
    n = min(len(R_est), len(R_true))
    return R_est[:n], np.arange(n) / fps, fps, R_true[:n], np.asarray(truth["pos"], float)[:n], R_cb


def reader_start(kind: str, unit: str):
    """Per-frame reader up vector (variant a for UZH, frozen for sim)."""
    if kind == "sim":
        r = np.load(Path("/home/kevin/Documents/Projects/awmv-l/data/l01/scores") / f"{unit}.npz")
        u = np.asarray(r["up_est"], float)
        return u / np.linalg.norm(u, axis=1, keepdims=True)
    rows = [json.loads(l) for l in (ROOT / f"data/n01/uzh_reader/{unit}__a.jsonl").read_text().splitlines() if l.strip()]
    n = max(r["index"] for r in rows) + 1
    u = np.full((n, 3), np.nan)
    for r in rows:
        v = np.asarray(r["gravity_up_cam"], float)
        u[r["index"]] = v / np.linalg.norm(v)
    return u


def main() -> int:
    import pandas as pd

    pairs = pd.concat([pd.read_csv(OUT / f) for f in ("pairs_sim.csv", "pairs_uzh.csv") if (OUT / f).exists()],
                      ignore_index=True)
    pairs = pairs[(pairs.status == "ok") & (pairs.gap_s == GAP)].copy()
    rows = []
    for (kind, unit), g in pairs.groupby(["kind", "unit"]):
        try:
            R_est, t, fps, R_true, pos, R_cb = unit_data(kind, unit)
            u_reader = reader_start(kind, unit)
        except Exception as exc:  # noqa: BLE001
            rows.append({"kind": kind, "unit": unit, "status": f"load_failed:{exc}"})
            continue
        g = g.sort_values("i")
        wins = {}
        for _, r in g.iterrows():
            wins.setdefault(int(r.i), r)
        starts = sorted(wins)
        for a, w1 in enumerate(starts):
            for w2 in starts[a + 1:]:
                if not (TURN_MIN <= t[w2] - t[w1] <= TURN_MAX):
                    continue
                r1, r2 = wins[w1], wins[w2]
                if not (np.isfinite(u_reader[w1]).all() and np.isfinite(u_reader[w2]).all()):
                    continue
                t_est1 = parse_vec(r1.t_est)
                t_est2 = parse_vec(r2.t_est)
                v1 = -(R_true[w1] @ parse_vec(r1.t_true))
                v2 = -(R_true[w2] @ parse_vec(r2.t_true))
                v1 = v1 / np.linalg.norm(v1)
                v2 = v2 / np.linalg.norm(v2)
                theta = np.degrees(np.arccos(np.clip(v1 @ v2, -1, 1)))
                # start at w1 with the reader, propagate both windows with DPVO
                R_pred1 = R_est[w1]  # relative construction: anchor the true attitude at w1
                R_pred1 = R_true[w1]
                R_pred2 = R_true[w1] @ R_est[w1].T @ R_est[w2]
                # the reader start is a constant correction to the propagated attitude
                u0 = u_reader[w1]
                # minimal body-frame rotation that takes the propagated up to the reader start
                C_start = Rotation.align_vectors([up_of(R_pred1)], [u0])[0].as_matrix()
                Rp1 = R_pred1 @ C_start
                Rp2 = R_pred2 @ C_start
                base = ang(up_of(Rp2), up_of(R_true[w2]))
                c1 = solve_correction([Rp2], [v2], [t_est2], two=False)
                if c1 is None:
                    continue
                one = ang(up_of(Rp2 @ body_corr(c1)), up_of(R_true[w2]))
                cc = solve_correction([Rp1, Rp2], [v1, v2], [t_est1, t_est2], two=True)
                if cc is None:
                    continue
                two_arm = ang(up_of(Rp2 @ body_corr(cc)), up_of(R_true[w2]))
                rows.append({"kind": kind, "unit": unit, "w1": w1, "w2": w2, "theta_deg": theta,
                             "baseline_deg": base, "one_constraint_deg": one,
                             "two_constraint_deg": two_arm, "status": "ok"})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "turn_roll.csv", index=False)
    ok = df[df.status == "ok"]
    out = {}
    for (kind,), g in ok.groupby(["kind"]):
        pass
    for kind in ("sim", "uzh"):
        g = ok[ok.kind == kind]
        out[kind] = {}
        for lo, hi in BINS:
            b = g[(g.theta_deg >= lo) & (g.theta_deg < hi)]
            out[kind][f"{lo:.0f}-{hi:.0f}"] = {
                "n": int(len(b)),
                "baseline_median_deg": float(b.baseline_deg.median()) if len(b) else None,
                "one_constraint_median_deg": float(b.one_constraint_deg.median()) if len(b) else None,
                "two_constraint_median_deg": float(b.two_constraint_deg.median()) if len(b) else None,
            }
    out["reading"] = {
        "supported_if": "theta >= 30: two-constraint median <= 5.0 deg and >= 30 % below baseline",
    }
    (OUT / "turn_summary.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if k != "reading":
            print(k, json.dumps(v, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
