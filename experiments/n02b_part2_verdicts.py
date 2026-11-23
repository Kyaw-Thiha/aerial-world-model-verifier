"""Track N, Part 2: verdict tiers on the rotation-propagated lean (UZH outdoor, descriptive).

Pre-registered in ``configs/n02b_part2_prereg.yaml`` before any DPVO run on the edited clips and
before any tier score. Reads the stored reader lean per condition, the DPVO poses on the same
edited frames, and the UZH truth.

    PYTHONPATH=. python experiments/n02b_part2_verdicts.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.depicted.uzh import published_mount
from src.verdict.lean import (G0, E3, depicted_lean_from_gravity, required_lean,
                              world_to_yaw_frame)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/n02b"
CLIPS = ROOT / "data/clips_v3"
POSES = ROOT / "data/n01/poses" / "uzh_edits"
KD_GRID = np.round(np.arange(0.0, 0.401, 0.05), 3)
TAU_CORR = 0.619653015661502
VIOL = ("tilt_removal", "scale_0.5", "scale_1.5", "mirror_roll")


def axis_of(l: np.ndarray) -> np.ndarray:
    """Unit thrust axis from a lean 2-vector (x, y) = horizontal components."""
    z2 = np.clip(1.0 - np.sum(l ** 2, axis=-1), 0.0, 1.0)
    a = np.concatenate([np.asarray(l, float), np.sqrt(z2)[..., None]], axis=-1)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def ang_lean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.sum(axis_of(a) * axis_of(b), axis=-1), -1.0, 1.0)))


def sequence_truth(seq: str):
    """500 Hz truth for the derivatives, plus the 15.07 Hz frame times to sample it at."""
    meta = json.loads((ROOT / f"data/uzh/{seq}/meta.json").read_text())
    truth = np.load(ROOT / f"data/uzh/{seq}/truth.npz")
    gt = np.asarray(truth["gt"], float)
    t0 = float(truth["gt_t0"])
    t500 = gt[:, 0] - t0
    pos = gt[:, 1:4]
    yaw = Rotation.from_quat(gt[:, 4:8]).as_euler("ZYX")[:, 0]
    frame_rel = np.asarray(truth["t_unix"], float) - t0
    R_cb, _ = published_mount(meta)
    return pos, yaw, t500, R_cb, frame_rel


def required_v3(pos, yaw, t):
    rl = required_lean(pos, yaw, t, window_s=0.37)
    q = rl.acc + G0 * E3 + 0.295 * rl.vel
    n = np.linalg.norm(q, axis=1)
    good = np.isfinite(n) & (n > 1e-9)
    u = np.full_like(q, np.nan)
    u[good] = q[good] / n[good, None]
    lean = world_to_yaw_frame(u[:, :2], yaw)
    mask = rl.mask & np.isfinite(lean).all(axis=1)
    return lean, rl.velocity_reg, mask


def up_cam_from_lean(l: np.ndarray, yaw: float, R_cb: np.ndarray) -> np.ndarray:
    """Camera-frame world up from a lean 2-vector (schema convention, yaw-free)."""
    from src.depicted.clips import up_body_from_lean
    u_b = np.asarray(up_body_from_lean(np.asarray(l, float)[None])[0], float)
    return R_cb.T @ u_b


def propagate_lean(clip_id: str, seq: str, yaw0: float, l0: np.ndarray, R_cb: np.ndarray,
                   yaw: np.ndarray, n: int) -> np.ndarray:
    npz = POSES / f"{clip_id}.npz"
    if not npz.exists():
        return np.full((n, 2), np.nan)
    d = np.load(npz)
    R = np.asarray(d["c2w"], float)[:, :3, :3]
    m = min(len(R), n)
    u0 = up_cam_from_lean(l0, yaw0, R_cb)
    up = np.full((m, 3), np.nan)
    for k in range(m):
        rel = R[k].T @ R[0]
        up[k] = rel @ u0
    lean = np.full((n, 2), np.nan)
    lean[:m] = depicted_lean_from_gravity(up, R_cb)
    return lean


def f0_score(l_dep, l_req, mask):
    ok = mask & np.isfinite(l_dep).all(1) & np.isfinite(l_req).all(1)
    if ok.sum() < 5:
        return float("nan"), int(ok.sum())
    return float(np.median(ang_lean(l_dep[ok], l_req[ok]))), int(ok.sum())


def c2_score(l_dep, l_req, vreg, mask):
    ok = mask & np.isfinite(l_dep).all(1) & np.isfinite(l_req).all(1) & np.isfinite(vreg).all(1)
    if ok.sum() < 8:
        return float("nan"), int(ok.sum())
    ld, lr, vr = l_dep[ok], l_req[ok], vreg[ok]
    best = (float("inf"), None)
    for kd in KD_GRID:
        r = ld - lr - kd * vr
        off = np.median(r, axis=0)
        res = ang_lean(r, np.zeros_like(r)) if False else ang_lean(ld, lr + kd * vr + off)
        s = float(np.median(res))
        if s < best[0]:
            best = (s, kd)
    return best[0], int(ok.sum())


def coherence_score(l_dep, l_req, mask) -> float:
    ok = mask & np.isfinite(l_dep).all(1) & np.isfinite(l_req).all(1)
    if ok.sum() < 8:
        return float("nan")
    rs = []
    for ax in range(2):
        a, b = l_dep[ok, ax], l_req[ok, ax]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            rs.append(0.0)
        else:
            rs.append(float(np.corrcoef(a, b)[0, 1]))
    return 1.0 - min(rs)


def main() -> int:
    rows = list(csv.DictReader(open(OUT / "clips.csv")))
    seq_cache = {s: sequence_truth(s) for s in {r["seq"] for r in rows}}
    req_cache = {s: required_v3(*seq_cache[s][:3]) for s in seq_cache}
    # sample the 500 Hz required side at the 15.07 Hz frame times
    frame_cache = {}
    for s, (_, _, t500, _, frame_rel) in seq_cache.items():
        lean500, vreg500, mask500 = req_cache[s]
        lean_f = np.column_stack([np.interp(frame_rel, t500, lean500[:, k]) for k in range(2)])
        vreg_f = np.column_stack([np.interp(frame_rel, t500, vreg500[:, k]) for k in range(2)])
        idx = np.clip(np.searchsorted(t500, frame_rel), 0, len(t500) - 1)
        frame_cache[s] = (lean_f, vreg_f, mask500[idx])
    out_rows = []
    for r in rows:
        cid, seq, cond = r["clip_id"], r["seq"], r["condition"]
        meta = json.loads((CLIPS / seq / f"{cid}.meta.json").read_text())
        d = np.load(CLIPS / seq / f"{cid}.npz")
        _pos, yaw_full, _t_full, R_cb, _fps = seq_cache[seq]
        l_req_full, vreg_full, mask_full = frame_cache[seq]
        a, b = meta["index"]
        l_req = l_req_full[a:b]
        vreg = vreg_full[a:b]
        mask = mask_full[a:b]
        n = len(l_req)
        l_reader = np.asarray(d["depicted_lean"], float)[:n]
        yaw = yaw_full[a:b]
        # start from the first frame valid in the required mask and finite in the reader
        start = next((i for i in range(n) if mask[i] and np.isfinite(l_reader[i]).all()), None)
        if start is None:
            l_prop = np.full((n, 2), np.nan)
        else:
            l_prop = propagate_lean(cid, seq, float(yaw[start]), l_reader[start], R_cb, yaw, n)
        rec = {"clip_id": cid, "seq": seq, "condition": cond, "n": n,
               "n_mask": int(mask.sum())}
        # pipeline diagnostic: oracle-start propagation error at 2 s (extractor validity only)
        try:
            from src.depicted.clips import up_body_from_lean
            npz = POSES / f"{cid}.npz"
            Rs = np.load(npz)["c2w"][:, :3, :3]
            i0 = int(np.argmax(mask))
            u_b = up_body_from_lean(l_req[i0][None])[0]
            u_cam0 = R_cb.T @ u_b
            step = int(round(2.0 * 15.07))
            ups = np.array([(Rs[k].T @ Rs[0]) @ u_cam0
                            for k in range(min(len(Rs), i0 + step + 1))])
            lean_or = depicted_lean_from_gravity(ups, R_cb)
            rec["oracle_err_2s"] = float(np.median(ang_lean(lean_or[i0:], l_req[i0: i0 + len(lean_or)])))
        except Exception:
            rec["oracle_err_2s"] = float("nan")
        # gate inputs (Part 3): reader-vs-propagated disagreement and the path's peak tilt
        ok = mask & np.isfinite(l_reader).all(1) & np.isfinite(l_prop).all(1)
        rec["disagreement_deg"] = (float(np.median(ang_lean(l_reader[ok], l_prop[ok])))
                                   if ok.sum() >= 5 else float("nan"))
        uz = np.sqrt(np.clip(1.0 - np.sum(l_req[mask] ** 2, axis=1), 0.0, 1.0))
        rec["peak_tilt_deg"] = (float(np.degrees(np.max(np.arccos(np.clip(uz, -1, 1)))))
                                if mask.sum() else float("nan"))
        for arm, lean in (("reader", l_reader), ("propagated", l_prop)):
            rec[f"F0_{arm}"], rec[f"n_{arm}"] = f0_score(lean, l_req, mask)
            rec[f"C2_{arm}"], _ = c2_score(lean, l_req, vreg, mask)
            rec[f"R_{arm}"] = coherence_score(lean, l_req, mask)
            rec[f"lean_used_{arm}"] = int(np.isfinite(lean).all(1).sum())
        rec["valid_extraction"] = bool(np.isfinite(rec["oracle_err_2s"]) and rec["oracle_err_2s"] <= 10.0)
        out_rows.append(rec)
    with open(OUT / "per_clip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # leave-one-sequence-out thresholds from the real clips, per arm and tier
    seqs = sorted({r["seq"] for r in out_rows})
    summary = {"per_condition": {}, "per_condition_valid": {}, "per_sequence": {},
               "thresholds": {}, "n_valid": {}}
    for tier in ("F0", "C2", "R"):
        for arm in ("reader", "propagated"):
            key = f"{tier}_{arm}"
            thr = {}
            for s in seqs:
                other = [r[key] for r in out_rows
                         if r["seq"] != s and r["condition"] == "real" and np.isfinite(r[key])]
                thr[s] = float(np.quantile(other, 0.95)) if other else float("nan")
            summary["thresholds"][key] = thr
    for cond in ("real", "null_edit", "yaw_control", "tilt_removal", "scale_0.5",
                 "scale_1.5", "mirror_roll"):
        summary["n_valid"][cond] = int(sum(r["valid_extraction"] for r in out_rows
                                           if r["condition"] == cond))
        block = {}
        for tier in ("F0", "C2", "R"):
            for arm in ("reader", "propagated"):
                key = f"{tier}_{arm}"
                flags = 0
                tot = 0
                for r in out_rows:
                    if r["condition"] != cond:
                        continue
                    v = r[key]
                    t = summary["thresholds"][key][r["seq"]]
                    if not (np.isfinite(v) and np.isfinite(t)):
                        continue
                    tot += 1
                    flags += int(v > t)
                block[f"{key}"] = f"{flags}/{tot}"
        summary["per_condition"][cond] = block
        block_valid = {}
        for key in [f"{t}_{a}" for t in ("F0", "C2", "R") for a in ("reader", "propagated")]:
            flags = tot = 0
            for r in out_rows:
                if r["condition"] != cond or not r["valid_extraction"]:
                    continue
                v = r[key]
                t = summary["thresholds"][key][r["seq"]]
                if not (np.isfinite(v) and np.isfinite(t)):
                    continue
                tot += 1
                flags += int(v > t)
            block_valid[key] = f"{flags}/{tot}"
        summary["per_condition_valid"][cond] = block_valid
    for s in seqs:
        summary["per_sequence"][s] = {
            cond: {f"{tier}_{arm}": float(np.median([r[f"{tier}_{arm}"] for r in out_rows
                                                     if r["seq"] == s and r["condition"] == cond]))
                   for tier in ("F0", "C2", "R") for arm in ("reader", "propagated")}
            for cond in ("real", "tilt_removal", "mirror_roll", "yaw_control", "null_edit")}
    summary["clusters"] = len(seqs)
    summary["note"] = ("descriptive: 3 sequence clusters, 9 clips per condition; LOFO threshold "
                       "at the 95th percentile of the other sequences' real clips")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    for cond, b in summary["per_condition"].items():
        print(cond, {k: v for k, v in b.items() if k.startswith(("F0", "R_"))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
