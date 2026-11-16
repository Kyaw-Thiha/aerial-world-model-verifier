"""Track F, stage 0: pixel-path verdict statistics on stored DPVO paths (UZH forward rigs).

Reads only stored artefacts: Track C's DPVO poses on the Modal volume ``awmv-c-data``
(``poses/dpvo/<sequence>.npz``, copied to ``data/pathrec/poses/dpvo``), Track E's v3 clips
with the official schema-v2 arrays (``data/clips_v2_w370ms``) and the UZH truth already
paired to the prepared frames (``data/uzh``). CPU only.

Pre-registration: ``configs/f01_pixel_path_prereg.yaml`` (commit 6b88d5a) and amendment 1
(``configs/f01_pixel_path_prereg_amend1.yaml``, commit 17d21c6): the path scale and yaw
offset are estimated on the real condition of a clip and held fixed for its edits.

    python experiments/f01_stage0.py --out results/f01_stage0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.depicted.clips import up_body_from_lean
from src.depicted.uzh import published_mount
from src.pathrec.align import umeyama
from src.pathrec.pathverdict import (G0, KD_DEFAULT, band_coherence, horizontal_basis,
                                     odd_window, preintegration_fit, physics_scale_check,
                                     shooting_fit, smooth_derivative, velocity_drag_fit)

CONDITIONS = ("real", "null_edit", "yaw_control", "tilt_removal", "mirror_roll",
              "scale_0.5", "scale_1.5")
EDITS = ("tilt_removal", "mirror_roll")
SCALE_EDITS = ("scale_0.5", "scale_1.5")
CONTROLS = ("null_edit", "yaw_control")
ARMS = ("oracle_attitude", "estimated_attitude")
PATHS = ("pixel", "oracle")
SEQUENCES = ("indoor_forward_3", "indoor_forward_5", "indoor_forward_6",
             "indoor_forward_7", "indoor_forward_9", "indoor_forward_10",
             "outdoor_forward_1", "outdoor_forward_3", "outdoor_forward_5")
PRIMARY_SEQS = ("indoor_forward_3", "indoor_forward_5", "indoor_forward_9",
                "indoor_forward_10")
EXCLUDED_SEQS = ("indoor_forward_7",)
MIN_CROP = 0.25
N_BOOT = 5000
SEED = 1
VARIANTS = ("V1_preintegration", "V2_shooting", "V3_band_coherence",
            "V4_velocity_drag", "V5_physics_scale")


# --------------------------------------------------------------------------- inputs
@dataclass
class Case:
    seq: str
    clip: str
    idx: int
    cond: str
    arm: str
    path: str
    t: np.ndarray
    X_h: np.ndarray
    yaw: np.ndarray
    lean: np.ndarray
    lean_req: np.ndarray
    tilt_req: np.ndarray
    conf: np.ndarray
    depth: np.ndarray
    meta: dict
    dt: float
    f95: float
    window: int
    up: np.ndarray
    full_path: bool = False

    @property
    def n(self) -> int:
        return len(self.t)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _py(o):
    """JSON-safe plain Python from numpy scalars/arrays."""
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def load_index(clips_root: Path) -> list[dict]:
    """Every clip with its meta, one row per (sequence, index, condition)."""
    out = []
    for seq in SEQUENCES:
        for meta_path in sorted((clips_root / seq).glob("*_real.meta.json")):
            meta = json.loads(meta_path.read_text())
            idx = int(meta["clip_id"].rsplit("_", 2)[-2])
            for cond in CONDITIONS:
                p = clips_root / seq / f"{seq}_{idx:02d}_{cond}.npz"
                if not p.exists():
                    continue
                out.append({"seq": seq, "idx": idx, "cond": cond, "npz": p,
                            "meta": json.loads(
                                p.with_suffix(".meta.json").read_text())})
    return out


def build_case(row: dict, arm: str, path_kind: str, roots: dict,
               window_rule: dict) -> Case | None:
    seq, idx, cond = row["seq"], row["idx"], row["cond"]
    meta = row["meta"]
    lo, hi = (int(x) for x in meta["index"])
    d = np.load(row["npz"])
    truth = np.load(roots["uzh"] / seq / "truth.npz")
    t_all = np.asarray(truth["t"], dtype=np.float64)
    pos_all = np.asarray(truth["pos"], dtype=np.float64)
    from scipy.spatial.transform import Rotation
    quat = np.asarray(truth["quat"], dtype=np.float64)
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    R_bw_all = Rotation.from_quat(quat).as_matrix()
    yaw_truth_all = Rotation.from_matrix(R_bw_all).as_euler("ZYX")[:, 0]

    c2w_all = np.load(roots["poses"] / f"{seq}.npz")["c2w"]
    full_path = bool(lo + 60 <= len(c2w_all))
    if path_kind == "pixel":
        c2w = c2w_all
        hi = min(hi, len(c2w))
        if hi - lo < 24:
            return None
        P = c2w[lo:hi, :3, 3]
        R_wc = c2w[lo:hi, :3, :3]
    else:
        P = pos_all[lo:hi]

    t = t_all[lo:hi]
    R_bw = R_bw_all[lo:hi]
    yaw_truth = yaw_truth_all[lo:hi]
    R_cb = np.asarray(meta["mount_rotation_R_cb"], dtype=np.float64)
    # the published mount, read the same way, must be what the clip stored
    R_cb_pub, _ = published_mount(json.loads((roots["uzh"] / seq / "meta.json").read_text()))
    if not np.allclose(R_cb, R_cb_pub, atol=1e-6):
        raise ValueError(f"{seq}: clip mount differs from the published Kalibr mount")

    n_avail = hi - lo
    lean_dep = np.asarray(d["depicted_lean"], dtype=np.float64)[:n_avail]
    z_b = R_bw[:, :, 2]
    lean_att = np.column_stack([np.cos(-yaw_truth) * z_b[:, 0] - np.sin(-yaw_truth) * z_b[:, 1],
                                np.sin(-yaw_truth) * z_b[:, 0] + np.cos(-yaw_truth) * z_b[:, 1]])

    if arm == "oracle_attitude":
        lean = lean_att
        u_b = R_bw[:, 2, :]                       # world up in body (third row)
        yaw = yaw_truth
    else:
        lean = lean_dep
        u_b = up_body_from_lean(lean_dep)
        yaw = None                                # path yaw for the pixel path, else truth

    if path_kind == "pixel":
        # body->path world = R_wc R_cb^T; map the body-frame world up into the path's world
        R_wb = np.einsum("nij,jk->nik", R_wc, R_cb.T)
        up_frames = np.einsum("nij,nj->ni", R_wb, u_b)
        # a NaN attitude frame must not poison the plane: average the finite frames only
        up_ok = np.isfinite(up_frames).all(axis=1)
        if up_ok.sum() < 24:
            return None
        up = up_frames[up_ok].mean(axis=0)
        if not np.isfinite(up).all() or np.linalg.norm(up) < 1e-9:
            return None
        up = up / np.linalg.norm(up)
        if arm == "estimated_attitude":
            yaw = np.arctan2(R_wb[:, 1, 0], R_wb[:, 0, 0])
    else:
        up = np.array([0.0, 0.0, 1.0])
        if arm == "estimated_attitude":
            yaw = yaw_truth

    e1, e2 = horizontal_basis(up)
    X_h = np.column_stack([(P - P[0]) @ e1, (P - P[0]) @ e2])
    tilt_req = np.asarray(d["tilt_req_deg"], dtype=np.float64)[:n_avail]
    lean_req = np.asarray(d["lean_req"], dtype=np.float64)[:n_avail]
    conf = np.asarray(d["depicted_conf"], dtype=np.float64)[:n_avail]
    depth = np.full(len(t), np.nan)
    dep_path = roots["depth"]
    if dep_path.exists():
        rec = next((r for r in json.loads(dep_path.read_text()) if r["seq"] == seq), None)
        if rec is not None:
            arr = np.asarray(rec["median_depth_m"], dtype=np.float64)
            m = max(0, min(len(arr) - lo, len(t)))
            if m > 0:
                depth[:m] = arr[lo:lo + m]

    # longest contiguous run where the path, the lean and the time base are all usable
    yaw = np.asarray(yaw, dtype=np.float64).ravel()
    ok = (np.isfinite(X_h).all(axis=1) & np.isfinite(lean).all(axis=1)
          & np.isfinite(t) & np.isfinite(yaw))
    if ok.sum() < 24:
        return None
    sl = slice(0, len(ok))
    if not ok.all():
        from src.pathrec.pathverdict import longest_finite_run
        sl = longest_finite_run(ok)
    if sl.stop - sl.start < 24:
        return None
    dt = float(np.median(np.diff(t[sl])))
    f95 = float(window_rule["per_sequence"][seq]["f95_hz"])
    window = odd_window(sl.stop - sl.start, dt, 0.5 / f95)
    return Case(seq=seq, clip=meta["clip_id"], idx=idx, cond=cond, arm=arm, path=path_kind,
                t=t[sl] - t[sl][0], X_h=X_h[sl], yaw=np.asarray(yaw)[sl], lean=lean[sl],
                lean_req=lean_req[sl], tilt_req=tilt_req[sl], conf=conf[sl],
                depth=depth[sl], meta=meta, dt=dt, f95=f95, window=window, up=up,
                full_path=full_path)


# --------------------------------------------------------------------------- statistics
def compute_variant(case_real: Case, case_cond: dict[str, Case]) -> dict:
    """All five variants for one clip: fit the real condition, score every condition."""
    out: dict[str, dict] = {}
    n, dt = case_real.n, case_real.dt
    Xr = case_real.X_h
    window = case_real.window

    # V2 shooting -------------------------------------------------------------
    real_fit = shooting_fit(Xr, case_real.yaw, case_real.lean, case_real.t)
    rows = {}
    if real_fit.ok:
        for cond, cc in case_cond.items():
            fit = (real_fit if cond == "real" else
                   shooting_fit(cc.X_h, cc.yaw, cc.lean, cc.t, scale=real_fit.scale,
                                psi=np.deg2rad(real_fit.psi_deg)))
            rows[cond] = {"statistic": fit.statistic, "psi_deg": fit.psi_deg,
                          "scale": fit.scale, "rel_se": fit.scale_rel_se}
        # sensitivity: refit psi per condition at the fixed scale
        refit = {}
        if case_real.arm == "estimated_attitude" and case_real.path == "pixel":
            coarse = np.deg2rad(np.arange(-180.0, 180.0, 5.0))
            for cond, cc in case_cond.items():
                f = shooting_fit(cc.X_h, cc.yaw, cc.lean, cc.t, scale=real_fit.scale,
                                 psi_grid=coarse)
                refit[cond] = {"statistic": f.statistic, "psi_deg": f.psi_deg}
        out["V2_shooting"] = {"rows": rows, "real_fit": _fit_dict(real_fit),
                              "psi_refit_sensitivity": refit}
    else:
        out["V2_shooting"] = {"rows": {}, "real_fit": _fit_dict(real_fit),
                              "psi_refit_sensitivity": {}}

    # V1 preintegration -------------------------------------------------------
    def vobs(c: Case) -> np.ndarray:
        return smooth_derivative(c.X_h, c.dt, c.window, 1)

    real_v = vobs(case_real)
    fit1 = preintegration_fit(real_v, case_real.yaw, case_real.lean, case_real.t)
    rows = {}
    if fit1.ok:
        for cond, cc in case_cond.items():
            f = (fit1 if cond == "real" else
                 preintegration_fit(vobs(cc), cc.yaw, cc.lean, cc.t, scale=fit1.scale,
                                    psi=np.deg2rad(fit1.psi_deg)))
            extra = {k: (float(np.real(v)) if isinstance(v, complex) else v)
                     for k, v in f.extra.items() if k.startswith(("r_", "s_"))}
            rows[cond] = {"statistic": f.statistic, "psi_deg": f.psi_deg,
                          "scale": f.scale, **extra}
        out["V1_preintegration"] = {"rows": rows, "real_fit": _fit_dict(fit1)}
    else:
        out["V1_preintegration"] = {"rows": {}, "real_fit": _fit_dict(fit1)}

    # V3 band coherence -------------------------------------------------------
    def acc(c: Case) -> np.ndarray:
        return smooth_derivative(c.X_h, c.dt, c.window, 2)

    rows = {}
    for cond, cc in case_cond.items():
        res = band_coherence(acc(cc), cc.yaw, cc.lean, cc.dt, f_lo=0.05, f_hi=cc.f95)
        rows[cond] = res
    out["V3_band_coherence"] = {"rows": rows}

    # V4 velocity drag -------------------------------------------------------
    fit4 = velocity_drag_fit(real_v, case_real.yaw, case_real.lean, case_real.t)
    rows = {}
    if fit4.ok:
        for cond, cc in case_cond.items():
            f = (fit4 if cond == "real" else
                 velocity_drag_fit(vobs(cc), cc.yaw, cc.lean, cc.t, gain=fit4.scale,
                                   psi=np.deg2rad(fit4.psi_deg)))
            rows[cond] = {"statistic": f.statistic, "psi_deg": f.psi_deg, "gain": f.scale}
    a_real = smooth_derivative(case_real.X_h, case_real.dt, case_real.window, 2)
    speed = np.linalg.norm(real_v, axis=1)
    acc_norm = np.linalg.norm(a_real, axis=1)
    keep = speed > 1e-9
    ratio = float(np.nanmedian(acc_norm[keep] / (KD_DEFAULT * speed[keep]))) if keep.any() \
        else float("inf")
    out["V4_velocity_drag"] = {"rows": rows if fit4.ok else {}, "real_fit": _fit_dict(fit4),
                               "acc_over_drag": ratio, "drag_dominated": bool(ratio < 0.5)}

    # V5 physics scale. The real-condition check is the registered one; per-condition free
    # fits are reported as the variant's own edit channel (a removed tilt has no acceleration
    # to scale to, so the scale stops being identifiable). Labelled exploratory in F01.
    steps = np.linalg.norm(np.diff(case_real.X_h, axis=0), axis=1)
    check = physics_scale_check(real_fit, steps, case_real.depth[1:])
    rows = {}
    free = {}
    for cond, cc in case_cond.items():
        f = real_fit if cond == "real" else shooting_fit(cc.X_h, cc.yaw, cc.lean, cc.t)
        chk = physics_scale_check(f, np.linalg.norm(np.diff(cc.X_h, axis=0), axis=1),
                                  cc.depth[1:])
        free[cond] = {**chk, "statistic": f.statistic}
        rows[cond] = {"statistic": float(chk["flag"]), "scale": f.scale,
                      "rel_se": f.scale_rel_se}
    out["V5_physics_scale"] = {"real_check": check, "free_per_condition": free,
                               "rows": rows}
    return out


def _fit_dict(fit) -> dict:
    return {"statistic": fit.statistic, "psi_deg": fit.psi_deg, "scale": fit.scale,
            "rel_se": fit.scale_rel_se, "r2": fit.r2, "ok": fit.ok}


# --------------------------------------------------------------------------- evaluation
def testability(case: Case, gate: dict, rule_src: str) -> dict:
    """E8/E9 population rules, applied to a real clip."""
    meta = case.meta
    test = meta.get("testability", {})
    keep = True
    if gate is not None:
        from src.verdict.abstain import gate_kept_from_conf
        keep = gate_kept_from_conf(case.conf, gate)
    reasons = []
    if not test.get("testable", True):
        reasons.extend(test.get("abstain_reasons", []))
    if not keep:
        reasons.append("confidence_gate")
    if case.meta.get("crop_fraction", 1.0) is not None and \
            float(case.meta.get("crop_fraction", 1.0)) < MIN_CROP:
        reasons.append("insufficient_field")
    if case.seq in EXCLUDED_SEQS:
        reasons.append("broken_export")
    if not rule_src:
        reasons.append("no_scale")
    return {"testable": not reasons, "reasons": reasons, "gate_kept": keep,
            "n_signal": test.get("n_signal")}


def loso_flags(rows: list[dict], value: str, primary_seqs: tuple[str, ...]) -> dict:
    """Leave-one-sequence-out conformal thresholds on the real condition.

    For a held-out sequence, the threshold is the 95th percentile of the real clips in the
    other sequences; the held-out real and edited clips are flagged against it. If a variant
    has no finite real values outside a sequence, its threshold is NaN and nothing flags.
    """
    real = [r for r in rows if r["cond"] == "real" and r.get("testable")
            and np.isfinite(r.get(value, np.nan))]
    per_seq = {}
    for seq in primary_seqs:
        train = [r[value] for r in real if r["seq"] != seq]
        tau = float(np.quantile(train, 0.95)) if len(train) >= 4 else float("nan")
        per_seq[seq] = tau
    for r in rows:
        tau = per_seq.get(r["seq"], float("nan"))
        r["tau"] = tau
        if r.get("testable") and np.isfinite(r.get(value, np.nan)) and np.isfinite(tau):
            r["flag"] = bool(r[value] > tau)
        else:
            r["flag"] = None
    return per_seq


def rate(rows: list[dict], cond: str) -> dict:
    sel = [r for r in rows if r["cond"] == cond and r.get("testable")
           and r.get("flag") is not None]
    n = len(sel)
    k = sum(1 for r in sel if r["flag"])
    seqs = sorted({r["seq"] for r in sel})
    return {"n": n, "k": k, "rate": (k / n) if n else None, "n_sequences": len(seqs)}


def clustered(rows: list[dict], cond: str, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    seqs = sorted({r["seq"] for r in rows if r.get("testable")})
    if len(seqs) < 3:
        return {"note": "fewer than 3 clusters; not computed"}
    rng = np.random.default_rng(seed)
    by = {s: [r for r in rows if r["seq"] == s and r["cond"] == cond
              and r.get("testable") and r.get("flag") is not None] for s in seqs}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(seqs), size=len(seqs), replace=True)
        sub = [r for i in pick for r in by[seqs[i]]]
        if sub:
            vals.append(np.mean([bool(r["flag"]) for r in sub]))
    if len(vals) < 20:
        return {"note": "too few bootstrap replicates"}
    return {"lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            "n_boot": len(vals), "n_clusters": len(seqs),
            "note": "sequence-clustered; descriptive below ~10 clusters"}


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/f01_stage0")
    ap.add_argument("--clips", default="data/clips_v2_w370ms")
    ap.add_argument("--uzh", default="data/uzh")
    ap.add_argument("--poses", default="data/pathrec/poses/dpvo")
    ap.add_argument("--window-rule", default="results/e_window_rule.json")
    ap.add_argument("--depth", default="results/pathrec_calib/depth.json")
    ap.add_argument("--gate", default="configs/gate_frozen.yaml")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    roots = {"uzh": Path(args.uzh), "poses": Path(args.poses),
             "depth": Path(args.depth)}
    window_rule = json.loads(Path(args.window_rule).read_text())
    import yaml
    gate = yaml.safe_load(Path(args.gate).read_text())
    index = load_index(Path(args.clips))
    print(f"{len(index)} (sequence, clip, condition) rows")

    # group by clip so the real condition can set the ruler
    groups: dict[tuple[str, int], dict[str, dict]] = {}
    for row in index:
        groups.setdefault((row["seq"], row["idx"]), {})[row["cond"]] = row

    out_rows: list[dict] = []
    skipped: list[dict] = []
    for (seq, idx), conds in sorted(groups.items()):
        if "real" not in conds:
            continue
        for arm in ARMS:
            for path_kind in PATHS:
                cases = {}
                for cond, row in conds.items():
                    c = build_case(row, arm, path_kind, roots, window_rule)
                    if c is not None:
                        cases[cond] = c
                if "real" not in cases:
                    skipped.append({"seq": seq, "idx": idx, "arm": arm, "path": path_kind,
                                    "why": "no full real path"})
                    continue
                res = compute_variant(cases["real"], cases)
                for variant in VARIANTS:
                    vres = res[variant]
                    for cond, cc in cases.items():
                        blob = vres.get("rows", {}).get(cond, {})
                        stat = blob.get("statistic")
                        out_rows.append({
                            "seq": seq, "clip": cc.clip, "idx": idx, "cond": cond,
                            "arm": arm, "path": path_kind, "variant": variant,
                            "statistic": stat,
                            "psi_deg": blob.get("psi_deg"),
                            "scale": blob.get("scale"),
                            "n_frames": cc.n,
                            "full_path": cc.full_path,
                            "roll_spread": float(np.nanstd(cc.lean[:, 1]))
                            if cond == "real" else None,
                            "pitch_spread": float(np.nanstd(cc.lean[:, 0]))
                            if cond == "real" else None,
                            "acc_over_drag": vres.get("acc_over_drag")
                            if variant == "V4_velocity_drag" else None,
                            "testability": testability(cc, gate, True) if cond == "real"
                            else None,
                        })

    # --- population rules and rates
    summary = {}
    for variant in VARIANTS:
        summary[variant] = {}
        for arm in ARMS:
            summary[variant][arm] = {}
            for path_kind in PATHS:
                sub = [r for r in out_rows if r["variant"] == variant and r["arm"] == arm
                       and r["path"] == path_kind]
                for r in sub:
                    if r["cond"] != "real":
                        base = next((b for b in sub if b["cond"] == "real"
                                     and b["seq"] == r["seq"] and b["idx"] == r["idx"]), None)
                        r["testable"] = bool(base and base["testability"]["testable"]) \
                            if base else False
                        r["testable"] = r["testable"] and np.isfinite(
                            r["statistic"]) if r["statistic"] is not None else False
                    else:
                        r["testable"] = r["testability"]["testable"] and \
                            r["statistic"] is not None and np.isfinite(r["statistic"])
                    if variant == "V4_velocity_drag" and not r.get("drag_dominated", False):
                        r["testable"] = False
                if variant == "V4_velocity_drag":
                    for r in sub:
                        if r["cond"] != "real":
                            base = next((b for b in sub if b["cond"] == "real"
                                         and b["seq"] == r["seq"] and b["idx"] == r["idx"]),
                                        None)
                            r["drag_dominated"] = bool(base and base.get("drag_dominated"))
                    for r in sub:
                        if r["cond"] == "real":
                            ratio = r.get("acc_over_drag")
                            r["drag_dominated"] = bool(ratio is not None
                                                       and np.isfinite(ratio) and ratio < 0.5)
                primary = [r for r in sub if r["seq"] in PRIMARY_SEQS and r["full_path"]]
                tau = loso_flags(primary, "statistic", PRIMARY_SEQS)
                pooled = [r["statistic"] for r in primary if r["cond"] == "real"
                          and r.get("testable") and np.isfinite(r["statistic"])]
                primary_ids = {(r["seq"], r["idx"]) for r in primary}
                if len(pooled) >= 4:
                    tau_pool = float(np.quantile(pooled, 0.95))
                    for r in sub:
                        if (r["seq"], r["idx"]) not in primary_ids:
                            r["tau"] = tau_pool
                            r["flag"] = bool(r.get("testable")
                                             and np.isfinite(r["statistic"])
                                             and r["statistic"] > tau_pool)
                primary_rates = {c: rate(primary, c) for c in CONDITIONS}
                expl = [r for r in sub if (r["seq"], r["idx"]) not in primary_ids]
                summary[variant][arm][path_kind] = {
                    "primary": {"n_clips": len({(r['seq'], r['idx']) for r in primary
                                                if r['cond'] == 'real'}),
                                "n_sequences": len({r['seq'] for r in primary
                                                    if r['cond'] == 'real'}),
                                "rates": primary_rates,
                                "tau": tau,
                                "clustered": {c: clustered(primary, c, args.n_boot)
                                              for c in CONDITIONS},
                                "abstained_real": sum(
                                    1 for r in primary if r["cond"] == "real"
                                    and not r["testable"])},
                    "exploratory": {"n_clips": len({(r['seq'], r['idx']) for r in expl
                                                    if r['cond'] == 'real'}),
                                    "n_sequences": len({r['seq'] for r in expl
                                                        if r['cond'] == 'real'}),
                                    "rates": {c: rate(expl, c) for c in CONDITIONS}},
                }

    # --- scale provenance and path quality (no truth in any statistic above)
    path_quality = []
    for seq in SEQUENCES:
        poses = np.load(Path(args.poses) / f"{seq}.npz")["c2w"]
        truth = np.load(Path(args.uzh) / seq / "truth.npz")
        n = min(len(poses), len(truth["pos"]))
        R, s, tr = umeyama(poses[:n, :3, 3], truth["pos"][:n], with_scale=True)
        est = s * poses[:n, :3, 3] @ R.T + tr
        err = float(np.sqrt(np.mean(np.sum((est - truth["pos"][:n]) ** 2, axis=1))))
        path_quality.append({"seq": seq, "n": n, "scale_sim3": float(s), "ate_sim3_m": err,
                             "path_len_units": float(np.linalg.norm(
                                 np.diff(poses[:n, :3, 3], axis=0), axis=1).sum())})

    provenance = {
        "commit": _git_head(),
        "prereg_commit": "6b88d5a",
        "amendment": "configs/f01_pixel_path_prereg_amend1.yaml",
        "inputs": {
            "poses": str(Path(args.poses)),
            "clips": str(Path(args.clips)),
            "uzh": str(Path(args.uzh)),
            "window_rule": str(Path(args.window_rule)),
            "depth": str(Path(args.depth)),
        },
        "pose_sha256": {s: _sha256(Path(args.poses) / f"{s}.npz") for s in SEQUENCES},
        "n_rows": len(out_rows),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = _py({"provenance": provenance, "summary": summary, "skipped": skipped,
         "path_quality": path_quality,
         "notes": [
             "Primary: 4 indoor-forward sequences, E's R2/coherence-testable real clips with "
             "a full 60-frame path. Exploratory: outdoor, and shorter overlaps.",
             "Thresholds are leave-one-sequence-out conformal 95th percentiles on real clips.",
             "Sequence-clustered intervals are descriptive below ~10 clusters.",
         ]})
    (out / "summary.json").write_text(json.dumps(payload, indent=1))
    import csv
    with open(out / "rows.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["seq", "clip", "idx", "cond", "arm", "path",
                                                "variant", "statistic", "psi_deg", "scale",
                                                "n_frames", "full_path", "roll_spread",
                                                "pitch_spread", "acc_over_drag", "testable",
                                                "tau", "flag"])
        writer.writeheader()
        for r in out_rows:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})
    print(json.dumps({"rows": len(out_rows), "skipped": len(skipped)}, indent=1))
    for variant in VARIANTS:
        block = summary[variant]["estimated_attitude"]["pixel"]["primary"]
        line = "  ".join(f"{c}={_fmt(block['rates'][c])}" for c in CONDITIONS)
        print(f"{variant:20s} est/pixel {line}")
    return 0


def _fmt(r: dict) -> str:
    if not r or r.get("rate") is None:
        return "n/a"
    return f"{r['rate']:.3f}({r['k']}/{r['n']})"


if __name__ == "__main__":
    raise SystemExit(main())
