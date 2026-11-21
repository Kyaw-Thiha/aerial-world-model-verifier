"""Track F2, part 1: spline-smoothed joint flatness fit on stored pixel paths (CPU).

Inputs and population conventions are carried over from ``experiments/f01_stage0.py`` on branch
``track-f-pixel-path`` (commit 3aa8a13) so the population, attitude arms and the amendment-1
pairing are identical; only the statistic is new (``src/pathrec/spline_fit.py``).

Pre-registration: ``configs/f04_round2_prereg.yaml`` (commit 63002b8).

    python experiments/f04_spline_fit.py --path dpvo --out results/f04_spline/dpvo
    python experiments/f04_spline_fit.py --path all  --out results/f04_spline
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from src.depicted.clips import up_body_from_lean
from src.depicted.uzh import published_mount
from src.pathrec.spline_fit import (KD_DEFAULT, dyn_fit, dyn_fit_nov0, dyn_score_psi,
                                    fit_spline_path, horizontal_basis)

CONDITIONS = ("real", "null_edit", "yaw_control", "tilt_removal", "mirror_roll",
              "scale_0.5", "scale_1.5")
ARMS = ("estimated_attitude", "oracle_attitude")
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
LEVELS = (0.90, 0.95, 0.99)
SETTINGS = ("rho", "rho_coarse", "rho_fine", "rho_kd0", "rho_nov0", "rho_psi_refit")


# --------------------------------------------------------------------------- helpers
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def _py(o):
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def longest_finite_run(mask: np.ndarray) -> slice:
    mask = np.asarray(mask, dtype=bool)
    best_start, best_len, start = 0, 0, None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start > best_len:
                best_start, best_len = start, i
            start = None
    if start is not None and len(mask) - start > best_len:
        best_start, best_len = start, len(mask)
    return slice(best_start, best_start + best_len)


def load_index(clips_root: Path) -> list[dict]:
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
                            "meta": json.loads(p.with_suffix(".meta.json").read_text())})
    return out


def build_case(row: dict, arm: str, path_kind: str, roots: dict, window_rule: dict):
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
        hi = min(hi, len(c2w_all))
        if hi - lo < 24:
            return None
        P = c2w_all[lo:hi, :3, 3]
        R_wc = c2w_all[lo:hi, :3, :3]
    else:
        P = pos_all[lo:hi]

    t = t_all[lo:hi]
    R_bw = R_bw_all[lo:hi]
    yaw_truth = yaw_truth_all[lo:hi]
    R_cb = np.asarray(meta["mount_rotation_R_cb"], dtype=np.float64)
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
        u_b = R_bw[:, 2, :]
        yaw = yaw_truth
    else:
        lean = lean_dep
        u_b = up_body_from_lean(lean_dep)
        yaw = None

    if path_kind == "pixel":
        R_wb = np.einsum("nij,jk->nik", R_wc, R_cb.T)
        up_frames = np.einsum("nij,nj->ni", R_wb, u_b)
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
    conf = np.asarray(d["depicted_conf"], dtype=np.float64)[:n_avail]

    yaw = np.asarray(yaw, dtype=np.float64).ravel()
    ok = (np.isfinite(X_h).all(axis=1) & np.isfinite(lean).all(axis=1)
          & np.isfinite(t) & np.isfinite(yaw))
    if ok.sum() < 24:
        return None
    sl = slice(0, len(ok)) if ok.all() else longest_finite_run(ok)
    if sl.stop - sl.start < 24:
        return None
    dt = float(np.median(np.diff(t[sl])))
    f95 = float(window_rule["per_sequence"][seq]["f95_hz"])
    return {"seq": seq, "clip": meta["clip_id"], "idx": idx, "cond": cond, "arm": arm,
            "path": path_kind, "t": t[sl] - t[sl][0], "X_h": X_h[sl],
            "yaw": yaw[sl], "lean": lean[sl], "tilt_req": tilt_req[sl], "conf": conf[sl],
            "meta": meta, "dt": dt, "f95": f95, "full_path": full_path}


# --------------------------------------------------------------------------- statistic
def score_clip(case_real: dict, cases: dict[str, dict], n_intervals: int | None = None,
               k_d: float = KD_DEFAULT) -> dict:
    """Spline + joint fit: real condition sets (s, psi, v0); every condition is scored."""
    t = case_real["t"]
    sp = fit_spline_path(t, case_real["X_h"], 0.5 / case_real["f95"],
                         n_intervals=n_intervals)
    n = len(t)
    sl = np.s_[3:n - 3]
    real = dyn_fit(sp.t, sp.a, case_real["yaw"][sl], case_real["lean"][sl], k_d=k_d)
    if not real.ok:
        return {"ok": False, "knots": sp.n_intervals}
    psi = np.deg2rad(real.psi_deg)
    v0 = np.array(real.v0)
    real_nov0 = dyn_fit_nov0(sp.t, sp.a, sp.v, case_real["yaw"][sl], case_real["lean"][sl],
                             k_d=k_d)
    rows = {}
    for cond, cc in cases.items():
        yaw_c, lean_c = cc["yaw"][sl], cc["lean"][sl]
        if cond == "real":
            fit = real
        else:
            fit = dyn_fit(sp.t, sp.a, yaw_c, lean_c, k_d=k_d, scale=real.scale, psi=psi, v0=v0)
        refit = dyn_score_psi(sp.t, sp.a, yaw_c, lean_c, real.scale, v0, k_d=k_d)
        if cond == "real" or not real_nov0.ok:
            nov0 = real_nov0
        else:
            nov0 = dyn_fit_nov0(sp.t, sp.a, sp.v, cc["yaw"][sl], cc["lean"][sl], k_d=k_d,
                                scale=real_nov0.scale, psi=np.deg2rad(real_nov0.psi_deg))
        rows[cond] = {"rho": fit.statistic, "rho_model": fit.statistic_model,
                      "rho_demeaned": fit.statistic_demeaned,
                      "rho_psi_refit": refit.statistic if refit is not None else float("nan"),
                      "rho_nov0": nov0.statistic if nov0 is not None else float("nan"),
                      "psi_deg": fit.psi_deg, "scale": fit.scale,
                      "v0_x": fit.v0[0], "v0_y": fit.v0[1]}
    return {"ok": True, "knots": sp.n_intervals, "n_eval": len(sp.t),
            "real_fit": {"rho": real.statistic, "psi_deg": real.psi_deg,
                         "scale": real.scale, "v0": real.v0}, "rows": rows}


# --------------------------------------------------------------------------- evaluation
def testability(case: dict, gate: dict) -> dict:
    meta = case["meta"]
    test = meta.get("testability", {})
    keep = True
    if gate is not None:
        from src.verdict.abstain import gate_kept_from_conf
        keep = gate_kept_from_conf(case["conf"], gate)
    reasons = []
    if not test.get("testable", True):
        reasons.extend(test.get("abstain_reasons", []))
    if not keep:
        reasons.append("confidence_gate")
    crop = meta.get("crop_fraction", 1.0)
    if crop is not None and float(crop) < MIN_CROP:
        reasons.append("insufficient_field")
    if case["seq"] in EXCLUDED_SEQS:
        reasons.append("broken_export")
    return {"testable": not reasons, "reasons": reasons, "gate_kept": keep}


def auc(pos, neg) -> float:
    pos = np.asarray([v for v in pos if np.isfinite(v)], dtype=float)
    neg = np.asarray([v for v in neg if np.isfinite(v)], dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort") + 1.0
    order = np.argsort(allv, kind="mergesort")
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def clustered(rows: list[dict], cond: str, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    seqs = sorted({r["seq"] for r in rows if r.get("testable")})
    if len(seqs) < 3:
        return {"note": "fewer than 3 clusters"}
    rng = np.random.default_rng(seed)
    by = {s: [r for r in rows if r["seq"] == s and r["cond"] == cond and r.get("testable")]
          for s in seqs}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(seqs), size=len(seqs), replace=True)
        sub = [r for i in pick for r in by[seqs[i]]]
        if sub:
            vals.append(np.mean([bool(r.get("flag")) for r in sub]))
    if len(vals) < 20:
        return {"note": "too few replicates"}
    return {"lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            "n_boot": len(vals), "n_clusters": len(seqs),
            "note": "sequence-clustered; descriptive below ~10 clusters"}


def evaluate(rows: list[dict], value: str, level: float, n_boot: int) -> dict:
    """LOSO flags, rates, AUCs and (for q0.95) clustered intervals on one value column."""
    for r in rows:
        r["flag"] = False
        r["tau"] = float("nan")
    primary = [r for r in rows if r["seq"] in PRIMARY_SEQS and r["full_path"]]
    prim_ids = {(r["seq"], r["idx"]) for r in primary}
    real = [r for r in primary if r["cond"] == "real" and r.get("testable")
            and np.isfinite(r.get(value, np.nan))]
    tau = {}
    for seq in PRIMARY_SEQS:
        train = [r[value] for r in real if r["seq"] != seq]
        tau[seq] = float(np.quantile(train, level)) if len(train) >= 4 else float("nan")
    for r in primary:
        t_j = tau.get(r["seq"], float("nan"))
        r["tau"] = t_j
        r["flag"] = bool(r.get("testable") and np.isfinite(r.get(value, np.nan))
                         and np.isfinite(t_j) and r[value] > t_j)
    pooled = [r[value] for r in real]
    tau_pool = float(np.quantile(pooled, level)) if len(pooled) >= 4 else float("nan")
    for r in rows:
        if (r["seq"], r["idx"]) not in prim_ids:
            r["tau"] = tau_pool
            r["flag"] = bool(r.get("testable") and np.isfinite(r.get(value, np.nan))
                             and np.isfinite(tau_pool) and r[value] > tau_pool)

    def rate(sub, cond):
        sel = [r for r in sub if r["cond"] == cond and r.get("testable")]
        k = sum(1 for r in sel if r["flag"])
        return {"n": len(sel), "k": k, "rate": (k / len(sel)) if sel else None,
                "n_sequences": len({r["seq"] for r in sel})}

    block = {"n_clips": len({(r["seq"], r["idx"]) for r in primary if r["cond"] == "real"}),
             "n_sequences": len({r["seq"] for r in primary if r["cond"] == "real"}),
             "rates": {c: rate(primary, c) for c in CONDITIONS},
             "tau": tau,
             "auc": {c: auc([r[value] for r in primary if r["cond"] == c and r.get("testable")],
                            [r[value] for r in real]) for c in CONDITIONS},
             "abstained_real": sum(1 for r in primary if r["cond"] == "real"
                                   and not r.get("testable"))}
    if abs(level - 0.95) < 1e-9:
        block["clustered"] = {c: clustered(primary, c, n_boot) for c in CONDITIONS}
    expl = [r for r in rows if (r["seq"], r["idx"]) not in prim_ids]
    block["exploratory"] = {
        "n_clips": len({(r["seq"], r["idx"]) for r in expl if r["cond"] == "real"}),
        "n_sequences": len({r["seq"] for r in expl if r["cond"] == "real"}),
        "rates": {c: rate(expl, c) for c in CONDITIONS}}
    block["gate"] = gate_verdict(block["rates"])
    return block


def gate_verdict(rates: dict) -> dict:
    """The pre-registered pass rule for one path at one level."""
    if not all(rates.get(c) and rates[c]["rate"] is not None
               for c in ("real", "tilt_removal", "mirror_roll")):
        return {"pass": False, "why": "missing rate"}
    frr = rates["real"]["rate"]
    rem = rates["tilt_removal"]["rate"]
    mir = rates["mirror_roll"]["rate"]
    return {"pass": bool(frr <= 0.15 and rem >= 0.90 and mir >= 0.90),
            "false_reject": frr, "removal": rem, "mirror": mir}


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="dpvo", choices=("dpvo", "vipe", "mast3r", "pi3x", "mapanything", "all"))
    ap.add_argument("--poses-root", default="data/pathrec/poses")
    ap.add_argument("--clips", default="data/clips_v2_w370ms")
    ap.add_argument("--uzh", default="data/uzh")
    ap.add_argument("--window-rule", default="results/e_window_rule.json")
    ap.add_argument("--gate", default="configs/gate_frozen.yaml")
    ap.add_argument("--out", default="results/f04_spline")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    import yaml
    window_rule = json.loads(Path(args.window_rule).read_text())
    gate = yaml.safe_load(Path(args.gate).read_text())
    index = load_index(Path(args.clips))
    paths = ("dpvo", "vipe", "mast3r", "pi3x", "mapanything") if args.path == "all" else (args.path,)
    out_root = Path(args.out)
    summary = {"provenance": {"commit": _git_head(),
                              "prereg": "configs/f04_round2_prereg.yaml",
                              "poses_root": args.poses_root, "clips": args.clips,
                              "n_index_rows": len(index),
                              "pose_sha256": {}},
               "paths": {}}

    for path_name in paths:
        roots = {"uzh": Path(args.uzh), "poses": Path(args.poses_root) / path_name}
        if not (roots["poses"] / "indoor_forward_3.npz").exists():
            summary["paths"][path_name] = {"missing": str(roots["poses"])}
            continue
        summary["provenance"]["pose_sha256"][path_name] = {
            s: _sha256(roots["poses"] / f"{s}.npz") for s in SEQUENCES
            if (roots["poses"] / f"{s}.npz").exists()}
        groups: dict[tuple[str, int], dict] = {}
        for row in index:
            groups.setdefault((row["seq"], row["idx"]), {})[row["cond"]] = row
        all_rows = []
        skipped = 0
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
                        skipped += 1
                        continue
                    headline = score_clip(cases["real"], cases)
                    if not headline["ok"]:
                        skipped += 1
                        continue
                    k = headline["knots"]
                    coarse = score_clip(cases["real"], cases, n_intervals=max(4, k - 1))
                    fine = score_clip(cases["real"], cases, n_intervals=k + 1)
                    kd0 = score_clip(cases["real"], cases, k_d=0.0)
                    base = testability(cases["real"], gate)
                    for cond, cc in cases.items():
                        def val(blob, key="rho"):
                            v = blob.get("rows", {}).get(cond, {}).get(key)
                            return v if v is not None and np.isfinite(v) else float("nan")
                        all_rows.append({
                            "seq": seq, "clip": cc["clip"], "idx": idx, "cond": cond,
                            "arm": arm, "path": path_kind,
                            "rho": val(headline), "rho_model": val(headline, "rho_model"),
                            "rho_demeaned": val(headline, "rho_demeaned"),
                            "rho_psi_refit": val(headline, "rho_psi_refit"),
                            "rho_coarse": val(coarse), "rho_fine": val(fine),
                            "rho_kd0": val(kd0), "rho_nov0": val(headline, "rho_nov0"),
                            "psi_deg": headline["rows"].get(cond, {}).get("psi_deg"),
                            "scale": headline["rows"].get(cond, {}).get("scale"),
                            "v0_x": headline["rows"].get(cond, {}).get("v0_x"),
                            "v0_y": headline["rows"].get(cond, {}).get("v0_y"),
                            "knots": k, "n_frames": len(cc["t"]), "dt": cc["dt"],
                            "window_s": 0.5 / cc["f95"], "full_path": cc["full_path"],
                            "real_testable": base["testable"],
                            "real_reasons": ";".join(base["reasons"]),
                        })
        # testability propagation
        by_key = {(r["seq"], r["idx"], r["cond"], r["arm"], r["path"]): r for r in all_rows}
        for r in all_rows:
            if r["cond"] == "real":
                r["testable"] = bool(r["real_testable"] and np.isfinite(r["rho"]))
            else:
                base = by_key.get((r["seq"], r["idx"], "real", r["arm"], r["path"]))
                r["testable"] = bool(base and base["testable"] and np.isfinite(r["rho"]))
        path_block = {"skipped": skipped, "by_arm": {}}
        for arm in ARMS:
            path_block["by_arm"][arm] = {}
            for path_kind in PATHS:
                sub = [r for r in all_rows if r["arm"] == arm and r["path"] == path_kind]
                block = {"primary": {}, "sensitivity": {}}
                for setting in SETTINGS:
                    for level in LEVELS:
                        block["primary"][f"{setting}|q{level:.2f}"] = evaluate(
                            sub, setting, level, args.n_boot)
                for key in ("rho_coarse", "rho_fine", "rho_kd0", "rho_psi_refit"):
                    vals = [r[key] for r in sub if r["cond"] == "real"
                            and r[key] is not None and np.isfinite(r[key])]
                    block["sensitivity"][key] = {
                        "median": float(np.median(vals)) if vals else None, "n": len(vals)}
                path_block["by_arm"][arm][path_kind] = block
        path_block["knots_median"] = float(np.median(
            [r["knots"] for r in all_rows if r["cond"] == "real"]))
        # one compact copy for downstream joins; the full row set is rows.csv
        path_block["rows"] = [{k: r.get(k) for k in ("seq", "idx", "cond", "arm", "path",
                                                     "rho", "testable")} for r in all_rows]
        summary["paths"][path_name] = path_block

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(_py(summary), indent=1))
    fields = ["seq", "clip", "idx", "cond", "arm", "path", "rho", "rho_model",
              "rho_demeaned", "rho_psi_refit", "rho_nov0", "rho_coarse", "rho_fine", "rho_kd0",
              "psi_deg", "scale", "v0_x", "v0_y", "knots", "n_frames", "dt", "window_s",
              "full_path", "testable", "tau", "flag", "real_reasons"]
    with open(out_root / "rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for pn in paths:
            for r in summary["paths"].get(pn, {}).get("rows", []):
                w.writerow({k: r.get(k) for k in fields})
    for pn in paths:
        b = summary["paths"].get(pn, {})
        if "by_arm" not in b:
            print(f"{pn}: missing")
            continue
        for arm in ARMS:
            for path_kind in PATHS:
                block = b["by_arm"][arm][path_kind]
                head = block["primary"]["rho|q0.95"]
                print(f"{pn:7s} {arm[:9]:9s} {path_kind:6s} real={_fmt(head['rates']['real'])} "
                      f"removal={_fmt(head['rates']['tilt_removal'])} "
                      f"mirror={_fmt(head['rates']['mirror_roll'])} "
                      f"AUC={head['auc']['tilt_removal']:.2f}/{head['auc']['mirror_roll']:.2f} "
                      f"gate={head['gate']['pass']}")
        print(f"{pn:7s} knots median {b['knots_median']:.1f} skipped={b['skipped']}")
    return 0


def _fmt(r: dict) -> str:
    if not r or r.get("rate") is None:
        return "n/a"
    return f"{r['rate']:.3f}({r['k']}/{r['n']})"


if __name__ == "__main__":
    raise SystemExit(main())
