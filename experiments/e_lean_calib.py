"""Estimator calibration curve and per-axis gains.

Two related measurements, both in lean space (the horizontal components of the thrust axis in
the yaw-aligned frame, i.e. the same quantity the shared schema calls ``depicted_lean``):

``--mode perframe``  calibration curve — depicted lean regressed on the **truth-attitude** lean,
                     per sequence (UZH, forward rigs) and per flight (TII), same estimator, same
                     commit. This asks "how much of the true lean does the estimator see", with
                     no path, no clip and no verdict in the loop.
``--mode clips``     per-axis gain and correlation of depicted lean against the **required** lean
                     (from the dense-truth ``lean_req`` of the provisional v2 clips), per
                     condition and per rig, with the 45-degree rigs and the outdoor rigs kept
                     separate and labelled.

Both report slope(s), intercept, residual noise and correlation. Slopes are dimensionless
(lean/lean); residual noise is given in degrees for readability.

    python experiments/e_lean_calib.py --mode perframe
    python experiments/e_lean_calib.py --mode clips --clips data/clips_v2
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from src.depicted.clips import lean_from_up_body
from src.depicted.uzh import all_sequences, is_forward, load_truth, rig_of

TII_RUN = "results/phase1_depicted/20261108T131825Z_c738aa8"


def _regress(y: np.ndarray, x: np.ndarray) -> dict:
    """2-D OLS with a shared intercept: y (N,2) on x (N,2); per-axis slopes as well."""
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    y, x = y[ok], x[ok]
    n = len(y)
    if n < 20:
        return {"n": int(n), "slope": None}
    A = np.column_stack([x.ravel(), np.ones(2 * n)])
    coef, *_ = np.linalg.lstsq(A, y.ravel(), rcond=None)
    pred = coef[0] * x.ravel() + coef[1]
    resid = y.ravel() - pred
    # Full 2x2 map l_dep = M l_req + b. This is what separates the two explanations of the
    # 45-rig discrepancy: a *rotation* (a mount error) gives M ~ R(delta) with equal singular
    # values, while a *view-dependent estimator bias* attenuates one lean axis, giving unequal
    # singular values and no rotation.
    X = np.column_stack([x, np.ones(n)])
    coef2, *_ = np.linalg.lstsq(X, y, rcond=None)
    M = coef2[:2, :].T
    U, S, Vt = np.linalg.svd(M)
    rot = U @ Vt
    gx, *restx = np.polyfit(x[:, 0], y[:, 0], 1)
    gy, *resty = np.polyfit(x[:, 1], y[:, 1], 1)
    return {
        "n": int(n),
        "slope": float(coef[0]),
        "intercept": float(coef[1]),
        "gx": float(gx), "gx_intercept": float(restx[0]),
        "gy": float(gy), "gy_intercept": float(resty[0]),
        "corr": float(np.corrcoef(x.ravel(), y.ravel())[0, 1]),
        "corr_x": float(np.corrcoef(x[:, 0], y[:, 0])[0, 1]),
        "corr_y": float(np.corrcoef(x[:, 1], y[:, 1])[0, 1]),
        "residual_rms_deg": float(np.degrees(np.sqrt(np.mean(resid ** 2)))),
        "x_spread_deg": float(np.degrees(np.std(x))),
        "matrix_M": M.tolist(),
        "sv1": float(S[0]), "sv2": float(S[1]),
        "matrix_rotation_deg": float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))),
        "matrix_anisotropy": float(S[0] / S[1]) if S[1] > 1e-9 else None,
    }


def perframe_tii(root: Path) -> dict:
    """TII: depicted vs truth-attitude lean, per flight, with the recorded uptilt as the mount."""
    from src.perception.frames import camera_to_body_pitched

    metrics = json.loads((root / TII_RUN / "metrics.json").read_text())
    out = {}
    for f in sorted((root / TII_RUN).glob("frames_flight-*.npz")):
        flight = f.name.replace("frames_", "").replace(".npz", "")
        d = np.load(f)
        if "est_up" not in d.files or "true_up" not in d.files:
            continue
        uptilt = metrics["flights"].get(flight, {}).get("uptilt_metadata_deg")
        if uptilt is None:
            continue
        R_cb = camera_to_body_pitched(np.deg2rad(uptilt))
        up_b_dep = np.einsum("ij,nj->ni", R_cb, d["est_up"])
        up_b_true = np.einsum("ij,nj->ni", R_cb, d["true_up"])
        rec = _regress(lean_from_up_body(up_b_dep), lean_from_up_body(up_b_true))
        rec["uptilt_deg"] = float(uptilt)
        out[flight] = rec
    return out


def perframe_uzh(uzh_root: Path, perframe_dir: Path, variant: str = "b") -> dict:
    """UZH: depicted vs truth-attitude lean, per sequence (forward rigs only by default)."""
    out = {}
    for seq in all_sequences():
        f = perframe_dir / f"frames_{seq}__{variant}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        truth = load_truth(uzh_root, seq)
        R_cb = d["R_cb"] if "R_cb" in d.files else truth.R_cb()
        up_b_dep = np.einsum("ij,nj->ni", R_cb, d["up_est"])
        up_b_true = truth.up_body
        rec = _regress(lean_from_up_body(up_b_dep), lean_from_up_body(up_b_true))
        rec["rig"] = rig_of(seq)
        rec["forward"] = bool(is_forward(seq))
        out[seq] = rec
    return out


def clips_table(clips: Path, conds: list[str]) -> dict:
    """Per condition and rig: gain and correlation of depicted lean against required lean."""
    rows: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    for npz in sorted(clips.glob("*/*.npz")):
        meta_p = npz.with_suffix(".meta.json")
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        cond = meta["condition"]
        if cond not in conds:
            continue
        d = np.load(npz)
        if "lean_req" not in d.files:
            raise ValueError(f"{npz}: no lean_req; run experiments/e_clips_v2.py first")
        rig = meta.get("rig") or _rig_from_seq(meta.get("sequence", ""))
        rows.setdefault((rig, cond), []).append((d["depicted_lean"], d["lean_req"]))
    out: dict[str, dict] = {}
    for (rig, cond), pairs in sorted(rows.items()):
        y = np.concatenate([p[0] for p in pairs])
        x = np.concatenate([p[1] for p in pairs])
        out.setdefault(rig, {})[cond] = _regress(y, x)
    return out


def _rig_from_seq(seq: str) -> str:
    for prefix, rig in (("indoor_forward", "indoor_forward"), ("outdoor_forward", "outdoor_forward"),
                        ("indoor_45", "indoor_45"), ("outdoor_45", "outdoor_45")):
        if seq.startswith(prefix):
            return rig
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("perframe", "clips"), default="perframe")
    ap.add_argument("--tii-root", default="../aerial-world-model-verifier")
    ap.add_argument("--uzh-root", default="data/uzh")
    ap.add_argument("--perframe", default="results/e_perframe/v1_all")
    ap.add_argument("--variant", default="b")
    ap.add_argument("--clips", default="data/clips_v2")
    ap.add_argument("--out", default="results/e_calib")
    ap.add_argument("--conds", default="real,null_edit,yaw_control,tilt_removal,scale_0.5,scale_1.5")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "perframe":
        tii = perframe_tii(Path(args.tii_root))
        uzh = perframe_uzh(Path(args.uzh_root), Path(args.perframe), args.variant)
        rec = {"tii": tii, "uzh": uzh, "variant": args.variant}
        (out / "calibration.json").write_text(json.dumps(rec, indent=1))

        def line(name, r):
            if r.get("slope") is None:
                return f"  {name:22s} n={r['n']:5d}  (too few)"
            return (f"  {name:22s} n={r['n']:5d}  slope={r['slope']:5.3f} "
                    f"gx={r['gx']:5.3f} gy={r['gy']:5.3f} corr={r['corr']:5.3f} "
                    f"resid={r['residual_rms_deg']:5.2f} deg spread={r['x_spread_deg']:5.2f} deg"
                    f" | M sv=({r['sv1']:4.2f},{r['sv2']:4.2f}) rot={r['matrix_rotation_deg']:+5.1f} deg")

        print("TII (8 flights, recorded uptilt mount):")
        for k, v in tii.items():
            print(line(k, v))
        fwd = {k: v for k, v in uzh.items() if v["forward"]}
        rest = {k: v for k, v in uzh.items() if not v["forward"]}
        print("\nUZH forward rigs (variant b):")
        for k, v in sorted(fwd.items()):
            print(line(k, v))
        if fwd:
            print(line("FORWARD POOLED", _pool(fwd)))
        print("\nUZH 45-degree rigs (separate, out of scope):")
        for k, v in sorted(rest.items()):
            print(line(k, v))
    else:
        conds = [c for c in args.conds.split(",") if c]
        tab = clips_table(Path(args.clips), conds)
        (out / "clips_gain.json").write_text(json.dumps(tab, indent=1))
        print(f"{'rig':16s} {'condition':14s} {'n':>5s} {'slope':>6s} {'gx':>6s} {'gy':>6s} "
              f"{'corr':>6s} {'corr_x':>6s} {'corr_y':>6s} {'resid':>6s}")
        for rig in ("indoor_forward", "outdoor_forward", "indoor_45", "outdoor_45"):
            if rig not in tab:
                continue
            for cond in conds:
                r = tab[rig].get(cond)
                if not r or r.get("slope") is None:
                    continue
                print(f"{rig:16s} {cond:14s} {r['n']:5d} {r['slope']:6.3f} {r['gx']:6.3f} "
                      f"{r['gy']:6.3f} {r['corr']:6.3f} {r['corr_x']:6.3f} {r['corr_y']:6.3f} "
                      f"{r['residual_rms_deg']:6.2f}   sv=({r['sv1']:4.2f},{r['sv2']:4.2f}) "
                      f"rot={r['matrix_rotation_deg']:+5.1f}")
            print()
    return 0


def _pool(recs: dict) -> dict:
    """Pool the per-sequence summaries (weighted by frames) for a rig-level line."""
    slope = np.average([r["slope"] for r in recs.values()],
                       weights=[r["n"] for r in recs.values()])
    gx = np.average([r["gx"] for r in recs.values()], weights=[r["n"] for r in recs.values()])
    gy = np.average([r["gy"] for r in recs.values()], weights=[r["n"] for r in recs.values()])
    corr = np.average([r["corr"] for r in recs.values()], weights=[r["n"] for r in recs.values()])
    resid = np.average([r["residual_rms_deg"] for r in recs.values()],
                       weights=[r["n"] for r in recs.values()])
    spread = np.average([r["x_spread_deg"] for r in recs.values()],
                        weights=[r["n"] for r in recs.values()])
    w = np.array([r["n"] for r in recs.values()], dtype=float)
    sv1 = float(np.average([r["sv1"] for r in recs.values()], weights=w))
    sv2 = float(np.average([r["sv2"] for r in recs.values()], weights=w))
    rot = float(np.average([r["matrix_rotation_deg"] for r in recs.values()], weights=w))
    return {"n": int(sum(r["n"] for r in recs.values())), "slope": float(slope),
            "sv1": sv1, "sv2": sv2, "matrix_rotation_deg": rot,
            "intercept": float("nan"), "gx": float(gx), "gy": float(gy),
            "corr": float(corr), "corr_x": float("nan"), "corr_y": float("nan"),
            "residual_rms_deg": float(resid), "x_spread_deg": float(spread)}


if __name__ == "__main__":
    raise SystemExit(main())
