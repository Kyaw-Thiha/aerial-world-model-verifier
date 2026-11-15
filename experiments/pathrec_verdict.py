"""Item 5: pixel-path verdict test — DPVO path + GeoCalib depicted tilt, on real vs edited clips.

Pre-registration: `configs/pathrec_prereg_verdict.yaml` (commit `518024e`).

Per clip, with no ground truth used in any decision:

1. depicted attitude: GeoCalib gravity per frame, gravity-aligned DPVO rotation, complementary
   filter (gain 0.2, Track A's value) -> body z axis in world;
2. implied attitude: DPVO path differentiated (0.5 s window, poly 3), flatness with k_d=0.295
   -> implied thrust axis;
3. residual r(t) = angle(depicted, implied); clip score = median over samples whose implied
   tilt is >= 10 deg;
4. verdict: score > 10 deg -> violated, else clean; fewer than 20 % testable -> abstain.

A "path-only" control repeats 2-4 with the true path, so the pixel path's contribution is
visible.

    python experiments/pathrec_verdict.py --poses /tmp/v/poses --gravity /tmp/v/gravity \
        --truth /tmp/ct2/uzh --out results/pathrec_verdict/v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.fusion.attitude import complementary_filter, gravity_align, relative_rotations
from src.pathrec.lean import differentiate, lean_xy, required_tilt_deg


def body_z_from_filter(c2w: np.ndarray, up_cam: np.ndarray, gain: float = 0.2) -> np.ndarray:
    R = c2w[:, :3, :3]
    gal = gravity_align(R, up_cam)
    R_al = np.einsum("ij,njk->nik", gal["R_align"], R)
    filt = complementary_filter(relative_rotations(R_al), up_cam, gain=gain)
    return np.einsum("nij,j->ni", filt.R, np.array([0.0, 0.0, 1.0]))


def residual_deg(z_dep: np.ndarray, z_imp: np.ndarray) -> np.ndarray:
    n = min(len(z_dep), len(z_imp))
    cos = np.clip((z_dep[:n] * z_imp[:n]).sum(1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def score_clip(c2w: np.ndarray, pos: np.ndarray, up_cam: np.ndarray, dt: float,
               k_d: float = 0.295, window_s: float = 0.5) -> dict:
    z_dep = body_z_from_filter(c2w, up_cam)
    _, v, a = differentiate(pos, dt=dt, window_s=window_s)
    tilt = required_tilt_deg(a, None, k_d=0.0)
    z_imp = np.zeros_like(a)
    from src.physics.flatness import implied_thrust_axis
    z_imp = implied_thrust_axis(a, v, k_d=k_d)
    # align the sample ranges: differentiate trims the edges
    half = (len(pos) - len(a)) // 2
    sl = slice(half, half + len(a))
    r = residual_deg(z_dep[sl], z_imp)
    testable = tilt >= 10.0
    out = {"n": int(len(r)), "testable_fraction": float(testable.mean()),
           "implied_tilt_median_deg": float(np.median(tilt))}
    if testable.sum() >= 3:
        out["score_deg"] = float(np.median(r[testable]))
    else:
        out["score_deg"] = float("nan")
    return out


def verdict(score: float, testable: float, tau: float = 10.0) -> str:
    if not np.isfinite(score) or testable < 0.2:
        return "abstain"
    return "violated" if score > tau else "clean"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--gravity", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", default="results/pathrec_verdict/v1")
    args = ap.parse_args()

    grav = {r["seq"]: r for r in json.loads(Path(args.gravity).read_text())}
    rows = []
    for seq, g in grav.items():
        npz = Path(args.poses) / f"{seq}.npz"
        if not npz.exists():
            print(f"{seq}: no poses"); continue
        c2w = np.load(npz)["c2w"]
        up = np.asarray([r["gravity_up_cam"] for r in g["rows"]], dtype=np.float64)
        up /= np.linalg.norm(up, axis=1, keepdims=True)
        n = min(len(c2w), len(up))
        tp = Path(args.truth) / seq.split("_tilt")[0].split("_yaw")[0] / "truth.npz"
        truth = np.load(tp) if tp.exists() else None
        meta = Path(args.truth) / seq.split("_tilt")[0].split("_yaw")[0] / "meta.json"
        fps = json.loads(meta.read_text())["fps"] if meta.exists() else 15.1
        dt = 1.0 / fps
        pix = score_clip(c2w[:n], c2w[:n, :3, 3], up[:n], dt)
        rec = {"seq": seq, **{f"pix_{k}": v for k, v in pix.items()},
               "pix_verdict": verdict(pix["score_deg"], pix["testable_fraction"])}
        if truth is not None:
            m = min(n, len(truth["pos"]))
            ref = score_clip(c2w[:m], truth["pos"][:m], up[:m], dt)
            rec.update({f"pathonly_{k}": v for k, v in ref.items()})
            rec["pathonly_verdict"] = verdict(ref["score_deg"], ref["testable_fraction"])
        rows.append(rec)
        print(f"{seq:40s} score {rec['pix_score_deg']:6.2f}  testable {rec['pix_testable_fraction']:.2f} "
              f"-> {rec['pix_verdict']:9s} | path-only {rec.get('pathonly_score_deg', float('nan')):6.2f} "
              f"-> {rec.get('pathonly_verdict', '-')}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
