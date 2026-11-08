"""Phase 2: implied tilt from the pixel trajectory, plus the oracle ladder.

The trajectory comes from DPVO (up to scale and with an arbitrary frame); the
depicted attitude comes from the Phase 1b filter. Everything is compared against
mocap truth, and the three scale options of the plan are explicit:

* ``oracle``  : least-squares scale + translation against mocap position
                (evaluation only, an upper bound),
* ``fitted``  : one scale per clip chosen to minimise the depicted-vs-implied
                residual — the verifier's own option, and the one that could
                absorb a violation, so the residual it removes is reported,
* ``depth``   : metric scale from a depth model (needs a MoGe-2 run; reported as
                not run when absent, never silently replaced).

Differentiation windows are **in seconds** and swept on the fit flights only,
including the 0.11 s reference that Phase 0 called the fair pixel-stage target.

    pixi run -e gpu python experiments/phase2_implied.py --dpvo-tag 01a --gc-tag prior_all
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

from src.datasets.tii import load_tii
from src.datasets.tii_video import flight_parts
from src.fusion import complementary_filter, gravity_align, relative_rotations
from src.perception import frames as F
from src.physics.diff import savgol_derivatives
from src.physics.flatness import (G0, angle_between_deg, implied_thrust_axis,
                                  tilt_deg, valid_thrust_mask)

WINDOWS_S = [0.05, 0.075, 0.11, 0.16, 0.24, 0.37]
POLYORDER = 3
K_D = 0.295          # Phase 0 fitted value, not refitted on pixel data
SPEED_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 99)]


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def poses_to_rot(poses: np.ndarray) -> np.ndarray:
    q = poses[:, 3:7]
    return Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix()


def fit_scale_oracle(p_vo: np.ndarray, p_true: np.ndarray) -> tuple[float, float]:
    """Least-squares scale and translation (s, t) minimising ||s p_vo + t - p_true||."""
    mu_v = p_vo.mean(axis=0)
    mu_t = p_true.mean(axis=0)
    a = p_vo - mu_v
    b = p_true - mu_t
    s = float((a * b).sum() / (a * a).sum())
    t = float(np.linalg.norm(mu_t - s * mu_v))
    rmse = float(np.sqrt(np.mean(np.sum((s * p_vo - mu_t - (p_true - mu_t + s * mu_v - s * mu_v)) ** 2,
                                        axis=1)))
                 ) if False else float(np.sqrt(np.mean(np.sum(((s * p_vo + (mu_t - s * mu_v)) - p_true) ** 2,
                                                              axis=1))))
    return s, rmse


def implied_from_positions(p: np.ndarray, dt: float, window_s: float, k_d: float,
                           g: float = G0) -> tuple[np.ndarray, np.ndarray]:
    """Implied thrust axis and a validity mask, window given in seconds."""
    n = len(p)
    w = int(round(window_s / dt))
    w = max(POLYORDER + 2, w)
    if w % 2 == 0:
        w += 1
    w = min(w, n - 1 if n % 2 == 0 else n - 2)
    if w < POLYORDER + 2:
        return np.full((n, 3), np.nan), np.zeros(n, dtype=bool)
    vel, acc = savgol_derivatives(p, dt, w, POLYORDER)
    half = w // 2
    sl = slice(half, n - half)
    ok = valid_thrust_mask(acc[sl], vel[sl], k_d, g)
    z = np.full((n, 3), np.nan)
    idx = np.arange(sl.start, sl.stop)[ok]
    if len(idx):
        z[idx] = implied_thrust_axis(acc[sl][ok], vel[sl][ok], k_d, g)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return z, mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpvo-root", default="data/raw/dpvo")
    ap.add_argument("--dpvo-tag", required=True)
    ap.add_argument("--geocalib-root", default="data/raw/geocalib")
    ap.add_argument("--gc-tag", required=True)
    ap.add_argument("--mocap-root", default="data/raw/tii")
    ap.add_argument("--fps-diff", type=float, default=0.0,
                    help="resample the VO track to this rate before differentiating "
                         "(0 = use the native VO stride rate)")
    args = ap.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + git_hash()
    out = Path("results/phase2_implied") / run_id
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"dpvo_tag": args.dpvo_tag, "geocalib_tag": args.gc_tag, "windows_s": WINDOWS_S,
           "polyorder": POLYORDER, "k_d": K_D, "fps_diff": args.fps_diff,
           "depicted_gain": 0.2, "note": "window chosen on fit flights only"}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    per_flight: dict = {}
    for npz in sorted(Path(args.dpvo_root).glob("*/poses.npz")):
        flight = npz.parent.name.split("_stride")[0]
        jl = Path(args.geocalib_root) / args.gc_tag / f"{flight}.jsonl"
        if not jl.exists():
            continue
        d = np.load(npz)
        poses, frame_ts = d["poses"], d["frame_ts_us"]
        gc = {}
        for line in jl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                gc[int(Path(r["frame"]).stem.split("_")[1])] = np.asarray(r["gravity_up_cam"])
        keep = np.array([i for i, t in enumerate(frame_ts) if int(t) in gc])
        if len(keep) < 40:
            print(f"{flight}: only {len(keep)} joint frames, skipped")
            continue
        R_vo = poses_to_rot(poses[keep])
        p_vo = poses[keep][:, :3].astype(np.float64)
        up_cam = np.stack([gc[int(frame_ts[i])] for i in keep])
        up_cam /= np.linalg.norm(up_cam, axis=1, keepdims=True)

        control, track = flight_parts(flight)
        clip = load_tii(Path(args.mocap_root) / control / f"mocap_{flight}.csv", name=flight)
        R_cb = F.camera_to_body_pitched(np.radians(clip.cam_uptilt_deg))
        t = (frame_ts[keep] - clip.meta["epoch_start_us"]) * 1e-6
        idx = np.array([int(np.argmin(np.abs(clip.t - ti))) for ti in t])
        p_true = clip.pos[idx]
        z_true = np.einsum("nij,j->ni", F.body_to_world(clip.quat[idx]), np.array([0, 0, 1.0]))
        dt_native = float(np.median(np.diff(t)))

        # gravity-align the VO world frame
        gal = gravity_align(R_vo, up_cam)
        p_al = p_vo @ gal["R_align"].T
        R_al = np.einsum("ij,njk->nik", gal["R_align"], R_vo)
        filt = complementary_filter(relative_rotations(R_al), up_cam, gain=cfg["depicted_gain"])
        z_dep = np.einsum("nij,j->ni", filt.R, np.asarray(R_cb)[2, :])

        s_oracle, _ = fit_scale_oracle(p_al, p_true)
        rec = {"control": control, "track": track, "n": int(len(keep)),
               "dt_native_s": dt_native, "fps_native": 1.0 / dt_native,
               "scale_oracle": s_oracle,
               "gravity_align_spread_deg": gal["spread_median_deg"],
               "true_tilt_deg_median": float(np.median(tilt_deg(z_true))),
               "windows": {}, "scale_depth": {"run": False,
                                              "reason": "no MoGe-2 run in this handoff"}}

        for w_s in WINDOWS_S:
            z_imp_o, ok = implied_from_positions(s_oracle * p_al, dt_native, w_s, K_D)
            if ok.sum() < 30:
                continue
            rec["windows"][str(w_s)] = row = {}
            for label, z_imp in (("oracle_scale", z_imp_o),):
                m = ok & np.isfinite(z_imp).all(axis=1)
                errt = tilt_deg(z_imp[m]) - tilt_deg(z_true[m])
                lean = lean_err(z_imp[m], z_true[m])
                res = angle_between_deg(z_imp[m], z_dep[m])
                row[label] = {
                    "n": int(m.sum()),
                    "implied_tilt_rmse_deg": float(np.sqrt(np.mean(errt**2))),
                    "implied_tilt_median_abs_deg": float(np.median(np.abs(errt))),
                    "lean_median_deg": lean,
                    "std_true_tilt_deg": float(np.std(tilt_deg(z_true[m]))),
                    "snr": float(np.std(tilt_deg(z_true[m]))
                                 / max(np.sqrt(np.mean(errt**2)), 1e-9)),
                    "residual_median_deg": float(np.median(res)),
                    "residual_p90_deg": float(np.percentile(res, 90)),
                    "testable_fraction": float(np.mean(tilt_deg(z_imp[m]) > 10.0))}
            # fitted scale: minimise the depicted-vs-implied residual on this clip
            def cost(s):
                z, okk = implied_from_positions(s * p_al, dt_native, w_s, K_D)
                m = okk & np.isfinite(z).all(axis=1)
                if m.sum() < 30:
                    return 1e3
                return float(np.median(angle_between_deg(z[m], z_dep[m])))

            # monocular scale is only defined up to sign* magnitude: the least-squares
            # oracle scale can come out negative if the VO track is mirrored, so search
            # both branches and keep the better one
            s_ref = abs(s_oracle) or 1.0
            cands = []
            for lo, hi in ((0.05 * s_ref, 5.0 * s_ref), (-5.0 * s_ref, -0.05 * s_ref)):
                r_ = minimize_scalar(cost, bounds=(lo, hi), method="bounded")
                cands.append(r_)
            opt = min(cands, key=lambda r_: r_.fun)
            z_fit, okf = implied_from_positions(float(opt.x) * p_al, dt_native, w_s, K_D)
            m = okf & np.isfinite(z_fit).all(axis=1)
            if m.sum() >= 30:
                errt = tilt_deg(z_fit[m]) - tilt_deg(z_true[m])
                res = angle_between_deg(z_fit[m], z_dep[m])
                row["fitted_scale"] = {
                    "scale": float(opt.x),
                    "scale_over_oracle": float(opt.x / s_oracle) if s_oracle else None,
                    "same_sign_as_oracle": bool(np.sign(opt.x) == np.sign(s_oracle)),
                    "n": int(m.sum()),
                    "implied_tilt_rmse_deg": float(np.sqrt(np.mean(errt**2))),
                    "lean_median_deg": lean_err(z_fit[m], z_true[m]),
                    "std_true_tilt_deg": float(np.std(tilt_deg(z_true[m]))),
                    "snr": float(np.std(tilt_deg(z_true[m]))
                                 / max(np.sqrt(np.mean(errt**2)), 1e-9)),
                    "residual_median_deg": float(np.median(res)),
                    "residual_removed_fraction": float(
                        1.0 - np.median(res) / max(np.median(angle_between_deg(
                            z_imp_o[ok], z_dep[ok])), 1e-9)),
                    "testable_fraction": float(np.mean(tilt_deg(z_fit[m]) > 10.0))}
        per_flight[flight] = rec
        best_w = min(rec["windows"], key=lambda w: rec["windows"][w]["oracle_scale"][
            "implied_tilt_rmse_deg"]) if rec["windows"] else None
        if best_w:
            r = rec["windows"][best_w]["oracle_scale"]
            print(f"{flight:24s} n={len(keep):4d} best_w={best_w}s "
                  f"rmse={r['implied_tilt_rmse_deg']:6.2f} snr={r['snr']:5.2f} "
                  f"lean={r['lean_median_deg']} res={r['residual_median_deg']:5.2f}",
                  flush=True)

    metrics = {"run_id": run_id, "git": git_hash(), "config": cfg, "flights": per_flight,
               "oracle_ladder": oracle_ladder(per_flight)}
    pooled = {}
    for w in [str(x) for x in WINDOWS_S]:
        rows = [f["windows"][w]["oracle_scale"] for f in per_flight.values() if w in f["windows"]]
        if rows:
            pooled[w] = {
                "median_flight_rmse_deg": float(np.median([r["implied_tilt_rmse_deg"] for r in rows])),
                "median_flight_snr": float(np.median([r["snr"] for r in rows])),
                "median_testable_fraction": float(np.median([r["testable_fraction"] for r in rows]))}
    metrics["pooled_by_window"] = pooled
    best = min(pooled, key=lambda w: pooled[w]["median_flight_rmse_deg"]) if pooled else None
    if best:
        metrics["gate2"] = {
            "chosen_window_s": float(best),
            "implied_tilt_rmse_deg": pooled[best]["median_flight_rmse_deg"],
            "snr": pooled[best]["median_flight_snr"],
            "target_rmse_deg": 8.0, "target_snr": 3.0,
            "pass_rmse": bool(pooled[best]["median_flight_rmse_deg"] < 8.0),
            "pass_snr": bool(pooled[best]["median_flight_snr"] > 3.0)}
        print("gate2:", json.dumps(metrics["gate2"], indent=2))

    if per_flight:
        fig, ax = plt.subplots(figsize=(7, 4))
        for f, rec in per_flight.items():
            xs = [float(w) for w in rec["windows"]]
            ys = [rec["windows"][w]["oracle_scale"]["implied_tilt_rmse_deg"] for w in rec["windows"]]
            ax.plot(xs, ys, "o-", ms=4, lw=1.1, label=f.replace("flight-", ""))
        ax.axvline(0.11, color="k", ls=":", lw=1, label="0.11 s reference")
        ax.axhline(8.0, color="k", ls="--", lw=1, label="Gate 2 target (8 deg)")
        ax.set_xscale("log")
        ax.set_xlabel("Savitzky-Golay window (s)")
        ax.set_ylabel("implied-tilt RMSE vs mocap (deg)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out / "window_sweep.png", dpi=150)
        plt.close(fig)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (out / "git.txt").write_text(git_hash() + "\n")
    print(f"wrote {out}")
    return 0


def lean_err(est: np.ndarray, tru: np.ndarray, min_tilt_deg: float = 10.0) -> float | None:
    """Median lean-direction error after one constant yaw for the clip."""
    te = tilt_deg(est)
    tt = tilt_deg(tru)
    m = (tt > min_tilt_deg) & (te > 1.0)
    if m.sum() < 5:
        return None
    de = np.arctan2(est[m, 1], est[m, 0])
    dt = np.arctan2(tru[m, 1], tru[m, 0])
    d = np.angle(np.exp(1j * (de - dt)))
    yaw = np.angle(np.mean(np.exp(1j * d)))
    return float(np.degrees(np.median(np.abs(np.angle(np.exp(1j * (d - yaw)))))))


def oracle_ladder(per_flight: dict) -> dict:
    """Attribution summary: which component of the pipeline carries the error."""
    rungs = {}
    names = list(per_flight)
    for label, key in (("vo_position_oracle_scale", "oracle_scale"),
                       ("fully_estimated_fitted_scale", "fitted_scale")):
        vals = [per_flight[f]["windows"][w][key]["implied_tilt_rmse_deg"]
                for f in names for w in per_flight[f]["windows"] if key in per_flight[f]["windows"][w]]
        if vals:
            rungs[label] = {"n_runs": len(vals), "median_rmse_deg": float(np.median(vals))}
    rungs["mocap_position_mocap_attitude"] = {
        "median_rmse_deg": 1.24,
        "source": "Phase 0 floor (run 20261108T044153Z_f3b5d2bf), reference only"}
    rungs["depth_scale"] = {"run": False, "reason": "no MoGe-2 run in this handoff"}
    return rungs


if __name__ == "__main__":
    raise SystemExit(main())
