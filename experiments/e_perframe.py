"""Track E Task 2: per-frame depicted-gravity error on UZH-FPV, three variants.

The metric is convention-free given a published mount: the angle between the estimated up
direction and the truth-implied up direction, **both in the camera frame**,

    err_i = angle( up_est,i , R_cb^T R_bw,i^T e3 )

with ``R_cb`` from the rig's published Kalibr ``T_cam_imu``. No clip, no path, no nuisance
fit. Everything about the verdict test is downstream of this number.

Blocks reported (all fixed in ``configs/e_prereg.yaml``):
  * overall / per sequence / per rig / per environment,
  * by truth speed bin and by blur quintile (blur = variance of the Laplacian on the frame
    the estimator actually saw),
  * with a 0.5 s and 1.0 s running median over the per-frame gravity (heavy-tailed errors),
  * the frozen S3 confidence gate applied unchanged to 4 s clips: AUROC for "clip median
    error > 10 deg", coverage per rig, and frames dropped.

    pixi run python experiments/e_perframe.py --root data/uzh --rows results/e_audit/geocalib
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.depicted.metrics import (SPEED_BINS, angle_deg, bootstrap_auroc, running_median,
                                  speed_bin_label, summarize)
from src.depicted.uzh import all_sequences, load_truth

GATE_TAU = -0.0016656352350220922   # configs/gate_frozen.yaml, read-only
CLIP_SECONDS = 4.0
CLIP_LABEL_DEG = 10.0


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def frame_table(root: Path, rows_dir: Path, seq: str, variant: str) -> dict | None:
    rows = load_rows(rows_dir / f"{seq}__{variant}.jsonl")
    if not rows:
        return None
    truth = load_truth(root, seq)
    idx = np.array([r["index"] for r in rows])
    if len(idx) != truth.n_frames:
        raise ValueError(f"{seq}/{variant}: {len(idx)} rows for {truth.n_frames} truth frames")
    up_est = np.array([r["gravity_up_cam"] for r in rows], dtype=np.float64)
    up_true = truth.up_cam_truth()
    err = angle_deg(up_est, up_true)
    from src.depicted.uzh import dense_kinematics

    gt = np.load(root / seq / "truth.npz")["gt"]
    kin = dense_kinematics(gt, window_s=0.10)
    t_gt = kin["t"] - (truth.t_unix[0] + truth.offset_s)
    v = np.column_stack([np.interp(truth.t, t_gt, kin["v"][:, k]) for k in range(3)])
    speed = np.linalg.norm(v, axis=1)
    lat = np.array([r.get("latitude_confidence_mean", np.nan) for r in rows], dtype=np.float64)
    upc = np.array([r.get("up_confidence_mean", np.nan) for r in rows], dtype=np.float64)
    unc = np.array([r.get("gravity_uncertainty_raw", np.nan) for r in rows], dtype=np.float64)
    blur = np.array([r.get("blur_var_raw" if variant == "b" else "blur_var_pinhole", np.nan)
                     for r in rows], dtype=np.float64)
    return {"seq": seq, "variant": variant, "rig": truth.rig, "env": truth.env,
            "t": truth.t, "up_est": up_est, "up_true": up_true, "err": err,
            "speed": speed, "lat_conf": lat, "up_conf": upc, "unc": unc, "blur": blur,
            "R_cb": truth.R_cb()}


def smooth_err(tab: dict, window_s: float) -> np.ndarray:
    up = running_median(tab["up_est"], tab["t"], window_s)
    return angle_deg(up, tab["up_true"])


def clips_of(tab: dict, seconds: float = CLIP_SECONDS) -> list[dict]:
    """Non-overlapping clips of ``seconds``; frame count follows the measured rate."""
    t = tab["t"]
    n = len(t)
    dt = float(np.median(np.diff(t)))
    k = max(1, int(round(seconds / dt)))
    out = []
    for lo in range(0, n - k + 1, k):
        sl = slice(lo, lo + k)
        err = tab["err"][sl]
        lat = tab["lat_conf"][sl]
        out.append({"index": [lo, lo + k], "t0": float(t[lo]), "t1": float(t[lo + k - 1]),
                    "median_err": float(np.nanmedian(err)),
                    "lat_conf_median": float(np.nanmedian(lat)),
                    "n_finite": int(np.isfinite(err).sum()), "n": k})
    return out


def table_by(rows: list[dict], key: str) -> dict:
    out: dict[str, list] = {}
    for r in rows:
        out.setdefault(r[key], []).append(r["value"])
    return {k: summarize(np.asarray(v)) for k, v in sorted(out.items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/uzh")
    ap.add_argument("--rows", default="results/e_audit/geocalib")
    ap.add_argument("--out", default="results/e_perframe")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--variants", default="abc")
    args = ap.parse_args()
    root = Path(args.root)
    rows_dir = Path(args.rows)
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    metrics: dict = {"tag": args.tag, "gate_tau": GATE_TAU, "clip_seconds": CLIP_SECONDS,
                     "variants": {}}
    for variant in args.variants:
        tabs = []
        for seq in all_sequences():
            tab = frame_table(root, rows_dir, seq, variant)
            if tab is None:
                continue
            tabs.append(tab)
            np.savez_compressed(
                out / f"frames_{seq}__{variant}.npz",
                t=tab["t"], up_est=tab["up_est"], up_true=tab["up_true"], err=tab["err"],
                speed=tab["speed"], lat_conf=tab["lat_conf"], up_conf=tab["up_conf"],
                unc=tab["unc"], blur=tab["blur"], R_cb=tab["R_cb"])
        if not tabs:
            continue

        all_err = np.concatenate([t["err"] for t in tabs])
        per_seq = {t["seq"]: summarize(t["err"]) for t in tabs}
        # pooled by rig and environment
        rig_rows, env_rows, spd_rows, blur_rows = [], [], [], []
        pooled_blur = np.concatenate([t["blur"] for t in tabs])
        edges = np.percentile(pooled_blur[np.isfinite(pooled_blur)], [20, 40, 60, 80])
        for t in tabs:
            for e, s, b in zip(t["err"], t["speed"], t["blur"]):
                rig_rows.append({"rig": t["rig"], "value": e})
                env_rows.append({"env": t["env"], "value": e})
                spd_rows.append({"bin": speed_bin_label(s), "value": e})
                q = int(np.searchsorted(edges, b, side="right"))
                blur_rows.append({"bin": f"q{q + 1}", "value": e})

        sm05 = np.concatenate([smooth_err(t, 0.5) for t in tabs])
        sm10 = np.concatenate([smooth_err(t, 1.0) for t in tabs])

        # ---- frozen S3 gate on 4 s clips
        clip_rows = []
        for t in tabs:
            for c in clips_of(t):
                clip_rows.append({"seq": t["seq"], "rig": t["rig"], "env": t["env"],
                                  "median_err": c["median_err"],
                                  "lat_conf_median": c["lat_conf_median"],
                                  "dropped": bool(np.isfinite(c["lat_conf_median"]) and
                                                  -c["lat_conf_median"] > GATE_TAU),
                                  "n": c["n"]})
        pos = [c["median_err"] for c in clip_rows if c["median_err"] > CLIP_LABEL_DEG]
        neg = [c["median_err"] for c in clip_rows if c["median_err"] <= CLIP_LABEL_DEG]
        pos_sig = [-c["lat_conf_median"] for c in clip_rows if c["median_err"] > CLIP_LABEL_DEG]
        neg_sig = [-c["lat_conf_median"] for c in clip_rows if c["median_err"] <= CLIP_LABEL_DEG]
        groups = np.concatenate([[c["seq"] for c in clip_rows if c["median_err"] > CLIP_LABEL_DEG],
                                 [c["seq"] for c in clip_rows if c["median_err"] <= CLIP_LABEL_DEG]])
        s3 = bootstrap_auroc(np.array(pos_sig), np.array(neg_sig), groups)
        per_rig_cov = {}
        for rig in sorted({c["rig"] for c in clip_rows}):
            sub = [c for c in clip_rows if c["rig"] == rig]
            per_rig_cov[rig] = {"n_clips": len(sub),
                                "clip_drop_rate": float(np.mean([c["dropped"] for c in sub])),
                                "frame_drop_rate": float(sum(c["n"] * c["dropped"] for c in sub)
                                                         / max(sum(c["n"] for c in sub), 1))}

        metrics["variants"][variant] = {
            "n_sequences": len(tabs), "n_frames": int(len(all_err)),
            "pooled": summarize(all_err),
            "pooled_smoothed_0p5s": summarize(sm05),
            "pooled_smoothed_1p0s": summarize(sm10),
            "per_sequence": per_seq,
            "by_rig": table_by(rig_rows, "rig"),
            "by_env": table_by(env_rows, "env"),
            "by_speed_bin": table_by(spd_rows, "bin"),
            "by_blur_quintile": table_by(blur_rows, "bin"),
            "gate_s3": {"auroc": s3, "n_clips": len(clip_rows),
                        "n_positive": len(pos), "n_negative": len(neg),
                        "label": f"clip median err > {CLIP_LABEL_DEG} deg",
                        "clip_seconds": CLIP_SECONDS, "per_rig": per_rig_cov,
                        "clips": clip_rows},
            "error_quantiles": {q: float(np.nanpercentile(all_err, q))
                                for q in (50, 75, 90, 95, 99)},
        }
        p = metrics["variants"][variant]["pooled"]
        print(f"variant {variant}: n={p['n']} median={p['median_deg']:.2f} "
              f"trim5={p['trimmed_rmse_5pct_deg']:.2f} trim10={p['trimmed_rmse_10pct_deg']:.2f} "
              f"rmse={p['rmse_deg']:.2f} >10deg={p['frac_over_10_deg']:.3f} "
              f"| smooth0.5 median={metrics['variants'][variant]['pooled_smoothed_0p5s']['median_deg']:.2f} "
              f"| S3 AUROC={s3['auroc']:.3f} clips={len(clip_rows)} pos={len(pos)}", flush=True)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"\nwrote {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
