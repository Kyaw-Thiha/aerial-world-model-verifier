"""Track E: clock and time-offset audit for UZH-FPV (Task 1).

The question this answers, without using any pixel estimator: on what clock are the image
timestamps, the IMU samples and the Leica ground truth, and how much are they offset?

Method. The IMU and the Leica prism are bolted to the same airframe, so the *magnitude* of
the angular velocity is the same signal on both. Cross-correlating |omega| from ``imu.txt``
with |omega| differenced from the ground-truth quaternions gives the clock offset directly,
with no camera, no extrinsics and no estimator in the loop. The Kalibr camera-IMU timeshift
(published per rig) then carries the image clock onto the IMU clock.

Why it matters: the indoor-45 rig's image timestamps are on a clock about 18 years away from
the Leica clock but its *ranges still overlap numerically*, so a naive "keep frames inside
the truth interval" filter (Track C's `prepare_uzh`) silently pairs frames with unrelated
truth samples. Any depicted-vs-truth error computed on a sequence prepared that way is
meaningless, so the pairing has to be rebuilt on a measured offset.

    pixi run modal run remote/e_sync.py --seqs indoor_forward_5,indoor_45_9
"""
from __future__ import annotations

import modal

app = modal.App("awmv-e-sync")

e_data = modal.Volume.from_name("awmv-e-data", create_if_missing=True)
E_DATA = "/e-data"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy<2", "scipy", "pyyaml")
)


def _read_rows(text: str, min_cols: int = 8):
    import numpy as np

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        v = [float(x) for x in line.split()]
        if len(v) >= min_cols:
            rows.append(v)
    out = np.asarray(rows, dtype=np.float64)
    return out[np.argsort(out[:, 0])]


@app.function(image=image, volumes={E_DATA: e_data}, timeout=1800, cpu=4)
def sync_probe(seq: str, max_lag_s: float = 40.0, bins: int = 3) -> str:
    """Report the zip's three clocks and the IMU<->ground-truth offset from |omega|."""
    import json
    import zipfile
    from pathlib import Path

    import numpy as np
    from scipy.spatial.transform import Rotation

    zdir = Path(E_DATA) / "zips"
    cand = sorted(zdir.glob(f"{seq}*with_gt.zip")) + sorted(zdir.glob(f"{seq}*.zip"))
    zpath = cand[0]
    with zipfile.ZipFile(zpath) as z:
        gt = _read_rows(z.read("groundtruth.txt").decode(), 8)
        imu_raw = z.read("imu.txt").decode()
        imu_head = imu_raw.splitlines()[:3]
        imu = _read_rows(imu_raw, 7)
        img_lines = [l for l in z.read("left_images.txt").decode().splitlines()
                     if l.strip() and not l.startswith("#")]
        img_ts = np.array([float(l.split()[1]) for l in img_lines])

    gt_t = gt[:, 0]
    quat = gt[:, 4:8] / np.linalg.norm(gt[:, 4:8], axis=1, keepdims=True)
    # GT angular speed at the GT rate
    dt = np.diff(gt_t)
    om_gt = np.zeros(len(gt_t))
    if len(gt_t) > 2:
        R = Rotation.from_quat(quat)
        rel = R[:-1].inv() * R[1:]
        om_gt[:-1] = np.linalg.norm(rel.as_rotvec(), axis=1) / dt
        om_gt[-1] = om_gt[-2]

    imu_t = imu[:, 0]
    # column layout is checked, not assumed: report the header and per-column scale
    col_stats = {f"col{i}": [float(np.median(np.abs(imu[:, i]))),
                            float(np.percentile(np.abs(imu[:, i]), 99))] for i in range(1, imu.shape[1])}

    def best_gyro():
        """Pick the three consecutive columns that behave like a gyro (small, ~1 rad/s)."""
        best, best_i = None, None
        for i in range(1, imu.shape[1] - 2):
            v = imu[:, i:i + 3]
            m = np.median(np.abs(v))
            if 0.01 < m < 5.0:
                score = abs(np.log(m / 0.3))
                if best is None or score < best:
                    best, best_i = score, i
        return best_i

    gi = best_gyro() or 1
    om_imu = np.linalg.norm(imu[:, gi:gi + 3], axis=1)

    grid_dt = 0.002
    off0 = float(np.median(gt_t)) - float(np.median(imu_t))
    lo = min(imu_t[0], imu_t[-1]) - 5.0
    hi = max(imu_t[0], imu_t[-1]) + 5.0
    n = int((hi - lo) / grid_dt) + 1
    grid = lo + np.arange(n) * grid_dt

    def resample(t, y):
        return np.interp(grid, t, y, left=np.nan, right=np.nan)

    a_imu = resample(imu_t, om_imu)

    def corr_at(tau, t0=None, t1=None):
        """Normalised correlation of |omega| at offset tau (gt_time = imu_time + tau)."""
        m = np.isfinite(a_imu)
        if t0 is not None:
            m &= (grid >= t0) & (grid <= t1)
        a = a_imu[m]
        b = np.interp(grid[m] + tau, gt_t, om_gt, left=np.nan, right=np.nan)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 50:
            return -1.0, 0, int(ok.sum())
        a, b = a[ok], b[ok]
        a = np.sqrt(np.abs(a))
        b = np.sqrt(np.abs(b))
        a = a - a.mean()
        b = b - b.mean()
        d = np.sqrt((a @ a) * (b @ b))
        return (float(a @ b / d) if d > 0 else -1.0), 0, int(ok.sum())

    taus = off0 + np.arange(-max_lag_s, max_lag_s, 0.002)
    # coarse then fine: 20 ms steps first, then 2 ms around the peak
    coarse = off0 + np.arange(-max_lag_s, max_lag_s, 0.020)
    cs = np.array([corr_at(t, )[0] for t in coarse])
    best_c = float(coarse[int(np.argmax(cs))])
    fine = best_c + np.arange(-0.040, 0.040, 0.002)
    fs = np.array([corr_at(t)[0] for t in fine])
    ib = int(np.argmax(fs))
    best = float(fine[ib])

    # drift check: repeat on equal thirds of the IMU span
    span = imu_t[-1] - imu_t[0]
    per_bin = []
    for k in range(bins):
        t0 = imu_t[0] + span * k / bins
        t1 = imu_t[0] + span * (k + 1) / bins
        loc = best + np.arange(-2.0, 2.0, 0.002)
        vals = np.array([corr_at(t, t0, t1)[0] for t in loc])
        per_bin.append({"t0_s": float(t0 - imu_t[0]), "t1_s": float(t1 - imu_t[0]),
                        "offset_s": float(loc[int(np.argmax(vals))]),
                        "peak_corr": float(vals.max())})

    rec = {
        "seq": seq, "zip": str(zpath.name),
        "clocks": {
            "images": {"n": int(len(img_ts)), "first": float(img_ts.min()),
                       "last": float(img_ts.max()), "span_s": float(img_ts.max() - img_ts.min()),
                       "median_dt_s": float(np.median(np.diff(np.sort(img_ts))))},
            "imu": {"n": int(len(imu_t)), "first": float(imu_t.min()), "last": float(imu_t.max()),
                    "span_s": float(imu_t.max() - imu_t.min()),
                    "median_dt_s": float(np.median(np.diff(imu_t)))},
            "gt": {"n": int(len(gt_t)), "first": float(gt_t.min()), "last": float(gt_t.max()),
                   "span_s": float(gt_t.max() - gt_t.min()),
                   "median_dt_s": float(np.median(np.diff(gt_t)))},
        },
        "imu_header": imu_head, "imu_col_stats": col_stats, "gyro_cols": [gi, gi + 1, gi + 2],
        "images_minus_imu_first_s": float(img_ts.min() - imu_t.min()),
        "images_minus_imu_last_s": float(img_ts.max() - imu_t.max()),
        "gt_minus_imu_first_s": float(gt_t.min() - imu_t.min()),
        "gt_minus_imu_last_s": float(gt_t.max() - imu_t.max()),
        "offset_gt_minus_imu_s": best,
        "offset_peak_corr": float(fs[ib]),
        "offset_coarse_corr": float(cs.max()),
        "offset_curve": {"lags": [float(x) for x in fine[::2]],
                         "corr": [float(x) for x in fs[::2]]},
        "drift_bins": per_bin,
        "drift_span_s": float(per_bin[-1]["offset_s"] - per_bin[0]["offset_s"]) if bins > 1 else None,
    }
    (Path(E_DATA) / "sync").mkdir(parents=True, exist_ok=True)
    (Path(E_DATA) / "sync" / f"{seq}.json").write_text(json.dumps(rec, indent=1))
    e_data.commit()
    print(json.dumps({k: rec[k] for k in ("seq", "offset_gt_minus_imu_s", "offset_peak_corr",
                                          "drift_bins", "gyro_cols")}), flush=True)
    return json.dumps(rec)


@app.local_entrypoint()
def main(seqs: str = "", max_lag_s: float = 40.0) -> None:
    for s in [x for x in seqs.split(",") if x]:
        print(sync_probe.remote(s, max_lag_s), flush=True)
