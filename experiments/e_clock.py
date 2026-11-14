"""Track E Task 1: clock audit run locally, straight from the UZH sequence zips.

Same measurement as ``remote/e_sync.py`` (which needs the whole zip on a container) but reads
only the three small members over HTTP ``Range``, so it runs in seconds per sequence and needs
no volume. Both implementations write the same JSON shape; when both have run for a sequence
their offsets are compared and any disagreement is reported.

    python experiments/e_clock.py --out results/e_audit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from src.depicted.sync import estimate_offset
from src.depicted.uzh import SEQUENCE_RIG, all_sequences, rig_of

UZH_BASE = "https://download.ifi.uzh.ch/rpg/web/datasets/uzh-fpv-newer-versions/v3"
CALIB = {"indoor_forward": "indoor_forward_calib_snapdragon.zip",
         "indoor_45": "indoor_45_calib_snapdragon.zip",
         "outdoor_forward": "outdoor_forward_calib_snapdragon.zip",
         "outdoor_45": "outdoor_45_calib_snapdragon.zip"}
CALIB_URL = "https://download.ifi.uzh.ch/rpg/web/datasets/uzh-fpv/calib"


def _rows(text: str, min_cols: int) -> np.ndarray:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        v = [float(x) for x in line.split()]
        if len(v) >= min_cols:
            out.append(v)
    a = np.asarray(out, dtype=np.float64)
    return a[np.argsort(a[:, 0])]


def kalibr_timeshift(rig: str, cache: Path) -> float | None:
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / CALIB[rig]
    if not p.exists():
        import urllib.request

        with urllib.request.urlopen(f"{CALIB_URL}/{CALIB[rig]}", timeout=600) as r:
            p.write_bytes(r.read())
    import zipfile

    with zipfile.ZipFile(p) as z:
        name = next(n for n in z.namelist() if "camchain-imucam" in n and n.endswith(".yaml"))
        chain = yaml.safe_load(z.read(name))
    return float(chain["cam0"]["timeshift_cam_imu"])


def audit(seq: str, members_dir: Path, cache: Path, drift_bins: int = 3) -> dict:
    """Measure the clocks from the three text members of one sequence."""
    d = Path(members_dir) / seq
    gt = _rows((d / "groundtruth.txt").read_text(), 8)
    # imu.txt rows are ``index timestamp wx wy wz ax ay az``: the first column is a row
    # counter, and the timestamp is the second one (verified from the file's own header)
    imu = _rows((d / "imu.txt").read_text(), 8)
    header = (d / "imu.txt").read_text().splitlines()[0].lstrip("# ").split()
    if header[:2] != ["timestamp", "ang_vel_x"] and len(header) == 8:
        if header[1] != "timestamp":
            raise ValueError(f"{d}/imu.txt: unexpected header {header}")
    lines = [l for l in (d / "left_images.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    img_ts = np.array([float(l.split()[1]) for l in lines])
    t_gt = gt[:, 0]
    quat = gt[:, 4:8] / np.linalg.norm(gt[:, 4:8], axis=1, keepdims=True)
    R = Rotation.from_quat(quat)
    # body-frame angular velocity: for R_{i+1} = R_i Exp(delta), rotvec(delta) is exactly
    # what a gyro bolted to the same body measures (in that body frame)
    om_gt = np.zeros((len(t_gt), 3))
    om_gt[:-1] = (R[:-1].inv() * R[1:]).as_rotvec() / np.diff(t_gt)[:, None]
    om_gt[-1] = om_gt[-2]

    t_imu = imu[:, 1]
    om_imu = imu[:, 2:5]
    est = estimate_offset(t_imu, om_imu, t_gt, om_gt, drift_bins=drift_bins)

    return {
        "seq": seq, "rig": rig_of(seq),
        "source": f"{UZH_BASE}/{seq}_snapdragon_with_gt.zip (text members only)",
        "clocks": {
            "images": {"n": int(len(img_ts)), "first": float(img_ts.min()),
                       "last": float(img_ts.max()),
                       "span_s": float(img_ts.max() - img_ts.min()),
                       "dt_s": float(np.median(np.diff(np.sort(img_ts))))},
            "imu": {"n": int(len(t_imu)), "first": float(t_imu[0]), "last": float(t_imu[-1]),
                    "span_s": float(t_imu[-1] - t_imu[0]),
                    "dt_s": float(np.median(np.diff(t_imu)))},
            "gt": {"n": int(len(t_gt)), "first": float(t_gt[0]), "last": float(t_gt[-1]),
                   "span_s": float(t_gt[-1] - t_gt[0]),
                   "dt_s": float(np.median(np.diff(t_gt)))},
        },
        "imu_columns_used": [2, 3, 4],
        "images_minus_imu_s": {"first": float(img_ts.min() - t_imu[0]),
                               "last": float(img_ts.max() - t_imu[-1])},
        "offset_gt_minus_imu_s": est["offset_s"],
        "offset_peak_corr": est["peak_corr"],
        "offset_overlap_s": est["overlap_s"],
        "offset_identifiable": est["identifiable"],
        "offset_resolution_s": est["resolution_s"],
        "offset_epoch_difference_s": float(est["offset_s"] - (t_gt[0] - t_imu[0])),
        "drift_bins": est["drift_bins"],
        "drift_span_s": est["drift_span_s"],
        "n_usable_bins": est.get("n_usable_bins"),
        "kalibr_timeshift_cam_imu_s": kalibr_timeshift(rig_of(seq), cache),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e_audit")
    ap.add_argument("--cache", default="data/raw/uzh_calib")
    ap.add_argument("--members", default="data/uzh_members",
                    help="directory of groundtruth.txt/imu.txt/left_images.txt per sequence")
    ap.add_argument("--seqs", default="")
    ap.add_argument("--drift-bins", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "sync").mkdir(parents=True, exist_ok=True)
    wanted = [s for s in args.seqs.split(",") if s] or all_sequences()
    recs = []
    for seq in wanted:
        rec = audit(seq, Path(args.members), Path(args.cache), args.drift_bins)
        (out / "sync" / f"{seq}.json").write_text(json.dumps(rec, indent=1))
        recs.append(rec)
        print(f"{seq:18s} {rec['rig']:16s} "
              f"tau={rec['offset_gt_minus_imu_s']:.3f} "
              f"(epoch diff {rec['offset_epoch_difference_s']:+.3f}) "
              f"corr={rec['offset_peak_corr']:.3f} overlap={rec['offset_overlap_s']:5.1f}s "
              f"ok={rec['offset_identifiable']} res={rec['offset_resolution_s']:.3f}s "
              f"drift={rec['drift_span_s']} "
              f"kalibr_shift={rec['kalibr_timeshift_cam_imu_s']} "
              f"img-imu={rec['images_minus_imu_s']['first']:+.2f}", flush=True)
    (out / "sync_summary.json").write_text(json.dumps(recs, indent=1))
    print(f"\nwrote {out / 'sync_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
