"""Track E: measure the image/IMU clock against the Leica truth clock (Task 1).

Per sequence it reads the ground-truth table, the IMU stream and the image list straight out of
the sequence zip and reports:

* the three clocks (first/last/span/rate) and whether the images share the IMU clock;
* the offset tau with ``gt_time = imu_time + tau`` measured by correlating |omega| from the IMU
  with |omega| differenced from the truth quaternions (``src.depicted.sync``);
* the correlation peak, the offset resolution, and per-bin offsets (clock drift);
* the published Kalibr camera-IMU timeshift for the rig, for comparison.

Why this is not optional: the indoor-45 rig stamps images on a clock about 18 years from the
Leica rows, and its numeric ranges still overlap, so a naive "keep the frames inside the truth
interval" filter pairs frames with unrelated truth. Any depicted-vs-truth error computed from
such a pairing is meaningless, and Track C's indoor_45_9 numbers were built that way.

    pixi run modal run remote/e_sync.py
"""
from __future__ import annotations

import modal

app = modal.App("awmv-e-sync")

e_data = modal.Volume.from_name("awmv-e-data", create_if_missing=True)
c_data = modal.Volume.from_name("awmv-c-data", create_if_missing=True)
E_DATA = "/e-data"
C_DATA = "/c-data"

UZH_BASE = "https://download.ifi.uzh.ch/rpg/web/datasets/uzh-fpv-newer-versions/v3"
CALIB = {"indoor_forward": "indoor_forward_calib_snapdragon.zip",
         "indoor_45": "indoor_45_calib_snapdragon.zip",
         "outdoor_forward": "outdoor_forward_calib_snapdragon.zip",
         "outdoor_45": "outdoor_45_calib_snapdragon.zip"}

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy<2", "scipy", "pyyaml")
    .add_local_python_source("src")
)


def _rows(text: str, min_cols: int):
    import numpy as np

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


def _zip_path(seq: str):
    """Prefer Track C's copy on its own volume: a volume-to-volume copy buys nothing."""
    import shutil
    import urllib.request
    from pathlib import Path

    stem = f"{seq}_snapdragon_with_gt.zip"
    own = Path(E_DATA) / "zips" / stem
    if own.exists():
        return own
    alt = Path(C_DATA) / "zips" / stem
    if alt.exists():
        return alt
    own.parent.mkdir(parents=True, exist_ok=True)
    tmp = own.with_suffix(".part")
    with urllib.request.urlopen(f"{UZH_BASE}/{stem}", timeout=900) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(own)
    return own


@app.function(image=image, volumes={E_DATA: e_data, C_DATA: c_data}, timeout=1800, cpu=4)
def sync_probe(seq: str, drift_bins: int = 3) -> str:
    import json
    import zipfile
    from pathlib import Path

    import numpy as np
    import yaml
    from scipy.spatial.transform import Rotation

    from src.depicted.sync import estimate_offset
    from src.depicted.uzh import rig_of

    zpath = _zip_path(seq)
    with zipfile.ZipFile(zpath) as z:
        gt = _rows(z.read("groundtruth.txt").decode(), 8)
        imu = _rows(z.read("imu.txt").decode(), 7)
        lines = [l for l in z.read("left_images.txt").decode().splitlines()
                 if l.strip() and not l.startswith("#")]
    img_ts = np.array([float(l.split()[1]) for l in lines])
    img_names = [l.split()[2] for l in lines]

    t_gt = gt[:, 0]
    quat = gt[:, 4:8] / np.linalg.norm(gt[:, 4:8], axis=1, keepdims=True)
    om_gt = np.zeros(len(t_gt))
    if len(t_gt) > 3:
        R = Rotation.from_quat(quat)
        rel = R[:-1].inv() * R[1:]
        dt = np.diff(t_gt)
        om_gt[:-1] = np.linalg.norm(rel.as_rotvec(), axis=1) / dt
        om_gt[-1] = om_gt[-2]

    t_imu = imu[:, 0]
    cols = imu.shape[1]
    # choose the three consecutive columns that behave like a gyro instead of assuming a layout
    best_i, best_score = 1, None
    for i in range(1, cols - 2):
        m = float(np.median(np.abs(imu[:, i])))
        if 0.01 < m < 5.0:
            s = abs(np.log(m / 0.3))
            if best_score is None or s < best_score:
                best_i, best_score = i, s
    om_imu = np.linalg.norm(imu[:, best_i:best_i + 3], axis=1)

    est = estimate_offset(t_imu, om_imu, t_gt, om_gt, drift_bins=drift_bins)

    # the rig's published camera-IMU timeshift, for comparison
    shift = None
    cp = Path(E_DATA) / "zips" / CALIB[rig_of(seq)]
    if not cp.exists():
        cp = Path(C_DATA) / "zips" / CALIB[rig_of(seq)]
    if cp.exists():
        with zipfile.ZipFile(cp) as zc:
            name = next(n for n in zc.namelist() if "camchain-imucam" in n and n.endswith(".yaml"))
            chain = yaml.safe_load(zc.read(name))
        shift = float(chain["cam0"]["timeshift_cam_imu"])

    imu_first, imu_last = float(t_imu[0]), float(t_imu[-1])
    rec = {
        "seq": seq, "rig": rig_of(seq), "zip": zpath.name,
        "clocks": {
            "images": {"n": int(len(img_ts)), "first": float(img_ts.min()),
                       "last": float(img_ts.max()),
                       "span_s": float(img_ts.max() - img_ts.min()),
                       "dt_s": float(np.median(np.diff(np.sort(img_ts))))},
            "imu": {"n": int(len(t_imu)), "first": imu_first, "last": imu_last,
                    "span_s": imu_last - imu_first,
                    "dt_s": float(np.median(np.diff(t_imu)))},
            "gt": {"n": int(len(t_gt)), "first": float(t_gt[0]), "last": float(t_gt[-1]),
                   "span_s": float(t_gt[-1] - t_gt[0]),
                   "dt_s": float(np.median(np.diff(t_gt)))},
        },
        "gui_columns_used": [int(best_i), int(best_i + 1), int(best_i + 2)],
        "images_minus_imu_s": {"first": float(img_ts.min() - imu_first),
                               "last": float(img_ts.max() - imu_last)},
        "gt_minus_imu_s": {"first": float(t_gt[0] - imu_first),
                           "last": float(t_gt[-1] - imu_last)},
        "offset_gt_minus_imu_s": est["offset_s"],
        "offset_peak_corr": est["peak_corr"],
        "offset_resolution_s": est["resolution_s"],
        "offset_epoch_difference_s": float(est["offset_s"] - (t_gt[0] - imu_first)),
        "drift_bins": est["drift_bins"],
        "drift_span_s": est["drift_span_s"],
        "kalibr_timeshift_cam_imu_s": shift,
        "first_image_name": img_names[0] if img_names else None,
    }
    out = Path(E_DATA) / "sync"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{seq}.json").write_text(json.dumps(rec, indent=1))
    e_data.commit()
    print(json.dumps({k: rec[k] for k in ("seq", "offset_gt_minus_imu_s", "offset_peak_corr",
                                          "offset_epoch_difference_s", "drift_span_s",
                                          "kalibr_timeshift_cam_imu_s")}), flush=True)
    return json.dumps(rec)


@app.local_entrypoint()
def main(seqs: str = "", drift_bins: int = 3) -> None:
    import json

    from src.depicted.uzh import all_sequences

    wanted = [s for s in seqs.split(",") if s] or all_sequences()
    handles = [(s, sync_probe.spawn(s, drift_bins)) for s in wanted]
    for s, h in handles:
        try:
            print(json.dumps(json.loads(h.get()), indent=1)[:400], flush=True)
        except Exception as exc:
            print(f"FAILED {s}: {exc}", flush=True)
