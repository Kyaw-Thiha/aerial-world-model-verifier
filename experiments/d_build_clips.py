"""Build the shared ``verdict_clip.npz`` files for Track D from the stored TII artifacts.

Inputs are the Track A/A2 products already in the repository:

* ``data/processed/p3a/p3a_v1/<flight>.npz``: per-frame timestamps, commanded position,
  mocap quaternion (yaw is derived here), frame names, camera uptilt;
* ``data/processed/p3a/p3a_v1/clips.json``: the 4 s clip table;
* ``results/gate_a2/tii/<flight>__<condition>.jsonl``: per-frame GeoCalib gravity and
  confidence for every edited condition.

The output carries **only** the shared schema fields. Mocap enters as ``pos`` and ``yaw``
and nothing else; the required lean is recomputed from those two arrays by the scorer.

    PYTHONPATH=. pixi run python experiments/d_build_clips.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.datasets.tii_video import flight_parts
from src.verdict.dense import dense_required_lean
from src.verdict.lean import depicted_lean_from_gravity
from src.verdict.schema import Clip, save_clip

ROOT = Path(__file__).resolve().parents[1]
FLIGHTS = ["flight-01a-ellipse", "flight-04a-ellipse", "flight-07a-lemniscate",
           "flight-13a-trackRATM", "flight-01p-ellipse", "flight-07p-lemniscate",
           "flight-13p-trackRATM", "flight-16p-trackRATM"]

# stored condition -> schema condition
CONDITION_MAP = {
    "identity": "real",
    "tilt_k0": "tilt_removal",
    "tilt_k05": "scale_0.5",
    "tilt_k15": "scale_1.5",
    "mirror_roll": "mirror_roll",
    "yaw_control": "yaw_control",
}
DELAYS_MS = [100, 200, 300]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["frame"]] = row
    return out


def up_and_conf(spec, rows: dict[str, dict]) -> tuple[np.ndarray, np.ndarray]:
    names = [str(n) for n in spec["names"]]
    up = np.full((len(names), 3), np.nan)
    conf = np.full(len(names), np.nan)
    for i, name in enumerate(names):
        row = rows.get(name)
        if row is None:
            continue
        up[i] = np.asarray(row["gravity_up_cam"], dtype=np.float64)
        c = row.get("latitude_confidence_mean")
        if c is not None:
            conf[i] = float(c)
    return up, conf


def shift_depicted(lean: np.ndarray, t: np.ndarray, delay_s: float) -> np.ndarray:
    """Depicted lean delayed by ``delay_s``: ``l(t) = l_true(t - delay_s)``."""
    target = t - delay_s
    out = np.full_like(lean, np.nan)
    for k in range(lean.shape[1]):
        col = lean[:, k]
        ok = np.isfinite(col)
        if ok.sum() < 2:
            continue
        out[:, k] = np.interp(target, t[ok], col[ok], left=np.nan, right=np.nan)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="data/processed/p3a/p3a_v1")
    ap.add_argument("--rows", default="results/gate_a2/tii")
    ap.add_argument("--out", default="data/processed/track_d/clips")
    args = ap.parse_args()
    specs_dir, rows_dir, out_root = ROOT / args.specs, ROOT / args.rows, ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    clips = json.loads((specs_dir / "clips.json").read_text())["clips"]
    manifest = {"inputs": {}, "clips": [], "conditions": {}, "counts": {}}
    n_written = 0
    for flight in FLIGHTS:
        spec = np.load(specs_dir / f"{flight}.npz", allow_pickle=False)
        t_all = np.asarray(spec["t"], dtype=np.float64)
        pos_all = np.asarray(spec["pos"], dtype=np.float64)
        R_bw = Rotation.from_quat(spec["quat"]).as_matrix()
        yaw_all = Rotation.from_matrix(R_bw).as_euler("ZYX")[:, 0]
        uptilt = float(spec["uptilt_deg"])
        from src.perception.frames import camera_to_body_pitched
        R_cb = camera_to_body_pitched(np.deg2rad(uptilt))
        control, _ = flight_parts(flight)
        dense = dense_required_lean(
            ROOT / f"data/raw/tii/{control}/mocap_{flight}.csv", t_all)
        manifest["inputs"][flight] = {
            "spec": str((specs_dir / f"{flight}.npz").relative_to(ROOT)),
            "spec_sha256": sha256(specs_dir / f"{flight}.npz")}

        rows_by_cond = {}
        for stored, schema_cond in CONDITION_MAP.items():
            path = rows_dir / f"{flight}__{stored}.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            rows_by_cond[schema_cond] = load_rows(path)
        full = {}
        for schema_cond, rows in rows_by_cond.items():
            up, conf = up_and_conf(spec, rows)
            full[schema_cond] = (depicted_lean_from_gravity(up, R_cb), conf)

        for clip in [c for c in clips if c["flight"] == flight]:
            lo, hi = clip["index"]
            sl = slice(int(lo), int(hi))
            t = t_all[sl]
            pos = pos_all[sl]
            yaw = yaw_all[sl]
            base_meta = {
                "dataset": "TII racing",
                "flight": flight,
                "sequence": flight,
                "clip_id": clip["clip_id"],
                "fps": float(spec["measured_fps"]),
                "frame_stride": 10,
                "mount_source": "recorded",
                "uptilt_deg": uptilt,
                "estimator": "GeoCalib",
                "estimator_variant": "simple_divisional on raw fisheye",
                "truth_frames": "mocap position and yaw; no other mocap quantity enters",
                "track": clip["track"],
                "control": clip["control"],
                "speed_median_mps": float(clip["speed_median"]),
                "depicted_conf_meaning": (
                    "GeoCalib latitude_confidence_mean per frame; higher is better; the "
                    "frozen S3 gate uses the negative clip median"),
                "trackratm": "trackRATM" in flight,
                "schema_version": 2,
                "required_lean_source": (
                    "lean_req/vel_req: dense mocap differentiated at 274.73 Hz with a "
                    "0.37 s / poly-3 Savitzky-Golay window, sampled at the frame times"),
                "required_window_samples": int(dense.window),
                "required_dt_s": float(dense.dt_s),
            }
            for schema_cond in full:
                lean_all, conf_all = full[schema_cond]
                meta = dict(base_meta, condition=schema_cond, edited=schema_cond != "real")
                clip_obj = Clip(t=t.copy(), pos=pos.copy(), yaw=yaw.copy(),
                                depicted_lean=lean_all[sl].copy(),
                                depicted_conf=conf_all[sl].copy(), meta=meta,
                                lean_req=dense.l_req0[sl].copy(),
                                vel_req=dense.vel_req[sl].copy(),
                                tilt_req_deg=dense.tilt_deg[sl].copy())
                save_clip(out_root / flight / f"{clip['clip_id']}__{schema_cond}", clip_obj)
                n_written += 1

            # null edit: TII's identity warp IS the real clip (A2 measured 0.13-0.16 deg
            # difference from the Phase 1 run); it is stored as its own condition so the
            # control is explicit rather than assumed.
            lean_all, conf_all = full["real"]
            meta = dict(base_meta, condition="null_edit", edited=False,
                        note="identity warp; the stored real clip, reported as a control")
            save_clip(out_root / flight / f"{clip['clip_id']}__null_edit",
                      Clip(t=t.copy(), pos=pos.copy(), yaw=yaw.copy(),
                           depicted_lean=lean_all[sl].copy(),
                           depicted_conf=conf_all[sl].copy(), meta=meta,
                           lean_req=dense.l_req0[sl].copy(),
                           vel_req=dense.vel_req[sl].copy(),
                           tilt_req_deg=dense.tilt_deg[sl].copy()))
            n_written += 1

            lean_all, conf_all = full["real"]
            for delay in DELAYS_MS:
                delayed = shift_depicted(lean_all, t_all, delay / 1000.0)[sl]
                meta = dict(base_meta, condition=f"delay_ms:{delay}", edited=True,
                            applied_delay_ms=delay,
                            note="depicted series delayed by delay_ms; analysis condition")
                save_clip(out_root / flight / f"{clip['clip_id']}__delay_ms_{delay}",
                          Clip(t=t.copy(), pos=pos.copy(), yaw=yaw.copy(),
                               depicted_lean=delayed,
                               depicted_conf=conf_all[sl].copy(), meta=meta,
                               lean_req=dense.l_req0[sl].copy(),
                               vel_req=dense.vel_req[sl].copy(),
                               tilt_req_deg=dense.tilt_deg[sl].copy()))
                n_written += 1

            manifest["clips"].append({"clip_id": clip["clip_id"], "flight": flight,
                                      "n_frames": int(len(t))})

    manifest["counts"] = {"clips": len(manifest["clips"]), "files": n_written}
    (out_root.parent / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {n_written} clip files under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
