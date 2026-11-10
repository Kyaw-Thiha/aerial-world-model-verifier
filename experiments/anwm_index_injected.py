"""Pairing table, index and R2 sync for the injected-attitude arm.

    python -m experiments.anwm_index_injected            # index only
    python -m experiments.anwm_index_injected --sync     # index + upload to track-b/clips_injected/

Every row is derived from the manifests on disk plus
`results/anwm_injected/coverage_level_original.json`; nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from src.generation.paths import K_D, PathSpec

ROOT = Path(__file__).resolve().parents[1]
LEVEL = ROOT / "data/processed/anwm_clips/C"
REGEN = ROOT / "data/processed/anwm_clips_regen/C"
INJ = ROOT / "data/processed/anwm_clips_injected"
COV = ROOT / "results/anwm_injected/coverage_level_original.json"
OUT_DIR = ROOT / "data/processed/anwm_clips_injected"
FIELDS = ["clip_id", "arm", "maneuver", "policy", "designed_tilt_deg",
          "commanded_tilt_1s_drag_free_deg", "injected_tilt_drag_free_deg",
          "injected_roll_max_deg", "injected_pitch_max_deg", "paired_level_id", "level_source",
          "repeat_noise_mae", "prior_cov_level_mean", "prior_cov_level_min",
          "prior_cov_inj_mean", "prior_cov_inj_min", "seed", "checkpoint_sha256", "code_commit",
          "n_frames"]


def frames(d: Path) -> list[np.ndarray]:
    return [np.asarray(Image.open(p), float) for p in sorted((d / "frames").glob("*.png"))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true")
    args = ap.parse_args()
    cov = json.loads(COV.read_text()) if COV.exists() else {}
    rows = []
    for arm_dir in sorted(INJ.glob("*")):
        arm = arm_dir.name
        for d in sorted((arm_dir / "C").iterdir()):
            m = json.loads((d / "manifest.json").read_text())
            cp = m["commanded_path"]
            spec = PathSpec(name=m["id"], regime="C", pos=np.asarray(cp["positions"]),
                            yaw=np.asarray(cp["yaw"]))
            roll = np.asarray(m.get("injected_roll_deg") or [], float)
            pitch = np.asarray(m.get("injected_pitch_deg") or [], float)
            ztilt = np.nan
            if len(roll) > 1:
                c = np.cos(np.radians(roll[1:])) * np.cos(np.radians(pitch[1:]))
                ztilt = float(np.degrees(np.arccos(np.clip(c, -1, 1))).max())
            c = cov.get(m["paired_level_id"], {})
            # repeat noise: level original versus its regenerated twin, same seed
            repeat = None
            lvl_id = m["paired_level_id"]
            if (REGEN / lvl_id).exists() and (LEVEL / lvl_id).exists():
                a, b = frames(LEVEL / lvl_id), frames(REGEN / lvl_id)
                n = min(len(a), len(b))
                repeat = float(np.mean([np.abs(a[i] - b[i]).mean() for i in range(n)]))
            rows.append({
                "clip_id": m["id"], "arm": arm,
                "maneuver": m.get("meta", {}).get("maneuver", ""),
                "policy": m.get("meta", {}).get("policy", ""),
                "designed_tilt_deg": m.get("designed_required_tilt_deg"),
                "commanded_tilt_1s_drag_free_deg": round(spec.peak_tilt(1.0, 0.0), 3),
                "injected_tilt_drag_free_deg": None if np.isnan(ztilt) else round(ztilt, 3),
                "injected_roll_max_deg": None if not len(roll) else round(float(np.abs(roll).max()), 3),
                "injected_pitch_max_deg": None if not len(pitch) else round(float(np.abs(pitch).max()), 3),
                "paired_level_id": lvl_id,
                "level_source": "regenerated" if repeat is not None else "original",
                "repeat_noise_mae": None if repeat is None else round(repeat, 4),
                "prior_cov_level_mean": (c.get("level") or {}).get("mean"),
                "prior_cov_level_min": (c.get("level") or {}).get("min"),
                "prior_cov_inj_mean": m.get("prior_coverage_mean"),
                "prior_cov_inj_min": m.get("prior_coverage_min"),
                "seed": m.get("seed"),
                "checkpoint_sha256": m.get("slim_checkpoint_sha256", "")[:16],
                "code_commit": (m.get("code_commit") or "")[:16],
                "n_frames": m.get("frames"),
            })
    rows.sort(key=lambda r: (r["arm"], r["clip_id"]))
    out = OUT_DIR / "clips_injected_index.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out.relative_to(ROOT)}")
    missing = [r["clip_id"] for r in rows if r["prior_cov_inj_min"] is None]
    if missing:
        print("rows without recorded coverage:", missing[:5])
    if args.sync:
        import boto3
        from botocore.config import Config

        from src.utils.r2 import load_env

        load_env()
        s3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}"
                                            ".r2.cloudflarestorage.com",
                          aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                          aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                          region_name="auto",
                          config=Config(max_pool_connections=32,
                                        retries={"max_attempts": 5, "mode": "standard"}))
        bucket = os.environ["R2_BUCKET"]
        files = [p for p in INJ.rglob("*") if p.is_file()]
        from concurrent.futures import ThreadPoolExecutor

        def put(p):
            s3.upload_file(str(p), bucket, "track-b/clips_injected/" + str(p.relative_to(INJ)))

        with ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(put, files))
        for extra in (out, COV,
                      ROOT / "results/anwm_clips_injected/specs.json",
                      ROOT / "results/anwm_injected/stage0_summary.json"):
            if extra.exists():
                s3.upload_file(str(extra), bucket, "track-b/clips_injected/" + extra.name)
        print(f"synced {len(files)} files + {4} index artifacts to s3://{bucket}/"
              f"track-b/clips_injected/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
