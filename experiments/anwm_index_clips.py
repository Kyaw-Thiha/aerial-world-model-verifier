"""Write clips_index.csv from the clip folders, optionally syncing the tree to R2.

    python experiments/anwm_index_clips.py            # index only
    python experiments/anwm_index_clips.py --sync     # index, then upload to track-b/clips/

Every row is derived from the clip's own manifest.json, so the index cannot drift from the
frames on disk. Required tilt is recomputed from the commanded path with src/generation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.generation.paths import K_D, PathSpec

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "data/processed/anwm_clips"
FIELDS = ["clip_id", "regime", "maneuver", "policy", "designed_tilt_deg", "reachable",
          "n_frames", "peak_speed_ms", "peak_yaw_rate_deg_s", "step_spacing_m",
          "required_tilt_1s_drag_free_deg", "required_tilt_1s_kd_deg", "start_traj",
          "start_time", "seed", "checkpoint_sha256"]


def row_for(manifest: dict) -> dict:
    cp = manifest["commanded_path"]
    spec = PathSpec(name=manifest["id"], regime=manifest["regime"],
                    pos=np.asarray(cp["positions"], float), yaw=np.asarray(cp["yaw"], float))
    s = spec.speeds()
    meta = manifest.get("meta", {})
    return {"clip_id": manifest["id"], "regime": manifest["regime"],
            "maneuver": meta.get("maneuver", ""), "policy": meta.get("policy", ""),
            "designed_tilt_deg": manifest.get("designed_required_tilt_deg"),
            "reachable": meta.get("reachable", ""), "n_frames": manifest.get("frames"),
            "peak_speed_ms": round(s["peak_speed_ms"], 3),
            "peak_yaw_rate_deg_s": round(s["peak_yaw_rate_deg_s"], 2),
            "step_spacing_m": round(cp.get("step_spacing_m", float("nan")), 3),
            "required_tilt_1s_drag_free_deg": round(spec.peak_tilt(1.0, 0.0), 2),
            "required_tilt_1s_kd_deg": round(spec.peak_tilt(1.0, K_D), 2),
            "start_traj": manifest["start_state"]["traj"],
            "start_time": manifest["start_state"]["time"],
            "seed": manifest.get("seed"),
            "checkpoint_sha256": manifest.get("slim_checkpoint_sha256", "")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true")
    args = ap.parse_args()
    rows, bad = [], []
    for mf in sorted(CLIPS.glob("*/*/manifest.json")):
        m = json.loads(mf.read_text())
        frames = sorted((mf.parent / "frames").glob("*.png"))
        if len(frames) != m.get("frames"):
            bad.append((mf.parent.name, len(frames), m.get("frames")))
        rows.append(row_for(m))
    rows.sort(key=lambda r: r["clip_id"])
    out = CLIPS / "clips_index.csv"
    CLIPS.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} clips -> {out.relative_to(ROOT)}")
    if bad:
        print("MISMATCH between manifest frame count and files on disk:", bad[:10])
    if args.sync:
        import os

        import boto3
        from botocore.config import Config

        from src.utils.r2 import load_env

        load_env()  # populate os.environ from .env (the worktree links it)

        s3 = boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}"
                                            ".r2.cloudflarestorage.com",
                          aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                          aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                          region_name="auto",
                          config=Config(max_pool_connections=32,
                                        retries={"max_attempts": 5, "mode": "standard"}))
        bucket = os.environ["R2_BUCKET"]
        files = [p for p in CLIPS.rglob("*") if p.is_file()]
        # the shared R2 helper reads every object back and re-hashes; for ~1000 small frames that
        # is minutes of round trips, so bulk upload here and verify by size + a manifest sample
        from concurrent.futures import ThreadPoolExecutor

        def put(p):
            rel = str(p.relative_to(CLIPS))
            s3.upload_file(str(p), bucket, "track-b/clips/" + rel)
            return rel

        with ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(put, files))
        bad = []
        for rel in [str(p.relative_to(CLIPS)) for p in files
                    if p.name in ("manifest.json", "clips_index.csv")]:
            head = s3.head_object(Bucket=bucket, Key="track-b/clips/" + rel)
            if head["ContentLength"] != (CLIPS / rel).stat().st_size:
                bad.append(rel)
        if bad:
            raise SystemExit(f"size mismatch after upload: {bad[:5]}")
        print(f"synced {len(files)} files to s3://{bucket}/track-b/clips/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
