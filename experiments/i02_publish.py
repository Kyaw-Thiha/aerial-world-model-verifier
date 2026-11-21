"""Publish rendered clips from the Modal volume to R2 (Track I, Step 2).

Rendering happens on a rented GPU; publishing happens here. That split is forced by a fact
worth recording: the Modal secret ``awmv-r2`` cannot write to ``track-i/`` (S3 ``Unauthorized``
on ``PutObject``), while the workstation's ``.env`` key can, and the project's rule is to write
only inside the track's own prefix. So the container leaves each clip's video and truth on the
volume and this script moves them out, one clip at a time, verifying the sha256 on both ends
and removing the local copy afterwards (the workstation has 9.7 GB free; 300 clips would
otherwise be 1.5 GB of transient files).

    python experiments/i02_publish.py --limit 4        # a few clips
    python experiments/i02_publish.py --all            # everything the volume holds
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/i02/manifest.csv"
FIELDS = ["clip_id", "split", "family", "scene_seed", "maneuver", "uptilt_deg", "label",
          "violation", "defect_family", "magnitude_deg", "magnitude_unit", "defect_magnitude",
          "interval_start_s", "interval_end_s", "interval_fraction", "duration_s", "n_frames",
          "n_gt", "peak_tilt_req_deg", "median_tilt_req_deg", "peak_thrust_ratio",
          "params_json", "renderer", "engine", "samples", "r2_video", "r2_gt",
          "sha256_video", "sha256_gt", "bytes_video", "bytes_gt", "commit", "generated_utc",
          "status"]


def _modal(*args: str) -> None:
    pixi = Path.home() / "Documents/Projects/aerial-world-model-verifier/.pixi/envs/default/bin/modal"
    exe = str(pixi) if pixi.is_file() else "modal"
    subprocess.run([exe, *args], check=True, cwd=ROOT)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows() -> dict[str, dict]:
    if not MANIFEST.is_file():
        return {}
    with MANIFEST.open() as fh:
        return {r["clip_id"]: r for r in csv.DictReader(fh)}


def save_rows(rows: dict[str, dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for cid in sorted(rows):
            w.writerow({k: rows[cid].get(k, "") for k in FIELDS})


def volume_clips() -> list[str]:
    """Clip ids that exist on the volume, so a resumed batch is published too."""
    out = subprocess.run(
        [str(_modal_exe()), "volume", "ls", "--json", "awmv-i-data", "i02/clips"],
        check=True, capture_output=True, text=True, cwd=ROOT).stdout
    entries = json.loads(out)
    return sorted(Path(e["filename"]).name for e in entries if e.get("type") == "dir")


def _modal_exe() -> Path | str:
    pixi = Path.home() / "Documents/Projects/aerial-world-model-verifier/.pixi/envs/default/bin/modal"
    return pixi if pixi.is_file() else "modal"


def publish(limit: int = 0) -> dict:
    from src.utils import r2 as r2mod

    rows = load_rows()
    # every clip on the volume is a candidate, not only the last batch's rows: a resumed
    # render reports "skipped_existing" and still needs publishing
    for cid in volume_clips():
        if cid not in rows:
            rows[cid] = {"clip_id": cid, "status": "ok"}
    todo = [cid for cid, r in sorted(rows.items())
            if not r.get("r2_video") and r.get("status") in ("ok", "skipped_existing", "")]
    if limit:
        todo = todo[:limit]
    if not todo:
        print("nothing to publish")
        return {"published": 0}
    env = r2mod.load_env()
    tmp = Path("/tmp/i02_publish")
    tmp.mkdir(parents=True, exist_ok=True)
    done = 0
    for cid in todo:
        row = rows[cid]
        local_dir = tmp / cid
        shutil.rmtree(local_dir, ignore_errors=True)     # `volume get` refuses an existing path
        local_dir.mkdir(parents=True, exist_ok=True)
        _modal("volume", "get", "awmv-i-data", f"i02/clips/{cid}/manifest_row.json", str(local_dir))
        with (local_dir / "manifest_row.json").open() as fh:
            meta = json.load(fh)
        keep = {k: rows.get(cid, {}).get(k, "") for k in ("r2_video", "r2_gt")}
        rows[cid] = {**meta, **keep}
        row = rows[cid]              # the loop's binding must see the refreshed row
        _modal("volume", "get", "awmv-i-data", f"i02/clips/{cid}/{cid}.mp4", str(local_dir))
        _modal("volume", "get", "awmv-i-data", f"i02/clips/{cid}/gt.npz", str(local_dir))
        mp4 = local_dir / f"{cid}.mp4"
        gt = local_dir / "gt.npz"
        if not (mp4.is_file() and gt.is_file()):
            print(f"{cid}: missing on the volume, skipping")
            continue
        key_video = f"track-i/videos/{row['split']}/{row['family']}/{cid}.mp4"
        key_gt = f"track-i/gt/{cid}.npz"
        up_v = r2mod.upload(mp4, key_video, env)
        up_g = r2mod.upload(gt, key_gt, env)
        if not (up_v["match"] and up_g["match"]):
            print(f"{cid}: sha256 mismatch on upload, left unpublished")
            continue
        row["r2_video"], row["r2_gt"] = key_video, key_gt
        row["sha256_video"], row["sha256_gt"] = up_v["sha256"], up_g["sha256"]
        row["bytes_video"], row["bytes_gt"] = str(up_v["bytes"]), str(up_g["bytes"])
        rows[cid] = row
        save_rows(rows)
        shutil.rmtree(local_dir, ignore_errors=True)
        done += 1
        print(f"{cid}: {up_v['bytes'] / 1e6:.1f} MB video, {up_g['bytes'] / 1e3:.0f} kB truth")
    save_rows(rows)
    print(f"published {done} clips")
    return {"published": done}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    publish(limit=0 if args.all else (args.limit or 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
