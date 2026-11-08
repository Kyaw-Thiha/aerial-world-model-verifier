"""Fetch TII RATM video (camera JPEG zips), frame timestamps and calibration.

Track A. Uses the same ranged reads as ``scripts/fetch_tii.py`` on the chunked
GitHub release, so only the requested members are transferred. Each flight gets

* ``camera_<flight>.zip``                inner zip of JPEG frames (stored as-is)
* ``<flight>_cam_ts_sync.csv``           per-frame row with the authors' sync'd telemetry
* ``camera_<flight>.csv``                frame timestamp (Unix us) -> filename
* ``metadata_<flight>.yaml``             camera angle, exposure, lighting

and the official OpenCV calibration JSON files (pinned to a repo commit) go to
``<out>/calibration/``.

Idempotent: a member whose local size matches the zip entry is skipped. Every
fetched member is read through the outer zip (CRC checked by ``zipfile``) and its
sha256 is recorded in ``<out>/manifest.json``.

    pixi run python scripts/fetch_tii_video.py --list
    pixi run python scripts/fetch_tii_video.py --out data/raw/tii_video --small-only
    # on Modal (writes to Volume awmv-a-data): pixi run modal run remote/a_fetch_video.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.fetch_tii import open_archive  # noqa: E402

# Flights chosen for Track A (same eight as Phase 0, see docs/decisions.md [A]).
FLIGHTS = {
    "autonomous": ["flight-01a-ellipse", "flight-04a-ellipse",
                   "flight-07a-lemniscate", "flight-13a-trackRATM"],
    "piloted": ["flight-01p-ellipse", "flight-07p-lemniscate",
                "flight-13p-trackRATM", "flight-16p-trackRATM"],
}

CALIB_REPO_COMMIT = "df4a2e11c46980d48afdc9aa76ef4faa673f5928"  # repo HEAD when fetched
CALIB_FILES = ["calib_a-trackRATM.json", "calib_ap-ellipse-lemniscate.json",
               "calib_p-trackRATM.json", "drone_to_camera.json"]
CALIB_URL = ("https://raw.githubusercontent.com/tii-racing/drone-racing-dataset/"
             f"{CALIB_REPO_COMMIT}/camera_calibration/{{name}}")


def member_names(flight: str, small_only: bool) -> list[str]:
    names = [f"{flight}_cam_ts_sync.csv", f"csv_raw/camera_{flight}.csv",
             f"metadata_{flight}.yaml"]
    if not small_only:
        names.insert(0, f"camera_{flight}.zip")
    return names


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def fetch(out: Path, small_only: bool = False, list_only: bool = False,
          flights: dict[str, list[str]] | None = None, log=print) -> dict:
    flights = flights or FLIGHTS
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    total = 0
    for kind, fl in flights.items():
        z, f = open_archive(kind)
        index = {i.filename: i for i in z.infolist() if not i.is_dir()}
        for flight in fl:
            for rel in member_names(flight, small_only):
                info = index[f"{kind}/{flight}/{rel}"]
                total += info.file_size
                dest = out / kind / Path(rel).name
                if list_only:
                    log(f"{info.file_size/1e6:9.1f} MB  {info.filename}")
                    continue
                if dest.exists() and dest.stat().st_size == info.file_size:
                    log(f"skip  {dest} (size matches)")
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                with z.open(info) as src, open(tmp, "wb") as dst:  # CRC checked at EOF
                    while chunk := src.read(1 << 22):
                        dst.write(chunk)
                if tmp.stat().st_size != info.file_size:
                    raise IOError(f"size mismatch for {dest}")
                tmp.rename(dest)
                manifest[str(dest.relative_to(out))] = {
                    "bytes": info.file_size, "sha256": sha256(dest),
                    "member": info.filename, "archive": kind}
                log(f"fetch {dest} ({info.file_size/1e6:.1f} MB)")
    if not list_only:
        cdir = out / "calibration"
        cdir.mkdir(exist_ok=True)
        for name in CALIB_FILES:
            dest = cdir / name
            if not dest.exists():
                with urllib.request.urlopen(CALIB_URL.format(name=name), timeout=60) as r:
                    dest.write_bytes(r.read())
            manifest[f"calibration/{name}"] = {"bytes": dest.stat().st_size, "sha256": sha256(dest),
                                               "source_commit": CALIB_REPO_COMMIT}
        mpath.write_text(json.dumps(manifest, indent=2))
    log(f"total selected: {total/1e9:.3f} GB")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/tii_video")
    ap.add_argument("--small-only", action="store_true", help="skip the camera JPEG zips")
    ap.add_argument("--list", action="store_true", help="print sizes and exit")
    args = ap.parse_args()
    fetch(Path(args.out), small_only=args.small_only, list_only=args.list)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
