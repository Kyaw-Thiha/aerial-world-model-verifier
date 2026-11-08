"""Fetch mocap/IMU CSVs (no images, no rosbags) for selected TII RATM flights.

Source: https://github.com/tii-racing/drone-racing-dataset (release v3.0.0).
The release ships `autonomous_zipchunk01..03` and `piloted_zipchunk01..07`; the
official downloader concatenates them into `autonomous.zip` / `piloted.zip`
(about 5 GB and 12 GB, mostly images). We never download whole chunks: the
chunks are exposed as one virtual file via HTTP Range requests, so `zipfile`
reads the central directory and extracts only the requested CSV members.

Usage:
    pixi run fetch-tii --list                 # list CSV members in both archives
    pixi run fetch-tii                        # fetch the flights in FLIGHTS
    pixi run fetch-tii --flights flight-01a-ellipse ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://github.com/Drone-Racing/drone-racing-dataset/releases/download/v3.0.0"
ARCHIVES = {
    "autonomous": [f"autonomous_zipchunk{i:02d}" for i in range(1, 4)],
    "piloted": [f"piloted_zipchunk{i:02d}" for i in range(1, 8)],
}
OUT_DIR = Path("data/raw/tii")
BLOCK = 8 * 1024 * 1024  # read-ahead block size for ranged reads


def _open(req: urllib.request.Request):
    return urllib.request.urlopen(req, timeout=120)


def chunk_size(name: str) -> int:
    req = urllib.request.Request(f"{BASE}/{name}", headers={"Range": "bytes=0-0"})
    with _open(req) as r:
        # Content-Range: bytes 0-0/<total>
        return int(r.headers["Content-Range"].rsplit("/", 1)[1])


class ChunkedHTTPFile:
    """Read-only seekable view over several concatenated remote files."""

    def __init__(self, names: list[str]):
        self.names = names
        self.sizes = [chunk_size(n) for n in names]
        self.total = sum(self.sizes)
        self.pos = 0
        self._cache: dict[int, bytes] = {}
        self.bytes_fetched = 0

    # -- file protocol -----------------------------------------------------
    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.total + off
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.total - self.pos
        end = min(self.pos + n, self.total)
        out = []
        while self.pos < end:
            blk = self.pos // BLOCK
            data = self._block(blk)
            lo = self.pos - blk * BLOCK
            take = data[lo : lo + (end - self.pos)]
            out.append(take)
            self.pos += len(take)
        return b"".join(out)

    # -- internals ---------------------------------------------------------
    def _block(self, blk: int) -> bytes:
        if blk in self._cache:
            return self._cache[blk]
        start = blk * BLOCK
        stop = min(start + BLOCK, self.total)
        data = self._fetch(start, stop)
        if len(self._cache) > 8:  # keep memory bounded
            self._cache.pop(next(iter(self._cache)))
        self._cache[blk] = data
        return data

    def _fetch(self, start: int, stop: int) -> bytes:
        # a block may straddle a chunk boundary
        parts, off = [], 0
        for name, size in zip(self.names, self.sizes):
            lo, hi = max(start, off), min(stop, off + size)
            if lo < hi:
                req = urllib.request.Request(
                    f"{BASE}/{name}",
                    headers={"Range": f"bytes={lo - off}-{hi - off - 1}"},
                )
                with _open(req) as r:
                    parts.append(r.read())
            off += size
        data = b"".join(parts)
        self.bytes_fetched += len(data)
        assert len(data) == stop - start, (len(data), stop - start)
        return data


def open_archive(kind: str) -> tuple[zipfile.ZipFile, ChunkedHTTPFile]:
    f = ChunkedHTTPFile(ARCHIVES[kind])
    return zipfile.ZipFile(f), f


# member kind -> substring that identifies the file inside a flight folder
MEMBER_KINDS = {
    "mocap": "csv_raw/mocap_",              # native mocap pose stream (275 Hz)
    "metadata": "metadata_flight-",          # per-flight yaml (track, control, camera angle)
    "imu": "csv_raw/ros2bag_dump/imu_",      # on-board IMU (optional)
    "camera": "csv_raw/camera_",             # frame timestamps (optional)
    "gate": "csv_raw/gate_corners_",         # gate poses (optional)
}


def members(z: zipfile.ZipFile, kinds: list[str]) -> list[zipfile.ZipInfo]:
    pats = [MEMBER_KINDS[k] for k in kinds]
    out = []
    for i in z.infolist():
        if i.is_dir():
            continue
        name = i.filename
        if name.endswith(".yaml") or name.endswith(".csv"):
            if any(p in name for p in pats):
                out.append(i)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list CSV members and exit")
    ap.add_argument("--flights", nargs="*", help="flight folder names to fetch")
    ap.add_argument("--members", default="mocap,metadata",
                    help=f"comma-separated subset of {sorted(MEMBER_KINDS)}")
    args = ap.parse_args()
    kinds = [k.strip() for k in args.members.split(",") if k.strip()]
    bad = [k for k in kinds if k not in MEMBER_KINDS]
    if bad:
        sys.exit(f"unknown member kinds {bad}; choose from {sorted(MEMBER_KINDS)}")

    if args.list:
        for kind in ARCHIVES:
            z, f = open_archive(kind)
            print(f"== {kind}: {f.total/1e9:.2f} GB virtual zip, {len(z.infolist())} members")
            for i in members(z, kinds):
                print(f"{i.file_size/1e6:9.1f} MB  {i.filename}")
            print(f"(fetched {f.bytes_fetched/1e6:.1f} MB to list)")
        return 0

    if not args.flights:
        sys.exit("pass --flights (see FLIGHTS in docs/decisions.md) or use --list")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    wanted = set(args.flights)
    for kind in ARCHIVES:
        z, f = open_archive(kind)
        for info in members(z, kinds):
            flight = Path(info.filename).parts[1]  # <kind>/<flight>/...
            if flight not in wanted:
                continue
            dest = OUT_DIR / kind / Path(info.filename).name
            if dest.exists() and dest.stat().st_size == info.file_size:
                print(f"skip  {dest} (already present)")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dest, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            manifest[str(dest)] = {"bytes": dest.stat().st_size, "sha256": sha256(dest),
                                   "member": info.filename, "archive": kind}
            print(f"fetch {dest} ({dest.stat().st_size/1e6:.1f} MB, "
                  f"{f.bytes_fetched/1e6:.0f} MB transferred so far)")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
