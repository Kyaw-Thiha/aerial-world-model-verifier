"""Locally fetch native ANWM test trajectories (full pickle + the 32 frames we need).

Why local: the same HTTP-range fetch that takes ~2 min/shard from this workstation did not
finish a single shard in ~25 min from a Modal container (see docs/track_b_report.md), so the
range reads run here and the result is pushed to the Volume with `modal volume put`.

    python experiments/anwm_fetch_native.py --want 5        # index until 5 found, then fetch
    python experiments/anwm_fetch_native.py --upload        # push data/raw/anwm_full to the Volume
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_anwm import RangeFile, index_shard, shard_url  # noqa: E402

IDX = ROOT / "data/raw/anwm/shard_index.json"


class ReadAhead(io.RawIOBase):
    """RangeFile with a large read-ahead window.

    `tarfile` asks for 512 bytes per header, i.e. one HTTP request per member, which is what made
    a shard walk take ~5.5 minutes (~7 members/s). Reading 8 MB blocks and serving headers from
    memory removes almost all of those requests while keeping the same semantics.
    """

    BLOCK = 8 << 20

    def __init__(self, inner):
        self.inner = inner
        self.pos = 0
        self.base = 0
        self.buf = b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def _fill(self, n: int) -> None:
        end = min(self.inner.size, self.base + self.BLOCK)
        if end > self.base:
            self.buf = self.inner.read_range(self.base, end - self.base)
        else:
            self.buf = b""

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.inner.size + off
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.inner.size - self.pos
        out = []
        while n > 0:
            if not (self.base <= self.pos < self.base + len(self.buf)):
                self.base = self.pos
                self._fill(n)
                if not self.buf:
                    break
            rel = self.pos - self.base
            chunk = self.buf[rel:rel + n]
            out.append(chunk)
            self.pos += len(chunk)
            n -= len(chunk)
            if len(chunk) == 0:
                break
        return b"".join(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)
OUT = ROOT / "data/raw/anwm_full"
CAND = (39, 79, 119, 159)
CTX = 16


def rollout_names() -> dict[str, int]:
    p = ROOT / "third_party/anwm/data/splits/airvln_16/test/rollout_16.pkl"
    entries = pickle.load(open(p, "rb"))
    first = {}
    for name, ct, _, _ in entries:
        first.setdefault(name, ct)
    return first


def _index_shard_fast(i: int) -> dict:
    """index_shard with the read-ahead wrapper."""
    import tarfile as _tf

    f = ReadAhead(RangeFile(shard_url(i)))
    out = {}
    with _tf.open(fileobj=f, mode="r:") as tf:
        for m in tf:
            if m.name.endswith("traj_data.pkl"):
                out[m.name.split("/")[0]] = [i, m.offset_data, m.size]
    return out


def index_until(wanted: int, max_shards: int = 100, workers: int = 16) -> dict:
    index = json.loads(IDX.read_text()) if IDX.exists() else {}
    first = rollout_names()
    todo = [i for i in range(342) if not any(v[0] == i for v in index.values())][:max_shards]
    done = 0
    ex = ThreadPoolExecutor(max_workers=workers)
    pending = {ex.submit(_index_shard_fast, i): i for i in todo}
    try:
        while pending:
            fin, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in fin:
                try:
                    index.update(fut.result())
                except Exception as e:
                    print("shard failed", type(e).__name__, e, flush=True)
                done += 1
            IDX.write_text(json.dumps(index))
            found = [n for n in first if n in index]
            if done % 5 == 0 or len(found) >= wanted:
                print(f"{done} shards, {len(found)}/{wanted} found", flush=True)
            if len(found) >= wanted:
                break
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    IDX.write_text(json.dumps(index))
    return index


def fetch(names: list[str], index: dict, first: dict[str, int]) -> None:
    for name in names:
        shard, off, size = index[name]
        ct = first[name]
        need = set(range(max(0, ct - CTX + 1), ct + 17))
        for c in CAND:
            need |= set(range(max(0, c - CTX + 1), c + 17))
        outd = OUT / name
        outd.mkdir(parents=True, exist_ok=True)
        f = ReadAhead(RangeFile(shard_url(shard)))
        nframes = 0
        with tarfile.open(fileobj=f, mode="r:") as tf:
            for m in tf:
                if m.name == f"{name}/traj_data.pkl":
                    src = tf.extractfile(m)
                    with open(outd / "traj_data.pkl", "wb") as fh:
                        shutil.copyfileobj(src, fh, 1 << 24)
                elif m.name.startswith(f"{name}/") and m.name.endswith(".jpg"):
                    frame = int(Path(m.name).stem)
                    if frame in need:
                        (outd / f"{frame}.jpg").write_bytes(tf.extractfile(m).read())
                        nframes += 1
        print(f"{name}: pickle {(outd/'traj_data.pkl').stat().st_size/1e6:.0f} MB, "
              f"{nframes} frames, fetched {f.inner.fetched/1e6:.0f} MB", flush=True)
        (OUT / "manifest.json").write_text(json.dumps(
            {n: {"shard": index[n][0], "curr_time": first[n], "start_candidates": list(CAND)}
             for n in names}, indent=1))


def upload() -> None:
    subprocess.run(["modal", "volume", "put", "awmv-b-native", str(OUT), "/airvln_16",
                    "--force"], check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--want", type=int, default=5)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--max-shards", type=int, default=100)
    args = ap.parse_args()
    if args.upload:
        upload()
        return 0
    first = rollout_names()
    index = index_until(args.want, args.max_shards)
    found = [n for n in first if n in index][: args.want]
    print("fetching", found)
    fetch(found, index, first)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
