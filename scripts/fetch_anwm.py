"""Fetch trajectory metadata (no images, no depth) from the ANWM-Dataset shards.

Source: https://huggingface.co/datasets/EmbodiedCity/ANWM-Dataset (apache-2.0),
342 tar shards `airvln_16-NNNNNN.tar`, 637 GB in total. Each trajectory is a folder
`<ID>_processed/` of numbered JPEGs plus `traj_data.pkl`. The pickle also carries a
per-frame float depth array and is therefore ~0.5 GB, but the small pose keys
(K, position, yaw, timestamps, images, pitch, roll, point) are written first.

So this script never downloads a shard. It mounts a shard over HTTP Range with one
persistent connection, lets `tarfile` walk the headers, and streams each wanted
`traj_data.pkl` through a restricted unpickler that (a) allows only numpy array
reconstruction and (b) aborts the moment a large key (`depth` / `pose`) is read,
keeping the entries already parsed. Cost is ~1 MB per trajectory instead of the
271-649 MB pickle. Nothing else from the file is executed.

Usage:
    pixi run python scripts/fetch_anwm.py --probe 0 170 341        # list members of shards
    pixi run python scripts/fetch_anwm.py --index --shards 0,9,18  # index (one shard ~2 min)
    pixi run python scripts/fetch_anwm.py --fetch-test [--limit N]  # pose keys of test trajectories
"""
from __future__ import annotations

import argparse
import http.client
import io
import json
import pickle
import tarfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

REPO = "https://huggingface.co/datasets/EmbodiedCity/ANWM-Dataset/resolve/main"
N_SHARDS = 342
OUT = Path("data/raw/anwm")
BIG_KEYS = ("depth", "pose")  # large per-frame arrays; everything after them is skipped
BLOCK = 64 * 1024
SPLIT_DIR = Path("third_party/anwm/data/splits/airvln_16")


def shard_url(i: int) -> str:
    return f"{REPO}/airvln_16-{i:06d}.tar"


class RangeFile(io.RawIOBase):
    """Seekable read-only view of one remote file via HTTP Range.

    Uses one persistent TLS connection, because tar walking issues a request per
    member header; a fresh handshake per header costs orders of magnitude more
    than the bytes themselves.
    """

    def __init__(self, url: str):
        self.url, self.pos, self.cache, self.fetched = url, 0, {}, 0
        # HF redirects to a CDN; resolve once with urllib (which follows redirects),
        # then speak to the final host over one persistent connection.
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
            final = r.geturl()
        u = urlparse(final)
        self._conn = http.client.HTTPSConnection(u.hostname, timeout=120)
        self._path = u.path + (f"?{u.query}" if u.query else "")
        self._hdr = {"Range": "bytes=0-0"}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _get(self, lo: int, hi: int) -> bytes:
        self._conn.request("GET", self._path, headers={"Range": f"bytes={lo}-{hi - 1}"})
        self._reconnects = getattr(self, "_reconnects", 0)
        r = self._conn.getresponse()
        data = r.read()
        if r.status not in (200, 206):
            raise IOError(f"range read failed: {r.status} {data[:200]!r}")
        self.fetched += len(data)
        return data

    def read(self, n=-1):
        if 0 < n <= 4096:  # tar header: fetch exactly, no block cache
            data = self._get(self.pos, self.pos + n)
            self.pos += len(data)
            return data
        end = self.size if n < 0 else min(self.pos + n, self.size)
        out = []
        while self.pos < end:
            b = self.pos // BLOCK
            if b not in self.cache:
                if len(self.cache) > 16:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[b] = self._get(b * BLOCK, min((b + 1) * BLOCK, self.size))
            blk = self.cache[b]
            lo = self.pos - b * BLOCK
            take = blk[lo : lo + (end - self.pos)]
            out.append(take)
            self.pos += len(take)
        return b"".join(out)

    def stream(self, lo: int, size: int):
        """Buffered raw stream over [lo, lo+size) that fetches 1 MB at a time."""
        return _SubStream(self, lo, size)

    def read_range(self, lo: int, n: int) -> bytes:
        """Direct ranged read that bypasses the block cache (for the large pickle heads)."""
        return self._get(lo, min(lo + n, self.size))


class _SubStream(io.RawIOBase):
    """Lazy view of a slice of a RangeFile, so an unpickler can stop early."""

    CHUNK = 1 << 20

    def __init__(self, parent, lo: int, size: int):
        self.parent, self.lo, self.size, self.pos = parent, lo, size, 0
        self._buf = b""
        self._buf_lo = 0

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def seekable(self):
        return True

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        lo, hi = self.pos, self.pos + n
        if not (self._buf_lo <= lo and hi <= self._buf_lo + len(self._buf)):
            start = lo
            self._buf = self.parent.read_range(self.lo + start, min(self.CHUNK, self.size - start))
            self._buf_lo = start
        out = self._buf[lo - self._buf_lo: hi - self._buf_lo]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


class SafeUnpickler(pickle._Unpickler):  # pure-Python unpickler: exposes .stack
    """Only allow numpy array reconstruction; nothing else can be imported."""

    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
        ("numpy", "ndarray"), ("numpy", "dtype"),
    }

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


class _StopAtBigKey(SafeUnpickler):
    """Unpickle only until the first large key, then keep the pairs gathered so far.

    The pickle is one dict: EMPTY_DICT, then key/value pairs, then SETITEMS. Keys
    still sit on the stack when their value has been read, so intercepting the key
    itself yields every completed pair without ever reading the depth payload. The
    stream is remote and lazy, so this costs a few hundred KB instead of ~500 MB.
    """

    def __init__(self, stream):
        super().__init__(stream)
        # the base class fills `dispatch` with its own bound methods, so rebind the
        # three unicode opcodes to this subclass or the override would never run
        for op, name in ((0x8C, "load_short_binunicode"), (0x58, "load_binunicode"),
                         (0x56, "load_unicode")):
            if op in self.dispatch:
                self.dispatch[op] = getattr(type(self), name)  # dispatch calls fn(self)

    # The pure-Python unpickler's load_* methods push onto self.stack and return
    # None, so the check has to read the stack rather than a return value.
    def _check(self) -> None:
        if self.stack and self.stack[-1] in BIG_KEYS:
            raise _StopBigKey()

    def load_binunicode(self):
        super().load_binunicode()
        self._check()

    def load_short_binunicode(self):
        super().load_short_binunicode()
        self._check()

    def load_unicode(self):
        super().load_unicode()
        self._check()


class _StopBigKey(Exception):
    pass


def _parse_pose_head(stream) -> dict:
    """Load the leading dict entries of a traj_data.pkl without its depth payload.

    CPython 3.12's unpickler keeps the dict being built on `metastack` and the
    pending key/value pairs on `stack`, so aborting at a large key leaves both
    readable.
    """
    u = _StopAtBigKey(stream)
    result = None
    try:
        result = u.load()
    except _StopBigKey:
        pass
    if isinstance(result, dict):
        return result
    pairs = list(u.stack)
    if pairs and pairs[-1] in BIG_KEYS:
        pairs.pop()
    if len(pairs) % 2:
        pairs.pop()
    out = {}
    if u.metastack and u.metastack[0] and isinstance(u.metastack[0][0], dict):
        out = dict(u.metastack[0][0])
    for i in range(0, len(pairs) - 1, 2):
        out[pairs[i]] = pairs[i + 1]
    if not out:
        raise ValueError("no dict entries recovered from traj_data.pkl")
    return out


def list_trajectories(f: RangeFile) -> dict:
    """{folder: (data_offset, size)} for every traj_data.pkl in the shard."""
    out = {}
    with tarfile.open(fileobj=f, mode="r:") as tf:
        for m in tf:
            if m.name.endswith("traj_data.pkl"):
                out[m.name.split("/")[0]] = (m.offset_data, m.size)
    return out


def read_traj(f: "RangeFile", offset: int, size: int) -> dict:
    """Read only the pose keys of one traj_data.pkl, streaming from the remote shard."""
    import io as _io

    stream = f.stream(offset, size)
    return _parse_pose_head(_io.BufferedReader(stream))


def index_shard(i: int) -> dict:
    f = RangeFile(shard_url(i))
    out = {}
    with tarfile.open(fileobj=f, mode="r:") as tf:
        for m in tf:
            if m.name.endswith("traj_data.pkl"):
                out[m.name.split("/")[0]] = [i, m.offset_data, m.size]
    return out


def run_index(shards: list[int], workers: int, index: dict, idx_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    lock = threading.Lock()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(index_shard, i): i for i in shards}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                got = fut.result()
            except Exception as e:  # one bad shard must not kill the scan
                print(f"shard {i}: FAILED {type(e).__name__}: {e}")
                continue
            with lock:
                index.update(got)
                done += 1
                if done % 10 == 0 or done == len(shards):
                    idx_path.write_text(json.dumps(index, indent=0))
                    print(f"{done}/{len(shards)} shards, {len(index)} trajectories indexed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", nargs="*", type=int, help="list trajectories in these shards")
    ap.add_argument("--index", action="store_true", help="index shards (threaded)")
    ap.add_argument("--shards", default=None,
                    help="comma-separated shard ids (default: all); one shard costs ~120 s of "
                         "range requests, so sample when a full index is not needed")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--fetch-test", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-shards", type=int, default=N_SHARDS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    idx_path = OUT / "shard_index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {}  # folder -> [shard, off, size]

    test_names = (SPLIT_DIR / "test/traj_names.txt").read_text().split()

    if args.probe is not None:
        for i in args.probe:
            f = RangeFile(shard_url(i))
            trajs = list_trajectories(f)
            for k, (off, sz) in trajs.items():
                index[k] = [i, off, sz]
            n_test = sum(k in set(test_names) for k in trajs)
            print(f"shard {i}: {f.size/1e9:.2f} GB, {len(trajs)} trajectories "
                  f"({n_test} in test), transferred {f.fetched/1e6:.0f} MB")
            for k in list(trajs)[:6]:
                print("   ", k, f"pkl {trajs[k][1]/1e6:.0f} MB", "TEST" if k in set(test_names) else "")
        idx_path.write_text(json.dumps(index, indent=1))
        return 0

    if args.index:
        wanted = ([int(x) for x in args.shards.split(",")] if args.shards
                  else list(range(N_SHARDS)))
        missing = [i for i in wanted if not any(v[0] == i for v in index.values())]
        print(f"indexing {len(missing)} shards with {args.workers} workers")
        run_index(missing, args.workers, index, idx_path)
        print(f"index: {len(index)} trajectories, "
              f"{len(set(test_names) & set(index))}/{len(test_names)} test found")
        return 0

    if args.fetch_test:
        names = [n for n in test_names if n in index][: args.limit]
        for n in names:
            dest = OUT / f"{n}.npz"
            if dest.exists():
                print("skip ", dest)
                continue
            i, off, sz = index[n]
            f = RangeFile(shard_url(i))
            d = read_traj(f, off, sz)
            np.savez_compressed(dest, **{k: np.asarray(v) for k, v in d.items()})
            print(f"fetch {dest} keys={sorted(d)} T={len(d.get('yaw', []))} "
                  f"({f.fetched/1e6:.1f} MB transferred)")
        return 0
    ap.error("pick --probe, --index or --fetch-test")


if __name__ == "__main__":
    raise SystemExit(main())
