"""Minimal HTTP-range zip reader, so a few members of a 1 GB sequence zip cost a few MB.

UZH archives each sequence as a single zip of ~1 GB whose interesting members are tiny
(``groundtruth.txt``, ``imu.txt``, ``left_images.txt``, tens to hundreds of kB). Downloading
the archive to read them is wasteful on a workstation with a small disk, and the same trick is
already used for the TII chunks in ``scripts/fetch_tii.py``.

Usage::

    z = HttpZip("https://.../indoor_forward_5_snapdragon_with_gt.zip")
    text = z.read("groundtruth.txt").decode()

Only what ``zipfile`` needs is implemented: ``read``, ``seek``, ``tell``, ``seekable``.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile

BLOCK = 1 << 20


class HttpFile(io.RawIOBase):
    """Read-only, seekable file over HTTP ``Range`` requests with a small block cache."""

    def __init__(self, url: str, length: int | None = None, timeout: float = 120.0):
        super().__init__()
        self.url = url
        self.timeout = timeout
        self._pos = 0
        self._blocks: dict[int, bytes] = {}
        if length is None:
            with urllib.request.urlopen(self._request(0, 0), timeout=self.timeout) as r:
                cr = r.headers.get("Content-Range", "")
                if "/" not in cr:
                    raise RuntimeError(f"{url}: server did not answer a Range request")
                length = int(cr.split("/")[-1])
        self._len = int(length)

    # ------------------------------------------------------------------ http
    def _request(self, start: int, end: int):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        return req

    def _block(self, idx: int) -> bytes:
        if idx not in self._blocks:
            start = idx * BLOCK
            end = min(start + BLOCK, self._len) - 1
            if end < start:
                return b""
            with urllib.request.urlopen(self._request(start, end), timeout=self.timeout) as r:
                self._blocks[idx] = r.read()
        return self._blocks[idx]

    # ------------------------------------------------------------------ io
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._len + offset
        else:
            raise ValueError(whence)
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self._len - self._pos
        n = max(0, min(n, self._len - self._pos))
        out = bytearray()
        while n > 0:
            idx, off = divmod(self._pos, BLOCK)
            blk = self._block(idx)
            take = min(n, BLOCK - off)
            out += blk[off:off + take]
            self._pos += take
            n -= take
        return bytes(out)

    def readat(self, n: int, offset: int) -> bytes:
        cur = self._pos
        self.seek(offset)
        data = self.read(n)
        self._pos = cur
        return data


class HttpZip:
    """Thin wrapper: ``read(name)`` for one member, ``names()`` for the listing."""

    def __init__(self, url: str, timeout: float = 120.0):
        self.url = url
        self._fh = HttpFile(url, timeout=timeout)
        self.zip = zipfile.ZipFile(self._fh)

    def names(self) -> list[str]:
        return self.zip.namelist()

    def read(self, name: str) -> bytes:
        return self.zip.read(name)

    def read_prefix(self, name: str, n: int) -> bytes:
        with self.zip.open(name) as fh:
            return fh.read(n)

    def close(self) -> None:
        try:
            self.zip.close()
        finally:
            self._fh.close()
