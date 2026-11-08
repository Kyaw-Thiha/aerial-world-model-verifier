"""Pull one path from a Modal Volume, used because ``modal volume get`` trips on
an existing local directory in this CLI version.

    pixi run python scripts/pull_volume.py awmv-a-data tii_frames data/raw/tii_frames
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal


def _drain(it) -> bytes:
    import asyncio

    if hasattr(it, "__aiter__"):
        async def _go():
            return b"".join([c async for c in it])
        return asyncio.run(_go())
    return b"".join(list(it))


def main(vol_name: str, remote: str, local: str) -> int:
    vol = modal.Volume.from_name(vol_name)
    dest = Path(local)
    n = 0
    stack = [remote.rstrip("/")]
    while stack:
        p = stack.pop()
        for entry in vol.listdir(p):
            if entry.type == modal.volume.FileEntryType.DIRECTORY:
                stack.append(entry.path)
                continue
            rel = entry.path[len(remote.rstrip("/")):].lstrip("/")
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)

            out.write_bytes(_drain(vol.read_file(entry.path)))
            n += 1
    print(f"wrote {n} files under {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
