"""Pull named files (not whole trees) from a Modal volume — Track V helper.

    python scripts/v_pull_files.py <volume> <manifest.json>

The manifest is a list of {"remote": "path/on/volume", "local": "path/on/disk"}.
"""
from __future__ import annotations

import json
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


def main(vol_name: str, manifest: str) -> int:
    vol = modal.Volume.from_name(vol_name)
    items = json.loads(Path(manifest).read_text())
    for it in items:
        dest = Path(it["local"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_drain(vol.read_file(it["remote"])))
        print(f"{it['remote']} -> {dest} ({dest.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
