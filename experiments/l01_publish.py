"""Track L, L01: publish the per-frame reader arrays to R2 under ``track-l/``, with a manifest.

Each stored ``<clip>.npz`` from the re-score (per-frame gravity, lean, truth, gate inputs) goes
to ``track-l/l01/per_frame/<clip>.npz``; the upload is read back and hashed again, and the
manifest records key, bytes and sha256 for every object. The arrays are the evidence behind
``results/l01/*.csv`` and are too large for git.

    PYTHONPATH=. python experiments/l01_publish.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from src.utils.r2 import upload

SCORES = Path("data/l01/scores")
OUT = Path("results/l01")
PREFIX = "track-l/l01/per_frame"


def main() -> int:
    files = sorted(SCORES.glob("i02_dev_*.npz"))
    if not files:
        raise SystemExit("no per-frame arrays: pull them from awmv-l-data /l01/scores first")
    rows = []
    for p in files:
        key = f"{PREFIX}/{p.name}"
        r = upload(p, key)
        if not r["match"]:
            raise SystemExit(f"{key}: remote hash does not match")
        rows.append({"key": key, "bytes": p.stat().st_size, "sha256": r["sha256"]})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {"prefix": PREFIX, "n": len(rows), "commit": commit,
                "note": "per-frame arrays behind results/l01; uploaded and re-hashed on R2",
                "files": rows}
    (OUT / "r2_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"uploaded {len(rows)} objects, {sum(r['bytes'] for r in rows)/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
