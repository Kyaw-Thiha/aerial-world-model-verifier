"""Track L, L02: publish the raw per-flight edit-reader outputs to R2, with a manifest.

Each ``<reader>__<flight>.json`` (per-frame leans for every condition) goes to
``track-l/l02/downstream/``; the durable summary (``results/l02/downstream.json``) and the fold
records stay in git. The upload is read back and hashed.

    PYTHONPATH=. python experiments/l02_publish.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.utils.r2 import upload

SRC = Path("results/l02/downstream")
OUT = Path("results/l02")
PREFIX = "track-l/l02/downstream"


def main() -> int:
    files = sorted(SRC.glob("*__flight-*.json"))
    if not files:
        raise SystemExit("no per-flight edit outputs")
    rows = []
    for p in files:
        key = f"{PREFIX}/{p.name}"
        r = upload(p, key)
        if not r["match"]:
            raise SystemExit(f"{key}: remote hash does not match")
        rows.append({"key": key, "bytes": p.stat().st_size, "sha256": r["sha256"]})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (OUT / "r2_manifest_l02.json").write_text(json.dumps({
        "prefix": PREFIX, "n": len(rows), "commit": commit,
        "note": "per-flight, per-frame edit-reader outputs behind results/l02/downstream.json",
        "files": rows}, indent=1))
    print(f"uploaded {len(rows)} objects, {sum(r['bytes'] for r in rows)/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
