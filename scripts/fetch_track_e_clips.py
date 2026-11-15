"""Bulk-download Track E's UZH clips from R2 (read-only; data/ is gitignored).

    PYTHONPATH=. pixi run python scripts/fetch_track_e_clips.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from src.utils.r2 import client, load_env, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/processed/track_e_uzh"


def main() -> int:
    env = load_env(ROOT / ".env")
    c = client(env)
    paginator = c.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=env["R2_BUCKET"], Prefix="track-e/clips/"):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith(".npz") or k.endswith(".meta.json") or k.endswith("index.json"):
                keys.append(k)
    keys.sort()
    DEST.mkdir(parents=True, exist_ok=True)
    n = 0
    for key in keys:
        rel = key[len("track-e/clips/"):]
        out = DEST / "clips" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        c.download_file(env["R2_BUCKET"], key, str(out))
        n += 1
        if n % 100 == 0:
            print(f"  {n}/{len(keys)}", flush=True)
    print(f"downloaded {n} objects under {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
