"""Mirror a directory tree to R2 under a prefix, verifying each object on the way back.

``scripts/sync_r2.py`` covers one file; phase 3 has thousands of small JSONL/JSON artifacts
and a few hundred MB of specs, so this walks a tree instead. Each upload is read back and
re-hashed by :func:`src.utils.r2.upload`, so a truncated object fails here.

    pixi run -e gpu python scripts/sync_r2_tree.py --src data/processed/p3a --prefix track-a/phase3a/specs
    pixi run -e gpu python scripts/sync_r2_tree.py --src results/phase3a_verdict --prefix track-a/phase3a/results --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.utils.r2 import list_objects, load_env, upload

SKIP_SUFFIX = (".npy", ".npz", ".pt", ".ckpt", ".mp4")  # large binaries need their own call


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-binary", action="store_true")
    ap.add_argument("--max-mb", type=float, default=50.0,
                    help="skip single files above this size unless --include-binary")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"no such path: {src}")
    env = load_env()
    files = [p for p in sorted(src.rglob("*")) if p.is_file()]
    manifest = []
    for p in files:
        rel = p.relative_to(src).as_posix()
        key = f"{args.prefix.rstrip('/')}/{rel}"
        mb = p.stat().st_size / 1e6
        if not args.include_binary and (p.suffix in SKIP_SUFFIX or mb > args.max_mb):
            manifest.append({"key": key, "bytes": p.stat().st_size, "sha256": sha256(p),
                             "skipped": True})
            continue
        if args.dry_run:
            manifest.append({"key": key, "bytes": p.stat().st_size, "sha256": sha256(p),
                             "skipped": False})
            continue
        r = upload(p, key, env)
        manifest.append({"key": key, "bytes": r["bytes"], "sha256": r["sha256"],
                         "remote_sha256": r["remote_sha256"], "skipped": False})
        print(f"{key} {r['bytes']} bytes ok", flush=True)
    if not args.dry_run:
        existing = {o["key"] for o in list_objects(args.prefix, env, limit=1000)}
        missing = [m["key"] for m in manifest if not m["skipped"] and m["key"] not in existing]
        print(f"uploaded {sum(1 for m in manifest if not m['skipped'])} files, "
              f"missing after upload: {len(missing)}")
        if missing:
            print("first missing:", missing[0])
    print(json.dumps(manifest[:5], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
