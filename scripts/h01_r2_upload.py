"""Upload a Track H clip tree to R2 under its own prefix and write a sha256 manifest.

    set -a; . ./.env; set +a
    pixi run python scripts/h01_r2_upload.py --src data/processed/h01_seed_clips \\
        --prefix track-h/clips/anwm-seeds --tag anwm-seeds
    pixi run python scripts/h01_r2_upload.py --src data/h01_lingbot_out --prefix track-h/clips/lingbot --tag lingbot

The manifest is committed under ``results/h01_r2/``; the objects are not. Each object is
read back by :func:`src.utils.r2.upload` and re-hashed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = (".npy",)


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
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-mb", type=float, default=200.0)
    ap.add_argument("--patterns", default="", help="comma-separated globs; empty = all files")
    args = ap.parse_args()

    from src.utils.r2 import load_env, upload

    load_env()
    src = ROOT / args.src
    files = sorted(p for p in src.rglob("*") if p.is_file())
    if args.patterns:
        pats = [q.strip() for q in args.patterns.split(",") if q.strip()]
        files = [p for p in files if any(p.match(q) for q in pats)]
    rows, skipped = [], []
    for p in files:
        rel = p.relative_to(src).as_posix()
        if p.suffix in SKIP:
            skipped.append({"path": rel, "reason": "npy (reproducible from the manifest)"})
            continue
        if p.stat().st_size > args.max_mb * 1e6:
            skipped.append({"path": rel, "reason": f">{args.max_mb} MB"})
            continue
        key = f"{args.prefix.rstrip('/')}/{rel}"
        row = {"path": rel, "key": key, "bytes": p.stat().st_size, "sha256": sha256(p)}
        if not args.dry_run:
            r = upload(str(p), key)
            row["remote_sha256"] = r["remote_sha256"]
            row["match"] = bool(r["match"])
            if not r["match"]:
                raise SystemExit(f"R2 hash mismatch for {key}")
        rows.append(row)
        print(("would upload " if args.dry_run else "uploaded ") + key,
              f"{row['bytes']} B", flush=True)
    out = ROOT / "results/h01_r2" / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"prefix": args.prefix, "src": args.src, "tag": args.tag,
                               "dry_run": bool(args.dry_run), "n_files": len(rows),
                               "bytes": sum(r["bytes"] for r in rows),
                               "files": rows, "skipped": skipped}, indent=1))
    print(f"{len(rows)} files, {sum(r['bytes'] for r in rows)/1e6:.1f} MB -> {args.prefix}; "
          f"manifest {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
