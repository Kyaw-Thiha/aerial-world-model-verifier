"""Round-trip one small file through R2 and verify sha256 on both sides.

    pixi run sync-r2 --put path/to/file     # upload, key = relative path under r2/
    pixi run sync-r2 --get <key> --dest p   # download
    pixi run sync-r2 --check                # report which R2 names are set (no values)
    pixi run sync-r2 --ls [prefix]

The upload is read back from R2 and hashed again, so a silently truncated object
fails here rather than three machines later.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils.r2 import config_report, download, list_objects, upload


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--put", metavar="FILE")
    g.add_argument("--get", metavar="KEY")
    g.add_argument("--check", action="store_true")
    g.add_argument("--ls", nargs="?", const="", metavar="PREFIX")
    ap.add_argument("--key", help="remote key (default: file name)")
    ap.add_argument("--dest", help="destination path for --get")
    args = ap.parse_args()

    if args.check:
        print("R2 credential presence:", config_report())
        return 0

    if args.put:
        key = args.key or Path(args.put).name
        r = upload(args.put, key)
        print(f"uploaded {r['key']} {r['bytes']} bytes")
        print(f"  local  sha256 {r['sha256']}")
        print(f"  remote sha256 {r['remote_sha256']}")
        print(f"  match: {r['match']}")
        return 0 if r["match"] else 1

    if args.get:
        dest = args.dest or Path(args.get).name
        r = download(args.get, dest)
        print(f"downloaded {r['key']} -> {r['path']} ({r['bytes']} bytes)")
        print(f"  match: {r['match']}")
        return 0 if r["match"] else 1

    for o in list_objects(args.ls or ""):
        print(f"{o['bytes']:12d}  {o['key']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
