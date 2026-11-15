"""Validate, index and publish Track E verdict clips (shared schema, handoff section 6).

Reads the npz + meta pairs that `remote/e_export.py` wrote on the Modal volume, checks every
file against the shared schema, writes an index, and (optionally) syncs the tree to R2 under
`track-e/clips/`.

Validation is strict on purpose: the schema is the contract with Track D's scorer, so a clip
that does not satisfy it must fail here rather than inside someone else's comparison.

    python scripts/export_verdict_clips.py --clips data/clips --r2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

CONDITIONS = {"real", "null_edit", "tilt_removal", "scale_0.5", "scale_1.5",
              "mirror_roll", "yaw_control"}
META_REQUIRED = ("dataset", "sequence", "clip_id", "condition", "fps", "mount_source",
                 "estimator", "estimator_variant")
ARRAYS = ("t", "pos", "yaw", "depicted_lean", "depicted_conf")


def check_clip(npz: Path, meta: Path) -> dict:
    """Return a record or raise; a malformed clip must not enter the index."""
    d = np.load(npz)
    m = json.loads(meta.read_text())
    missing = [k for k in ARRAYS if k not in d.files]
    if missing:
        raise ValueError(f"{npz.name}: missing arrays {missing}")
    t = np.asarray(d["t"], dtype=np.float64)
    pos = np.asarray(d["pos"], dtype=np.float64)
    yaw = np.asarray(d["yaw"], dtype=np.float64)
    lean = np.asarray(d["depicted_lean"], dtype=np.float64)
    conf = np.asarray(d["depicted_conf"], dtype=np.float64)
    n = len(t)
    if pos.shape != (n, 3):
        raise ValueError(f"{npz.name}: pos is {pos.shape}, expected {(n, 3)}")
    if yaw.shape != (n,):
        raise ValueError(f"{npz.name}: yaw is {yaw.shape}")
    if lean.shape != (n, 2):
        raise ValueError(f"{npz.name}: depicted_lean is {lean.shape}, expected {(n, 2)}")
    if conf.shape != (n,):
        raise ValueError(f"{npz.name}: depicted_conf is {conf.shape}")
    if n > 1 and not np.all(np.diff(t) > 0):
        raise ValueError(f"{npz.name}: t is not strictly increasing")
    if not np.isfinite(t).all() or not np.isfinite(pos).all() or not np.isfinite(yaw).all():
        raise ValueError(f"{npz.name}: non-finite t/pos/yaw")
    finite_lean = np.isfinite(lean).all(axis=1)
    bad = [k for k in META_REQUIRED if k not in m]
    if bad:
        raise ValueError(f"{meta.name}: missing meta fields {bad}")
    if m["condition"] not in CONDITIONS:
        raise ValueError(f"{meta.name}: condition {m['condition']!r} not in the schema")
    return {
        "clip_id": m["clip_id"], "sequence": m["sequence"], "condition": m["condition"],
        "n_frames": int(n), "duration_s": float(t[-1] - t[0]) if n > 1 else 0.0,
        "fps": float(m["fps"]), "estimator": m["estimator"],
        "estimator_variant": m["estimator_variant"], "mount_source": m["mount_source"],
        "depicted_finite_fraction": float(finite_lean.mean()),
        "depicted_conf_median": float(np.nanmedian(conf)) if np.isfinite(conf).any() else None,
        "depicted_lean_norm_median": float(np.nanmedian(np.linalg.norm(lean, axis=1)))
        if finite_lean.any() else None,
        "sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "path": str(npz), "meta_path": str(meta),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/clips", help="directory with <seq>/<clip>.npz")
    ap.add_argument("--out", default="results/e_clips")
    ap.add_argument("--r2", action="store_true", help="upload to R2 under track-e/clips/")
    ap.add_argument("--prefix", default="track-e/clips")
    args = ap.parse_args()
    root = Path(args.clips)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records, problems = [], []
    for npz in sorted(root.glob("*/*.npz")):
        meta = npz.with_suffix(".meta.json")
        try:
            if not meta.exists():
                raise ValueError(f"{npz.name}: no sidecar meta.json")
            records.append(check_clip(npz, meta))
        except Exception as exc:  # collected, then reported; never silently skipped
            problems.append(str(exc))

    by_cond: dict[str, int] = {}
    by_seq: dict[str, int] = {}
    for r in records:
        by_cond[r["condition"]] = by_cond.get(r["condition"], 0) + 1
        by_seq[r["sequence"]] = by_seq.get(r["sequence"], 0) + 1
    index = {"n_clips": len(records), "n_problems": len(problems), "problems": problems,
             "by_condition": by_cond, "by_sequence": by_seq, "clips": records,
             "schema": "handoff 09 section 6; arrays " + ", ".join(ARRAYS)}
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(json.dumps({k: index[k] for k in ("n_clips", "n_problems", "by_condition")}, indent=1))
    if problems:
        for p in problems[:10]:
            print("PROBLEM:", p, file=sys.stderr)

    if args.r2:
        from src.utils.r2 import load_env, upload  # boto3 + .env, project helper

        env = load_env()
        uploaded = []
        for npz in sorted(root.glob("*/*.npz")):
            for p in (npz, npz.with_suffix(".meta.json")):
                if not p.exists():
                    continue
                key = f"{args.prefix.rstrip('/')}/{p.relative_to(root).as_posix()}"
                r = upload(p, key, env)
                uploaded.append({"key": key, "bytes": r["bytes"],
                                 "remote_sha256": r["remote_sha256"]})
        idx_key = f"{args.prefix.rstrip('/')}/index.json"
        r = upload(out / "index.json", idx_key, env)
        uploaded.append({"key": idx_key, "bytes": r["bytes"],
                         "remote_sha256": r["remote_sha256"]})
        (out / "r2_upload.json").write_text(json.dumps(uploaded, indent=1))
        print(f"uploaded {len(uploaded)} objects under {args.prefix}")
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
