"""Build the Track A2 results manifest and mirror artifacts to R2.

Only the artifacts that cannot be regenerated cheaply are mirrored; everything else is
either in git or reproducible from a committed script plus the Modal Volume. Each upload is
read back and hashed, so a truncated object fails here rather than later.

    PYTHONPATH=. pixi run -e gpu python experiments/a2_publish.py --dry-run
    PYTHONPATH=. pixi run -e gpu python experiments/a2_publish.py --upload
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (local path, R2 key, note)
ARTIFACTS = [
    ("results/gate_a2/floor/floor_224.jsonl", "track-a/a2/geocalib/floor_224.jsonl",
     "186 released ANWM level frames, 224 px, with uncertainty/confidence"),
    ("results/gate_a2/anwm/depicted.jsonl", "track-a/a2/geocalib/anwm_depicted.jsonl",
     "82 generated clips x 16 frames, 224 px, with uncertainty/confidence"),
    ("results/gate_a2/edits/edits.jsonl", "track-a/a2/geocalib/edits.jsonl",
     "edit-recovery control: 12 generated clips x 7 conditions"),
    ("results/gate_a2/tii", "track-a/a2/geocalib/tii",
     "8 TII flights x 6 conditions, with uncertainty/confidence"),
    ("results/a2_gate", "track-a/a2/gate",
     "gate validation tables + the frozen-gate fit"),
    ("results/a2_verdict_gated", "track-a/a2/verdict_gated",
     "gated verdict tables for k_d in {0, 0.15, 0.295}"),
    ("results/a2_anwm", "track-a/a2/anwm_required",
     "required tilt from the commanded paths (both interpolators, dt sweep)"),
    ("results/a2_anwm_score", "track-a/a2/anwm_scored",
     "ANWM scoring tables, controls and minimum consistent dt"),
    ("configs/gate_frozen.yaml", "track-a/a2/gate_frozen.yaml", "the frozen gate"),
    ("results/a2_parity", "track-a/a2/parity",
     "ANWM residuals with the verdict's own nuisance fits (uptilt, lag)"),
    ("results/a2_followups", "track-a/a2/followups",
     "Regime C cells with/without parity, k_d flips, gate selection test, censored dt_min"),
    ("results/a2_checks", "track-a/a2/checks", "consolidated follow-up checks"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def walk(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    if not p.exists():
        return []
    return sorted(q for q in p.rglob("*") if q.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-mb", type=float, default=25.0)
    ap.add_argument("--out", default="results/a2_manifest.md")
    args = ap.parse_args()

    rows, to_upload = [], []
    for local, key, note in ARTIFACTS:
        p = ROOT / local
        files = walk(p)
        if not files:
            rows.append((local, key, "-", 0, "-", note + " (MISSING)"))
            continue
        total = sum(f.stat().st_size for f in files)
        for f in files:
            rel = f.relative_to(p) if p.is_dir() else Path(f.name)
            rows.append((local, f"{key}/{rel}" if p.is_dir() else key,
                         f"{f.stat().st_size/1e6:.2f} MB", len(files), sha256(f), note))
            if f.stat().st_size <= args.max_mb * 1e6:
                to_upload.append((f, f"{key}/{rel}" if p.is_dir() else key))

    lines = ["# Track A2 artifact manifest", "",
             f"{len(rows)} objects, {sum(1 for r in rows if r[2] != '-')} present.", "",
             "| local | R2 key | size | sha256 (first 16) | note |", "| --- | --- | --- | --- | --- |"]
    for local, key, size, _n, digest, note in rows:
        lines.append(f"| `{local}` | `{key}` | {size} | `{digest[:16] if digest != '-' else '-'}` | {note} |")
    (ROOT / args.out).write_text("\n".join(lines) + "\n")
    print(f"manifest: {len(rows)} objects -> {args.out}")

    if args.upload:
        from src.utils.r2 import upload
        results = []
        for f, key in to_upload:
            r = upload(str(f), key)
            results.append({"key": r["key"], "bytes": r["bytes"], "match": r["match"]})
            print(f"  {key}: {r['bytes']} bytes match={r['match']}")
        (ROOT / "results/a2_manifest_upload.json").write_text(json.dumps(results, indent=1))
        bad = [r for r in results if not r["match"]]
        print(f"uploaded {len(results)}, mismatches {len(bad)}")
        return 1 if bad else 0
    print(f"{len(to_upload)} objects would be uploaded (use --upload)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
