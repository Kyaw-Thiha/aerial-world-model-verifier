"""Shared I/O, provenance and formatting helpers for report asset scripts.

Every asset script:
  * reads its numbers from a stored result file (never hand-typed),
  * writes report/data/<asset>.csv (the exact numbers) and
    report/data/<asset>.src.json (paths, commit, sha256, script, notes),
  * writes a LaTeX table fragment and/or a vector PDF figure.

Source worktrees are resolved from AWMV_SOURCES (default:
/home/kevin/Documents/Projects) and the fixed branch checkout names.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1]
DATA = REPORT / "data"
TABLES = REPORT / "tables"
FIGURES = REPORT / "figures"

SOURCES = Path(os.environ.get("AWMV_SOURCES", "/home/kevin/Documents/Projects"))
WORKTREES = {
    "a": SOURCES / "awmv-track-a",
    "a2": SOURCES / "awmv-track-a2",
    "b": SOURCES / "awmv-track-b",
    "c": SOURCES / "awmv-track-c",
    "d": SOURCES / "awmv-track-d",
    "e": SOURCES / "awmv-track-e",
}

R2_PREFIXES = {
    "a": "track-a/",
    "a2": "track-a/a2/",
    "b": "track-b/",
    "c": "track-c/",
    "d": "track-d/",
    "e": "track-e/",
}


def src(track: str, rel: str) -> Path:
    p = WORKTREES[track] / rel
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_json(track: str, rel: str):
    with open(src(track, rel), "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(track: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(WORKTREES[track]), "rev-parse", "HEAD"], text=True
    ).strip()


def provenance(asset: str, sources, notes: str = "", derived: bool = False):
    """Write <asset>.src.json next to the CSV in report/data/."""
    entries = []
    for s in sources:
        if isinstance(s, dict):
            entries.append(s)
            continue
        track, rel = s
        p = src(track, rel)
        entries.append(
            {
                "worktree": track,
                "branch": subprocess.check_output(
                    ["git", "-C", str(WORKTREES[track]), "rev-parse", "--abbrev-ref", "HEAD"],
                    text=True,
                ).strip(),
                "commit": git_commit(track),
                "path": rel,
                "abs_path": str(p),
                "sha256": sha256(p) if p.is_file() else None,
                "r2_prefix": R2_PREFIXES.get(track),
            }
        )
    payload = {
        "asset": asset,
        "script": f"report/scripts/{Path(inspect.stack()[1].filename).name}",
        "derived_by_report_script": derived,
        "notes": notes,
        "sources": entries,
    }
    out = DATA / f"{asset}.src.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return out


def write_csv(asset: str, rows, header):
    out = DATA / f"{asset}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return out


def tex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def write_tex(asset: str, body: str):
    """Write a bare tabular fragment.

    Long prose notes are dropped from the fragment (captions live in the report text);
    wide tables get a smaller font so the fragment is usable without hand-editing.
    """
    lines = []
    for line in body.strip().splitlines():
        if line.lstrip().startswith("\\multicolumn") and len(line) > 140:
            continue
        lines.append(line)
    cols = max((line.count("&") + 1) for line in lines if "&" in line) if any("&" in l for l in lines) else 1
    prefix = ""
    if cols >= 8:
        prefix = "\\scriptsize\n"
    elif cols >= 5:
        prefix = "\\footnotesize\n"
    name = asset if asset.startswith("tab_") else f"tab_{asset}"
    out = TABLES / f"{name}.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(prefix + "\n".join(lines) + "\n")
    return out


def fmt(x, nd=2):
    """Format a number to nd decimals; em dash for None."""
    if x is None:
        return "--"
    return f"{x:.{nd}f}"


def pct(x, nd=1):
    if x is None:
        return "--"
    return f"{100.0 * x:.{nd}f}"


def rate(k, n):
    return (k / n) if n else None
