"""Shared I/O, provenance and formatting helpers for report asset scripts.

Every asset script:
  * reads its numbers from a stored result file (never hand-typed),
  * writes report/data/<asset>.csv (the exact numbers) and
    report/data/<asset>.src.json (paths, commit, sha256, script, notes),
  * writes a LaTeX table fragment and/or a vector PDF figure.

Source checkouts are resolved in this order, per track:

  1. ``$AWMV_SOURCES/awmv-track-<track>`` (default ``AWMV_SOURCES`` is
     /home/kevin/Documents/Projects) - the pre-consolidation per-branch worktrees;
  2. this repository's own root - the consolidated layout, where every track's
     ``results/`` and ``reports/`` live on one branch.

Both layouts place the files at the same relative paths, so the assets are built
from identical bytes either way. Provenance (``data/<asset>.src.json``) keeps the
per-branch identity recorded in ``report/cutoff.json`` under layout 2, and
re-reads it from git under layout 1.
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
REPO_ROOT = REPORT.parent


def _checkout(track: str) -> Path:
    """Per-track worktree if it exists, else the consolidated repository root."""
    d = SOURCES / f"awmv-track-{track}"
    return d if d.is_dir() else REPO_ROOT


WORKTREES = {t: _checkout(t) for t in ("a", "a2", "b", "c", "d", "e")}

# Branch names and the per-branch HEADs recorded when the assets were last built
# from the per-branch worktrees (report/cutoff.json). Used only under the
# consolidated layout, where re-reading them from git would report the single
# consolidation branch instead of the six source branches.
TRACK_BRANCH = {
    "a": "track-a-perception", "a2": "track-a2-gate", "b": "track-b-generation",
    "c": "track-c-pathrec", "d": "track-d-verdict-stat", "e": "track-e-depicted-uzh",
}

# Pre-registration commit of each frozen config, as recorded in
# report/tables/tab_frozen-configs_1.tex (the commit that last touched the file on
# its own branch). Used only under the consolidated layout.
FROZEN_CONFIG_COMMITS = {
    "configs/phase1.yaml": "e4fd5d9",
    "configs/verdict_preregistration.yaml": "0834204",
    "configs/gate_frozen.yaml": "9e1656d",
    "configs/pathrec_prereg.yaml": "6a16297",
    "configs/gain_frozen.yaml": "7e8c42e",
    "configs/coherence_frozen.yaml": "7e8c42e",
    "configs/window_bandwidth_prereg.yaml": "c7af793",
}

_RECORDED = None


def recorded_heads() -> dict:
    """Per-branch head/date snapshot from report/cutoff.json (may be empty)."""
    global _RECORDED
    if _RECORDED is None:
        try:
            with open(REPORT / "cutoff.json", encoding="utf-8") as fh:
                _RECORDED = json.load(fh).get("branches", {})
        except (OSError, ValueError):
            _RECORDED = {}
    return _RECORDED


def on_consolidated(track: str) -> bool:
    """True when this track is read from the consolidated repository root."""
    return WORKTREES[track] == REPO_ROOT


def branch_name(track: str) -> str:
    if on_consolidated(track):
        return TRACK_BRANCH[track]
    return subprocess.check_output(
        ["git", "-C", str(WORKTREES[track]), "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()


def commit_of(track: str) -> str:
    if on_consolidated(track):
        rec = recorded_heads().get(TRACK_BRANCH[track], {})
        if rec.get("head"):
            return rec["head"]
    return subprocess.check_output(
        ["git", "-C", str(WORKTREES[track]), "rev-parse", "HEAD"], text=True
    ).strip()


def head_short(track: str) -> str:
    return commit_of(track)[:7]


def head_date(track: str) -> str:
    if on_consolidated(track):
        rec = recorded_heads().get(TRACK_BRANCH[track], {})
        if rec.get("head_date"):
            return rec["head_date"][:10]
    return subprocess.check_output(
        ["git", "-C", str(WORKTREES[track]), "log", "-1", "--format=%ad", "--date=short"],
        text=True,
    ).strip()

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
    return commit_of(track)


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
                "branch": branch_name(track),
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
