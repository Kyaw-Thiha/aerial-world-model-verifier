"""Write report/cutoff.json: per-branch commit and date read, R2 prefix and object counts."""
from __future__ import annotations

import json
import subprocess

import common as C

BRANCHES = {
    "track-a-perception": "a", "track-a2-gate": "a2", "track-b-generation": "b",
    "track-c-pathrec": "c", "track-d-verdict-stat": "d", "track-e-depicted-uzh": "e",
}


def git(track, *args):
    return subprocess.check_output(["git", "-C", str(C.WORKTREES[track]), *args], text=True).strip()


def build():
    payload = {
        "generated_utc": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
        "note": ("Commit and date actually read from each worktree when the assets were built. "
                 "Track D's worktree advanced from 59b3aaf (handoff) to the commit below during "
                 "this build; the handoff value is recorded beside it. R2 counts were measured "
                 "on 2026-11-15 from the updated credentials."),
        "r2_object_counts": {"status": "see report/data/r2_counts.json"},
        "branches": {},
    }
    for branch, track in BRANCHES.items():
        payload["branches"][branch] = {
            "head": git(track, "rev-parse", "HEAD"),
            "head_short": git(track, "rev-parse", "--short", "HEAD"),
            "head_date": git(track, "log", "-1", "--format=%ad", "--date=iso"),
            "worktree": str(C.WORKTREES[track]),
            "r2_prefix": C.R2_PREFIXES[track],
            "r2_object_count": None,
            "handoff_head_note": ("handoff ~59b3aaf; advanced during build" if track == "d" else
                                  ("handoff ~5961c09; actual dd209bf" if track == "e" else "")),
        }
    counts_file = C.DATA / "r2_counts.json"
    if counts_file.exists():
        with open(counts_file, encoding="utf-8") as fh:
            measured = json.load(fh)
        payload["r2_object_counts"] = {
            "status": "measured",
            "measured_utc": measured["measured_utc"],
            "method": measured["method"],
            "note": measured["note"],
            "prefixes": measured["prefixes"],
        }
        for branch, info in payload["branches"].items():
            pref = info["r2_prefix"]
            if pref in measured["prefixes"]:
                info["r2_object_count"] = measured["prefixes"][pref]["count"]
                info["r2_bytes"] = measured["prefixes"][pref]["bytes"]
    with open(C.REPORT / "cutoff.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(build(), indent=2)[:600])
