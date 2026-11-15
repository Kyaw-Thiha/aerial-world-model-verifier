# `awmv-report-assets-2026-11-15.zip` — contents preserved in the repository

The report-assets agent delivered a zip on 2026-11-15 (795 KB, sha256 `10134d9e2c10ad4361a4a7b91e28076480c869d3d83c069c4a12592c39cd6d19`,
`MANIFEST.txt` inside it records git HEAD `02f4e72` on branch `report-assets`).

This directory holds the part of that zip that is **not** already a file on `main`, so that the
zip itself can be deleted without losing anything. Verified member by member (203 members):

| members | status on `main` |
| --- | --- |
| 127 | byte-identical to the file at the same path (`report/…`) |
| 3 | `report/scripts/{common,cutoff,methodology}.py` — the zip has the originals; those exact bytes are still reachable at commit `59ca0b6` (verified by sha256: `1598c6fa…`, `540c7719…`, `d647c5fd…`). `main` now carries a later, layout-independent revision; the diff is only the source-resolution helpers |
| 72 + 1 | `source_reports/` (below) and `MANIFEST.txt` — **copied here**, byte-identical to the zip |

## `source_reports/`

The markdown the report-assets agent read, grouped by branch, as they were on 2026-11-15T10:10Z:
each branch's numbered reports, plus that branch's `decisions.md` and (for D) `prototype_v0.1.md`.

Two things make this snapshot worth keeping beyond provenance:

* the per-branch `decisions.md` copies are the **pre-merge** versions, which is what
  `report/data/corrections.csv` cites by line number (`docs/decisions.md L219`, `L257`, `L289`, …).
  The consolidated `docs/decisions.md` reorders sections, so those numbers address the merged file
  differently; the branch copies here (and `git show archive/track-d-pre-consolidation:docs/decisions.md`)
  still resolve them exactly;
* they show which report text was current when each asset was built, which is what the `.src.json`
  provenance files assert.

Nothing here is edited. The live, current versions of every report are in `reports/` at the
repository root, and `docs/reports/INDEX.md` is the one-line index of them.

## If the zip is ever needed again

It can be rebuilt exactly: `cd report && python3 scripts/make_all.py && python3 scripts/check_numbers.py`
regenerates `report/`, and the `source_reports/` half is this directory.
