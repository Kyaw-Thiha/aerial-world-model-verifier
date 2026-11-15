# Consolidation report — 2026-11-15

Seven track branches were merged into `main` of `aerial-world-model-verifier`, with history
preserved and nothing re-analysed. This file records what was merged, every conflict, what was
left alone, the verification results, and how to roll back.

Headline: `main` went from `492033f` (25 tests) to a repository carrying 298 additional commits
from every track — 221 tests passing, 3 skipped, the report assets rebuilding to the same numbers,
7 of 7 frozen configs unchanged, and all 64 in-repo commit hashes cited anywhere in `reports/`,
`report/` or `docs/` still reachable.

---

## 1. What was merged, and in what order

All eight folders were **git worktrees of one repository** (one `.git`, one remote,
`git@github.com:Kyaw-Thiha/aerial-world-model-verifier.git`), so no clone-to-clone reconciliation
was needed. The DAG is nearly linear — `main → a → {a2 → d → report-assets, c → e}`, with `b`
branching off `main` on its own — which dictated the order:

| # | branch | pre-consolidation tip | commits | merge commit | conflicts |
| --- | --- | --- | --- | --- | --- |
| — | `main` | `492033f` | — | — | — |
| 1 | `track-a-perception` | `164c5e0` | 60 | `44b9c7e` | none |
| 2 | `track-a2-gate` | `428dd3f` | 92 | `c50749d` | none |
| 3 | `track-b-generation` | `9ca9e90` | 95 | `a29d2bf` | `docs/decisions.md` |
| 4 | `track-c-pathrec` | `ed76a08` | 91 | `fc062d9` | `docs/decisions.md` |
| 5 | `track-d-verdict-stat` | `2a91765` | 111 | `a4f7889` | `docs/decisions.md` |
| 6 | `track-e-depicted-uzh` | `dd209bf` | 140 | `e852020` | `docs/decisions.md` |
| 7 | `report-assets` | `02f4e72` | 114 | `59ca0b6` | none |
| 8 | (consolidation docs/scripts) | — | 1 | `1072052` | — |

Every merge was `git merge --no-ff` with a message naming the track and stating what was resolved.
No squash, no rebase, no force push, no history rewrite. Per-branch tests were run **before**
merging and the full suite **after** each merge:

| after merging | suite |
| --- | --- |
| `main` (before) | 25 passed, 6 skipped |
| a | 59 passed |
| a2 | 101 passed |
| b | 137 passed, 2 skipped |
| c | 166 passed, 2 skipped |
| d | 207 passed, 2 skipped |
| e | 221 passed, 3 skipped |
| report-assets | 221 passed, 3 skipped |

The count rises monotonically; the union of collected test IDs was checked against each branch's
own collection (`--collect-only`, parametrisation collapsed) and no test node present on a branch
is missing from `main`.

`pixi.toml` and `pixi.lock` were **byte-identical on all eight branches** (`63abe36`, `bc98800`),
so there was no dependency union to perform and the lock was **not regenerated**. `.gitignore`
differed by one line (`logs/` on a2/d/report-assets) and merged to the union.

## 2. Conflicts, and how each was resolved

Only one file ever conflicted: **`docs/decisions.md`**, on merges 3-6. Nothing else — not
`configs/`, `results/`, `reports/`, `src/`, `tests/`, `experiments/`, `README` — needed a manual
resolution.

Resolution rule (agreed with the user at checkpoint 1): *keep every entry from every branch, in
date order, labelled by track; nothing dropped, nothing rewritten.* Mechanically:

1. split the file into `##` sections;
2. 3-way merge each section by heading (`git merge-file`), so edits to a *shared* section survive
   from both sides;
3. order the track sections by the date of the commit that first introduced their heading —
   B (2026-11-09 02:37) → A phase 3a (11-09 03:01) → A2 (11-09 15:03) → C (11-10 02:28) →
   D (11-13 14:02) → E (11-13 14:25);
4. verify by set comparison that every bullet line of both sides appears in the result.

The verification found exactly two bullet lines that do **not** survive verbatim, in each of the
merges where their pre-correction wording was carried by the later branch:

* Track A phase 3a, Gate 2 "VO path 41 m against mocap 479 m" — `main` already carried the
  2026-11-10 replacement ("units were mixed; the replayed path is 960 camera units with ATE
  5.97 m; the failure is direction, not magnitude"), added by A2. The later correction is kept.
* The same section's Task 4 UZH-calibration bullet — `main` carried A2's correction ("the release
  does ship a Kalibr camchain; Track C found it — the blocker was a wrong-path probe"), which
  supersedes C's and E's inherited older wording.

Both are corrections *of the very claim* the older line states, dated later, already on `main`
before the merge; keeping the older text would have re-introduced a withdrawn claim. No other
information is lost, and the merge commit messages say so per merge.

Final file: 483 lines, 7 track sections plus 6 shared sections, conflict-marker free.

## 3. Not merged, and why

* **`report_tex.zip` (the LaTeX report sources).** Not present on disk when the consolidation ran,
  despite the brief expecting it, and still absent after the user approved deleting it once
  integrated. It is the only Phase 3 item that could not be done: `main.tex` and `sections/` are
  absent, so the *full* report cannot be compiled here. `report/assets_test.tex` (the assets-only
  document) is in the repository and compiles. The integration is the one open step; the zip's
  deletion and that of `awmv-report-assets-2026-11-15.zip` both wait on it. See §4, item 5.
* **Nothing from the report-assets zip was left behind**: its 203 members were audited one by one
  (127 already byte-identical on `main`, 3 pre-consolidation scripts still reachable at `59ca0b6`,
  73 committed to `docs/archive/report-assets-2026-11-15/`).
* **Handoff briefs 02-10.** Never committed by the tracks and not found anywhere on the machine;
  `docs/handoffs/README.md` records their absence. Handoff 01 exists (as the first agent's
  condensed record) and handoff 11 (this brief) is now committed.
* **Branches, worktrees, bundles, zips, R2 objects, Modal volumes.** Nothing was deleted or moved.
  Removal is Phase 5 and waits for explicit approval (checkpoint 2).
* **`temp.md`, `temp.py`, `temp_proj`** in `~/Documents/Projects` — out of scope, untouched. The
  user confirmed at checkpoint 1 to leave them untouched.
* **Science.** No discrepancy in `report/CHECKS.md`'s "Known open discrepancies" was resolved, no
  report text was edited, no frozen config was touched, and no experiment was run.

One consequence of the union merge worth stating: `report/data/corrections.csv` cites
`docs/decisions.md` by line number (`L67`, `L123`, `L219`, … `L289`). Those line numbers were
correct for the pre-consolidation D branch and no longer address the same lines in the merged,
date-ordered file. They **do** still address exactly the intended entries at
`git show archive/track-d-pre-consolidation:docs/decisions.md`. The numbers themselves are a
report asset and were not re-pointed (that would be an edit to the report, not a tidy-up); the
working reference is the archive tag.

## 4. Verification results

| # | check | result |
| --- | --- | --- |
| 1 | full test suite on `main` | **221 passed, 3 skipped** (46 s) |
| 2 | environment installs from `pixi.lock` on a clean clone | **pass** — `git clone` to `/tmp/awmv-clean`, `pixi install --locked` → "The default environment has been installed"; `pixi run pytest -q` → 198 passed, 9 skipped, 17 failed, all 17 in `tests/test_generation_paths.py` because `third_party/anwm` is a gitignored clone; with that clone linked, **215 passed, 9 skipped, 0 failed** |
| 3 | `report/scripts/make_all.py` + `check_numbers.py` | **pass** — 16/16 asset groups built; `CHECKS.md` baseline reproduced exactly: **103 values checked, 100 match, 2 mismatch, 1 not checkable** |
| 4 | frozen-config sha256 vs `report/tables/tab_frozen-configs_1.tex` | **pass, 7 of 7** (`c5d3b55e…`, `c8ffac93…`, `dc0cc4ff…`, `845f4c69…`, `28f91f7f…`, `40a40bea…`, `75b208da…`) |
| 5 | LaTeX compiles, no undefined references | **partial** — `report/assets_test.tex` → `assets_test.pdf` with `latexmk`, 0 undefined references; the full report **cannot** be built: `report_tex.zip` never arrived |
| 6 | every cited commit reachable from `main` | **pass** — 64 in-repo commit hashes found in `reports/`, `report/` and `docs/`; `git merge-base --is-ancestor` holds for **all 64**. The remaining hex tokens are 4 external-repo commits (TII `df4a2e11…`, GeoCalib `1dedb29f…`, DPVO `0ac95b6…`, ANWM `657a802`) and 22 sha256 prefixes of frozen configs |
| 7 | tag the result | `consolidated-2026-11-15` |

Asset determinism, measured: after a rebuild, all 48 `.tex` fragments, the `.csv`/`.json` data and
`CHECKS.md` came back **byte-identical**; the 11 vector PDFs differ only in the `/CreationDate`
metadata matplotlib writes (identical byte size, identical content once the date is normalised).
Provenance (`report/data/*.src.json`) is byte-identical when the pre-consolidation worktrees are
present; without them the scripts fall back to the consolidated repo and only the `abs_path` field
changes (verified by running both ways).

Additional checks performed:

* no `.env` (or any key material) has ever been committed on any branch; `.gitignore` covers
  `.env`, `data/`, `third_party/`, `.pixi/`, caches, and the large binary suffixes under `results/`;
  no tracked file matches a credentials pattern;
* every branch's tree is a subset of `main`'s at the file level (0 files missing per branch);
* Modal: `modal app list --limit 0 --json` → `[]` (nothing running), 9 `awmv-*` volumes plus one
  unrelated `rl-outputs` volume present, untouched;
* R2: `rclone size` against the bucket returned **403 AccessDenied**; the recorded counts in
  `report/data/r2_counts.json` (2026-11-15T10:03Z) were **not** re-measured and nothing was
  written or deleted.

## 5. Rollback

Three independent layers, none of which depends on the others.

**a. Tags in the repository** (also in any future clone):

```
git checkout archive/main-pre-consolidation                   # 492033f
git checkout archive/track-a-pre-consolidation                # 164c5e0
git checkout archive/track-a2-pre-consolidation               # 428dd3f
git checkout archive/track-b-pre-consolidation                # 9ca9e90
git checkout archive/track-c-pre-consolidation                # ed76a08
git checkout archive/track-d-pre-consolidation                # 2a91765
git checkout archive/track-e-pre-consolidation                # dd209bf
git checkout archive/report-assets-pre-consolidation          # 02f4e72
```

To put `main` itself back before the consolidation without rewriting anything published:

```
git checkout main
git reset --hard archive/main-pre-consolidation     # local only; origin/main was never moved
```

**b. Bundles** in `~/Documents/Projects/awmv-archive/backups/` — one per branch, every one
`git bundle verify`-clean, sha256 in `SHA256SUMS.txt`:

```
main-pre-consolidation.bundle                  0cdc4cc3…     (14 MB)
track-a-pre-consolidation.bundle               54482d2e…     (26 MB)
track-a2-pre-consolidation.bundle              2533b79d…     (31 MB)
track-b-pre-consolidation.bundle               c8622aac…    (122 MB)
track-c-pre-consolidation.bundle               b3c02626…     (26 MB)
track-d-pre-consolidation.bundle               22831957…     (33 MB)
track-e-pre-consolidation.bundle               f3147f31…     (29 MB)
report-assets-pre-consolidation.bundle         bcf37609…     (33 MB)
```

Restore into an empty directory:

```
mkdir /tmp/restore && cd /tmp/restore && git init -q
git fetch /path/to/awmv-archive/backups/track-d-pre-consolidation.bundle \
    'refs/heads/*:refs/heads/*'
git checkout track-d-verdict-stat
```

**c. Standalone artefacts kept as found:**

* `track-d-verdict-stat.bundle` (32 MB, sha256 `a38e33ec…454545`, contains `2a91765` and a complete
  history) — still at the original path; a copy is also on R2 at
  `track-d/track-d-verdict-stat.bundle` per the D report;
* `awmv-report-assets-2026-11-15.zip` (795 KB, sha256 `10134d9e…cd6d19`) — verified byte-identical
  to `report/` on the `report-assets` branch (only `__pycache__` extra), and its `source_reports/`
  carries every track's markdown reports and decisions log;
* `awmv-archive/untracked/track-c/` — the single untracked file found
  (`intrinsics.json`, from `results/pathrec_calib/`) and Track C's 103 MB `data/` tree;
* `awmv-archive/untracked/data-manifest.tsv` — 42,535 paths with sizes and mtimes for the 3.3 GB
  `data/` tree (not in git, not in any bundle);
* `awmv-archive/inventory/INVENTORY.md` — the pre-change inventory this work was planned from;
* `awmv-archive/inventory/merge_decisions.py` — the resolver used for the decisions-log unions.

**After Phase 5** (if approved) the branches and worktrees are removed; recovery is then via the
tags (which live in `main`'s history) or the bundles, as above.

## 6. Unknowns raised with the user

| question | answer |
| --- | --- |
| `temp.md`, `temp.py`, `temp_proj/` in `~/Documents/Projects` — AWMV material? | left untouched (out of scope) |
| `report_tex.zip` was not attached: wait for it, or proceed and integrate later? | wait; it never arrived, so the LaTeX integration is outstanding (§3, §4 item 5) |
| Merge order: as planned (a → a2 → b → c → d → e → report-assets) or strict DAG order? | as planned |
| Phase 5: remove merged worktrees/branches and move the zip + bundle into `awmv-archive/`? | **not yet answered** — checkpoint 2 |

Other things noticed and *not* acted on, recorded for the user:

* the R2 credentials in `.env` no longer list the bucket (`403 AccessDenied`); the recorded counts
  stand as the last measurement;
* `third_party/anwm` is a plain gitignored clone, not a submodule, so a clean clone cannot pass 17
  tests until it is cloned — documented in `README.md` and `docs/REPRODUCE.md`;
* `report/data/corrections.csv` line references point at the pre-merge decisions log (§3).

## 7. Final layout (two levels)

```
aerial-world-model-verifier/
  README.md                        project, layout, setup, where things are
  pixi.toml  pixi.lock             environment (unchanged across all branches)
  pytest.ini  .gitignore  .env.example   (.env is gitignored and never committed)
  configs/                         22 files: 7 frozen configs, pre-registrations, proposals
  src/                             datasets depicted fusion generation negatives pathrec
                                   perception physics score utils verdict (+remote_common.py)
  experiments/                     75 scripts, one per experiment phase / track step
  remote/                          19 Modal apps (GPU batch work; nothing runs locally)
  scripts/                         13 fetch/sync/export helpers
  tests/                           25 test modules + tests/pathrec/
  docs/
    decisions.md                   the log, per-entry track tags, incl. [consolidation]
    EXPERIMENTS.md  DATA.md  REPRODUCE.md  CONSOLIDATION_REPORT.md
    project_structure.md  project_context.md  paper_tracker.csv  prototype_v0.1.md
    anwm_feasibility.md  verdict_prototype.md  verdict_clip_schema_v2.md  track_*.md
    handoffs/                      01_setup_phase0_anwm.md, 11_consolidation.md, README.md
    reports/                       INDEX.md (every report, one line, with status)
    archive/                       README.md (what is superseded, where it stays)
  reports/                         33 numbered reports: 01-08, b01-b06, C01-C09, D01-D04, E01-E05
  report/                          ASSETS.md CHECKS.md MISSING.md cutoff.json assets_test.tex
    tables/                        48 LaTeX fragments
    figures/                       11 vector PDFs
    data/                          44 CSV + provenance JSON files
    scripts/                       21 build/check scripts
  results/                         64 run directories + manifests (the report scripts read these)
  data/                            gitignored, 3.3 GB (see docs/DATA.md)
  third_party/                     gitignored ANWM clone
```

Outside the repository, in `~/Documents/Projects/`:

```
awmv-archive/
  README.md  backups/ (8 bundles + SHA256SUMS.txt)  untracked/ (track-c/, data-manifest.tsv)
  inventory/ (INVENTORY.md, merge_decisions.py)
awmv-track-a/ awmv-track-a2/ awmv-track-b/ awmv-track-c/ awmv-track-d/ awmv-track-e/ awmv-report/
                                  the worktrees — still present, removable after approval
awmv-report-assets-2026-11-15.zip  track-d-verdict-stat.bundle   (kept in place)
temp.md  temp.py  temp_proj/       out of scope, untouched
```

## 8. What changed in the code (and why)

Only the report scripts changed, and only to make them layout-independent:

* `report/scripts/common.py` — resolves each track from `$AWMV_SOURCES/awmv-track-<track>` when
  that worktree exists, else from the repository root; keeps the recorded per-branch
  `branch`/`commit`/`head_date` from `report/cutoff.json` and the frozen-config commits from
  `tab_frozen-configs_1.tex` under the consolidated layout;
* `report/scripts/methodology.py`, `report/scripts/cutoff.py` — use those helpers instead of
  shelling out to git in a worktree that may not exist.

No model, dataset, experiment, result file, table, figure, config, report or test was modified.
