# Handoff 11 — consolidate the AWMV repository

Committed verbatim as the record of the consolidation task (supplied in-conversation on
2026-11-15; not previously in the repository). The outcome, the conflicts and the verification
are in `docs/CONSOLIDATION_REPORT.md`.

---

You are a new coding agent. Read this whole file before acting. Nothing in it requires prior
context. You run on the user's own computer, in `~/Documents/Projects`.

## What the project is (one paragraph)

`aerial-world-model-verifier` (AWMV) is a research project that builds a physics-based verifier
for aerial video world models: a quadrotor thrusts only along its body axis, so the tilt a flight
path requires can be compared with the tilt the video depicts. Several coding agents worked on
separate **tracks** (A, A2, B, C, D, E, and a report-assets agent), each in its own folder and
branch. The user now wants **everything consolidated into one clean repository on `main`**, with
logs and data saved in a way a human can browse.

## Your goal

1. Every track's work is merged into `main` of `aerial-world-model-verifier`, with **history
   preserved**.
2. The repository has a clear structure, an index of experiments, a data index, and reproduction
   instructions.
3. Nothing is lost, nothing is silently changed, and the whole thing can be rolled back.
4. The scientific content is **not** altered. You are tidying, not re-analysing.

## In scope and out of scope

In scope, inside `~/Documents/Projects`: `aerial-world-model-verifier`, `awmv-track-a`,
`awmv-track-a2`, `awmv-track-b`, `awmv-track-c`, `awmv-track-d`, `awmv-track-e`, `awmv-report`,
`awmv-report-assets-2026-11-15.zip`, `track-d-verdict-stat.bundle`.

**Out of scope: do not open, move, edit or delete anything else** in that directory (it holds the
user's unrelated projects). `temp.md`, `temp.py` and `temp_proj` are unknown: do not touch them;
list them in your report and ask.

The user will attach `report_tex.zip` (the LaTeX report sources) to the conversation. Integrate it
(see Phase 3).

## Hard rules

- **Never rewrite history.** No squash, no rebase of shared branches, no `--force`, no
  `git filter-repo`. Use merge commits. Several reports cite specific commit hashes (for example D
  at `2a91765`, E at `dd209bf`, B at `9ca9e90`, A2 and others); those hashes must stay reachable.
- **Never delete** a branch, worktree, folder, bundle, zip or R2/Modal object without the user's
  explicit approval. When approved, use `git worktree remove` / `git branch -d` (not `-D`, not
  `rm -rf`).
- **Frozen configs and pre-registrations are immutable.** Their content must not change (their
  sha256 values are recorded in `report/tables/tab_frozen-configs_1.tex` and in the configs'
  commit history). Do not move or rename them either; add indexes instead.
- **Do not fix science.** If two files disagree (see "Known open discrepancies" in
  `report/CHECKS.md`), record it; do not resolve it.
- **Secrets:** never print, copy or commit `.env` or credentials. Check that `.gitignore` covers
  them.
- **R2 (Cloudflare) is read-only** for you: listing and size queries only. **Modal:** read-only
  (list apps, volumes); report anything running; do not stop or delete without asking.
- **No experiments and no GPU spend.** CPU-only checks (tests, scripts) are fine.
- If anything is ambiguous or surprising, stop and ask the user.

## What exists (verify all of it, do not trust this list)

- Track branches and folders: probably git worktrees of one repository, but they might be separate
  clones. Find out (`git worktree list` from the main repo; otherwise check each folder's `.git`).
- Known branch names: `track-d-verdict-stat`; `report-assets` (branched from
  `track-d-verdict-stat`, only adds `report/`). Others: derive them.
- Known recorded HEADs: track D `2a91765`; track E `dd209bf` (reports/E05_window_rule.md); track B
  generation `9ca9e90`.
- `track-d-verdict-stat.bundle`: a git bundle of the D branch (also stored on R2 as
  `track-d/track-d-verdict-stat.bundle`, sha256 starts `a38e33ec`). Keep it.
- R2 object counts measured on 2026-11-15: track-a 288 (including a2 122), track-b 3,641, track-c
  0, track-d 67, track-e 595. See `report/data/r2_counts.json` and `report/cutoff.json` on the
  `report-assets` branch.
- Modal apps and volumes are prefixed `awmv-a-`, `awmv-c-`, `awmv-e-` (and similar).
- `awmv-report-assets-2026-11-15.zip`: the report-assets agent's output; its contents should equal
  the `report/` folder on the `report-assets` branch. Check, then archive.
- Documents of record inside the repo: `docs/decisions.md`, `docs/prototype_v0.1.md`, `reports/`
  (numbered track reports, D01-D04, E04/E05, A reports 01-08), `configs/` (frozen configs,
  `*_prereg.yaml`, proposals), `results/` (stored outputs the report scripts read), and the
  earlier handoffs and `project_structure.md`.

## Phase 0: inventory (read-only)

Produce `INVENTORY.md` (in a scratch location, not yet committed) with, for each folder and
branch: path, whether worktree or clone, branch name, HEAD, ahead/behind versus `main`,
uncommitted or untracked files (names and sizes), stashes, tags, remotes (`git remote -v`), disk
usage, large files (>10 MB) and whether they are tracked. Also: unpushed commits, the merge-base
graph between branches, and any files present in one worktree but untracked everywhere.

## Phase 1: safety net (before any change)

- For every branch and for `main`: create a tag `archive/<track>-pre-consolidation` at its current
  tip and a `git bundle` into `~/Documents/Projects/awmv-archive/backups/` (create it). Record
  each bundle's sha256.
- Copy any **untracked but valuable** files (logs, results, data) into
  `awmv-archive/untracked/<track>/` before touching their folder. Do not copy secrets.
- **Checkpoint 1: post the inventory and your merge plan to the user and wait for approval.** The
  plan states the merge order, the conflicts you expect, and the target structure.

## Phase 2: merge

- Order by dependency using the merge-base graph (earlier tracks first, `report-assets` last,
  since it branches from D). Merge with `git merge --no-ff` and a message naming the track.
- Expect conflicts in `docs/decisions.md`, `pixi.toml`, `pixi.lock`, `.gitignore`, `README`/index
  files, and `results/` indexes. Resolve by **keeping both sides' content** (decisions log: keep
  every entry, in date order, labelled by track). For `pixi.toml`/`pixi.lock`: union the
  dependencies, regenerate the lock, and note it. Never edit a frozen config.
- Run each branch's tests before merging it and the full suite after each merge. Record results.
  The D branch had a full suite of 142 passing and 41 passing from a clean clone; use those as
  references.
- Large data stays out of git (it lives on R2); small result summaries stay in git, because
  `report/scripts/*` read them.

## Phase 3: structure and human-readable indexes

Follow the existing `project_structure.md` (source under `src/`, no `verifier` nesting, folders
added per phase). **Do not mass-move code.** Prefer adding indexes over reshuffling. Add, at the
repository root or under `docs/`:

- `README.md`: what the project is, the layout, how to set up the environment (pixi), where the
  report is, where the data is.
- `docs/EXPERIMENTS.md`: a table mapping each report experiment (E1 physics floor, E2 depicted
  tilt TII, E3 verdict real flights, E4 ANWM data audit, E5 ANWM verdict, E6 pose injection, E7
  path from pixels, E8 gain and coherence tests, E9 UZH transfer, E10 window rule and prototype)
  to: owning track, branch, commit(s), config(s), stored result directory, report(s), R2 prefix,
  and the script that regenerates its tables/figures.
- `docs/DATA.md`: what lives where (git, R2 prefixes with object counts and sizes, Modal volumes),
  how to restore it, and what is not recoverable.
- `docs/REPRODUCE.md`: how to rebuild the report assets and the LaTeX report, and what to expect
  (`report/CHECKS.md` baseline: 103 values checked, 100 match, 2 mismatch, 1 not checkable).
- `docs/reports/INDEX.md`: every report (numbered), one line each, and which are superseded (for
  example D01 by D02; A report 07 has duplicated sections; ANWM scoring report 08 has stale
  sections). Mark them; do not edit their text.
- `docs/handoffs/`: the handoff briefs (handoff_01 to handoff_11 and `project_structure.md`), if
  they are not already in the repository.
- `report/`: keep the report-assets agent's `report/` layout (`figures/`, `tables/`, `data/`,
  `scripts/`, `ASSETS.md`, `CHECKS.md`, `MISSING.md`, `cutoff.json`). Add the LaTeX sources from
  `report_tex.zip` (`main.tex`, `sections/`) so that it **builds in place** against
  `report/tables` and `report/figures` (adjust the include paths in the macros if needed; use
  `latexmk` or `pdflatex` twice). Commit the PDF only if small; otherwise leave it out and
  document the build command.
- `archive/` or `docs/archive/` for superseded material (moved, not deleted).

## Phase 4: verification (all must pass before you report done)

1. Full test suite passes on `main`.
2. The environment installs from `pixi.lock` on a clean clone.
3. `report/scripts/make_all.py` (CPU only) regenerates the tables and figures, and
   `check_numbers.py` reproduces the **baseline in `report/CHECKS.md`** (100 match, 2 mismatch, 1
   not checkable). Any change is a finding to report, not to fix.
4. The sha256 of every frozen config still matches `report/tables/tab_frozen-configs_1.tex`.
5. The LaTeX report compiles and has no undefined references.
6. Every commit hash cited in `docs/reports/` and `report/` is reachable from `main`
   (`git merge-base --is-ancestor`). List any that are not.
7. Tag the result `consolidated-2026-11-15`.

## Phase 5: tidy up (needs approval)

**Checkpoint 2: after Phase 4, show the user the result and ask before removing anything.** With
approval only: `git worktree remove` the merged worktrees, delete merged branches with
`git branch -d` (the archive tags keep them reachable), and move the zip and the bundle into
`awmv-archive/`. Leave R2 and Modal untouched unless the user says otherwise. Report any Modal
apps still running.

## Deliverable

`docs/CONSOLIDATION_REPORT.md` (committed on `main`) containing: what was merged and in what
order; every conflict and how it was resolved; anything not merged and why; the Phase 4 results;
the rollback procedure (which bundles/tags restore which state); the list of unknowns you asked
the user about; and the final directory tree (two levels deep). Then tell the user, in a few
sentences, what is where.

## Rules of thumb

- Preserve provenance over prettiness. A slightly untidy but traceable repository beats a tidy one
  with broken references.
- Record every decision in `docs/decisions.md` (tagged "consolidation").
- If you are unsure whether a file matters, keep it and note it.
