# Consolidation report — round 2, wave 1 (2026-11-22)

Seven finished branches from the second round were merged into `main` of
`aerial-world-model-verifier` with history preserved and nothing re-analysed. This file records
what was merged and in what order, every conflict and how it was resolved, what was left out and
why, the verification results, the findings that were recorded rather than fixed, and how to roll
back.

Headline: `consolidation-round2` is **85 commits ahead of `main` `41d1dad`** — 77 branch commits,
7 merge commits and one editorial commit. The suite is **258 passed, 3 skipped, 0 failed** (17
failures, and only those, when the gitignored ANWM clone is absent); the report assets rebuild
**16/16** with `CHECKS.md` unchanged at **103 checked / 100 match / 2 mismatch / 1 not
checkable**; all **7 frozen configs** still hash their recorded values; **95 of the 99** in-repo
commit tokens cited under `docs/`, `reports/` and `report/` are reachable (the four exceptions
are listed in §5); `report/assets_test.tex` compiles with 0 undefined references. The LaTeX
report sources were still not supplied, so their integration remains outstanding. No GPU job, no
Modal app and no R2 write occurred.

## 1. What was merged, and in what order

Every branch was a direct descendant of `main` `41d1dad` (none contained another), so the order
is the one in the brief, which is also the order the documents use. Each merge is
`git merge --no-ff`; no squash, rebase, force-push or history rewrite.

| # | branch | tip | commits | merge commit | conflicts | suite after |
| --- | --- | --- | --- | --- | --- | --- |
| — | `main` (before) | `41d1dad` | — | — | — | 221 passed, 3 skipped |
| 1 | `track-f-pixel-path` | `a9dc5cf` | 14 | `6db3ba8` | none | 231 passed, 3 skipped |
| 2 | `track-f2-pixel-path` | `c4f6285` | 8 | `017c5b0` | ledger, DATA, EXPERIMENTS, decisions, INDEX | 237 passed, 3 skipped |
| 3 | `track-g-segments2` | `c189727` | 17 | `2561220` | ledger, EXPERIMENTS, decisions | 246 passed, 3 skipped |
| 4 | `track-g2-segments` | `61401ae` | 4 | `2db9031` | ledger, DATA, EXPERIMENTS, decisions | 246 passed, 3 skipped |
| 5 | `track-h-generator-pilot` | `8b5e979` | 7 | `4fadcd6` | ledger, EXPERIMENTS, decisions, INDEX | 249 passed, 3 skipped |
| 6 | `track-h2-generator-coverage` | `97fda68` | 17 | `bbd4b16` | ledger, DATA, EXPERIMENTS, decisions | 250 passed, 3 skipped |
| 7 | `track-j-reader` | `f7ba108` | 10 | `62eef3e` | ledger, EXPERIMENTS, decisions, INDEX, `segments.py` | 258 passed, 3 skipped |

`pixi.toml` and `pixi.lock` are byte-identical to `main`; nothing was added to the environment.
Before each merge the branch's new pre-registrations were checked to have been added in exactly
one commit and never edited afterwards, and the seven frozen configs were checked to be
untouched on the branch. Both held for all seven branches.

`track-g2-segments` copied `src/verdict/segments.py` and `tests/test_segments.py` from
`track-g-segments2`: the blobs are identical (`cb3a3cd9…`, `256377c7…`), so merge 4 is a no-op on
those two files.

## 2. Conflicts, and how each was resolved

Only five documents ever conflicted, plus one code file on the last merge. A `git merge-tree`
preview before merging predicted exactly those paths; nothing in `configs/`, `results/`,
`reports/` or `report_candidates/` conflicted, and no result file, report, config or frozen
pre-registration was edited.

The rule was the agreed one: **keep every branch's content, ordered by date and labelled by
track; drop nothing and rewrite nothing.** The five documents were rebuilt deterministically
from committed snapshots (`main` plus the branches merged so far) by
`awmv-archive/inventory/build_round2_docs.py`, so the result depends on the set of branches, not
on the order they happened to merge in. The builder then verifies that **every non-blank line a
branch adds relative to `main` appears in the output**; the only permitted transformation is the
E-number mapping in §3.

* **`docs/decisions.md`** — 14 sections became 22; every new track section is kept and ordered by
  first-introduction date: F, G, H (2026-11-16), then F2, G2, H2, J (2026-11-21). The round-2
  `[consolidation-round2]` entry is appended last.
* **`docs/EXPERIMENTS.md`** — summary rows and sections from all seven branches are kept; the
  G01-G05 rows are added to the summary table (the G branches kept their own table inside their
  section, so the summary had no G rows). H's `track-h/` prefix line is kept and extended with
  `track-h2/`.
* **`docs/DATA.md`** — F's and F2's paragraphs in §4, J's §5a, G's §7 and G2's §8 are kept in
  numeric order; the R2 table gains the `track-h/` and `track-h2/` rows; §9 (round-2 backups) is
  added. The access-status paragraph was rewritten once to the longest variant so it names both
  new prefixes instead of carrying a stale copy beside a new one, and F's `pull_volume` line was
  put back inside the code fence it was written in.
* **`docs/reports/INDEX.md`** — F, F2, H and J contributed sections; G, G2 and H2 did not update
  the index on their branches, so their sections were added from their reports. `## Notes`,
  which sat between Tracks E and F, is placed after all track sections. No report text was
  edited.
* **`docs/COST_LEDGER.md`** — this file **did not exist on `main`**; all seven branches added it
  independently (an add/add conflict on every merge after the first). It is now one file with
  one section per track and **each branch's table verbatim** — the column sets differ, so a
  single normalised table would have rewritten rows. The two lines of shared price basis are
  stated once in the header; no spend was added, removed or recomputed.
* **`src/verdict/segments.py`** (merge 7 only) — Track J's file is byte-identical to the merged
  G2 copy plus a four-line provenance header saying it is a copy and must not be edited there.
  The diff against G2 was checked line by line; J's version is kept. `tests/test_segments.py` is
  identical between G and G2 and did not conflict.

No conflict was resolved by choosing one side over the other.

## 3. Experiment IDs: one collision, resolved as documentation only

F, H and H2 each called their experiment **E11** — each branched from `main` independently — and
F2 called itself **E12**. A naive union would leave three E11 rows. On the user's decision
(checkpoint 1), `docs/EXPERIMENTS.md` uses unique IDs and records the mapping:

| branch label | ID in `docs/EXPERIMENTS.md` | experiment |
| --- | --- | --- |
| F: E11 | **E11** | pixel-path verdict, stage 0 |
| F2: E12 | **E12** | pixel-path round 2 |
| H: E11 | **E13a** | generator pilot |
| H2: E11 | **E13b** | generator coverage with path control |
| G01-G03 | G01-G03 (own series) | segment-level detection, combined score, reader fine-tune |
| G04-G05 | G04-G05 (own series) | path-demand windows, simulated-set design |
| J01-J04 | J01-J04 (own series) | reader anatomy, noise budget, per-clip calibration, learned confidence |

The branches' reports and `report_candidates/*.tex` were **not** edited: they keep their authored
labels, which is why a reader will still find "E11" in `report_candidates/e11_pixel_path_verdict.tex`
and in H's/H2's candidates.

## 4. Not merged, and why

* **`track-i-simset`** (`955c0b3`) — wave 2, waiting for the user. Its manifest and frozen splits
  are what matter, and the brief says rendering may still be in progress.
* **`track-h3-reader-validity`** and **`track-k-calibration`** — both branches point at
  `main` `41d1dad`; they contain no commits yet.
* **`track-g-segments`** (`cdfdbf3`) — a **stale branch**. It was created at Track F's amendment
  commit `17d21c6`. Its four commits are (a) F's two pre-registration commits `6b88d5a` and
  `17d21c6`, both reachable from `track-f-pixel-path`, and (b) `3fe20cc` and `cdfdbf3`, which add
  G pre-registration YAMLs **byte-identical** to `track-g-segments2`'s (`5ac7fa52…`,
  `2cbf719a…`) and an older `src/pathrec/pathverdict.py` / `tests/test_pathverdict.py`
  (497/136 lines) superseded by Track F's final versions (525/150). It is left in place, tagged
  and bundled like the others, and not deleted; nothing in it is lost by not merging it.
* **The LaTeX report sources.** `report_tex` (`main.tex`, `sections/`, `sections/experiments/`)
  is in no branch, worktree, archive or user directory found by a search of the repository, all
  worktrees, `~/Documents`, `~/Downloads` and `~/Desktop`. Only `report/assets_test.tex` exists.
  Integration is **outstanding, not done**; the build is documented in `docs/REPRODUCE.md`.
* **`report_candidates/*.tex`** — 12 new candidate files stay as candidates. They are not spliced
  into `report/`, not built by `make_all.py`, and not part of `assets_test.tex`; the user's
  research lead does that.
* **Branches and worktrees** — none was removed or moved. Removal is checkpoint 2, after the user
  sees the result.
* **R2 and Modal** — nothing was written, deleted, started or stopped. No app was run.

## 5. Verification results

| # | check | result |
| --- | --- | --- |
| 1 | full test suite, with the `third_party/anwm` clone | **258 passed, 3 skipped, 0 failed** (46 s) |
| 1a | same suite without the clone | **17 failed, 241 passed, 3 skipped**; all 17 are `tests/test_generation_paths.py` and fail on the missing `third_party/anwm/anwm/config.py`, exactly the known dependency |
| 2 | `report/scripts/make_all.py` + `check_numbers.py` | **16/16 asset groups**; `CHECKS.md` unchanged: **103 checked, 100 match, 2 mismatch, 1 not checkable**. All 48 `.tex` fragments, the CSVs/JSONs and `CHECKS.md` came back byte-identical; the 11 figures differ only in matplotlib's `/CreationDate` (restored, not committed) |
| 3 | frozen-config sha256 vs `report/tables/tab_frozen-configs_1.tex` | **7 of 7** (`c5d3b55e…`, `c8ffac93…`, `dc0cc4ff…`, `845f4c69…`, `28f91f7f…`, `40a40bea…`, `75b208da…`) |
| 4 | every commit hash cited under `docs/`, `reports/`, `report/` reachable from the consolidation result | **95 of 99** resolve and are reachable; the four exceptions are below |
| 5 | LaTeX | `report/assets_test.tex` builds to a 9-page, 404 KB PDF with **0 undefined references**; the full report cannot be built because its sources are absent |
| 6 | tag | `consolidated-round2-wave1` on the commit that adds this report |

Note on #4: `main` itself still points at `41d1dad` — the user merges, not this agent — so the
check is against the consolidation result (`consolidation-round2`). Once `main` is fast-forwarded
to the tagged commit, the same 95 are reachable from `main`. The four exceptions, all cited in
in-repo text, are:

| commit | cited in | why not reachable |
| --- | --- | --- |
| `3fe20cc` | `reports/G01_segment_detection.md` (as the commit for `configs/g01_segment_prereg.yaml`) and `docs/decisions.md` | it is the stale `track-g-segments` line; the merged config comes from `track-g-segments2` `ebec2d7` and is byte-identical (`5ac7fa52…`) |
| `cdfdbf3` | `docs/decisions.md` (round-2 entry, describing the stale branch) | stale branch, deliberately not merged |
| `2d815cc` | `docs/decisions.md` `[F2]`, `reports/F06_feedforward_and_sim.md` | Track I, wave 2 |
| `5f0fa3f` | `docs/decisions.md` `[G2]`, `reports/G05_segments_on_simulated_set.md` | Track I, wave 1 spike, wave 2 |

These are recorded, not fixed: editing report text to point at another commit would be a
provenance change, and the stale branch's config bytes are provably the same.

## 6. Findings and open items (recorded, not resolved)

* **The experiment-ID collision** in §3: three branches independently used E11. Resolved in the
  index only; the branches' authored labels are unchanged.
* **`docs/COST_LEDGER.md` was never on `main`.** Each branch invented its own column set. The
  consolidated file keeps seven tables verbatim rather than normalising them; it computes no
  totals, so a reader must read the USD column the same way the branch did (several rows are
  estimates, several are actuals, and the notes say which).
* **`reports/G01_segment_detection.md` cites `3fe20cc`** for the G01 pre-registration. The bytes
  are the same as the reachable `ebec2d7`, so the provenance is intact even though the hash is
  not reachable. Not edited.
* **`report/data/corrections.csv` still cites `docs/decisions.md` by line number** from before
  round 1; the numbers no longer address the same lines. This is the round-1 note, unchanged.
  The working reference is the archive tags.
* **Round 1's open science discrepancies are untouched.** The 14.73 deg drag-free/lag-only
  conflict, the UZH window numbers, the D01/D02 split, report 08's stale §9 and the duplicated
  sections of A2 report 07 are all still recorded in `report/CHECKS.md`, `report/MISSING.md` and
  `docs/reports/INDEX.md`; nothing was resolved.
* **No broken result path was found** in the new `docs/EXPERIMENTS.md` rows: `results/h01_lingbot/`,
  `h01_lingbot_b/`, `h01_r2/` and `results/h02_inventory.json` all exist. The search was prompted
  by the fact that several of these directories are small and easy to forget.
* **The four unreachable citations** in §5 are the only provenance exceptions in the merged
  trees; two are the stale branch and two are Track I (wave 2).

## 7. Rollback

Three layers, none depending on the others.

**Tags** (in the repository, and in every clone after the user merges):

```
archive/main-pre-round2                   41d1dad
archive/track-f-pixel-path-pre-round2     a9dc5cf
archive/track-f2-pixel-path-pre-round2    c4f6285
archive/track-g-segments2-pre-round2      c189727
archive/track-g2-segments-pre-round2      61401ae
archive/track-h-generator-pilot-pre-round2 8b5e979
archive/track-h2-generator-coverage-pre-round2 97fda68
archive/track-j-reader-pre-round2         f7ba108
archive/track-g-segments-pre-round2       cdfdbf3   (stale branch)
archive/<round-1 tags>                    unchanged
```

**Bundles** in `~/Documents/Projects/awmv-archive/backups/`, one per ref, every one
`git bundle verify`-clean, sha256 in `SHA256SUMS-round2.txt`:

| bundle | sha256 (first 16) |
| --- | --- |
| `main-pre-round2.bundle` | `a156ec946e07d4ef` |
| `track-f-pixel-path-pre-round2.bundle` | `4f9751216e104598` |
| `track-f2-pixel-path-pre-round2.bundle` | `72ae6d85d4186e40` |
| `track-g-segments2-pre-round2.bundle` | `e58d4a33c2214dbb` |
| `track-g2-segments-pre-round2.bundle` | `196d1a84cf0162c2` |
| `track-h-generator-pilot-pre-round2.bundle` | `63dd2073e774538b` |
| `track-h2-generator-coverage-pre-round2.bundle` | `47bf979421672d55` |
| `track-j-reader-pre-round2.bundle` | `d5693d70043b83e6` |
| `track-g-segments-pre-round2.bundle` | `a9d7173791b4ade8` |

Restore one into an empty directory:

```
mkdir /tmp/restore && cd /tmp/restore && git init -q
git fetch /path/to/track-j-reader-pre-round2.bundle 'refs/heads/*:refs/heads/*'
git checkout track-j-reader
```

**`main` was never moved.** It is still `41d1dad`; the round-2 work exists only on
`consolidation-round2`. To discard the consolidation on this machine without touching any
branch: `git checkout consolidation-round2 && git reset --hard archive/main-pre-round2` (local
only; nothing was pushed).

## 8. What changed in the repository (and what did not)

Changed **only documentation**, plus one code file's conflict resolution:

* `docs/decisions.md` (+8 sections), `docs/EXPERIMENTS.md` (E11-E13a/b, G rows, sections),
  `docs/DATA.md` (§5a, §7, §8, §9, two R2 rows), `docs/reports/INDEX.md` (F, F2, G, G2, H, H2, J),
  `docs/COST_LEDGER.md` (new, seven per-track tables), `docs/REPRODUCE.md` (test count, LaTeX
  status), `README.md` (layout, prefixes, repository state).
* `src/verdict/segments.py` is Track J's copy (G2 plus the provenance header), unchanged from
  the branch content.
* Everything else under `src/`, `experiments/`, `remote/`, `scripts/`, `tests/`, `configs/`,
  `results/`, `reports/` and `report_candidates/` is exactly the branches' files, added as-is.
* `report/` (tables, figures, data, scripts, `CHECKS.md`, `assets_test.tex`) is byte-identical to
  `main`; the rebuild was used only as a check and its PDF timestamps were restored.
* `pixi.toml`, `pixi.lock`, `.gitignore` and `pytest.ini` are unchanged.

## 9. What this report does not do

It does not re-analyse, reinterpret or fix any result. It does not resolve any recorded
discrepancy. It does not remove a branch or worktree, move R2 data, or stop a Modal app. It does
not integrate the LaTeX report, because the sources do not exist on this machine. Wave 2
(`track-i-simset` and any new branches) starts only when the user asks.

## Appendix: pre-registration files added in wave 1

Every file below was added in exactly one commit and never edited afterwards on its branch
(`git log main..<branch> -- <file>` returns one commit). None is a frozen config; the seven
frozen configs on `main` were not touched.

| file | adding commit | commits touching it | sha256 (first 16) |
| --- | --- | --- | --- |
| `configs/f01_pixel_path_prereg.yaml` | `6b88d5a` | 1 | `1a55b30d4fc7944a` |
| `configs/f01_pixel_path_prereg_amend1.yaml` | `17d21c6` | 1 | `109fd31bba3a5877` |
| `configs/f04_round2_prereg.yaml` | `63002b8` | 1 | `3c395e1593ced705` |
| `configs/g01_segment_amendment_a.yaml` | `7d44d3c` | 1 | `f3f334998d5a1458` |
| `configs/g01_segment_prereg.yaml` | `ebec2d7` | 1 | `5ac7fa5220bc609f` |
| `configs/g02_combined_prereg.yaml` | `5e1d82c` | 1 | `2cbf719a740d07fa` |
| `configs/g03_tilt_reader_prereg.yaml` | `7074f9d` | 1 | `74924429bafdbb4b` |
| `configs/g04_path_demand_prereg.yaml` | `4c6eacf` | 1 | `4bdbe04d441d06cd` |
| `configs/g05_round2_design_prereg.yaml` | `7a98870` | 1 | `4741379c427db8a6` |
| `configs/h01_pilot_prereg.yaml` | `245f7d9` | 1 | `3529b9208d8bc8e4` |
| `configs/h01_pilot_prereg_amendment1.yaml` | `f959cd1` | 1 | `eabd250c4a8c491a` |
| `configs/h02_pilot_prereg.yaml` | `5c6e5c6` | 1 | `0f6287237dc4131f` |
| `configs/h02_pilot_prereg_amendment1.yaml` | `78bee7e` | 1 | `683a8e27297864eb` |
| `configs/j01_anatomy_amendment_a.yaml` | `0c57c7f` | 1 | `dbf70b35b4ddc7f9` |
| `configs/j01_anatomy_amendment_b.yaml` | `0c57c7f` | 1 | `b6283ed313d0abc8` |
| `configs/j01_anatomy_prereg.yaml` | `9652848` | 1 | `64091358c99f66e0` |
| `configs/j02_noise_budget_prereg.yaml` | `9652848` | 1 | `dfba27ccae0ed67c` |
| `configs/j03_perclip_prereg.yaml` | `d01d398` | 1 | `726b17ec751e197e` |
| `configs/j04_learned_confidence_prereg.yaml` | `1dbdd55` | 1 | `e2e03d915c9e0086` |

---

# Part II — wave 2 (2026-11-23)

Seven more branches were merged into `main` with history preserved and nothing re-analysed. This
part continues the numbering of Part I: it records what was merged and in what order, every
conflict and how it was resolved, the one stop condition, the verification results, the Modal
billing check against the track ledgers, the findings, and how to roll back.

Headline: `main` moved from `6bc7b5d` (`consolidated-round2-wave1`) to the tagged wave-2 result —
**169 commits** (161 branch commits, 7 merge commits, 1 editorial commit). The suite is
**328 passed, 4 skipped, 0 failed** (17 failures, and only those, without the ANWM clone); the
report assets rebuild **16/16** with `CHECKS.md` unchanged at **103 checked / 100 match / 2
mismatch / 1 not checkable**; all seven wave-1 frozen configs still hash their recorded values and
the three wave-2 frozen values you named match; **142 of the 144** in-repo commit tokens cited in
the trees are reachable (the two exceptions are the stale `track-g-segments` commits);
`report/assets_test.tex` still compiles with 0 undefined references. The LaTeX report sources
remain absent. No worktree or branch was removed, nothing was pushed, and no GPU job was run.

## 10. What was merged, and in what order

Every branch was merged with `git merge --no-ff`, in the order of the brief. I, H3, K, L and G3
branch from `41d1dad`; M and N branch from the wave-1 result `6bc7b5d`. No wave-2 branch contains
another, so the order is the content dependency order.

| # | branch | tip | commits | merge commit | conflicts | suite after |
| --- | --- | --- | --- | --- | --- | --- |
| — | `main` (before) | `6bc7b5d` | — | — | — | 258 passed, 3 skipped |
| 1 | `track-i-simset` | `272bb93` | 25 | `55b7945` | ledger, DATA, EXPERIMENTS, decisions | 282 passed, 3 skipped |
| 2 | `track-h3-reader-validity` | `20a93cb` | 18 | `5cee0a3` | ledger, DATA, EXPERIMENTS, decisions, INDEX | 287 passed, 4 skipped |
| 3 | `track-k-calibration` | `b4ddc9f` | 15 | `26c7cc2` | DATA, EXPERIMENTS, decisions, INDEX | 296 passed, 4 skipped |
| 4 | `track-l-reader-simreal` | `2fd64d8` | 25 | `e88dbf5` | ledger, DATA, EXPERIMENTS, decisions, INDEX, manifest, build script | 296 passed, 4 skipped |
| 5 | `track-g3-segments` | `e76411e` | 32 | `7f12e15` | ledger, DATA, EXPERIMENTS, decisions, `segments.py` | 298 passed, 4 skipped |
| 6 | `track-m-gates` | `d80abc5` | 24 | `ba829a9` | ledger, DATA, EXPERIMENTS, decisions | 323 passed, 4 skipped |
| 7 | `track-n-rotation` | `8e280f3` | 22 | `49bc1d3` | ledger, DATA, EXPERIMENTS, decisions | 328 passed, 4 skipped |

`pixi.toml`, `pixi.lock`, `.gitignore` and `pytest.ini` are unchanged. Before each merge the
branch's new pre-registrations were checked for immutability (two exceptions, §16), and the seven
wave-1 frozen configs were checked to be untouched on every branch. Both held.

## 11. Conflicts, and how each was resolved

Only the five documents conflicted, plus two data/code add/adds. The documents were rebuilt from
committed snapshots by the same union approach as wave 1 (`awmv-archive/inventory/build_round2_wave2_docs.py`):
`decisions.md`, `EXPERIMENTS.md` and `INDEX.md` take new sections only; `DATA.md` is merged
per section with a real 3-way merge using each branch's own base (it has mid-section insertions);
`COST_LEDGER.md` takes new track sections. After each rebuild the script verifies that **every
line a branch adds relative to its own base appears in the output**; the only permitted
transformations are the E-number mapping in §12 and the absorbed ledger preamble in §14.

* **`docs/COST_LEDGER.md`** — Track I, H3, L and G3 added their own file (base predates wave 1);
  M and N add a section to the wave-1 file. The result keeps the wave-1 header and seven track
  sections, then the new sections in branch order: I (I01, I02, full-set, booked), H3/H4, L
  (L01, L02, running total), G3, M, N (round 1 and round 2). Each table is verbatim; no spend was
  added, removed or recomputed.
* **`docs/DATA.md`** — I inserts an R2 row, a Modal row and a generated dataset block inside
  existing sections; H3, K, L, G3, M and N add new sections. The 3-way merge kept I's insertions
  in place; the generated block was then moved to its own section 10 (its
  `<!-- track-i-dataset:begin/end -->` markers are intact, so `i02_build_manifest.py` can still
  regenerate it), and the branch-local numbers 5/7/9/10 were renumbered 10-16 because they
  collided with wave 1's sections 5/7/9/10. No line was dropped.
* **`docs/EXPERIMENTS.md`** — summary rows and sections from all seven; H3's E11/E12 rows and
  headings are carried as E14/E15. The table was re-sorted by family (E, G, I, J, K, L, M, N) and
  given summary rows for G06-G07, I, K, L, M and N (their branches kept those rows inside their
  own section tables). Track M's own section has its M04/M05 table rows after two prose
  paragraphs — a markdown table break in the branch's text; it is left as authored.
* **`docs/decisions.md`** — every new section kept, in branch order: I, H3, H4, K, L, G3,
  G3-G07, M, N, then the `[consolidation-round2-wave2]` entry.
* **`docs/reports/INDEX.md`** — H3/H4, K and L added sections; I, G3, M and N did not, so their
  sections were added from their reports. The order groups families (G3 with G, H3/H4 with H, I
  before J). No report text was edited.
* **`results/i02/manifest.csv`** (add/add) — Track I's 301-row manifest is a strict superset of
  L's 165-row copy: all 165 of L's lines are present, no L-only line exists. I's version is kept.
  L copied the manifest before the full 300-clip set was rendered; nothing is lost.
* **`experiments/i02_build_manifest.py`** (add/add) — I's version adds `--full-hash`; L's is the
  earlier copy with no other difference. I's version is kept.
* **`src/verdict/segments.py`** (add/add) — G3's file is the merged G2 copy (231 lines) without
  Track J's four-line provenance header. The merged 235-line file is kept; the diff is exactly
  those four comment lines, so no code changed.

No conflict was resolved by discarding a side.

## 12. Experiment IDs

Track H3 independently used E11/E12. `docs/EXPERIMENTS.md` carries them as **E14** (reader
validity on generated frames, H03) and **E15** (SolarWM continuous-attitude campaign, H04). The
other new tracks keep their own series: I01-I02, K01-K03, L01-L02, G06-G07, M01-M05 and
N01-N06. The branches' reports and `report_candidates/*.tex` keep their authored labels.

## 13. The stop condition: Track M's frozen Gate B file

`configs/m02_gate_b_frozen.yaml` was frozen as **v1** at `tau = 3.0` (`42a2cf3`, marked
"IMMUTABLE after this commit") and **replaced by v2** at `tau = 4.5` in `846e8bf`, the same commit
that reports M02/M03 results. The v2 file also changes the selection unit (per-clip real-condition
disagreement instead of any-condition row) and the accepted/coverage numbers, and
`configs/m02_gate_b_amend2.yaml` (same commit) documents the correction and says it was written
before the H04/H01 generator clips were read. The branch's report, amendment and final frozen file
are internally consistent (all use 4.5).

This matches the posted stop condition — a frozen config whose content and sha changed after
freezing. It was posted with the merge plan and the user approved **merging M as-is**: the v1→v2
edit is recorded here and in `docs/decisions.md`, and the file is not repaired or reverted.

## 14. Verification

| # | check | result |
| --- | --- | --- |
| 1 | full suite with `third_party/anwm` | **328 passed, 4 skipped, 0 failed** (54 s) |
| 1a | same suite without the clone | **17 failed, 311 passed, 4 skipped**; all 17 are `tests/test_generation_paths.py` on the missing `third_party/anwm/anwm/config.py` — the known dependency. The extra skip (3→4) is a new skip on the wave-2 branches, not a failure |
| 2 | `make_all.py` + `check_numbers.py` | **16/16 asset groups**; `CHECKS.md` unchanged at **103 / 100 match / 2 mismatch / 1 not checkable**; text assets byte-identical; the 11 figures differ only in matplotlib's `/CreationDate` (restored) |
| 3 | frozen configs | wave-1 **7 of 7** (`c5d3b55e…`, `c8ffac93…`, `dc0cc4ff…`, `845f4c69…`, `28f91f7f…`, `40a40bea…`, `75b208da…`); `sim_splits_frozen.json` `cdf66e28…`; `sim_prereg.yaml` `245b68bf…`; `g06_eval_frozen.yaml` `615f228f…`; `m01_gate_a_frozen.yaml` `8ebe72e7…`; `m02_gate_b_frozen.yaml` `a8d652b9…` (v2, see §13); `sim_splits_frozen_readonly.json` `cdf66e28…` |
| 4 | cited commit hashes under `docs/`, `reports/`, `report/` reachable from the result | **153 resolve, 151 reachable** with this report included (144/142 before it was added); the two exceptions are `3fe20cc` and `cdfdbf3`, the stale `track-g-segments` line (unchanged from Part I; the Track I citations are now reachable) |
| 5 | LaTeX | `report/assets_test.tex` builds to a 9-page, 404 KB PDF with **0 undefined references**; the full report still cannot be built — its sources are absent |
| 6 | tag and `main` | `consolidated-round2-wave2` on the commit that adds this section; `main` fast-forwarded to it |

## 15. Modal billing against the track ledgers

`modal billing report --start 2026-11-16 --end 2026-11-24 --json` returns 291 entries totalling
**USD 121.27** for the window. Per-track billing is the sum over apps whose name starts
`awmv-<prefix>-`; the ledger total is what the track's ledger states, or the sum of its actual rows
when it states no total. Raw export, comparison and method:
`results/consolidation_round2/`.

| track | ledger total (basis) | Modal billing | difference |
| --- | --- | --- | --- |
| F | 5.20 (sum of actual rows; no total row) | 7.78 | +2.58 |
| F2 | 0.70 (actual; the 8.00 estimate was not spent) | 0.80 | +0.10 |
| G | 10.23 (total row; includes the 2.20 wasted run) | 5.48 | −4.75 |
| G2 | 0.00 (CPU only) | 0.00 | 0 |
| G3 | 8.02 (both passes; G07 actual 2.75) | 8.11 | +0.09 |
| H | ~13 (estimate) | 17.52 | +4.5 |
| H2 | ~5 (estimate) | 7.73 | +2.7 |
| H3/H4 | ~30 (actual, "about") | 17.24 | −12.8 |
| I | ~25.2 (planned total; render actual 7.55) | 23.27 | −1.9 vs planned |
| J | 0.00 (CPU only) | 0.00 | 0 |
| K | 0.00 (CPU only; no ledger) | 0.00 | 0 |
| L | 29.40 (booked total) | 22.89 | −6.51 |
| M | 2.95 (total row) | 4.37 | +1.42 |
| N | 4.45 (sum of actual rows; no total row) | 6.08 | +1.63 |

The differences are accounting, not science and not a finding about any result: the ledgers mix
estimates, planned totals and wall-clock arithmetic at list prices, the billing export is actual
charge by app name in a fixed window, CPU work does not appear as an app, and queued capacity is
not billed. They are recorded, not reconciled. The largest gaps are H3/H4 (ledger about 30,
billing 17.24) and L (29.40 vs 22.89); both ledgers contain estimates or failed-attempt hours.

## 16. Findings and open items (recorded, not resolved)

* **The Track M frozen-file edit** in §13 is the only stop condition that fired. Merged as-is on
  the user's approval; the v1→v2 history is intact.
* **Track I's `sim_splits_frozen.json` was edited once after creation** (second commit `2d815cc`
  adds `uptilt_deg` / `mount_yaw_deg` labels per clip). The final sha is the specified
  `cdf66e28…`, so the frozen object you named is correct; the edit is recorded because a frozen
  file was touched.
* **Track L's `l02_reader_prereg.yaml` was edited once** (`2bf2cb2`: per-epoch budget 1200→600,
  on measured throughput, before any fold outcome; the file states this and every arm keeps the
  same number of steps). Recorded, not treated as clean.
* **The E11/E12 collision** in §12 is resolved in the index only.
* **`docs/DATA.md` branch-local section numbers** collided (three different "7", two different
  "9"); they are renumbered 10-16 in the consolidation, with the archived section text unchanged.
* **Billing differences** in §15 are listed with their basis and left unreconciled.
* **Track M's EXPERIMENTS section** keeps its authored M04/M05 table rows after two prose
  paragraphs (a markdown table break); left as authored.
* **The two unreachable citations** are the stale `track-g-segments` commits; the branch is still
  present, tagged and bundled, and the G prereg bytes it carries are identical to the merged ones.
* **Round 1's open science discrepancies are untouched**, as are all wave-1 and wave-2 reports.

## 17. Rollback

**Tags** (in the repository; `main` moved forward from `6bc7b5d`):

```
consolidated-round2-wave1                  6bc7b5d   (the state before this wave)
archive/main-pre-round2-wave2              6bc7b5d
archive/track-i-simset-pre-round2-wave2    272bb93
archive/track-h3-reader-validity-pre-round2-wave2 20a93cb
archive/track-k-calibration-pre-round2-wave2 b4ddc9f
archive/track-l-reader-simreal-pre-round2-wave2 2fd64d8
archive/track-g3-segments-pre-round2-wave2 e76411e
archive/track-m-gates-pre-round2-wave2     d80abc5
archive/track-n-rotation-pre-round2-wave2  8e280f3
```

**Bundles** in `~/Documents/Projects/awmv-archive/backups/`, all `git bundle verify`-clean,
sha256 in `SHA256SUMS-round2-wave2.txt`:

| bundle | sha256 (first 16) |
| --- | --- |
| `main-pre-round2-wave2.bundle` | `bdde0327df925991` |
| `track-i-simset-pre-round2-wave2.bundle` | `d4261e3ed685f557` |
| `track-h3-reader-validity-pre-round2-wave2.bundle` | `109e536be5ec9af8` |
| `track-k-calibration-pre-round2-wave2.bundle` | `435104e240d18ade` |
| `track-l-reader-simreal-pre-round2-wave2.bundle` | `38f19efa8a117549` |
| `track-g3-segments-pre-round2-wave2.bundle` | `7fee31f4c6ea277f` |
| `track-m-gates-pre-round2-wave2.bundle` | `ec90094532004a66` |
| `track-n-rotation-pre-round2-wave2.bundle` | `60ed068e42c82409` |

To put `main` back before wave 2 without rewriting published history:
`git checkout main && git reset --hard consolidated-round2-wave1` (local only; nothing was
pushed).

## 18. What changed in the repository

Only documentation, the new billing result files, and the two conflict resolutions on existing
files (`results/i02/manifest.csv` and `experiments/i02_build_manifest.py` keep Track I's
supersets; `src/verdict/segments.py` keeps the merged 235-line version). Everything else under
`src/`, `experiments/`, `remote/`, `scripts/`, `tests/`, `configs/`, `results/`, `reports/` and
`report_candidates/` is exactly the branches' content, added as-is. `report/` is byte-identical to
wave 1. `pixi.toml`, `pixi.lock`, `.gitignore` and `pytest.ini` are unchanged. No worktree,
branch, R2 object or Modal app was removed or started.
