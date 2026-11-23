# Track N2 state — verdict tiers on the propagated lean over the simulated set

Branch `track-n2-sim-tiers`, worktree `~/Documents/Projects/awmv-n2`, started at the tip of
`track-n-rotation` (`8e280f3`) because `track-n-rotation` is **not** merged into `main`
(`git merge-base --is-ancestor track-n-rotation main` is false). The handoff allowed either; this
is the branch-from-`track-n-rotation` case.

Frozen inputs verified before any work: `configs/sim_splits_frozen.json` sha256
`cdf66e28…c120dbd` and `configs/sim_prereg.yaml` sha256 `245b68bf…3e75fb`, both matching the
handoff. Neither was edited. The Track I manifest is staged read-only as
`results/n2/inputs/manifest.csv`, sha256 `50aacbb2…ce6794`.

## What the handoff brief predates

The brief says "Track N has used 0.41 of 30" and asks for Parts A–D. Round 2 of Track N already
ran most of A (N05: C2 and coherence on the level-start propagated arm, all 200 dev clips) and
B(i)–(ii) (N06), and clarified C (N04). Those reports are read and extended, not repeated. The
gaps this branch fills are all in the pre-registrations.

## Status — all parts done

| part | state | report |
| --- | --- | --- |
| Pre-registrations (N07 + amendment A, N08, N09) | done, committed before scoring | — |
| **Part A** — F0 + M04 sign + C2 + coherence, 5 start arms, 200 dev clips, family × mount | **done** | `reports/N2A_verdict_tiers_propagated_f0.md` |
| **Part B(iii)** — two-draw robust aggregation | **done** | `reports/N2B_mount_drift_draw_aggregation.md` |
| **Part C** — SolarWM start / horizon / like-for-like | **done** | `reports/N2C_part4_provenance.md` |
| **Part D** — agreement gate on coverage | **not run**, as pre-registered: no deployable arm met the forest criterion under any tier | — |
| Docs, ledger, LaTeX candidate | done | `report_candidates/n2_propagated_lean_sim.tex` |

## Decisions in one line each

* **Part A negative on detection, positive on specificity.** At a matched 8/80 pooled feasible
  false-reject, F0 detects 13/27 removals on the reader-start arm against the reader's 22/27
  (forest 4/9 vs 9/9); the pre-registered forest criterion fails (the reader is saturated at
  9/9), the high-mount criterion passes. The propagated F0 cut (16–18°) is set by a 45° feasible
  tail (5–6 of 80 clips above 20°) the reader does not have. A reader start imports the forest
  bias; a 2° external attitude behaves like the oracle and matches the reader's forest curve with
  half its forest false-rejects.
* **Sign tier unusable on sim** at a 5–10 % budget (bimodal feasible scores at the ceiling).
* **Part B(iii) negative**: two-draw averaging does not meet its reduction; the draws disagree by
  a median 15.6° at 45° in forest.
* **Part C correction**: Part 4 compares a relative 1 s error to an absolute per-frame error; with
  a reader start the absolute error is not better than the reader's. N04's numbers stand.

## Spend

Ceiling 25 USD for this track. Track N total before this branch: **4.45 USD** (0.41 round 1 +
4.04 round 2). This branch spent **0.00 USD**: every number is a CPU reanalysis of stored N05
reader scores, stored DPVO poses and stored H04 reader/path data. No Modal app, volume or R2
write. Ledger: `docs/COST_LEDGER.md`.

## Data-cache note (integrity)

`data/n06/poses_repeat/scale1` no longer holds the third DPVO draw N06 reported as "run 2": it now
duplicates draw 1 on 188/200 clips. `data/n05/poses/sim200/scale1_rep2` does reproduce N05's
published draw-2 C2 values exactly (median |diff| 0.0 over 200 clips) and is the second draw used
by N07/N08. N06's published run-2 medians are not recomputed. Recorded in `docs/decisions.md`
and N2B. `data/` is gitignored and symlinked read-only from `awmv-n`.

## Resume command

```
cd ~/Documents/Projects/awmv-n2
PYTHONPATH=. pixi run python experiments/n07_sim_tiers.py
PYTHONPATH=. pixi run python experiments/n08_mount_draws.py
PYTHONPATH=. pixi run python experiments/n09_part4_provenance.py
PYTHONPATH=. pixi run python experiments/n2_checks.py      # 54 report-number checks, expect 0 failed
```

All three are CPU, seconds, zero spend. No detached Modal app or job exists for this track.

## Open items

* Whether fixing the 45° drift would unlock the forest detection gain (N2A predicts yes; Part
  B(iii) did not find a cheap remedy, so a new estimator or a per-clip failure flag is the next
  question).
* A deployed generator that does not expose its conditioning pose would need the start from
  pixels, which N2C shows is no better than the reader.
* No held-out simulated clip has been read by this branch.
