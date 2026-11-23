# Track N2 state — verdict tiers on the propagated lean over the simulated set

Branch `track-n2-sim-tiers`, worktree `~/Documents/Projects/awmv-n2`, started at the tip of
`track-n-rotation` (`8e280f3`) because `track-n-rotation` is **not** merged into `main`
(checked 2026-11-23: `git merge-base --is-ancestor track-n-rotation main` is false). The handoff
allowed either; this is the branch-from-`track-n-rotation` case.

Frozen inputs verified before any work: `configs/sim_splits_frozen.json` sha256
`cdf66e28…c120dbd` and `configs/sim_prereg.yaml` sha256 `245b68bf…3e75fb`, both matching the
handoff. Neither was edited; they are not on this branch (they live on `track-i-simset`) and were
not needed as files (the split was already filtered into `results/n05/clips.csv`). The Track I
manifest is staged read-only as `results/n2/inputs/manifest.csv`, sha256 `50aacbb2…ce6794`.

## What the handoff brief predates

The brief says "Track N has used 0.41 of 30" and asks for Parts A–D. Round 2 of Track N already
ran most of A (N05: C2 and R on the level-start propagated arm, all 200 dev clips) and B(i)–(ii)
(N06: two-view chain and resolution), and clarified C (N04). Those reports are read and extended,
not repeated. The **remaining** gaps this branch fills:

* **Part A:** the **F0 tier** (N05's pre-registration explicitly excluded it: "F0 is not run"),
  the **M04 sign sub-tier** (coherence is undefined on ~60 % of feasible clips), the **reader
  start** and **oracle start** arms, full family × mount stratification, and pre-registered
  numeric success criteria for forest and for the 30/45 mounts. Plus a 2°-start sensitivity arm
  (N05's named open question).
* **Part B(iii):** one further single change — two-draw robust aggregation.
* **Part C:** a short provenance note (N04 already states the start and horizon).
* **Part D:** conditional on Part A; gate condition pre-registered in `n07`.

## Status

| part | state |
| --- | --- |
| Pre-registrations (N07, N08) | **done**, committed before scoring |
| Part A (N07) | running |
| Part B(iii) (N08) | pending |
| Part C (N09) | pending |
| Part D | gated on Part A |

## Spend

Ceiling 25 USD for this track (handoff). Track N total before this branch: **4.45 USD**
(0.41 round 1 + 4.04 round 2), carried in `docs/COST_LEDGER.md`. This branch plans **0.00 USD**:
every number is a CPU reanalysis of stored N05 reader scores and stored DPVO poses. No Modal app,
no volume, no R2 write. If a job is ever needed it will be detached, recorded here, and stopped.

## Data-cache note (integrity)

`data/n06/poses_repeat/scale1` no longer holds the third DPVO draw N06 reported as "run 2": it now
duplicates draw 1 on 188/200 clips. `data/n05/poses/sim200/scale1_rep2` does reproduce N05's
published draw-2 C2 values exactly (median |diff| 0.0 over 200 clips) and is the second draw used
here. N06's published run-2 medians are not recomputed; the discrepancy is recorded in
`docs/decisions.md` and in N2B. `data/` is gitignored and symlinked read-only from `awmv-n`.

## Resume command

```
cd ~/Documents/Projects/awmv-n2
pixi run python experiments/n07_sim_tiers.py     # Part A, CPU, minutes
pixi run python experiments/n08_mount_draws.py   # Part B(iii), CPU, minutes
```
