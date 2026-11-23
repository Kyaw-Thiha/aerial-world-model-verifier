# G07 — Frozen evaluation of the G3 segment design on the held-out set

**Track G3.** Branch `track-g3-segments`, worktree `~/Documents/Projects/awmv-g3`. Run once on
2026-11-23 with `configs/g06_eval_frozen.yaml` (committed `3023889`, unedited), driver
`experiments/g07_heldout.py` (`759b01f`), scores `results/g07/` (`100/100` clips, one pass, no
re-scoring). Intervals are scene-seed-clustered (sim, 6 clusters) or flight-clustered (real, 2
clusters): **descriptive spreads, daggered everywhere**, never 95 % confidence intervals.

## Decision

**Held:** only boundary refinement (S4, +0.172 median IoU on the three flagged clips), because the
pre-registered primary cannot be evaluated on the cells it was written for — the 12-frame
coherence scan has an eligible window on **9 of 40** null clips and on **0 of 1** clip in each
25 %-coverage cell, so S1, S2 and S6 are **not evaluable** and 25 %-coverage detection has no
denominator at all. **Did not hold:** the max-over-windows residual scans detect 1/13 full-clip
removals and 2/11 mirrored rolls at a 3.1 % null, while the within-clip Benjamini–Hochberg rule on
the same residual statistic detects **8/13** and **10/11** at the same 3.1 % null (but with a
realised false-discovery proportion of 0.38 on partial clips), and the multi-scale joint rule
flags **10/10** null clips — the failures are thresholds that do not transfer across scenes, not
the window statistic itself.

## Setup

- **Populations.** 100 held-out simulated clips (`field_settlement`, `coastal_cliff`, seeds
  201-206): 40 feasible (the null control), 13 full-clip tilt removals, 13 full-clip mirrored
  rolls, 17 full-clip tilt scalings, 7 thrust-limit clips, and 10 partial-interval clips — 4 at
  10 %, 3 at 25 %, 3 at 50 %, one per defect type per fraction. The 18 real held-out clips of
  `flight-13p-trackRATM` and `flight-07a-lemniscate`, of which **3 are gate-kept and testable**.
- **Thresholds, fitted on dev only, applied unchanged.** Simulated: empirical 95th percentile on
  all 80 feasible dev clips — `coh12_p1` 1.9345, `coh12_p2` 1.8741, `coh8_p1` 1.9698, `res12_p1`
  61.363, `res8_p1` 75.124, `res8_p2` 66.581, `sign12_p1` 1.0000, whole-clip median 11.179 deg,
  `a5_joint` 4.254, `a4_lag_coh12` 1.937. Real: empirical 95th percentile on the 69 gate-kept dev
  clips — `coh12_p1` 1.694, `coh8_p1` 1.748, `res8_p1` 32.879, `res12_p1` 24.144, `sign12_p1`
  0.692, whole-clip median 12.549 deg, `a5_joint` 4.345.
- **Arms.** Exactly the 13 arms of the frozen recipe; nothing was added or dropped.
- **Controls.** Simulated block: the **null** is the unchanged feasible clip, beside every rate.
  The recipe states the yaw slice is a dev instrument and is **not re-rendered for held-out
  scenes**, so there is **no yaw control on the simulated block** — stated, not omitted. Real
  block: both Track D controls (`yaw_control`, `null_edit`) appear beside every real rate.
- **Provenance and the two deviations, disclosed.** (i) The frozen recipe's `arms_evaluated`
  block is not valid YAML (nested braces in a flow mapping); the driver reads it line by line
  against a name→knobs table cross-checked against each line, and the frozen file is unedited.
  (ii) The driver's real-block path has no held-out toggle, so the dev smoke executed the real
  held-out block once before the simulated scores existed; nothing changed afterwards and the
  path is deterministic, so the real numbers are that first execution's.
- **Verification against G06 (dev only, before the held-out scores existed).** The driver's
  all-dev thresholds reproduce G06's `all_dev` values exactly (`coh12_p1` 1.9345, `coh12_p2`
  1.8741, `coh8_p1` 1.9698) and the `coh12_p1` dev false-reject is 1/17 as in G06.

## Results

### Simulated held-out scenes

**Testability is the first result.** The coherence tier needs ≥9 usable frames and non-trivial
lean variance in a 12-frame window, and on these scenes it abstains almost everywhere:

| arm | testable null clips | false-reject | false segments / min |
| --- | --- | --- | --- |
| **coh12_p1** (primary) | **9 / 40** | **1/9 (11.1 %) †** | 0.40 |
| coh12_p2 | 8 / 40 | 0/8 (0.0 %) † | 0.00 |
| coh8_p1 | 5 / 40 | 0/5 (0.0 %) † | 0.00 |
| res12_p1 | 32 / 40 | 0/32 (0.0 %) | 0.00 |
| res8_p1 (comparison) | 32 / 40 | 1/32 (3.1 %) † | 0.24 |
| res8_p2 | 32 / 40 | 0/32 (0.0 %) | 0.00 |
| sign12_p1 | 9 / 40 | 0/9 (0.0 %) † | 0.00 |
| whole-clip median (baseline) | 40 / 40 | 0/40 (0.0 %) | 0.00 |
| a2_bh_coh12_q10 | 9 / 40 | 2/9 (22.2 %) † | 1.21 |
| a2_bh_res8_q10 | 32 / 40 | 1/32 (3.1 %) † | 0.24 |
| a5_joint | 10 / 40 | **10/10 (100 %) †** | 3.60 |
| a4_lag_coh12 | 8 / 40 | 1/8 (12.5 %) † | 0.97 |
| a3_refined | 9 / 40 | 1/9 (11.1 %) † | 0.40 |

† fewer than ten clusters (6 scene seeds) or fewer than ten testable clips; the `coh12_p1` null
interval is [0.00, 0.50] and `res8_p1`'s [0.00, 0.05], both descriptive.

**Detection, primary arm and baseline, with the testable denominator printed** (k / testable,
clips in the cell in brackets; IoU is the median over testable clips with undetected counted as 0):

| cell | primary `coh12_p1` | whole-clip median |
| --- | --- | --- |
| full-clip tilt removal | 0/1 [1] — IoU 0.00 | 8/13 [13] — IoU 0.998 |
| full-clip mirrored roll | 2/2 [2] — IoU 0.185, onset +20 f | 10/13 [13] — IoU 1.000 |
| full-clip tilt scaling | 0/3 [3] — IoU 0.00 | 8/17 [17] — IoU 0.00 |
| thrust limit (family with no power claimed) | 0/0 [7] | **7/7 [7] — a false reject of physical clips** |
| partial removal, 10 % | 0/0 [2] | 1/2 [2] — IoU 0.05, onset −290 f |
| partial removal, 25 % | **0/0 [1]** | 1/1 [1] — IoU 0.251, onset −314 f |
| partial removal, 50 % | 0/0 [1] | 1/1 [1] — IoU 0.498, onset −35 f |
| partial mirror, 10 % | 0/0 [1] | 0/1 [1] |
| partial mirror, 25 % | **0/0 [1]** | 0/1 [1] |
| partial mirror, 50 % | 1/1 [1] — IoU 0.04, onset +177 f | 1/1 [1] — IoU 0.499 |
| partial scaling, 10 % | 0/1 [1] | 0/1 [1] |
| partial scaling, 25 % | **0/0 [1]** | 0/1 [1] |
| partial scaling, 50 % | 0/0 [1] | 0/1 [1] |

Three things this table says. (i) **Every 25 %-coverage figure has zero testable clips** — the
cells hold one clip each and the coherence tier abstains on it; that is why S1/S2 are not
evaluable rather than failed. (ii) The baseline's IoU on partial edits is bounded at 0.251 /
0.498 / 0.050 by construction: it flags the whole clip, so its IoU *is* the interval fraction, and
its onset errors are −290 to −314 frames, i.e. it points at the clip start. (iii) The primary's
only testable partial detection (mirror, 50 %) localises nothing (IoU 0.04, onset +177 frames).

**The one arm that recovers the residual tier is the within-clip FDR rule** — same windows, same
statistic, different decision:

| cell | `a2_bh_res8_q10` (q = 0.10) | `res8_p1` (max-over-windows) |
| --- | --- | --- |
| full-clip tilt removal | **8/13** — IoU 0.832 | 1/13 — IoU 0.00 |
| full-clip mirrored roll | **10/11** — IoU 0.963 | 2/11 — IoU 0.00 |
| full-clip tilt scaling | 5/13 — IoU 0.00 (detected-only 0.952) | 2/13 — IoU 0.00 |
| partial removal, 25 % | 1/1 [1] — IoU 0.366 | 0/1 [1] |
| partial removal, 50 % | 1/1 [1] — IoU 0.646 | 0/1 [1] |
| partial mirror, 50 % | 1/1 [1] — IoU 0.730 | 0/1 [1] |
| null | 1/32 (3.1 %) † | 1/32 (3.1 %) † |
| realised false-discovery proportion, partial clips | **0.383** (820 flagged frames) | undefined (0 flagged frames) |

The max-over-windows thresholds are 66-75 deg of window residual, fitted on dev feasible clips;
no held-out window reaches them except at the thrust limit. The BH rule compares each window with
its dev distribution instead, and fires. Its cost is measured too: 38 % of the frames it flags on
partial clips are outside the true interval, which is the S3 failure below.

**Ablations.** `a4_lag_coh12` (per-clip lag): 1/8 null †, mirror full-clip 2/2 with IoU 0.166
vs 0.185 without the lag, and false segments 0.97/min vs 0.40 — the lag makes the primary worse
on every measured quantity. `a3_refined`: the only arm that improves on its comparator, median
IoU 0.153 → 0.325 over the 3 flagged defect clips (+0.172) † (a fourth flagged clip is
the null clip, unchanged at 0). `a5_joint`: standardised across lengths
8/12/16 and calibrated at 4.254, it flags **10 of 10 testable null clips** and still localises
nothing (IoU 0.025-0.053, onsets 152-322 frames): its dev standardisation does not transfer to
unseen scenes at all.

**Minimum localisable violation: not estimable on this block.** No cell has more than two
testable clips, the largest partial testable cell is n = 1, and the best median IoU among
testable partial detections is 0.04 (primary) — so the contour the pre-registration asks for
cannot be drawn here, and no localisation claim is made. The noise-floor arithmetic that fixes
the floor is unchanged from G06: a window median's standard error is ~2.4 deg at 5-6 deg per-frame
spread, so nothing under a few windows is localisable.

### Real held-out flights

Three of the 18 clips are gate-kept and testable (`13p` 3, `07a` 0); every rate below is k/3.

| condition | `coh12_p1` (tau 1.694) | whole-clip median (tau 12.549 deg) |
| --- | --- | --- |
| real | **0/3** | 2/3 (two false rejects) |
| null edit | **0/3** | 2/3 |
| yaw control | **0/3** | 2/3 |
| tilt removal | 3/3 — IoU 0.51 | 3/3 — IoU 1.00 |
| mirrored roll | 3/3 — IoU 0.75 | 3/3 — IoU 1.00 |
| x0.5 | 0/3 | 2/3 |
| x1.5 | 1/3 | 2/3 |
| removal 10 / 25 / 50 % | 0/3 / 2/3 (IoU 0.33) / 3/3 (IoU 0.60) | 2/3 / 3/3 (0.25) / 3/3 (0.50) |
| mirror 10 / 25 / 50 % | 1/3 / 1/3 / 3/3 (IoU 0.50) | 2/3 / 2/3 (0.25) / 3/3 (0.50) |

On these three clips the coherence primary behaves better than the baseline in exactly the way
the design intended — no false reject on the real clip or on either control, where the baseline
rejects two of three — and reaches full detection at 50 % coverage with IoU 0.50-0.60. Three
clips is not a rate; it is a direction, daggered.

## Success criteria, as pre-registered, evaluated once

| criterion | outcome | evidence |
| --- | --- | --- |
| **S1** ≥50 % of 25 %-coverage removal and mirror, IoU > 0.3 | **not evaluable** | 0 testable in both cells (1 clip each); the coherence tier abstains on both |
| **S2** ≥+10 pp over the 8-frame residual at matched false-reject | **not evaluable** | same cells; and the matched rule is undefined because the baseline's held-out false-reject is 0/40 (α = 0) |
| **S3** BH q = 0.10: realised FDP ≤ 0.20, false segments ≤ +0.5/min vs the max rule | **failed** | `a2_bh_coh12_q10` 1.21/min vs 0.40 (+0.81, over the 0.5 bar); its FDP clause is not evaluable (0 flagged frames on partial clips). For information `a2_bh_res8_q10` has FDP 0.383 (> 0.20) and 0.24/min |
| **S4** median IoU improves ≥0.05 | **met** | +0.172 over 3 flagged clips † |
| **S5** false segments fall ≥20 % or IoU +0.05 | **failed** | 0.40 → 0.97/min with the lag (+139 %); IoU delta 0.000 |
| **S6** joint ≥+5 pp over the best single length at matched false-reject | **not evaluable** | 0 testable at the cells; and `a5_joint` flags 10/10 null clips, so it has no usable operating point |

**Matched-false-reject coverage, stated because the recipe asks for it.** On the simulated block
the matching rule is *undefined*: the baseline's held-out false-reject is 0/40, so there is no
empirical false-reject to match and the quantile at α = 0 does not exist; nothing was substituted.
On the real block the driver computed nominal-threshold rates only — matched thresholds were
implemented for the simulated block and not for the real one, a gap in the driver found after the
run and left as it is rather than patched post hoc.

## What this supports

* **A negative, attributable result about transfer.** Every failure co-locates: thresholds fitted
  on three dev scene families (urban checker, forest ridge, industrial) do not transfer to two
  unseen families (field settlement, coastal cliff). The residual max-over-windows thresholds
  (66-75 deg) and the multi-scale joint threshold flag nothing or everything; the null rate of
  the multi-scale rule is 10/10.
* **A positive, measured result for item 3 that the dev pass could not see.** The same residual
  windows, decided by within-clip BH at q = 0.10 instead of a global conformal threshold, detect
  8/13 removals and 10/11 mirrored rolls at a 3.1 % null — with a realised FDP of 0.38 on partial
  clips, which is why S3 still fails as written. Item 3 is the only mechanism in the frozen set
  that survived contact with unseen scenes.
* **A testability finding that outranks the detection numbers.** The coherence tier's variance
  floor makes it eligible on 22 % of null clips and on none of the 25 %-coverage partial clips;
  any future version must either lower the floor with a stated cost to its null rate or report
  the testable denominator as its headline.
* **Boundary refinement works where a region exists** (+0.17 IoU) and **per-clip lag does not**
  (worse on every measured quantity, with 3 testable clips).

## What this does not show

* **Not a refutation or a confirmation of S1/S2/S6.** They are not evaluable: 25 %-coverage
  detection has 0 testable clips and the matched-false-reject rule has no denominator. Reporting
  them as "failed" would misstate the evidence; they are void.
* **Nothing about localisation.** No cell has enough testable clips for the minimum-localisable-
  violation contour; the best testable partial IoU is 0.04.
* **Nothing about the yaw control on simulated data.** The frozen recipe does not re-render the
  yaw slice for held-out scenes, so the simulated block has a null control only.
* **Nothing about the real block beyond a direction.** Three testable clips, two flight clusters.
* **Nothing about thrust-limit clips as a detection rate.** The family is physical by
  construction and the machinery has no power on it; the baseline's 7/7 flag rate there is a
  false reject of physical clips, not detection.
* **No claim that BH is ready.** Its realised FDP on partial clips (0.38) is nearly twice the
  pre-registered bar.

## Caveats

* **Scene clustering.** Six held-out scene seeds; every interval is a descriptive spread. The
  bootstrap intervals printed for the headline cells (`coh12_p1` null [0.00, 0.50]) show the
  spread, not a confidence statement.
* **Partial-cell size.** The held-out set was frozen by Track I with 4/3/3 partial clips; one per
  defect type per fraction. Any partial-interval rate here is a count of one to four clips.
* **Real population.** 18 held-out clips, 3 gate-kept: the gate removes 15, mostly `07a`.
* **Frame-rate asymmetry.** Window lengths are frames: 12 frames is 0.40 s at the simulated 30 fps
  and 1.0 s at the real 12 Hz. The same arm therefore means different durations on the two blocks;
  the frozen recipe fixes frames, so this is reported, not corrected.
* **Control truth.** Control compositions carry an empty truth mask, so their rows are false-flag
  rates, not IoU.

## Reproduce

```
# scores (done once, 2026-11-23; per-clip idempotent, resumes by re-running)
MODAL=... ./scripts/g07_score_heldout.sh
# pull + the single frozen evaluation
MODAL=... PY=... ./scripts/g07_pull_and_run.sh
# outputs: results/g07/{sim_clips,sim_arms,sim_false_reject,sim_fdp,success_criteria,
#                       real_clips,real_arms}.csv, sim_summary.json
```
