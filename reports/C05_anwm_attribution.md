# Report C05 — T3 attribution: ANWM generated vs ground-truth frames

Track C. Written 2026-11-10. Answer to item 2 of the follow-up.

## Plain-language summary

On the ANWM clips the extractor works on the **ground-truth frames and fails on the
generated frames**, from the same start, against the same commanded path. That is the
attribution the design was for: the gap between the two runs is the model's contribution,
and it is large — **median acceleration-direction gap 44 deg for Pi3 and 38 deg for VGGT**.

But the sample that can carry the claim is small: at ANWM's 1 s-per-step time base, only
**6 of 25** Regime-A clips have enough tilt (≥10 deg on ≥20 % of samples) to be testable at
all. On those 6, the extractor alone passes the pre-registered criteria on 4, and on the
generated frames it passes **0**.

## Setup

- Clips: ANWM Regime A, 25 clips (`A/A00` … `A/A18` plus `_s1`/`_s2` variants), 16 generated
  frames and 16 same-start ground-truth frames each, 224×224.
- Truth: the **commanded path** from the manifest, `positions[1:17]` (frame *i* depicts the
  pose after the step ending at `positions[i+1]`), at the documented **1 s per step**.
- Differentiation: Savitzky-Golay window 5 steps, polyorder 3, `k_d` = 0.295.
- Extractors: Pi3, VGGT. Criteria unchanged from `configs/pathrec_prereg.yaml`: median
  direction ≤ 30 deg, magnitude ratio in [0.7, 1.4], coverage ≥ 20 % per clip, pass rate ≥ 50 %.

## Results — Regime A (both conditions)

| extractor | condition | median dir err (deg) | median coverage | testable clips | clips passing |
| --- | --- | --- | --- | --- | --- |
| Pi3 | ground-truth frames | 22.7 | 0.08 | 6 / 25 | **4 / 6** |
| Pi3 | generated frames | 68.4 | 0.08 | 6 / 25 | **0 / 6** |
| VGGT | ground-truth frames | 29.0 | 0.08 | 6 / 25 | **4 / 6** |
| VGGT | generated frames | 74.5 | 0.08 | 6 / 25 | **0 / 6** |

Per-clip attribution (the clips where the ground-truth run is best):

| clip | step (m) | Pi3 gt / gen (deg) | VGGT gt / gen (deg) |
| --- | --- | --- | --- |
| A/A11 | 2 | 2.9 / 7.3 | 1.0 / 35.1 |
| A/A14 | 5 | 2.5 / 19.1 | 8.0 / 104.4 |
| A/A04 | 5 | 4.3 / 41.7 | 3.1 / 48.4 |
| A/A12 | 5 | 5.9 / 24.4 | — |
| A/A05 | 5 | 3.6 / 40.9 | 15.8 / 71.3 |
| A/A07 | 5 | — | 14.7 / 131.2 |

Median (generated − ground truth) direction gap: **43.7 deg (Pi3), 38.2 deg (VGGT)**.

## What this supports

- **The extractor is not the limiting factor on these clips.** Fed the same-start
  ground-truth frames, both models recover the commanded path's acceleration direction to
  1–8 deg on the best clips, well inside the pre-registered 30 deg.
- **The generated frames are.** On the identical clips and commanded paths the same
  extractors land at 7–131 deg, and on none of the testable clips do they meet the criteria.
  Under the attribution logic this gap belongs to the model, not to the extractor.
- **The 1 s time base is the binding constraint on coverage**, not the extractor: only 6 of
  25 clips clear the 20 % testable floor, because most commanded paths are short and
  hover-dominated (2–5 m steps at 1 s/step).

## What this does not show

- Nothing about the *physical impossibility* of any generated flight. The commanded path is a
  command; the comparison is between "what the pixels imply" and "what was asked for".
- Nothing about regimes B and C beyond testability counts (reported separately below).
- The ground-truth frames are a level reference by construction (Track B's own note), so the
  "extractor error alone" condition measures against a level camera, which is favourable to
  any extractor: a level camera is easier to track than a banking one.

## Caveats

- 6 testable clips is a pilot, not a rate. The pass/fail counts are reported as counts, not
  percentages, and no confidence interval is claimed.
- Magnitude ratios on the untestable clips are meaningless (some exceed 1e12: their true lean
  is essentially zero, so the ratio is a division by noise). Only the testable clips are read.
- At 1 s/step, two consecutive frames are 5 m apart; the differentiation window spans 5 s of
  flight. A 16-frame clip gives 12 interior samples, so a single bad frame moves the median.
