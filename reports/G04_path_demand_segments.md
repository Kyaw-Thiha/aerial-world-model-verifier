# G04 — Path-demand windows on real clips (item 1, preliminary)

Track G2. Written 2026-11-21. Pre-registration `configs/g04_path_demand_prereg.yaml` (commit
`4c6eacf`, before any statistic was computed). Code `experiments/g04_path_demand.py` (`0f2f701`),
`src/verdict/segments.py` (copied unchanged from round 1, `track-g-segments2` `463bfcc`).
Results `results/g04_path_demand/` (`0f2f701`). CPU only, **zero GPU spend**.

**Status: confirmatory against the pre-registration, preliminary against the track.** The
development set the protocol requires — Track I's labelled simulated flights (report I02) — does
not exist: `track-i-simset` carries only the I01 spike, three 3 s clips with full-clip
violations. No design choice below could therefore be validated on labelled data, so every table
is marked **preliminary (dev block, real clips only)** and the two held-out real flights remain
unscored (`flight-13p-trackRATM`, `flight-07a-lemniscate`, frozen in round 1).

## Decision

**Held:** the every-window rule reproduces round 1's published cells exactly, and a
quiet-window test does respond to tilt the path does not require. **Did not hold:** selecting
demand windows changes nothing that matters — it flags exactly the same five real clips as
scoring every window, leaves the false-segment rate identical, loses up to 9 percentage points
of mirrored-roll detection at persistence 2, and makes 15 of 69 admissible clips untestable —
while the quiet test's real-clip false-reject is 25 % (4/16), so neither half of item 1 is
usable as designed on TII.

## Plain-language summary

1. **Demand-window selection is a no-op on this corpus.** Windows whose path requires at least
   15 deg of tilt are 59 % of all windows, yet the five real clips the every-window rule
   false-rejects are *the same five* the demand rule rejects (5/53 versus 5/53 on the matched
   clip set), the false-segment rate is identical to two decimals (2.00/min at one window,
   1.55/min with persistence), and the losing cells are all in detection: mirrored roll at 25 %
   and 50 % coverage drops from 28 to 23 and from 38 to 34 of 53 clips at persistence 2.
2. **The selection has a real cost and no measured benefit**: 15 of 69 admissible dev clips
   (22 %) have no 8-frame window with 6 frames at 15 deg or more, so the demand statistic cannot
   score them at all. The mechanism is visible: the maximum-residual window is a demand window
   in only 47 % of admissible clips, but the windows that actually cross the threshold are
   almost always demand windows, so removing the others changes almost nothing (the flagged
   frame masks differ in 2 of 82 real clips).
3. **The quiet test finds unrequited tilt, but its threshold does not transfer.** On real
   gate-kept clips with a quiet window the false-reject is 25 % (4/16) at persistence 1 and 20 %
   (3/15) at 2, not the nominal 5 %: the failure is a per-flight *level* bias, and it sits
   exactly on the one flight where the frozen reader is least trustworthy. The injected
   unrequited tilt (10 and 17 deg) is found in 16/16 clips — but that is close to mechanical,
   because the injected region *is* the set of quiet windows the statistic is computed on, so
   the IoU of 1.00 carries no information.
4. **Localisation remains the reader's, not the method's.** At a full-clip 4 s mirrored roll
   the segment IoU is 0.48 (round 1: 0.5); partial mirrored roll reaches IoU 0.30 only at
   24 frames (2 s), partial tilt removal only 0.23 at 2 s and 0.33 at the whole clip. The
   reader's ~5.5 deg per-frame spread gives a window median a standard error of ~2.4 deg, so
   nothing shorter than a few windows can be localised at all.
5. **Controls behave**: yaw control and null edit sit at the false-reject count in every arm
   (5/53 each on the matched set), so the demand numbers are not an artifact of the selection.

## Question

Does selecting windows from the path alone — demand windows where the path requires a tilt above
the detection floor, quiet windows where it requires almost none — improve segment detection or
false-segment economy over scoring every window, and does a quiet-window test add the capability
the verdict lacks: detecting tilt the path does not require?

## Setup

**Population.** 100 TII real clips (8 flights), schema-v2 `verdict_clip.npz` from Track D; dev
block = six flights (82 clips, 69 gate-kept by the frozen S3 confidence gate); held-out
`13p-trackRATM` and `07a-lemniscate` reserved and **not scored**. The gate-kept calibration
population is round 1's amendment G01-A choice, reused as prior art.

**Path-demand selection** (from `lean_req`/`tilt_req_deg` only; the video is never consulted):

| term | rule | where the number comes from |
| --- | --- | --- |
| demand window | ≥6 of its 8 frames require ≥15 deg of tilt | midpoint of the published 14-17 deg detection floor (reports 04, 07); 6-of-8 mirrors the frozen usable-frame rule |
| quiet window | all 8 frames require ≤3 deg | below the frozen 10 deg signal floor and below the reader's ~4 deg median error |
| window | 8 frames, hop 4 | round 1's rule, unchanged so the ablation isolates selection |

Measured on all 100 clips before the pre-registration was written: 1,095 windows, 650 demand
(59 %), 176 quiet (16 %), 76 clips with a demand window, 22 with a quiet window (16 gate-kept).
On the dev block the same rule gives 905 windows, 519 demand, 137 quiet, 67 clips with demand and
17 with quiet (16 gate-kept).

**Statistics.** Demand windows: the frozen median residual, i.e. the median per-frame angle
between the depicted and required thrust axis (`src.verdict.segments.frame_residual_deg`, the
round-1 construction). Quiet windows: the median depicted tilt, `asin(|depicted lean|)` — tilt the
path does not ask for. A window abstains below 6 usable frames; the every-window arm additionally
keeps round 1's per-window signal requirement (≥2 frames at ≥10 deg), demand windows satisfy it by
construction, and quiet windows are exempt because the absence of required tilt is the thing being
measured.

**Calibration.** Leave-one-flight-out empirical 95th percentile of the clip's **maximum** over its
eligible windows, fitted on the gate-kept real clips of the fit flights, separately for each arm
and each persistence. A clip with no eligible window is not testable for that arm and is excluded
from that arm's denominator (never counted as a pass). Persistence 1 is primary, persistence 2
(the maximum over consecutive window pairs of their minimum, own calibration) is the
pre-registered secondary.

**Injections.** Partial edits composed on the depicted side exactly as in round 1 (inside a
centred contiguous interval the depicted lean of the same clip's full-clip edit is used), at
10/25/50/100 % of the clip, for tilt removal, mirrored roll, x0.5 and x1.5, with the yaw control
and null edit at every fraction. Plus a **synthetic** unrequited-tilt injection in quiet windows:
`l_dep → l_dep + tan(Δ)·û`, Δ ∈ {10, 17} deg, labelled synthetic in every table.

**Statistics.** Flight-clustered bootstrap, 5,000 replicates, `numpy.random.default_rng(1)`,
6 clusters: **descriptive spreads, not 95 % confidence intervals**. k/n everywhere. The gated
dev block is 4.51 real minutes of footage.

## Results

### 0. Correctness check: the every-window arm reproduces round 1 exactly

| arm (gated, this run) | 10 % | 25 % | 50 % | 100 % | round 1 (gated, same rule) |
| --- | --- | --- | --- | --- | --- |
| removal, P1 | 6/58 | 9/58 | 13/58 | 21/58 | 6/69, 9/69, 13/69, 21/69 |
| removal, P2 | 4/58 | 16/58 | 28/58 | 49/58 | 4/69, 16/69, 28/69, 49/69 |
| mirrored roll, P1 | 11/58 | 19/58 | 31/58 | 48/58 | 11/69, 19/69, 31/69, 48/69 |
| mirrored roll, P2 | 4/58 | 28/58 | 39/58 | 54/58 | 4/69, 28/69, 39/69, 54/69 |

The *counts* match round 1 cell for cell (ungated too: removal P1 7, 9, 10, 12 of 82). The
denominators differ because a clip with no eligible window is now excluded from that arm
(11 gated clips have no window with 2 signal frames), which is what the pre-registration asks
for; round 1 counted them as not flagged. This is the check that the ablation compares the same
statistic.

### 1. Demand selection versus every window (gated, primary)

| statistic | false-reject | testable clips | false segments / real min |
| --- | --- | --- | --- |
| every window, P1 | 5/58 (8.6 %) | 58 | 2.00 |
| every window, P2 | 4/58 (6.9 %) | 58 | 1.55 |
| demand windows, P1 | 5/54 (9.3 %) | 54 | 2.00 |
| demand windows, P2 | 4/53 (7.5 %) | 53 | 1.55 |
| quiet windows, P1 | 4/16 (25.0 %) | 16 | 1.11 |
| quiet windows, P2 | 3/15 (20.0 %) | 15 | 0.67 |
| frozen whole-clip median (reference) | 3/69 (4.3 %) | 69 | — |

On the matched clip set that both residual arms can score (n = 53) the false-reject is
**identical**: 5/53 for both arms at P1 and 4/53 at P2. Demand selection does not change which
real clips are rejected.

**Detection on that matched set** (all | demand):

| family | 25 % P1 | 50 % P1 | 100 % P1 | 25 % P2 | 50 % P2 | 100 % P2 |
| --- | --- | --- | --- | --- | --- | --- |
| tilt removal | 9 \| 9 | 13 \| 12 | 21 \| 20 | 16 \| 16 | 28 \| 26 | 49 \| 49 |
| mirrored roll | 19 \| 18 | 31 \| 29 | 48 \| 48 | 28 \| **23** | 38 \| **34** | 53 \| 53 |
| x0.5 | 5 \| 5 | 5 \| 5 | 5 \| 5 | 6 \| 6 | 10 \| 9 | 21 \| 21 |
| x1.5 | 5 \| 5 | 5 \| 5 | 5 \| 5 | 5 \| 5 | 6 \| 5 | 7 \| 6 |
| yaw control | 5 \| 5 | 5 \| 5 | 5 \| 5 | 4 \| 4 | 4 \| 4 | 5 \| 5 |
| null edit | 5 \| 5 | 5 \| 5 | 5 \| 5 | 4 \| 4 | 4 \| 4 | 5 \| 5 |

The losses that matter are mirrored roll at P2: −5 clips (−9.4 pp) at 25 % and −4 (−7.5 pp) at
50 %. Every other cell is within 2 clips, and the controls are identical between arms.

### 2. Why the selection is a no-op, and what it costs

* The maximum-residual window is a demand window in **47 %** of admissible clips — so selection
  does change each clip's statistic — but the windows that actually cross the threshold are
  demand windows anyway: the flagged frame masks differ between arms in **2 of 82** real clips.
* Cost: **15 of 69 admissible clips (22 %)** have no demand window and cannot be scored by that
  statistic (54 testable versus 58). On TII that is a fifth of the corpus, including every
  near-stationary segment.

### 3. The quiet-window test

Quiet windows exist in 16 gate-kept dev clips (6 of them on `13a-trackRATM`, 5 on `04a`, 3 on
`01a`, 1 each on `01p` and `07p`). The LOFO thresholds are 5.28-7.25 deg of depicted tilt.

| quantity | value | note |
| --- | --- | --- |
| real-clip false-reject, P1 | 4/16 (25.0 %) | nominal 5 %; 4 clusters in the bootstrap, descriptive only |
| real-clip false-reject, P2 | 3/15 (20.0 %) | |
| quiet-window depicted tilt, non-13a flights | 2.9-5.3 deg (9 clips) | 04a 2.9-3.2, 07p 3.2, 01p 4.2, 01a 4.2-5.3 |
| quiet-window depicted tilt, `13a-trackRATM` | 4.1-7.8 deg (6 clips) | the flight the frozen reader is known to fail on |
| synthetic unrequited tilt, Δ = 10 deg | 16/16 detected, IoU 1.00 | **mechanical**: the injected region is the quiet-window set |
| synthetic unrequited tilt, Δ = 17 deg | 16/16 detected, IoU 1.00 | same |

The threshold fails by a per-flight *level* offset of 2-5 deg: on the fit flights the 95th
percentile of quiet-window tilt is 5.3-7.3 deg, and the held-out flight's own quiet tilt sits at
4-8 deg. A level test on a reader whose per-flight bias is of the same size as the violation
cannot work at a nominal 5 % — which is why the frozen verdict carries a ±7 deg mount-nuisance
fit, and why this test, as designed, does not. The 16/16 detection is the statistic responding to
a large injected signal, not evidence that a 10 deg unrequited tilt would be caught in a video
whose reader bias is unknown.

### 4. Localisation and the noise floor

| family, P2 | 25 % (12 frames, 1 s) | 50 % (24 frames, 2 s) | 100 % (48 frames, 4 s) |
| --- | --- | --- | --- |
| tilt removal, IoU (all \| demand) | 0.15 \| 0.15 | 0.24 \| 0.23 | 0.35 \| 0.33 |
| mirrored roll, IoU (all \| demand) | 0.27 \| 0.22 | 0.35 \| 0.30 | 0.48 \| 0.42 |
| onset error, frames (demand) | −2 | −4 | 0 to 4 |

The smallest violation at which localisation reaches IoU > 0.3 is **24 frames (2 s) of mirrored
roll**; partial tilt removal never reaches it, and reaches 0.33 only when the whole clip is
violated. That is consistent with the noise floor and should not be read as a method limit that
tuning could remove: the TII per-frame read error has a trimmed spread of 5-6 deg (report 02), so
an 8-frame window's median residual has a standard error of
`1.25 · 5.5 / sqrt(8) ≈ 2.4 deg`, and the difference of two such medians roughly 3.4 deg; a
violation has to lift a window median by 6-7 deg before it is resolvable at two standard errors,
and 12 frames cannot do that when the violation is a partial removal of an already-uncertain tilt.

## Success criteria, as pre-registered

* **S1, demand selection (fails in part).** P1 cells all lose ≤3.8 pp (pass); at P2, mirrored roll
  loses 9.4 pp at 25 % and 7.5 pp at 50 % (fail, bar was 5 pp); and the "lowers false segments per
  real minute" clause fails outright — the rate is *identical* (2.00 and 1.55 per minute in both
  arms).
* **S2, quiet test (first half fails).** The 5 % false-reject hold does not: it is 25 % (4/16).
  The second half (≥half of quiet clips detect a 17 deg unrequited tilt) passes trivially, 16/16,
  for the structural reason above.
* **S3, localisation (reported).** IoU > 0.3 from 24 frames of mirrored roll, not for partial
  removal; the noise-floor arithmetic beside it.

## What this supports

* Among the six method changes handoff 18 lists, **item 1's demand half is not worth further
  work on TII as specified**: it is a no-op for false rejects and false segments, it costs a
  fifth of the corpus, and it loses mirrored-roll detection at persistence 2. That is a
  pre-registered negative result, not a tuning failure.
* **The quiet half is the useful half, and its obstacle is measurable**: a plain level test
  needs a per-clip notion of the reader's own null tilt, because the per-flight bias (2-5 deg) is
  the same size as the smallest violation worth catching.
* The every-window rule's stability is now independently reproduced (counts identical to round 1
  on two populations), which makes later ablations cheaper to trust.

## What this does not show

* **Nothing confirmatory.** No simulated development set exists, so no number here is a frozen
  evaluation; the two held-out real flights are untouched and remain the confirmatory block.
* **Nothing about the other five items.** FDR control, robust/confidence-weighted statistics,
  boundary refinement, per-clip time alignment and multi-scale windows are specified but not run
  (`configs/g05_round2_design_prereg.yaml`), because the protocol requires design on labelled
  data first.
* **No claim that the quiet test cannot work**, only that *this* form does not: the failure is
  located in the level-transfer step, and a version that estimates the clip's own null from the
  path-side nuisance fit was not tested here.
* **Nothing about ANWM.** Its 1 Hz, 16-point paths cannot support a window statistic; no
  generated clip is scored.
* Nothing about flights: 6 dev flights, 4.51 real minutes, 6-cluster bootstrap spreads.

## Caveats

* **The quiet population is peculiar.** 137 of 905 dev windows are quiet and most come from
  near-stationary segments of `13a-trackRATM`, the flight whose reader fails catastrophically
  elsewhere (round 1: real-clip median residual 76-85 deg on three of its clips). A quiet test
  needs hover-and-tilt footage with a working reader; TII barely has any.
* **Per-arm denominators.** Excluding clips with no eligible window raises the apparent
  false-reject (5/58 = 8.6 % rather than round 1's 5/69 = 7.2 %); both are reported, and the
  matched-set column (5/53) is the fair arm-to-arm comparison. Eleven admissible clips have no
  window with two signal frames at all.
* **The synthetic injection is a depicted-side manipulation**, not a pixel edit, and its IoU is
  1.00 by construction; it is a sensitivity check on the statistic only.
* The `13a` quiet clips' path data says the drone is nearly stationary (required tilt 0.16 deg,
  speed ~0.005 m/s in one clip); whether that is a genuine hold or a path-recovery artefact is
  outside this report.
* Flight-clustered intervals with 6 clusters are descriptive.

## Regenerate

```
PYTHONPATH=. python experiments/g04_path_demand.py
```

Writes `results/g04_path_demand/summary.json` (all blocks, thresholds, per-arm and matched rates)
and `variants.csv` (one row per clip, family, fraction, arm and statistic). Config:
`configs/g04_path_demand_prereg.yaml`. Round-1 comparison: `reports/G01_segment_detection.md` on
branch `track-g-segments2`.
