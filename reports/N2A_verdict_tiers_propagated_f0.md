# N2A — Verdict tiers on the propagated lean over the simulated set (F0, sign, C2, coherence)

**Decision.** F0 on the propagated lean does **not** improve forest removal detection at a matched
pooled false-reject, and it could not: the reader already detects **9/9** forest removals at every
pooled operating point from 5/80 to 10/80, while the deployable reader-start propagated arm
detects **4/9**. What the propagated lean buys instead is **specificity and mirror sensitivity**,
and only with a start that is not the reader: with a 2° external attitude the propagated F0
matches the reader's 9/9 forest removals at 10/80 pooled while flagging **2/26** instead of
**10/26** forest feasible clips and **9/9** instead of 6/9 forest mirrors — a reader start imports
the forest bias (F0 forest feasible median 8.4° against 8.7° for the reader) and therefore does
not buy it. The F0 operating point of every propagated arm is set by the **45° mount drift** (5–6
of 80 feasible clips above 20°), the M04 sign sub-tier cannot be run at a 5–10 % budget (its
feasible scores are bimodal at the ceiling), and C2 and coherence remain negative, reproducing
N05.

Pre-registration `configs/n07_sim_tiers_prereg.yaml` + amendment A, both committed before any
F0/sign/start score existed. Code `experiments/n07_sim_tiers.py`. Results `results/n2/n07/`.
All 200 development clips; **no held-out clip was read**. CPU only, **zero spend**.

*Reading of the brief.* The handoff predates Track N's round 2: N05 (C2 and coherence on the
propagated lean), N06 (the 30/45° drift diagnosis) and the N04 clarification already exist on
`track-n-rotation`. This branch extends those reports rather than repeating them, and the three
N2 reports are numbered N2A–N2C so they do not collide with N05/N06.

## Plain-language summary

1. **What was missing and is now done.** N05 compared two of the three verdict tiers (the
   per-clip fit C2 and the scale-free coherence R) on the propagated lean, and its
   pre-registration explicitly excluded F0. F0 is the fixed-calibration tier — the only one that
   cannot absorb a scene bias and the only one that can see a gain error — so the comparison was
   incomplete. This report adds F0 and the M04 sign sub-tier, adds the two start arms N05 did not
   run (reader start, and a 2°-error start that prices an external attitude source), and
   stratifies by scene family and mount.
2. **The reader is already perfect on forest removals.** At every pooled false-reject budget from
   5/80 to 10/80, F0 on the reader's lean catches 9/9 forest tilt removals. There is no room for
   a detection gain, and the pre-registered "+3 of 9" forest criterion could not be met by any
   arm. That mis-calibration is reported, not repaired. The cost is visible too: the reader spends
   its whole pooled budget in forest, where its false-reject is 5/26 at pooled 5/80, 7/26 at 7/80
   and 10/26 at 10/80.
3. **The propagated arm's threshold is set by the high mounts.** At the matched point the reader's
   F0 cut is 9.3°, the reader-start propagated cut is 18.0°, and the oracle-start cut is 16.4°.
   The reason is a heavy tail: at 45° mount, 5 of 20 feasible clips have an F0 residual above 20°
   (oracle start), and the reader has none at any mount. The drift N06 diagnosed now sets the
   verdict's operating point.
4. **A good start is the dividing line.** A reader start keeps the forest bias
   (F0 forest feasible median 8.4° vs 8.7° for the reader); an oracle start removes it (1.7°); a
   2°-perturbed oracle start removes almost all of it (2.5°). At a pooled 10/80 the 2° arm matches
   the reader on forest removals (9/9), halves its forest false-rejects (2/26 vs 10/26) and beats
   it on forest mirrors (9/9 vs 6/9). The reader-start arm cannot: 4/9 forest removals.
5. **The sign sub-tier does not transfer.** Its feasible scores are 0.0 or 1.0 with nothing in
   between, so a 95th-percentile cut lands on the ceiling and detects nothing; every removal and
   mirror clip scores exactly 1.0. At a fixed 0.5 cut it flags 18/80 feasible (22.5 %) and detects
   15/27 removals and 23/27 mirrors — M04's real-clip number (4/56) does not transfer to a
   low-noise simulated reader.
6. **C2 and coherence reproduce N05, with one new detail.** A reader start recovers C2 to the
   oracle-start level exactly (6/27 removals at 7/80 false-reject, the same as the oracle arm), so
   N05's start problem is real and a small start error is absorbed by the offset fit; but C2 still
   loses to the reader (16/27) because the offset absorbs constant-bank removals.

## Question

At a matched feasible false-reject count, does the propagated lean beat the reader's lean on the
simulated development set once F0 and the sign sub-tier are included, and how much of any answer
is bought by the start attitude?

## Setup

**Population.** All 200 development clips of Track I's frozen split: 80 feasible (the null), 27
tilt-removed, 27 mirrored, 33 scaled (17 ×0.5, 16 ×1.5), 20 partial (8/6/6 at 10/25/50 %), 13
thrust-limited (a path-side label, never pooled with camera defects). Families urban_checker /
forest_ridge / industrial (66/67/67); mounts 0/15/30/45° (50 each, 20 feasible each). The 100
held-out clips were filtered before any file was opened.

**Arms.** The reader is the frozen GeoCalib-224 pinhole lean (L01 settings, stride 2). The
propagated up is `up_k = (R_est(k)^T R_est(t0)) u_cam(t0)` with DPVO draw 1, converted to a lean
by `depicted_lean_from_gravity`. The start `u_cam(t0)` differs by arm:

| arm | start | deployable? |
| --- | --- | --- |
| `reader` | the reader's own per-frame lean | reference |
| `dpvo_level` | `R_cb^T e3` ("level body at t0"), N05's arm | yes (no oracle, no reader) |
| `dpvo_reader_start` | the reader's up at frame 0 (`up_est[0]`) | **yes** (one reader call) |
| `dpvo_oracle` | the true attitude at frame 0 | no — upper bound |
| `dpvo_oracle_2deg` | the oracle up rotated by exactly 2.0° about a random tangent axis; 8 draws, `rng(1)`, clip statistic = median over draws | no — prices a 1–3° external attitude source |

Draw 2 (`scale1_rep2`, which reproduces N05's published draw-2 C2 values exactly) is run for the
reader-start and oracle arms.

**Tiers.**
* **F0** — clip median of the thrust-axis angle between depicted and required lean at the frozen
  required side (schema v3 exact, k_d = 0.295), **no nuisance**. This is the K02/K03 statistic,
  re-implemented from `up_body_from_lean` and `angle_between_deg`; the difference from N05's
  lean-vector angle is a median 0.001–0.03° over the relevant 5–20° range.
* **C2** — N05's per-clip offset + k_d grid fit, reused byte-identically.
* **R** — N05's per-axis correlation, reused byte-identically.
* **S** — the M04 sign sub-tier with the reader-noise dead-band, eps = sin(1.0 × 1.9106°),
  8-frame windows, hop 4.

**Thresholds.** Natural: leave-one-scene-seed-out 95th percentile of the feasible clips of the
other five seeds. Matched: every arm cut at its own (k_ref+1)-th largest pooled feasible score,
with k_ref the reader arm's natural count for that tier — 8/80 for F0, 7/80 for C2. For R and S
the reader's natural count is 0 (a discrete statistic at its ceiling), so amendment A falls back
to the common 7/80. Detections are also reported at fixed pooled 5/7/10 per 80, and in the
descriptive within-stratum curves of `results/n2/n07/curve.csv`.

## Results — F0, the pre-registered primary comparison

### At the matched point (8/80 pooled feasible false-reject for every arm)

| condition | reader | level | **reader start** | oracle | oracle + 2° |
| --- | --- | --- | --- | --- | --- |
| feasible | 8/80 | 8/80 | 8/80 | 8/80 | 8/80 |
| tilt_removed | **22/27** | 1/27 | 13/27 | 13/27 | 14/27 |
| mirrored_roll | 15/27 | 6/27 | 16/27 | **19/27** | **20/27** |
| tilt_scaled | **15/33** | 3/33 | 7/33 | 8/33 | 10/33 |
| partial | 4/20 | 3/20 | 6/20 | 6/20 | 6/20 |
| thrust_limit (path-side) | 13/13 | 11/13 | 13/13 | 3/13 | 3/13 |

The matched thresholds are 9.33° (reader), 44.33° (level), 17.98° (reader start), 16.44° (oracle)
and 16.05° (2°). Every arm is at 8/80 by construction; the detection columns are the comparison.

| stratum | reader | level | reader start | oracle | oracle + 2° |
| --- | --- | --- | --- | --- | --- |
| forest feasible | 8/26 | 3/26 | 3/26 | 2/26 | 2/26 |
| forest removed | **9/9** | 0/9 | 4/9 | 4/9 | 5/9 |
| forest mirrored | 6/9 | 4/9 | 6/9 | **9/9** | **9/9** |
| high-mount feasible | 8/40 | 8/40 | 8/40 | 7/40 | 8/40 |
| high-mount removed | 13/13 | 1/13 | **13/13** | **13/13** | **13/13** |
| high-mount mirrored | 6/13 | 3/13 | 8/13 | 10/13 | **11/13** |

### Why the threshold is high: the 45° tail

Feasible F0 residual, median by mount (degrees):

| arm | 0° | 15° | 30° | 45° |
| --- | --- | --- | --- | --- |
| reader | 1.18 | 1.30 | 1.28 | 2.45 |
| level | 4.94 | 15.69 | 9.11 | 43.65 |
| reader start | 2.05 | 4.20 | 4.55 | 12.00 |
| oracle | 0.48 | 1.56 | 3.11 | 5.47 |
| oracle + 2° | 2.00 | 2.48 | 3.63 | 6.09 |

Feasible clips with F0 > 20°: reader **0/20 at every mount**; oracle **1/20 at 30° and 5/20 at
45°**; reader start the same (1 and 5). Six feasible clips, all high-mount, lift the oracle arm's
8/80 cut from **9.8°** (the 8th-largest of the remaining 74 feasible scores with those six set
aside) to 16.4°, and the reader-start arm's from 15.6° to 18.0°; the reader's cut stays at 9.3°
because it has no such tail. This is the N06 drift, expressed as a verdict operating point.

### At a matched *pooled* false-reject of 10/80 (pre-registered curve point)

| stratum | reader | reader start | oracle | oracle + 2° |
| --- | --- | --- | --- | --- |
| forest feasible | 10/26 | 5/26 | 2/26 | **2/26** |
| forest removed | 9/9 | 4/9 | 7/9 | **9/9** |
| forest mirrored | 6/9 | 6/9 | 9/9 | **9/9** |
| forest scaled | 9/11 | 6/11 | 8/11 | 8/11 |
| high-mount mirrored | 6/13 | 9/13 | 11/13 | **11/13** |
| high-mount scaled | 3/17 | 3/17 | 9/17 | 8/17 |

At the same pooled budget the 2°-start arm matches the reader's forest removals, halves its forest
false-rejects, and beats it on forest and high-mount mirrors. The reader-start arm does not match
them (4/9). **Descriptive, not a pre-registered success.**

### Descriptive within-stratum curve (threshold fitted on that stratum's own feasible clips)

Forest, forest false-rejects 3/26 (the reader's nearest natural point): reader 9/9 removals at an
11.8° cut; oracle **9/9 at 9.8°**; 2° **9/9 at 10.1°**; reader start 4/9 at 18.0°. At 0/26: reader
6/9, every propagated arm 0/9 (its cut is ~47°, the lost 45° clip). At 8/26: all arms 9/9. So the
oracle/2° arms are at least as good as the reader at equal forest false-reject, the reader-start
arm is not, and the difference is the inherited bias.

### Start-error sensitivity (exploratory)

Median F0 by condition, oracle vs 2°-perturbed oracle: feasible 2.30° vs 3.08°, removed 16.18° vs
16.10°, mirrored 25.66° vs 25.58°, scaled 13.52° vs 13.81°, partial 3.17° vs 3.49°. At 45° mount the
feasible medians are 5.5° and 6.1°. **A 2° start error costs about 0.8° on feasible clips and
nothing on the defect conditions**: the start stops being the limiter once it is good to ~2°, and
what remains is the drift. This answers N05's open item.

## Results — C2, coherence and the sign sub-tier

**C2, matched at 7/80.** Reader 16/27 removals, 10/27 mirrors, 12/33 scaled, 5/20 partial. Reader
start 6/27, 7/27, 4/33, 5/20; oracle 6/27, 7/27, 5/33, 6/20; level 0/27. The reader start
reproduces the oracle arm exactly on removals, so a small start error *is* absorbed by C2's offset
fit — N05's level start (10–30° wrong) is not, because it is not a yaw-frame lean offset. C2 still
loses to the reader because the offset absorbs constant-bank removals (N05 §5).

**Coherence, matched at 7/80.** Reader 10/27 removals, 10/27 mirrors; reader start 6/27 and 9/27;
oracle 6/27 and 9/27. Reproduces N05. It does not abstain in this implementation; where a lean
component has zero variance the per-axis correlation is taken as 0 and the statistic saturates at
its ceiling (1.0), which is the N05 finding (60 % of feasible clips hold a constant required lean)
and why the sign sub-tier was added.

**Sign sub-tier, matched at 7/80.** 0/80 feasible and 0 detections on every condition and arm: the
feasible scores are `{0.0: 42, 1.0: 16, ...}` for the reader and `{0.0: 35, 1.0: 28, ...}` for the
reader start, so a 95th-percentile cut is the ceiling (1.0) and nothing is strictly above it. Every
removal and mirror clip that is scoreable scores exactly 1.0. At a **fixed** cut of 0.5 (M04's own
sim comparison): reader 18/80 feasible, 15/27 removed, 23/27 mirrored; reader start 28/80, 19/27,
23/27; 2° 23/80, 22/27, 23/27. The tier separates only at a ~22–29 % false-reject. **M04's real-clip
operating point (4/56 at a deadline tied to the reader noise) does not transfer**, because the
simulated reader is quiet enough that a fully removed tilt has no sign-bearing frame left.

## Yaw control and the null

The simulated split has **no yaw-warp family** and a yaw warp cannot be constructed without
re-rendering; a world-yaw warp is analytically invisible to the propagated lean (the up vector is
invariant), so its tier score would equal the feasible score by construction — a statement, not a
measurement. The measured yaw control is the UZH one (N02b): reader F0 2/6 and propagated F0 2/6 on
6 clips, and propagation did not remove the reader's view dependence. The **null** is the 80
feasible clips; there is no separate null-edit rendering for the simulated set. Every detection
table above prints the feasible (null) rate beside it.

## Pre-registered criteria, scored

| criterion | result | numbers |
| --- | --- | --- |
| `primary_forest` (F0, reader start, matched) | **FAIL** | Δremoved 4/9 − 9/9 = **−5** (needs ≥ +3); Δforest FR 3/26 − 8/26 = −5 (needs ≤ 0) |
| `primary_high_mount` (F0, reader start, matched) | **PASS** | ΔFR 8/40 − 8/40 = 0 (needs ≤ +2); Δremoved 13/13 − 13/13 = 0 (needs ≥ 0) |
| `upper_bound_forest` (F0, oracle, matched) | **FAIL** | Δremoved 4/9 − 9/9 = −5 (needs ≥ +4) |
| Part D gate (any deployable arm, any tier, forest) | **not met** | C2 forest 1/9 vs 6/9; R forest 0/9 vs 0/9; S 0 everywhere |

The forest criterion was written expecting the reader not to be saturated. It was saturated
(9/9 at every pooled point), so no arm could meet a "+3" criterion. That is a mis-calibration of
the criterion, not of the arms; it is reported as a fail and the substantive reading is left to
the pre-registered curve and the descriptive within-forest curve. **Part D is therefore not run**,
as pre-registered.

## What this supports

* **The propagated lean does not improve F0 detection on the simulated set at a matched pooled
  false-reject.** The reader's F0 is already at 9/9 forest removals and 22/27 overall, and the
  propagated arm's cut is set by a 45° feasible tail the reader does not have.
* **The gain that exists is specificity and mirror sensitivity, and it needs a non-reader start.**
  A 2° external attitude gives 9/9 forest removals at 10/80 pooled with 2/26 forest false-rejects
  and 9/9 forest mirrors; a reader start gives 4/9 and 5/26, because it imports the forest bias.
* **The high-mount criterion passes trivially** (both arms 13/13 at 8/40), but the within-stratum
  curve shows the reader is better at a 10 % budget (13/13 vs 9/13) — the pass is not evidence of a
  propagation advantage.
* **The sign sub-tier is not a usable verdict tier on sim** at a 5–10 % budget; it is a ~25 %
  false-reject detector.
* **C2 and coherence remain negative** (N05 reproduced), and a reader start recovers C2 to the
  oracle level, confirming the start rather than the propagation was C2's problem.

## What this does not show

* Whether fixing the 45° drift (Part B) would unlock the forest detection gain. The tail analysis
  predicts it would — the oracle arm with a low cut is at 9/9 forest removals by a 9.8° threshold —
  but that is a prediction, not a measurement.
* Anything about a *deployable* non-reader attitude source: the 2° arm is an oracle with noise, not
  an IMU. It prices the requirement; it does not demonstrate hardware.
* Anything about TII (excluded by instruction) or the held-out simulated clips (not read).
* A gain-error reading at a per-clip threshold: the scaled edits are reported at the pooled point
  only. F0 does improve both arms' scale detection over C2 (reader 15/33 vs 12/33, propagated 7/33
  vs 4/33), but the reader still leads.

## Caveats

* Cells are 2–7 clips (forest 9 removals, high mount 13). Every number is k/n; no rate is given
  without its denominator, and the six-seed cluster count makes any interval descriptive.
* The pooled matched point is one global operating point; the within-forest curve is descriptive
  and would need a stratum-aware deployment to be usable.
* The matched point equalises *pooled* false-rejects, not per-family ones; that is the explicit
  pre-registered rule and it is what makes the forest comparison unfavourable to the propagated
  arm.
* DPVO is not bit-reproducible. The F0 counts are identical on draw 2 (reader start 13/27,
  mirror 16/27, scaled 7/33; oracle 13/27, 19/27, 8/33), so the conclusion is draw-robust.
* F0's exact axis angle and N05's lean-vector angle differ by < 0.1°; C2/R reuse N05's function, so
  cross-tier comparisons carry that sub-0.1° inconsistency.
* The 2° arm is a median over 8 perturbations; a single perturbation gives noisier numbers but the
  same reading.

## Provenance

* Pre-registration `configs/n07_sim_tiers_prereg.yaml`, amendment A
  `configs/n07_sim_tiers_prereg_amend_a.yaml`; decisions `[N2]`.
* Inputs: `data/n05/{gt,scores}`, `data/n05/poses/sim200/{scale1,scale1_rep2}` (read-only symlinks
  to the Track N caches); the Track I manifest staged at `results/n2/inputs/manifest.csv`
  (sha256 `50aacbb2…ce6794`).
* Code `experiments/n07_sim_tiers.py`; results `results/n2/n07/{per_clip,census,curve}.csv`,
  `summary.json`. Camera-side computed from the 250 Hz truth and the stored reader/pose arrays.
* Compute: local CPU, about 20 s, **USD 0.00**. No Modal app, no volume, no R2 write.
