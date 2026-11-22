# M01 — Gate A: is there enough required lean to give a verdict?

Track M, Stage 1. Written 2026-11-22. Branch `track-m-gates`. Pre-registration
`configs/m01_gate_a_prereg.yaml` (commit `ee3aa56`, written before any threshold was fitted),
frozen thresholds `configs/m01_gate_a_frozen.yaml` (written before any evaluation set was
scored). Code `src/verdict/{gate_a,selective}.py`, `experiments/m01_gate_a.py`. Results
`results/m01/`. CPU only, **zero GPU spend**.

## Decision

**A path-side signal gate does not deliver the abstention guarantee the handoff asked for: no
signal threshold reaches a 5 % yaw-control flag rate at any coverage above 6 %, and the
conformal risk-control selection therefore fails on the pre-registered rule and is reported as
failed rather than tuned.** The gate does its narrower job — it makes the 20 % of gate-kept
clips where the coherence tier cannot be computed at all an explicit abstention instead of a
silent non-detection, and it exposes that the yaw-control failure is a property of the tier and
the reader, not of the path: under the exact required lean the yaw warp adds a systematic
**+0.61** to the coherence score on **45 of 51** clips, uncorrelated with signal strength.

## Plain-language summary

1. **The yaw-control count is reconciled, and there were three different numbers because there
   are three different constructions.** `9/69` (J03) is the natural 95th-percentile threshold on
   schema v2's *drag-free* required lean. `16/69` is the same threshold on the *exact*
   `k_d = 0.295` required lean — this is Track K's own `R/window` row in
   `results/k03/20261122T194910Z/A_table.json`, and this track reproduces it clip for clip on
   false-reject, yaw control, removal, mirror and null. `23/69` (K03's headline read) is the
   same exact tier with the threshold *lowered* to match F0's 5/69 false-reject
   (per-arm LOFO α = 0.088). None of the three is wrong; they are not the same statistic.
2. **Why exact physics is worse here.** Adding the drag term changes the *shape* of the required
   lean (a velocity-proportional component that rotates with the path). The yaw control rotates
   the depicted lean by 2.3–2.6° at the median clip (p90 of clip medians 6.9°); against a
   required lean of σ ≈ 3–4° that rotation moves coherence from a median 0.87 to 1.29 under the
   drag-free side (+0.20) and from 0.49 to 1.23 under the exact side (+0.61). A statistic that
   loses half its margin to a tilt-preserving warp cannot be rescued by asking whether the path
   had enough signal.
3. **The signal statistic is right in theory and uninformative in practice.** The governing
   quantity for `1 − min(r_x, r_y)` is the smaller per-axis spread `sigma_min`. On the gate-kept
   development clips it runs from 0.14° to 24.4° (medians per flight: 04a 0.14, 13a 0.17, 01a
   0.49, 01p 3.24, 07p 4.04, 16p 24.39). But its rank correlation with the coherence damage the
   yaw control causes is **−0.01**: the damage is additive and systematic, so no threshold on it
   separates the clips that flip from the ones that do not. The risk–coverage curve is
   consequently non-monotone (yaw-control rate 21/65 at full coverage, 23/50 at λ = 2°, 16/22 at
   λ = 3.82°, 0/6 at λ = 5.5°).
4. **What the gate does buy.** At the middle pre-registered SNR threshold, `sigma_min ≥ 2 × 1.91
   = 3.82°` plus the near-vertical rule, 22 of 69 development clips are accepted (31.9 %): 43 are
   `weak_signal`, 4 are `near_vertical`, none both. Among the 22 accepted, false-reject is 1/22
   and the null edit 1/22, while detection stays high (removal 21/22, mirror 22/22, x0.5 15/22,
   x1.5 4/22) — the gate does not buy its false-reject by discarding the detections, unlike the
   ungated tier where 14 clips simply have no value. The price is the yaw control: **16/22**.
5. **The signal reading depends on which noise number you use, and that is the honest caveat.**
   The robust per-axis reader noise is 1.07° / 1.91° (MAD); the RMS is 8.1° / 14.2° because the
   racing flights carry 30–45° tails. The SNR argument uses the robust value because the
   statistic is a median-type one; with the RMS the same k would demand 16–28° of `sigma_min` and
   nothing would survive. Both readings are stated.
6. **Two yaw-invariant statistics were tried and both failed (exploratory, not pre-registered).**
   A complex correlation `|Σ l_dep conj(l_req)| / √(Σ|l_dep|²Σ|l_req|²)` is exactly invariant to a
   global azimuth rotation, and a magnitude correlation `corr(|l_dep|, |l_req|)` is invariant to
   it too; on the same clips they flag the yaw control at 6/69 and 6/69 — but they also lose the
   detections (removal 5/69 and 7/69 against the per-axis tier's 54/69). The invariance is
   bought with the discrimination, which is why a *gate* cannot fix this and the report says so.

## Question

Does the path's required lean carry enough signal for the scale-free coherence tier to be
meaningful, and can a path-side abstention gate make the tier's false-reject and yaw-control
rates acceptable without throwing away the population it can judge?

## Setup

**Population.** 100 TII real clips (8 flights, schema-v2 clip files), the frozen S3 gate keeping
72; the six development flights give 69 and `13p` / `07a` are held out and were **not scored**.
Conditions: identity (`real`), `tilt_removal`, `mirror_roll`, `scale_0.5`, `scale_1.5`,
`yaw_control`, `null_edit`, plus the three analysis-only delay conditions reported separately.

**Required side.** Schema v3 exact: the unit thrust axis `unit(a + g e3 + 0.295 v)` computed at
the native 274.73 Hz mocap rate (Savitzky–Golay, 101 samples, polyorder 3) and sampled at the
clip's frame times, expressed in the clip's path-yaw frame. Airframe model M0 and the kinematics
are imported byte-identically from `track-k-calibration` at `3110976`
(`src/airframe/{models,kinematics,schema_v3,fit}.py`, sha256 in `docs/decisions.md`). Never
`lean_req + (k_d − 0.295) vel_req`.

**Two reproductions before use.**

| check | target | this track | outcome |
| --- | --- | --- | --- |
| J03's R arm, v2 side, natural 95 % LOFO (69 clips) | fr 3, yaw 9, null 3, removal 54, mirror 55, x0.5 6, x1.5 11 | identical, all seven | matches |
| F0 clip threshold, exact side, gate-kept dev | 8.87° (K02, K03) | 8.8707° | matches |

**Statistics.** `sigma_min = min(std(l_x), std(l_y))` over the clip's finite required-lean frames,
in degrees; `sigma_p` (principal-axis spread), the tilt range and the near-vertical fraction are
reported beside it. The gate is `accept iff sigma_min ≥ λ and tilt_max ≤ 80°`, with reason codes
`weak_signal` / `near_vertical`. Conformal risk control on the two risks (coherence false-reject,
yaw-control flag rate), bounded by `(k+1)/(n+1)`, α = 0.05, with the threshold fitted on five
flights and applied to the sixth (nested LOFO within development). Intervals are flight-clustered
bootstraps (5,000 draws, `default_rng(1)`); with six clusters they are **descriptive**.

**Reader noise.** Per-axis MAD × 1.4826 of (depicted − required) on the development flights' real
clips: 1.07° / 1.91°, `sigma_n = 1.91°` (3,921 frames). Candidate k = 1.5 / 2 / 3 give
2.87° / 3.82° / 5.73°.

## Results

### 1. Signal by flight (gate-kept development clips, medians)

| flight | clips | sigma_min ° | sigma_p ° | tilt med ° | tilt max ° |
| --- | --- | --- | --- | --- | --- |
| flight-01a-ellipse | 5 | 0.49 | 5.31 | 0.18 | 20.4 |
| flight-01p-ellipse | 26 | 3.24 | 8.91 | 23.7 | 39.9 |
| flight-04a-ellipse | 6 | 0.14 | 0.18 | 0.16 | 0.63 |
| flight-07p-lemniscate | 25 | 4.04 | 9.72 | 23.7 | 36.2 |
| flight-13a-trackRATM | 6 | 0.17 | 0.25 | 0.96 | 1.51 |
| flight-16p-trackRATM | 1 | 24.39 | 32.24 | 57.3 | 123.2 |

Two flights carry almost no per-axis signal at all (04a, 13a: `sigma_min` 0.14–0.17°), the
piloted ellipse/lemniscate sit at 3–4°, and the single surviving racing clip has plenty of tilt
but passes through 123° where a lean vector is not a valid representation.

### 2. Risk–coverage (development, nested-LOFO thresholds; 69 clips)

| λ ° | accepted | coverage | false-reject | yaw control | null edit |
| --- | --- | --- | --- | --- | --- |
| 0 (near-vertical rule only) | 65 | 0.94 | 4/65 | 21/65 | 4/65 |
| 2.0 | 50 | 0.72 | 5/50 | 23/50 | 5/50 |
| 3.0 | 32 | 0.46 | 4/32 | 18/32 | 4/32 |
| 3.75 | 22 | 0.32 | 1/22 | 16/22 | 1/22 |
| 4.5 | 11 | 0.16 | 1/11 | 2/11 | 1/11 |
| 5.5 | 6 | 0.087 | 0/6 | 0/6 | 0/6 |

The curve is non-monotone in both risks, which is the diagnostic: the gate is not ordering the
clips by risk. Conformal risk control therefore reports **no threshold reaches the target** — the
yaw bound is 0.333 at full coverage and the false-reject bound 0.076 — and the pre-registered
selection outcome is `failed`.

### 3. The three candidate SNR thresholds (development)

| k | λ ° | accepted | false-reject | yaw | null | removal | mirror | x0.5 | x1.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.5 | 2.87 | 35/69 | 4 | 19 | 4 | 34 | 35 | 14 | 7 |
| 2.0 | 3.82 | 22/69 | 1 | 16 | 1 | 21 | 22 | 15 | 4 |
| 3.0 | 5.73 | 4/69 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Detection is quoted among accepted clips and the yaw control and null edit sit beside it, as the
protocol requires. At k = 3 the gate keeps four clips and the tier detects nothing on them: the
coverage collapses before the yaw rate does, which is the shape of the failure.

### 4. Where the yaw damage comes from (v2 versus v3, 51 gate-kept dev clips under 80°)

| construction | real coherence (median) | yaw-control coherence (median) | Δ median | clips with Δ > 0 | Spearman(Δ, sigma_min) |
| --- | --- | --- | --- | --- | --- |
| v2 drag-free required lean | 0.870 | 1.292 | +0.197 | 39/51 | +0.049 |
| v3 exact `k_d = 0.295` | 0.485 | 1.226 | +0.608 | 45/51 | −0.009 |

The null edit is bit-identical to the real condition on every clip (Δ = 0.000 exactly, as it must
be: the null edit is the identity warp), so the +0.61 is the yaw rotation itself. The yaw warp's
effect on the depicted lean is a median 2.55° (p90 of clip medians 6.88°), which is the same order
as the required lean's per-axis spread on the clips the gate accepts.

### 5. The three published yaw-control counts, reconciled

| source | required side | threshold | yaw-control flags |
| --- | --- | --- | --- |
| J03 `R/window`, natural | v2 drag-free | per-flight 95th percentile (median 1.75) | 9/69 |
| K03 `R/window`, natural | exact `k_d = 0.295` | per-flight 95th percentile (median 1.56) | 16/69 |
| K03 `R/window`, matched | exact | per-arm LOFO α = 0.088 to match F0's 5/69 | 23/69 |
| this track, ungated | exact | per-flight 95th percentile (median 1.58) | 16/69 |
| this track, Gate A λ = 3.82 | exact | recalibrated on accepted clips (0.93) | 16/22 |

The `16/69` here is reproduced from K03's stored table row for row on false-reject, yaw control,
removal, mirror and null; the two scale families differ by one clip each (x0.5 13 vs 14, x1.5 9 vs
7) because K03 masks frames with required tilt above 85° and this track uses an explicit 80°
near-vertical abstention instead. The apparent J03-vs-K03 disagreement is a construction and
operating-point difference, not an error in either.

## What this supports

* **A precise negative result with a mechanism.** The yaw-control false flag is a systematic
  additive damage to the coherence statistic (+0.61 under exact physics), uncorrelated with the
  path's signal, so a path-side gate cannot suppress it. This quantifies and explains the 7–9/69
  J03 reported and the 23/69 K03 reported, and it predicts that no signal-strength gate will fix
  the coherence tier's specificity.
* **A validated, cheap validity filter.** `weak_signal` (14 of 69 gate-kept dev clips have no
  computable coherence value at all) and `near_vertical` (4 clips above 80° of required tilt,
  where a lean vector degenerates) are decidable from the path alone, before any pixels are read,
  and they remove exactly the clips where a verdict would be meaningless rather than wrong.
* **The coverage number the paper needs.** With the middle SNR candidate the verifier can
  *judge* 32 % of the gate-kept real clips; that is a finding about the population, not a failure
  of the gate.

## What this does not show

* Nothing about the held-out flights (13p, 07a), the simulated set, or the generated clips: none
  was read.
* Nothing about whether Gate B fixes the yaw-control rate — that is M02, and the mechanism above
  predicts it will not, because both readers see the same warp.
* The SNR reading is not identified: it depends on whether `sigma_n` is the robust MAD (1.91°) or
  the RMS (8.1–14.2°), and the two differ by a factor of four to seven. Both are reported and no
  claim is made that either is "the" reader noise.
* `sigma_min` is a second-moment statistic on a clip of 48 frames; its own sampling error is not
  propagated, and with 69 development clips no interval on the coverage is tighter than a
  descriptive band.

## Caveats

* Six development flights, at most two of which contribute accepted clips at λ = 3.82; every rate
  in table 2 beyond that point is a handful of clips and is descriptive, not a 95 % interval.
* The near-vertical rule changes the calibration population and therefore the coherence
  threshold: excluding the 4 near-vertical clips from the threshold fit moves the threshold from
  1.36 to 0.93 and the ungated yaw rate from 16/69 to 21/65. Both are reported; neither is
  "correct", and the choice is stated rather than hidden.
* The exploratory yaw-invariant statistics (complex correlation, magnitude correlation) were
  computed after the yaw problem was seen and are labelled exploratory wherever they appear.
* Gate A's near-vertical limit of 80° is a representation choice, not a fitted threshold; K03's
  85° mask is a different choice on the same continuum, and the one-clip differences in the scale
  families come from it.

## Provenance

* **Code**: `experiments/m01_gate_a.py` (commit with the results), `src/verdict/gate_a.py`,
  `src/verdict/selective.py`; `src/airframe/*` copied byte-identically from
  `track-k-calibration` `3110976`.
* **Results** (in git): `results/m01/{signal_by_clip.csv, coherence_v3.csv,
  risk_coverage_dev.csv, nested_dev.csv, selection_dev.json, candidate_k_dev.json,
  frozen_lambda_dev.json, noise.json, taus_dev.json, yaw_damage.json, j03_reproduction.json,
  f0_verification.json}`.
* **Inputs**: schema-v2 clips `data/processed/track_d/clips` (Track D, read-only); mocap
  `data/raw/tii`; the frozen S3 gate table `results/a2_gate/20261109T201449Z_298ff20/clip_table.csv`
  and `configs/gate_frozen.yaml` (sha256 `dc0cc4ff…`).
* **Frozen**: `configs/m01_gate_a_frozen.yaml` (λ = 3.82°, near-vertical 80°, coherence τ
  recalibrated on accepted clips).
