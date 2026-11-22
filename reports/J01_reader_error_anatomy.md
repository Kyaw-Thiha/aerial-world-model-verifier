# J01 — What the depicted-tilt error is made of

Track J. Written 2026-11-21. Pre-registration `configs/j01_anatomy_prereg.yaml`, amendment A
(`configs/j01_anatomy_amendment_a.yaml`) and B (`configs/j01_anatomy_amendment_b.yaml`), all
committed before the numbers they govern were computed. Code `src/depicted/anatomy.py`,
`experiments/j01_anatomy.py`. Results `results/j01_anatomy/`. CPU only, **zero GPU spend**.

## Decision

**The reader's bias is a real, almost purely pitch offset of +2.15 to +2.69 deg (SE 0.03-0.18)
on the five non-racing flights, and how much of the squared error it explains depends on the
population: 30 % over all frames, 43 % on the clips the frozen gate actually keeps.** The
pre-registered rule that would have opened the per-clip-calibration branch (bias share ≥ 40 % of
the *trimmed* variance) does not fire as written — 35.2 %, and the trimmed split is not even
identified — but it does fire on the gate-kept stratum (42.8 %); that conflict is reported, not
resolved.

## Plain-language summary

1. **The reader's error has three parts and they are comparable in size.** Pooled over the
   eight TII flights and both axes: 30 % of the squared error is each flight's constant offset,
   40 % is drift below 0.5 Hz, 30 % is everything above 0.5 Hz. That is the frozen GeoCalib
   reader on 5,007 frames; the G03 fine-tuned reader gives 25 / 37 / 38.
2. **The split is not the same on every flight, and the pooled number is three flights'
   opinion.** The three racing flights carry **83 %** of the corpus's squared error (`13a`
   alone 44 %, `13p` 24 %, `16p` 15 %); `13a`'s pitch-axis RMS is 31.7 deg. On the two quiet
   piloted flights (`01p`, `07p`) the pitch constant is **68 % and 72 %** of the squared error —
   they are *bias-dominated*; on the other three non-racing flights it is 21 %, 24 % and 14 %, and
   the corpus as a whole is noise- or drift-dominated.
3. **The bias is a pitch offset, not a random direction.** Pitch bias is positive on all eight
   flights and sits at **+2.15 to +2.69 deg** on the five non-racing ones, with standard errors of
   0.03-0.18 deg: this is not sampling noise. Roll bias on the same flights is −0.52 to
   +1.43 deg and changes sign. A constant rotation fitted per flight removes about 1-1.5 deg of
   median error on four of the five non-racing flights.
4. **But it is not one mount rotation for the rig.** A single rotation fitted leave-one-flight-
   out moves the pooled median error from 3.90 to 3.74 deg and *makes the answer worse* on
   `flight-07p`, `flight-13p` and `flight-16p` (3.60→4.14, 7.75→8.32, 7.32→8.49). The offset is
   per flight (and partly per heading: the roll error depends on yaw at t ≈ 2.4), which is what
   the verdict's per-clip mount fit is for.
5. **The error is correlated on a 0.2-5 s scale.** The integral correlation time of the scalar
   error is 0.22, 0.77, 0.26, 3.17, 0.86, 5.23, 0.34 and 0.29 s per flight (median 0.56 s).
   For an error that is one third fast, that means the "fast" band is 0.5-6 Hz, not per-frame
   white noise: a filter that averages windows shorter than ~0.5 s buys far less than √n.
6. **The >10 deg tail is a flight property, and the frozen gate already removes 88 % of it.**
   12.75 % of frames are above 10 deg (reproducing report 02's 12.7 %). They concentrate on the
   three racing flights (33.6 %, 35.4 %, 34.6 %) and are almost absent on the fast piloted
   ones (0.8 %, 0.5 %). 558 of the 638 tail frames sit in clips the frozen S3 gate drops;
   80 survive. Only 21.6 % of tail frames are in short (≤3-frame) runs; 39.5 % are in runs of
   a second or more, and 9 of the 12 long runs are identifiable from tilt, speed, confidence
   or blur.
7. **Visual odometry is good enough on a calibrated rig and not on TII.** On the UZH indoor
   forward sequences DPVO and ViPE carry the gravity direction with 0.06-0.14 deg error per
   1/15 s step and 0.19-0.29 deg after one second — twenty times better than the 1.5 deg the
   handoff asks for. On TII, DPVO gives 2.80 deg per step (median, frame-weighted) and 7.76 deg
   after one second, with the two hopeless flights at 7.6-8.6 deg per step. **The fusion branch
   the handoff proposed has no usable rotation source on the dataset that matters.**

## Question

The reader's per-frame error is ~4 deg at the median and 22 deg at the RMSE, and two rounds of
segment work hit the same wall. Before building any fix: is that error a constant per-flight
offset (removable by calibration), slow drift (removable by a per-clip fit), or fast noise
(removable by fusing with video rotation)? How long does it correlate? Is the bias a rigid mount
rotation? Where is the tail, and what would a rotation source have to deliver?

## Setup

**Population.** 5,007 per-frame estimates from the 100 real TII clips (8 flights, 11.9-12.0 Hz),
the exact frame set the verdict scores and the G03 fine-tune was evaluated on. The frames are a
continuous ~12 Hz subsample of each whole flight (largest gap 118 ms), so each flight is one
contiguous run — this is amendment A, and it is why the slow band reaches from ~0.005 Hz to
0.5 Hz instead of being limited to a 4 s clip.

**Two readers, one label.** The frozen reader (`results/gate_a2/tii/*__identity.jsonl`) and the
G03 out-of-fold fine-tuned reader (`results/j_inputs/g03_folds/`, staged from
`track-g-segments2` at `cfea5c8`). The reference is the mocap label `up_cam` built from the
mocap quaternion and the flight's recorded uptilt, i.e. the same label both readers were scored
against in G03. The join on frame name is asserted identical; a mismatch raises.

**Error definition.** Both up-vectors are taken to the body frame with `R_cb` from the recorded
uptilt, and the estimate is log-mapped at the truth onto the orthonormal tangent basis
`(∂u/∂pitch, ∂u/∂roll)`. The two components are therefore an **exact orthogonal decomposition of
the scalar E2/G03 angle**: `hypot(pitch, roll) == angle(up_est, up_truth)` to 2e-14 deg on every
frame (checked in `input_check.json`). Axis 0 is forward (pitch), axis 1 is right (roll).

**Split.** The per-flight mean is the bias. The residual is split with an orthogonal DCT-II basis
at 0.5 Hz per contiguous run, so the run's energy is exactly `bias² + slow + fast` with no cross
term and no filter edge effect (`closure` = 0.000 on every all-frame row of
`decomposition.csv`). Shares are of the total second moment unless labelled "of centred".

**Reconstruction check.** The all-frame tail fraction comes out at 638/5,007 = 12.75 % against
the published 12.7 %, and per flight 0.8 % (01p), 33.6 % (13a), 35.4 % (13p), 34.6 % (16p)
against G03's 0.8 / 33.6 / 35.4 / 34.6. The pipeline reproduces the published numbers before it
adds anything.

## Results

### 1. Variance decomposition

Pooled over the eight flights and both axes (frame-weighted; flight-clustered bootstrap,
5,000 replicates, `rng(1)`, 8 clusters — descriptive spreads, not 95 % confidence intervals):

| population | reader | bias | slow (<0.5 Hz) | fast (≥0.5 Hz) | fast of centred |
| --- | --- | --- | --- | --- | --- |
| all 5,007 frames | frozen | **29.6 %** [12.4, 41.7] | 40.4 % | 30.0 % | 39.8 % [27.8, 59.5] |
| gate-kept clips (72/100) | frozen | **42.8 %** | 40.3 % | 19.8 % | 32.1 % |
| trimmed (E2 rule) | frozen | 35.2 % [16.5, 49.7] | *not identified* | *not identified* | 32.0 % |
| all 5,007 frames | fine-tuned | 25.0 % [6.5, 37.2] | 37.3 % | 37.7 % | 46.8 % |
| gate-kept clips | fine-tuned | 40.1 % | 37.4 % | 23.3 % | 36.6 % |

Equal weight per flight instead of per frame: bias 20.9 % (all) / 44.4 % (gate-kept) for the
frozen reader.

Per flight and axis, all frames (frozen reader; `decomposition.csv` has both readers and all
three strata):

| flight | axis | bias (deg) | RMS (deg) | bias % | slow % | fast % |
| --- | --- | --- | --- | --- | --- | --- |
| 01a-ellipse | pitch / roll | 5.05 / 1.03 | 10.94 / 9.19 | 21 / 1 | 15 / 23 | 63 / 76 |
| 01p-ellipse | pitch / roll | 2.24 / 1.76 | 2.71 / 2.70 | **68** / **42** | 12 / 48 | 20 / 10 |
| 04a-ellipse | pitch / roll | 3.70 / 0.33 | 7.62 / 7.23 | 24 / 0 | 19 / 27 | 57 / 73 |
| 07a-lemniscate | pitch / roll | 5.92 / 1.17 | 16.11 / 23.41 | 14 / 0 | 30 / 74 | 57 / 26 |
| 07p-lemniscate | pitch / roll | 2.51 / −0.08 | 2.97 / 2.98 | **72** / 0 | 13 / 94 | 15 / 6 |
| 13a-trackRATM | pitch / roll | 18.29 / 0.20 | 31.65 / 31.05 | 33 / 0 | 40 / 71 | 26 / 29 |
| 13p-trackRATM | pitch / roll | 10.08 / 8.15 | 25.80 / 21.65 | 15 / 14 | 30 / 38 | 54 / 48 |
| 16p-trackRATM | pitch / roll | 10.58 / 5.57 | 25.97 / 17.21 | 17 / 11 | 31 / 31 | 52 / 59 |

The two regimes are different problems. The two quiet piloted flights are bias-dominated at a
2.7-3.0 deg RMS; everything else is 7-32 deg RMS and noise- or drift-dominated, and no offset fix
touches those.

**The trimmed column is not usable and is reported as such.** Trimming to the smallest 90 % of
|error| is a selection on the *sum* of the components, which induces a large negative
covariance between slow and fast: the trimmed shares sum to 2.04 rather than 1 (closure
−1.04 × the total; amendment B). The rule is therefore also evaluated on the all-frame and
gate-kept strata, and both are printed above.

### 2. Time structure

Integral correlation time per flight (scalar error, residual after the flight mean; truncated at
the first non-positive autocorrelation):

| flight | τ_int (s) | ρ(1 frame) |
| --- | --- | --- |
| 01a-ellipse | 0.22 | 0.52 |
| 01p-ellipse | 0.77 | 0.77 |
| 04a-ellipse | 0.26 | 0.58 |
| 07a-lemniscate | 3.17 | 0.89 |
| 07p-lemniscate | 0.86 | 0.88 |
| 13a-trackRATM | 5.23 | 0.87 |
| 13p-trackRATM | 0.34 | 0.70 |
| 16p-trackRATM | 0.29 | 0.69 |

Median 0.56 s, spread 0.22-5.23 s. Per axis (`acf.csv`) the pitch axis is the shorter one
(0.19-2.71 s) and roll the longer (0.30-2.08 s); on `07p` the roll axis has τ_int 2.08 s with
ρ₁ = 0.98, i.e. an almost constant roll offset over the flight.

The periodograms peak between 0.2 and 0.8 Hz in both axes on every flight, and the pooled
spectrum is broad: there is no clean separation into a white floor and a DC term. That is the
quantitative reason a complementary filter cannot simply average away the fast part — the
"fast" band the DCT split calls above 0.5 Hz is itself correlated over several frames.

### 3. Structure of the bias

Per-flight mean error on the trimmed frames, with the frame-clustered standard error:

| flight | pitch bias (deg) | roll bias (deg) |
| --- | --- | --- |
| 01a-ellipse | +2.69 ± 0.18 | −0.33 ± 0.11 |
| 01p-ellipse | +2.15 ± 0.03 | +1.43 ± 0.05 |
| 04a-ellipse | +2.43 ± 0.10 | −0.41 ± 0.07 |
| 07a-lemniscate | +2.41 ± 0.17 | −0.52 ± 0.29 |
| 07p-lemniscate | +2.23 ± 0.03 | +0.04 ± 0.08 |
| 13a-trackRATM | +12.05 ± 0.76 | −3.09 ± 1.23 |
| 13p-trackRATM | +4.27 ± 0.62 | +3.29 ± 0.51 |
| 16p-trackRATM | +3.23 ± 0.45 | +1.61 ± 0.54 |

Sign consistency: **pitch positive on 8 of 8 flights**, and pinned to +2.2 to +2.7 deg on the
five non-racing ones. Roll is −0.5 to +1.4 deg and changes sign. Whatever the reader is doing, it is
biased in the forward-tilt direction by about 2.4 deg on the flights where it works.

**A constant rotation per domain** (orthogonal Procrustes on the up vectors, so it is the
best-case rigid mount correction):

| domain | rotation fitted | median before | median after | RMS before | RMS after |
| --- | --- | --- | --- | --- | --- |
| TII, all 8 flights | 4.9 deg | 3.97 | 3.42 | 22.34 | 22.06 |
| TII piloted (4 flights) | 5.2 deg | 3.82 | 3.68 | 17.10 | 16.74 |
| TII autonomous (4 flights) | 11.7 deg | 4.60 | 2.91 | 31.01 | 30.22 |

Leave-one-flight-out (fit on seven, apply to the eighth), median of the eight held-out medians:
3.90 → 3.74 deg. Per flight it helps `01p` 3.13→2.38, `04a` 3.23→2.15, `07a` 2.54→1.56, `01a`
4.20→3.33, and hurts `07p` 3.60→4.14, `13p` 7.75→8.32, `16p` 7.32→8.49. A robust (worst-10 %-
dropped, iteratively reweighted) refit gives the same picture in `bias_structure.json`. So: a
per-flight constant offset is real (and is what the verdict's per-clip uptilt nuisance fits),
but a single rig-level mount rotation does not generalise.

Within-flight regression on speed, tilt magnitude and heading (flight fixed effects,
flight-clustered standard errors, 8 clusters — descriptive):

| reader / axis | speed (deg per m/s) | tilt (deg per rad) | cos yaw | sin yaw | within R² |
| --- | --- | --- | --- | --- | --- |
| frozen / pitch | −0.14 ± 0.17 | 7.4 ± 3.9 | 0.26 ± 0.38 | 0.04 ± 0.69 | 0.071 |
| frozen / roll | −0.21 ± 0.08 | 3.5 ± 2.6 | **1.35 ± 0.56** | 3.99 ± 2.77 | 0.035 |
| fine-tuned / pitch | 0.08 ± 0.10 | 8.3 ± 4.3 | 0.12 ± 0.32 | 0.77 ± 0.89 | 0.139 |
| fine-tuned / roll | −0.13 ± 0.06 | 1.8 ± 2.2 | **0.78 ± 0.33** | 2.10 ± 0.91 | 0.015 |

The only consistent within-flight dependence is on heading (roll error moves with yaw, t ≈ 2.4
in both readers) — the view-dependence report 04 already recorded at 4.75 deg. Speed and
attitude magnitude do not explain the bias.

### 4. The tail

| quantity | frozen | fine-tuned |
| --- | --- | --- |
| frames above 10 deg | 638 / 5,007 (12.75 %) | 539 / 5,007 (10.77 %) |
| number of maximal runs | 134 | 127 |
| median run length | 2.5 frames (0.21 s) | 2.0 frames (0.17 s) |
| share of tail frames in runs ≤ 3 frames | 21.6 % | 23.9 % |
| share of tail frames in runs ≥ 1 s | 39.5 % | 30.6 % |
| tail frames in clips the frozen gate drops | 558 (87.5 %) | 468 (86.8 %) |
| tail frames surviving the gate | 80 | 71 |

Per flight the tail is a property of the flight, not of the frame: 0.8 % (01p), 0.5 % (07p),
5.7 % (04a), 11.1 % (01a), 17.6 % (07a), 33.6 % (13a), 35.4 % (13p), 34.6 % (16p).

What the tail frames look like, counting flights rather than pooling (per-flight medians are in
`tail.json`; pooling would let the three racing flights speak for the corpus):

| covariate | tail vs rest, per flight | count |
| --- | --- | --- |
| body tilt | tail higher on every flight (19.8 vs 1.3, 38.6 vs 24.2, 20.2 vs 1.2, 55.6 vs 3.0, 33.2 vs 22.8, 61.5 vs 1.2, 75.4 vs 59.1, 77.3 vs 64.1 deg) | 8/8 |
| speed | tail faster on 6 of 8 (9.6 vs 0.3 m/s on `13a`; slower on `01p` and `07p`) | 6/8 |
| S3 confidence | tail less confident on 5 of 8 (0.0008 vs 0.0022 on `13a`; higher on `01a`, `04a`; equal on `07a`) | 5/8 |
| blur proxy | tail blurrier on 7 of 8 (300 vs 556 on `01a`; the exception is `07p`, 764 vs 499) | 7/8 |

So the tail is high-attitude on every flight and fast on most, and the estimator's own S3
confidence already flags it on the majority. The blur proxy (variance of the Laplacian of a
320-px greyscale copy) is a proxy: it confounds motion blur with texture, and its separation is
the least consistent of the four. 44 of the 134 tail runs are single frames, so the tail is
mostly flicker plus a handful of multi-second failures.

The 12 runs of ≥ 1 s, and whether the recorded covariates mark them out (a run is "identifiable"
when ≥ 70 % of its frames sit in the flight's top tilt/speed quartile or bottom
confidence/blur quartile):

| flight | start (s) | length (s) | best factor | identifiable |
| --- | --- | --- | --- | --- |
| 01a | 15.1 | 1.67 | blur | no |
| 07a | 15.5 | 1.17 | tilt | yes |
| 07a | 17.3 | 1.34 | tilt | yes |
| 07a | 19.3 | 1.09 | tilt | yes |
| 13a | 21.8 | 3.08 | confidence | yes |
| 13a | 27.5 | 3.58 | confidence | yes |
| 13a | 32.6 | 1.17 | blur | yes |
| 13a | 33.9 | 3.58 | confidence | yes |
| 13p | 16.8 | 1.34 | confidence | no |
| 13p | 21.2 | 1.00 | tilt | yes |
| 16p | 17.9 | 1.00 | tilt | yes |
| 16p | 29.4 | 1.00 | confidence | no |

### 5. Video-rotation quality

Per-step relative-rotation error and the error after carrying the *true* gravity forward with the
*estimated* rotations (the quantity a complementary filter consumes). Error split into roll and
pitch in the body frame; rates are the stored ones (TII DPVO stride 8 at 15 Hz; UZH at 15.07 Hz).

| source | unit | per step (deg) | 0.5 s (deg) | 1.0 s (deg) |
| --- | --- | --- | --- | --- |
| TII DPVO | 01a-ellipse | 0.26 | 1.05 | 2.59 |
| TII DPVO | 01p-ellipse | 1.96 | 6.52 | 10.67 |
| TII DPVO | 04a-ellipse | 0.15 | 0.30 | 0.55 |
| TII DPVO | 07a-lemniscate | 1.04 | 2.59 | 4.85 |
| TII DPVO | 07p-lemniscate | 3.17 | 7.01 | 11.87 |
| TII DPVO | 13a-trackRATM | 0.37 | 1.69 | 2.61 |
| TII DPVO | 13p-trackRATM | 7.61 | 28.12 | 39.69 |
| TII DPVO | 16p-trackRATM | 8.60 | 32.07 | 41.41 |
| TII DPVO | **pooled** | **2.80** | — | **7.76** (median of flight medians) |
| UZH DPVO | indoor_forward_3 / 5 / 9 | 0.136 / 0.065 / 0.100 | 0.19 / 0.12 / 0.18 | 0.23 / 0.19 / 0.18 |
| UZH ViPE | indoor_forward_3 / 5 / 9 | 0.130 / 0.072 / 0.099 | 0.21 / 0.13 / 0.17 | 0.29 / 0.21 / 0.24 |

Per-step roll and pitch components are in `rotation.json`; on UZH they are balanced (roll
0.031-0.080 deg, pitch 0.028-0.058 deg). On TII the two hopeless flights fail in both axes.

The UZH numbers clear the handoff's bar (≤ 0.3 deg per step, ≤ 1.5 deg over 1 s) on every
sequence and both extractors. The TII numbers miss it by an order of magnitude on the median and
by 20-70× on the two flights that matter most, and the failure is not marginal: on `16p` DPVO's
estimated relative rotation is 1.5 % of the true rotation budget — its rotation is close to
frozen — and `13p` reaches 28 deg of propagation error after half a second.

## The three priority rules

Applied as written, frozen reader, in `results/j01_anatomy/rules.json`:

| rule | statistic | threshold | outcome |
| --- | --- | --- | --- |
| R1 bias | trimmed pooled bias share 35.2 % (all-frame 29.6 %, gate-kept 42.8 %) | ≥ 40 % | **does not fire as written**; fires on the gate-kept stratum |
| R2 fast | fast-of-centred 32.0 % trimmed / 39.8 % all-frame; DPVO 2.80 deg/step and 7.76 deg/1 s | ≥ 40 % and ≤ 0.3 and ≤ 1.5 | **does not fire** (both clauses fail on TII) |
| R3 tail | 21.6 % of tail frames in runs ≤ 3 frames; 12 long runs, 9 identifiable | > 50 % (1b) / any (1c) | **1c fires** on the literal reading; 1b does not |

So, in the end: **1b was not run** (no rule selects it, and the rotation source it needs does not
exist on TII); **1a was not run** (its rule is silent on the pre-registered stratum, and a branch
is not started on a stratum the pre-registration did not name); **1c was run**, CPU-only and at
zero cost, because its rule fires on the literal reading — report J04 — and is reported as
selected-but-marginal, since 88 % of the tail is already excluded by the gate. Nothing was run
silently; the 1a conflict is in `docs/decisions.md` [J].

## What limits the verifier

Putting the numbers together: on the flights where the reader works, the error is a ~2.4 deg
forward pitch offset plus a ~1.5 deg wobble correlated over 0.5+ s, and the verdict's
per-clip mount fit already removes the first part — which is why the clip-level false-reject is
4-7 % there. On the racing flights the error is 17-32 deg RMS with a five-second correlation
time and 88 % of it lands in clips the gate already discards, so the binding constraint is not
the estimator's noise but the fact that a third of those flights is unreadable. The handoff's
three proposed fixes map onto this as follows: a per-clip calibration is worth ~1-1.5 deg of
median error on the non-racing flights and nothing on the racing ones; a complementary filter needs a
rotation source that TII does not have (DPVO is 10-70× short) but UZH does (0.1 deg/step); and a
learned confidence would be trained on a tail that is 88 % already excluded. **The measurable
headroom is in the tail's admission decision, not in the fast component.**

## What this supports

* A parameter sheet with denominators for every quantity a Stage-1 design needs: bias share
  (29.6 % all / 42.8 % gate-kept), slow/fast split, τ_int per flight (0.22-5.23 s), the
  per-flight bias vector with standard errors, the tail anatomy, and per-step plus horizon
  rotation error for four extractor/dataset combinations.
* **The bias is a systematic pitch offset**, not a random per-flight draw: +2.15 to +2.69 deg
  with SE 0.03-0.18 on five independent flights. That is the first direct measurement of the
  number G2 inferred at 2-5 deg from quiet windows.
* **The tail is a flight property and the frozen gate already removes 88 % of it** — a
  quantitative answer to "what remains after the gate" (80 frames out of 5,007).
* **The rotation bar is met on a calibrated rig and missed by 10-70× on TII**, measured as the
  propagation error a filter would actually consume rather than as a rotation-matrix norm.

## What this does not show

* **Nothing causal about the reader.** The offset is measured, not explained: this track does
  not show whether it is the fisheye model, the uptilt convention, or the network.
* **Not a mount estimate for production.** The per-flight Procrustes is fitted on mocap and is
  an upper bound on what a calibration could remove; the leave-one-flight-out version is the
  honest number and it does not generalise.
* **Not a fusion result.** No filter was built; R2 exists to say whether one would have a usable
  input, and on TII it would not.
* **Not UZH verdict evidence.** The UZH rotation numbers are per-frame only; no UZH clip
  statistic is computed (E05's population is too small to calibrate).
* **Not Stage 2.** Stage 2 (re-running the selected pipeline on Track I's simulated scenes) needs
  a selected detection pipeline and a labelled simulated development set; no pipeline was
  selected here, and report G05 already recorded that `track-i-simset` carries only the I01 spike
  (one flight, three renders, full-clip violations). Nothing was re-run on it.
* **The "slow" band is not drift over the whole flight in the sense of a fixed budget.** The
  flight mean is the bias, so slow covers 0.005-0.5 Hz around it; a drift with a period longer
  than the flight would be absorbed into the bias and is not measurable here.

## Caveats

* **The trimmed variance decomposition is invalid** (closure −1.04 × total). It was the
  pre-registered population for rule 1; the all-frame and gate-kept rows are printed beside it
  and the rule outcome depends on which is used. Amendment B records this; the conflict is in
  `docs/decisions.md`.
* **The pooled share is dominated by three flights.** `13a`/`13p`/`16p` carry 44/24/15 % of the
  total squared error between them. The equal-weight-per-flight numbers (20.9 % bias all frames,
  44.4 % gate-kept) are printed beside the pooled ones for exactly that reason, and they tell
  the same story in the same direction.
* **Eight flights.** Every flight-clustered interval here has 8 clusters and is a descriptive
  spread, not a 95 % confidence interval.
* **The blur proxy is a proxy.** Laplacian variance tracks texture as well as blur, and it is
  the weakest of the four tail covariates.
* **TII DPVO poses are stride-8 (15 Hz)**, so the finest relative rotation available is 1/15 s,
  not the camera's 120 Hz; the 0.5 s and 1 s horizons have 7 and 15 samples. Amendment A records
  this.
* The gate accounting assigns a frame to the clip whose index range in
  `data/processed/p3a/p3a_v1/clips.json` contains it; 53 of the 5,007 frames fall outside every
  range and are treated as kept. Five of them are tail frames, so the 87.5 % removal figure is
  conservative (88.2 % if those are excluded) and the conclusion is unaffected.
* The fine-tuned reader's per-frame predictions are the G03 out-of-fold ones; a frame's error is
  measured by a model that never saw its flight, which is the honest number but also the reason
  its tail is slightly better than the frozen reader's.

## Regenerate

```
PYTHONPATH=. python experiments/j01_anatomy.py            # ~25 s, needs data/g03_frames
PYTHONPATH=. python experiments/j02_noise_budget.py       # item 6
```

Outputs: `results/j01_anatomy/{input_check.json, decomposition.csv, decomposition_pooled.json,
acf.csv, psd.csv, bias.csv, bias_structure.json, tail.json, blur.csv, rotation.json,
rules.json}`.
