# Report 06 — A label-free confidence gate for the depicted tilt (Track A2, Task 1)

Track A2. Written 2026-11-09. Pre-registration: `configs/verdict_preregistration.yaml`,
**amendment 2** (commit `d8941e8`), which supersedes amendment 1. Gate fitted by
`experiments/a2_gate_validate.py` and frozen in `configs/gate_frozen.yaml` (commit
`9e1656d`). Extraction: Modal app `awmv-a-gate`, rows under `results/gate_a2/tii/`.

## Plain-language summary

Gate 3's operating point was set after three flights had been removed on the strength of
their scores. That is choosing who counts by looking at the answer: it inflates the
accepted rate and it cannot be done on generated video, where no mocap exists.

This report replaces it with a gate that decides, from **pixels alone**, whether a clip's
depicted tilt can be trusted. It is fitted against the estimator's actual error against
mocap, not against verdict outcomes, and no flight is ever excluded by identity.

**It works as a detector of bad clips, and it is expensive in coverage.** The chosen signal
(GeoCalib's latitude confidence) ranks a clip whose depicted tilt is more than 10° wrong
above a good one with **AUROC 0.946** (flight-bootstrap 95 % interval 0.884–1.000), and it
keeps clips whose median error is **3.5°** while dropped clips sit at **8.0°**. All eight
bad clips in the set — the five on flight 13a, the 81° clip on 07a, and one clip each on 13p
and 16p — are dropped. The price: held-out coverage is **0.60**, and on the three flights
that host most of the bad clips it collapses to 0.00–0.14, i.e. whole flights become "not
testable" rather than just their bad clips.

**What it says about the complaint in the handoff.** The gate uses no mocap, no path and no
implied tilt, so it is applicable to generated video; it is applied to Track B's clips in
report 08 after a transfer check.

## The question and the setup

**Question.** Which pixel-only signal, and which threshold, best predicts that a clip's
depicted tilt is more than 10° wrong — and what does using it cost in testable clips?

**Candidate signals, fixed before any value was computed** (amendment 2). All are
clip-level and oriented so that larger means less trustworthy:

| signal | definition | single-frame? |
| --- | --- | --- |
| S1 | median GeoCalib `gravity_uncertainty` | yes |
| S2 | negative median image-mean `up_confidence` | yes |
| S3 | negative median image-mean `latitude_confidence` | yes |
| S4 | 0.5 s running-median jitter of the gravity direction | no |
| S5 | agreement of GeoCalib's frame-to-frame change with DPVO's rotation | no |
| S6 | fraction of frames with uncertainty above twice the clip median | yes |

Only single-frame signals may be frozen: the gate must also run on a 16-frame, 224×224
generated clip with 1 s between frames, where a 0.5 s window holds one sample and DPVO has
nothing to track. S4 and S5 are computed and reported, and are ineligible.

**Label.** The per-frame depicted-gravity error is the angle between the estimated up
direction in the camera frame and the up direction implied by mocap attitude through the
recorded mount (`camera_to_body_pitched(uptilt)`), with no fitted nuisances. A clip is
labelled positive when the median of that error exceeds 10°. Computed on the real
(identity) condition for all 100 clips of the Gate 3 clip set, including clips the
excitation rule abstains on. The label is never an input to a signal.

**Why the estimator had to be re-run.** The Gate 3 rows stored only the gravity direction.
GeoCalib's uncertainty and confidence outputs exist at the pinned commit
(`lm_optimizer.estimate_uncertainty`, `extractor.calibrate`) but were not written. The same
frames were re-run with the same weights, camera model and commit, into a new directory;
the Gate 3 artifacts were not touched. The re-run reproduces the Gate 3 gravity direction to
0.03–0.17° per flight — the estimator's own run-to-run variation, measured directly in
report 07 — and it reproduces the published 0.57° noise floor on the level frames
(median 0.579°, p99 4.17°).

**Selection rule (fixed in advance).** Highest AUROC among single-frame signals, with a
flight-level bootstrap interval; not validated if the interval's lower bound is ≤ 0.5 or
fewer than 5 clips are positive. Threshold maximises (TPR − FPR) on fit clips subject to
keeping at least 60 % of them; leave-one-flight-out, with the frozen threshold fitted on all
eight flights.

## Results

### Labels

| label threshold | positives | where |
| --- | --- | --- |
| 5° | 30 | 13a 10, 13p 10, 16p 7, 07a 2, 01a 1 |
| **10°** (primary) | **8** | 13a 5, 07a 1, 13p 1, 16p 1 |
| 20° | 5 | 13a 4, 07a 1 |

The 10° label picks out exactly the clips the Gate 3 report already flagged as estimator
failures, including the 07a clip whose verdict score was 48.8° (its depicted-vs-truth error
is 81.5°).

### Per-signal validation (identity condition, 100 clips)

| signal | AUROC | 95 % interval | bootstrap usable | LOFO coverage | kept median error | dropped median error |
| --- | --- | --- | --- | --- | --- | --- |
| S1 uncertainty | 0.787 | 0.424–0.922 | 1.00 | 0.40 | 3.53° | 5.06° |
| S2 up confidence | 0.890 | 0.732–0.984 | 1.00 | 0.65 | 3.48° | 8.18° |
| **S3 latitude confidence** | **0.946** | **0.884–1.000** | 1.00 | **0.60** | **3.51°** | **8.04°** |
| S4 temporal jitter (ineligible) | 0.984 | 0.925–1.000 | 1.00 | 0.80 | 3.57° | 9.87° |
| S5 VO agreement (ineligible) | 0.989 | 0.954–1.000 | 1.00 | 0.83 | 3.64° | 11.24° |
| S6 uncertainty instability | 0.749 | 0.633–0.993 | 1.00 | 0.67 | 3.64° | 6.31° |

The two ineligible signals are the strongest, which is expected — both use motion the
estimator cannot explain — and neither can be transferred to a 16-frame clip at 1 s spacing.

### The frozen gate

`configs/gate_frozen.yaml`, commit `9e1656d`, fitted on all eight flights:

```
signal = S3_latitude_confidence   (drop when signal > tau)
tau    = -0.0016656               (i.e. keep when median latitude confidence > 0.00167)
AUROC  = 0.946 [0.884, 1.000]   n = 100 clips, 8 positive
LOFO coverage = 0.60             kept 3.51° vs dropped 8.04°
```

### Coverage per flight (leave-one-flight-out thresholds)

| flight | coverage | positives dropped | negatives dropped | n |
| --- | --- | --- | --- | --- |
| 01a-ellipse | 1.00 | 0 | 0 | 5 |
| 01p-ellipse | 1.00 | 0 | 0 | 26 |
| 04a-ellipse | 0.86 | 0 | 1 | 7 |
| 07a-lemniscate | **0.00** | 1 | 6 | 7 |
| 07p-lemniscate | 0.96 | 0 | 1 | 26 |
| 13a-trackRATM | 0.73 | 3 | 0 | 11 |
| 13p-trackRATM | **0.09** | 1 | 9 | 11 |
| 16p-trackRATM | **0.14** | 1 | 5 | 7 |

The gate does not isolate the bad clips on a bad flight; it discards the flight. That is the
direct cost of a pixel-only rule: when the estimator is unreliable on a flight, every clip
on that flight looks unreliable, because it is.

### The specific clips the handoff asked about

- **flight 13a**: all 5 positive clips are dropped at the global threshold, and 8 of its 11
  clips are dropped. Held out, coverage is 0.73.
- **the 07a clip that scored 48.8°** (`flight-07a-lemniscate#04`): depicted-vs-truth error
  81.5°, gate signal −0.00090 against a threshold of −0.00167, **dropped**. Every clip on
  07a is dropped when 07a is held out (coverage 0.00).

## What this supports

- A **working, label-free indicator** of whether a clip's depicted tilt can be trusted, with
  an interval that excludes 0.5 by a wide margin, validated against mocap error and not
  against verdict outcomes.
- A **quantified cost**: coverage 0.60 overall, near zero on the flights whose estimator is
  broken. Any acceptance number computed under this gate is an acceptance number on the
  well-behaved subset, and the reports say so.

## What this does not show

- **Not** that the gate is a substitute for the estimator being right. It abstains; it does
  not correct.
- **Not** a calibrated probability. The signal is used as a ranking with a threshold fitted
  for TPR − FPR, not as \(P(\text{error} > 10°)\).
- **Not** validated independently of the flights it was fitted on: eight positives in four
  flights, and the bootstrap interval for S3 rests on draws that contain them. The interval
  is reported; it is not a promise about a ninth flight.
- **Not** transferable by assumption. 8.5 % of Track B's generated clips are dropped by this
  gate and 0 of 10 ground-truth clips are (report 08), which is the evidence that it
  transfers at all.
- S1–S3 come from one estimator at one commit. A different estimator would need its own
  gate; nothing here is a property of "confidence" in general.

## Caveats

- GeoCalib was re-run rather than re-read, so every number here rests on rows produced in
  this round. The re-run agrees with the Gate 3 rows to the estimator's own run-to-run
  noise (0.03–0.17° median per flight); the stricter 0.05° bound written into amendment 2
  is exceeded on six of eight flights and the deviation is recorded in amendment 4 with its
  measured effect on every downstream number.
- The label uses the recorded mount and no nuisance fit, so ~1–2° of systematic mount error
  is inside every label; at the 10° threshold this does not change which clips are positive.
- S3 is a mean over an image of a per-pixel confidence map; its threshold is therefore tied
  to the image size (224 px for generated clips, 640×480 for TII). The transfer check is the
  only evidence that the same tau is meaningful in both.
