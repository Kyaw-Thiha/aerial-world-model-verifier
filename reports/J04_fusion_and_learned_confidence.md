# J04 — Learned reader confidence (Stage 1c), and why there is no fusion result

Track J. Written 2026-11-21. Pre-registration `configs/j04_learned_confidence_prereg.yaml`
(committed at `1dbdd55`, before any fit). Code `experiments/j04_learned_confidence.py`.
Results `results/j04_confidence/`. CPU only, **zero GPU spend**.

**Scope note.** Only Stage 1c ran. Stage 1b (innovation-gated fusion of the reader with video
rotation) was not selected by any rule: rule 2 needs a rotation source within 0.3 deg per frame
and 1.5 deg per second, and TII DPVO delivers 2.80 and 7.76 (J01 §5). Stage 1a (per-clip
calibration) is CPU-only but its rule does not fire as written (J01 §rules); it is reported to
the user rather than run silently. The pre-registration allows a Modal GPU for this branch; it
was run on CPU because the reader's own features are already stored per frame — a **declared
scope reduction**, not a silent substitution.

## Decision

**Learned confidence ranks the bad frames better than the frozen gate's signal and is not
calibrated**: per-frame AUROC **0.946** [0.882, 0.973] against 0.851 for the same logistic fitted
on S3 alone and 0.866 for a physics-only (tilt, speed) model, and clip level 0.961 against the
published 0.946 — but the held-out recalibration slope is **0.55** and the intercept −1.87, well
outside the pre-registered [0.8, 1.25], and **using it as a weight or a mask in the window
statistic changes no detection cell** at the same false-reject.

## Question

J01 measured that 88 % of the >10 deg frames sit in clips the frozen S3 gate already drops, and
that the tail is high-attitude, fast and (on 5 of 8 flights) low-confidence. Can a learned
per-frame confidence predict the reader's error better than S3 on the frames that survive, and
does weighting the window statistic by it improve segment detection?

## Setup

**Frames.** 5,007 per-frame frozen-reader outputs, 8 flights, the same set as J01. The label is
`theta > 10 deg` (12.7 % prevalence), the J01 error.

**Features (all stored reader outputs; no pixels).** The eight raw per-frame quantities
(`gravity_uncertainty_raw`, `roll_uncertainty_raw`, `pitch_uncertainty_raw`, `up_confidence_mean`,
`up_confidence_p10`, `latitude_confidence_mean`, `latitude_confidence_p10`, `valid_frac`), and for
each of them a 0.5 s centred window's median, max, min, std and the frame's deviation from its own
median; plus body tilt, speed and heading. 44 features.

**Model.** L2 logistic regression by Newton/IRLS, standardised per fold on the fit flights, λ
chosen from {1e-3, 1e-2, 1e-1} by the fit-fold log-loss (1e-3 won in all 8 folds). Leave-one-
flight-out over the eight TII flights: nothing from the held-out flight enters standardisation,
λ selection or the fit. Baselines: the same model on S3 alone, and on tilt + speed alone.

**Downstream.** The predicted probability enters the G-round window statistic (8 frames, hop 4,
maximum over complete windows) as a weight `1 - p` floored at 0.1 (weighted median) and as a mask
dropping frames above the flight's 90th percentile of `p`. Four arms — plain, weighted, masked,
both — each calibrated leave-one-flight-out on the real clips and scored on the full-clip and
25 %/50 % partial edits composed on the depicted side exactly as in G01. Dev flights only; the
two held-out flights are untouched.

## Results

### Predictive quality, held out per flight

| held-out flight | n | bad frames | learned AUROC | S3 alone | tilt + speed |
| --- | --- | --- | --- | --- | --- |
| 01a-ellipse | 270 | 30 | 0.785 | 0.434 | 0.767 |
| 01p-ellipse | 1267 | 10 | 0.930 | 0.918 | 0.746 |
| 04a-ellipse | 350 | 20 | 0.952 | 0.383 | 0.774 |
| 07a-lemniscate | 370 | 65 | 0.970 | 0.550 | 0.976 |
| 07p-lemniscate | 1284 | 7 | 0.892 | 0.741 | 0.683 |
| 13a-trackRATM | 557 | 187 | 0.963 | 0.925 | 0.968 |
| 13p-trackRATM | 536 | 190 | 0.847 | 0.858 | 0.765 |
| 16p-trackRATM | 373 | 129 | 0.800 | 0.769 | 0.695 |

Pooled: **0.946** [0.882, 0.973] (flight-clustered bootstrap, 5,000 replicates, 8 clusters —
descriptive). S3 alone on the same frames: 0.851. Tilt + speed alone: 0.866. Clip level (the mean
predicted probability per clip, the unit of the published number): **0.961** on 114 clips, 9
positive, against the frozen gate's published 0.946.

The learned model wins on 7 of 8 folds by a wide margin on the three flights where S3 is
uninformative (04a 0.95 vs 0.38, 07a 0.97 vs 0.55) and loses narrowly on `13p` (0.847 vs 0.858).
That is the useful finding: S3 is a *clip-level* signal being used per frame, and the per-frame
features recover most of the gap.

### Calibration, held out

| quantity | value |
| --- | --- |
| Brier, learned | 0.082 |
| Brier, S3-alone model | 0.221 |
| recalibration slope (logistic on the held-out logit) | **0.553** |
| recalibration intercept | **−1.868** |

| decile | predicted | observed | n |
| --- | --- | --- | --- |
| 1 | 0.056 | 0.010 | 501 |
| 2 | 0.085 | 0.000 | 501 |
| 3 | 0.103 | 0.000 | 500 |
| 4 | 0.120 | 0.002 | 501 |
| 5 | 0.138 | 0.002 | 500 |
| 6 | 0.164 | 0.010 | 501 |
| 7 | 0.217 | 0.034 | 501 |
| 8 | 0.320 | 0.104 | 500 |
| 9 | 0.576 | 0.405 | 501 |
| 10 | 0.959 | 0.707 | 501 |

The ordering is right and the scale is not: the model is over-confident at both ends. The
pre-registered success criterion required a slope in [0.8, 1.25]; it fails.

### Downstream: the window statistic

Dev block, real false-reject and detection at the same leave-one-flight-out 5 % threshold
(k/n of 82 dev clips):

| arm | real FRR | removal 25 / 50 / 100 % | mirror 25 / 50 / 100 % | yaw / null (100 %) |
| --- | --- | --- | --- | --- |
| plain | 6/82 | 9 / 10 / 12 | 8 / 12 / 12 | 8 / 6 |
| weighted (`1-p`) | 7/82 | 9 / 9 / 10 | 8 / 12 / 12 | 7 / 7 |
| masked (`p` < q90) | 5/82 | 6 / 7 / 10 | 6 / 12 / 12 | 7 / 5 |
| weighted + masked | 6/82 | 6 / 6 / 7 | 6 / 12 / 12 | 6 / 6 |

**No cell improves.** The weighted arm is identical at 25 % and *worse* at 50 % and full coverage;
the masked arm lowers every detection together with the false-reject because the mask removes
frames that carry the signal. The pre-registered secondary criterion (≥ 5 percentage points
anywhere in the 25/50 % cells) is missed everywhere.

## What this supports

* **A per-frame confidence exists and is substantially better than the current signal at ranking
  frames**: 0.946 against 0.851, with the gain concentrated exactly where S3 is uninformative.
  If a confidence is wanted for *admissibility* or for reporting, the stored features carry it.
* **Ranking is not enough for the decision layer.** The G-round statistic already takes a median
  over a window, which is a robust aggregation; re-weighting an already-robust median by a
  monotone transform of the same information changes almost nothing — the same conclusion G04
  reached from the gate's side.
* **The tail is not separable from the flight.** The two flights the model handles worst
  (`13p` 0.847, `16p` 0.800) are the two whose tail is 27-33 % of frames: there, "bad" is the
  normal state and there is little contrast to rank.

## What this does not show

* **Not a calibrated probability.** The slope and intercept say the numbers must be recalibrated
  (Platt or isotonic, fitted inside the folds) before any of them is used as a probability. That
  was not pre-registered and was not run; the AUROC is unaffected by it.
* **Not a fusion result.** Stage 1b was not run, for the reason in the scope note.
* **Not the pixel-side question.** No image is read, so nothing here says whether a CNN on the
  frames would do better; it says the reader's own stored outputs already contain the signal.
* **Not held-out evidence.** Every number is dev-block or leave-one-flight-out; the two frozen
  held-out flights are unscored, as pre-registered.

## Caveats

* The clip-level AUROC (0.961 on 114 clips, 9 positive) and the gate's 0.946 are computed on
  different clip sets (114 vs 100) and different labels (frame-median error here, the gate's own
  label there); they are comparable in magnitude, not statistically.
* The label is `theta > 10 deg`, the same threshold the gate uses; a different threshold would
  move every AUROC (the gate's own sensitivity table shows 5 deg moves the positive count from 8
  to 30).
* Eight folds, 8 clusters: the bootstrap interval is descriptive.
* λ was selected on fit-fold log-loss, not by nested cross-validation inside the fit folds; with
  44 standardised features and 4,000+ fit frames the choice is not delicate (1e-3 won everywhere).

## Regenerate

```
PYTHONPATH=. python experiments/j04_learned_confidence.py     # ~25 s, CPU
```

Outputs: `results/j04_confidence/per_frame.json` (per-fold AUROC, pooled metrics, reliability),
`downstream.json` (four arms × controls), `rows.csv` (one row per clip, condition, coverage, arm).
