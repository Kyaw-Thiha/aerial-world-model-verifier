# G01 — Segment-level detection: windows, a scan statistic, and what it can localise

Track G. Written 2026-11-16. Pre-registration `configs/g01_segment_prereg.yaml`; amendment
`configs/g01_segment_amendment_a.yaml`. Code: `src/verdict/segments.py`,
`experiments/g01_segment_detection.py`. Results: `results/g01_segments/` (two population
blocks × two window lengths, committed at `2ba24a3`). CPU only; **zero GPU spend**.

## Plain-language summary

1. **The pre-registered design fails, and the reason is one flight.** Sliding 8-frame windows
   over a 4 s clip and calibrating the *maximum* over windows conformally puts the threshold at
   **89-94 deg** instead of ~15 deg, because four clips of `flight-13a-trackRATM` carry a
   single glitchy window above 89 deg. Every fold whose fit set contains that flight inherits
   it. The detector then flags nothing: detection sits at or a hair above the false-reject rate
   for every edit family, and the flags it does produce come from windows that were already
   above the threshold in the *real* clip. In the gate-kept block (below) that shows up as
   literally identical counts across families (5 of 69 for `x0.5`, `x1.5`, the yaw control and
   the null edit at every injected fraction), which is the diagnostic.
2. **The frozen confidence gate is exactly the instrument for this**, so amendment G01-A
   calibrates on the gate-kept real clips (69 of 82 on dev). The gate drops 28 of the 100 real
   clips and every clip whose max-window residual exceeds 80 deg. Thresholds become sane
   (residual 21-39 deg at one window, 16-25 deg with persistence; whole-clip median 12.5-14 deg).
3. **With the gate and persistence of two windows, segment detection beats the whole-clip
   median where it matters: on violations that cover only part of the clip.** Both rules are
   calibrated to a nominal 5 % (empirical 5.8 % for the scan statistic, 4.3 % for the
   baseline). Partial tilt removal is caught in **23.2 %** of clips at 25 % coverage (baseline
   8.7 %) and **40.6 %** at 50 % (baseline 24.6 %); partial mirrored roll in **40.6 %** at 25 %
   (baseline 10.1 %) and **56.5 %** at 50 % (baseline 27.5 %). At full
   coverage the two rules are equal (71.0 % vs 60.9 % removal, 78.3 % vs 75.4 % mirror).
4. **It does not meet the pre-registered success bar and it does not localise well.** The bar
   was half the clips at 25 % and 50 % coverage with interval IoU above 0.5; the achieved IoU
   is 0.15 and 0.24 (onset error −2 to −3.5 frames). Segments are real but blurry: a window
   straddles the edit boundary, and pre-existing glitch windows elsewhere in a clip add false
   segments (1.3-2.0 false segments per real minute).
5. **Scale edits are the wrong target for this machinery.** A x0.5/x1.5 edit is caught by the
   whole-clip median (69.6 % at full coverage for x1.5) and barely at all by a window
   residual. That is the gain tier's job, and report G02 shows it does it well.
6. **The yaw and null controls sit at the false-reject rate** in every gated block
   (5.8-7.2 %), which is what makes the partial-edit numbers meaningful.

## Question

The frozen verdict is one clip-level median. Does a window-based scan statistic (a) reject a
clip whose violation covers part of it, (b) localise the violation in time, and (c) keep the
whole-clip false-reject at the conformal target while doing so?

## Setup

**Population.** 100 real TII clips (8 flights), schema-v2 `verdict_clip.npz` from Track D,
47-49 frames at 11.9-12.0 Hz, each with the four edit families and both controls
(`tilt_removal`, `mirror_roll`, `scale_0.5`, `scale_1.5`, `yaw_control`, `null_edit`).
Dev block = six flights (82 clips); held-out block = `flight-13p-trackRATM` and
`flight-07a-lemniscate` (18 clips), frozen in the pre-registration and scored once, at the end.

**Windows.** The label-free rule (`0.5 / f95`, 8-sample floor at the truth rate, no clamp)
returns 0.05-0.22 s per TII clip at 274.73 Hz — 0.6-2.6 frames at the 12 Hz clip rate, which no
window statistic can be computed from, and the rule's own `abstain_if` clause declares such a
clip not testable (the defect Track E measured on the 15 Hz UZH export, report E05). The
8-sample floor is therefore applied at the rate the statistic is actually computed on: **8
clip frames (0.66 s), hop 4 frames**, with the pre-registered 12-frame sensitivity. Ten to
eleven windows per clip.

**Per-window tiers** (all read from the schema alone):

| tier | definition | window abstains when |
| --- | --- | --- |
| residual | median per-frame angle between depicted and required thrust axis (deg), the frozen median-residual construction (`up_body_from_lean` → ZYX attitude → body z) | <6 usable frames or <2 signal frames (required tilt ≥ 10 deg) |
| coherence | `1 - min(r_x, r_y)`, per-axis Pearson over the window, lag 0 | either lean has std < 0.01 lean units on an axis |
| sign | `1 - min(s_x, s_y)` | same |
| gain | **not computed**: the frozen gain tier abstains below 24 frames and a window has 8 | always |

**Calibration.** Leave-one-flight-out empirical 95th percentile of the clip **maximum** over
windows. Calibrating the maximum, not the single window, gives the clip one family-wise budget:
`P(max_w T_w > tau) ≤ alpha` under exchangeability. The persistence variant calibrates
`max_i min(T_i, T_{i+1})` on its own (amendment G01-A; the pre-registration said "same
calibration", which would have been a rate at an unstated false-reject).

**Injections.** Partial edits are composed on the depicted side: inside a centred contiguous
interval of `m = max(1, round(f N))` frames the depicted lean comes from the same clip's
full-clip edit of that family, outside it from the real clip. Frame names, times, positions,
yaw and `lean_req` are asserted identical across conditions before any composition. This is
exact rather than approximate because the depicted reader is per-frame — GeoCalib estimates
gravity from one frame, the clip files record no temporal smoothing — so the array equals what
a partial-edit re-run would produce, at zero GPU cost. Families × fractions = 4 × 3, plus the
full-clip (f = 1) reference point and both controls at every fraction.

**Statistics.** Flight-clustered bootstrap, 5,000 replicates, `numpy.random.default_rng(1)`,
6-8 clusters: **descriptive spreads, not 95 % confidence intervals**. k/n beside every rate.

## Results

### 1. The pre-registered primary (ungated calibration): the threshold is poisoned

| fold | tau residual (deg) | tau whole-clip median (deg) |
| --- | --- | --- |
| 01a-ellipse | 90.7 | 23.8 |
| 01p-ellipse | 93.6 | 48.9 |
| 04a-ellipse | 90.6 | 25.8 |
| 07p-lemniscate | 93.6 | 48.9 |
| 13a-trackRATM | 58.8 | 17.7 |
| 16p-trackRATM | 89.0 | 23.8 |

Real-clip false-reject: residual 7.3 % (6/82), coherence 6.1 % (5/82), whole-clip median
4.9 % (4/82) — all at the nominal 5 %. Detection at that threshold, residual tier, dev block
(82 clips): tilt removal 8.5 / 11.0 / 12.2 / 14.6 % at 10 / 25 / 50 / 100 % coverage; mirrored
roll 7.3 / 8.5 / 14.6 / 14.6 %; **x0.5 6.1 / 7.3 / 7.3 / 8.5 %; x1.5 11.0 / 11.0 / 11.0 /
12.2 %; yaw control 7.3 / 8.5 / 8.5 / 8.5 %; null edit 7.3 / 7.3 / 7.3 / 7.3 %**. The scale
and control rows are the tell: they do not respond to the edit at all, they are the same 5-9
clips over and over. This block is reported in full and is *not* the headline.

Attribution, not reinterpretation: `flight-13a-trackRATM` scores a real-clip median residual of
68-85 deg on five of its eleven clips (report 04 recorded the same flight as untestable), and
`flight-07a-lemniscate` has one clip at 79.8 deg. A maximum over eleven windows sits on the
tail of a heavy-tailed per-frame error by construction (12.7 % of TII frames are above 10 deg
error), so a few clips with catastrophic windows set a whole-flight-agnostic threshold.

### 2. Amendment G01-A: calibrate on the frozen gate-kept population

The frozen S3 confidence gate (label-free, per-clip, fixed 2026-11-09 on 100 clips) drops 28 of
the 100 real clips here — and every clip whose max-window residual exceeds 80 deg is among
them. Calibrating on the kept clips applies the verdict's existing admissibility rule instead
of removing clips because they score badly.

**Calibration (dev, 69 gate-kept clips):**

| window | persistence | tau residual per fold (deg) | tau coherence | tau whole-clip median (deg) |
| --- | --- | --- | --- | --- |
| 8 frames | 1 | 21.1 / 38.5 / 21.3 / 38.2 / 32.9 / 28.5 | 1.73-1.84 | 12.5-14.1 |
| 8 frames | 2 | 16.1 / 23.1 / 17.4 / 23.1 / 21.4 / 17.6 | 1.73-1.84 | 12.5-14.1 |
| 12 frames | 2 | 15.9 / 24.7 / 17.8 / 24.7 / 23.5 / 17.4 | 1.85-1.98 | 12.5-14.1 |

The coherence threshold (≈1.8) says the 95th percentile of a *clip's* worst 8-frame window is a
correlation near −0.8: at eight frames the correlation is so noisy that the max-over-windows
threshold is uninformative. That is the single most important number for interpreting the
coherence column below.

**Real clips:**

| block | residual FRR | coherence FRR | sign FRR | whole-clip FRR | false segments per real minute |
| --- | --- | --- | --- | --- | --- |
| ungated, 8 frames, P1 | 7.3 % (6/82) | 6.1 % (5/82) | 6.1 % (5/82) | 4.9 % (4/82) | 1.68 / 1.12 / 1.87 |
| gated, 8 frames, P1 | 7.2 % (5/69) | 4.3 % (3/69) | 4.3 % (3/69) | 4.3 % (3/69) | 2.00 / 0.67 / 0.67 |
| gated, 8 frames, P2 | 5.8 % (4/69) | 4.3 % (3/69) | 4.3 % (3/69) | 4.3 % (3/69) | 1.55 / 0.67 / 0.89 |
| gated, 12 frames, P2 | 5.8 % (4/69) | 5.8 % (4/69) | 5.8 % (4/69) | 4.3 % (3/69) | 1.33 / 0.89 / 0.89 |

Detection at the **nominal 5 % threshold** (empirical 5.8 % for the scan statistic, 4.3 % for
the whole-clip baseline), gated, 8-frame windows, persistence 2. Columns are residual tier |
whole-clip median baseline | coherence tier | IoU | F1 | onset error in frames:

| family | 10 % | 25 % | 50 % | 100 % |
| --- | --- | --- | --- | --- |
| tilt removal | 5.8 % (4/69) \| 4.3 % (3/69) \| 31.9 % \| 0.01 \| 0.01 \| −9 | **23.2 % (16/69)** \| 8.7 % (6/69) \| 47.8 % \| 0.15 \| 0.18 \| −2 | **40.6 % (28/69)** \| 24.6 % (17/69) \| 50.7 % \| 0.24 \| 0.29 \| −3.5 | 71.0 % (49/69) \| 60.9 % (42/69) \| 65.2 % \| 0.35 \| 0.45 \| 4 |
| mirrored roll | 5.8 % (4/69) \| 4.3 % (3/69) \| 20.3 % \| 0.01 \| 0.01 \| −9 | **40.6 % (28/69)** \| 10.1 % (7/69) \| 39.1 % \| 0.27 \| 0.32 \| −2 | **56.5 % (39/69)** \| 27.5 % (19/69) \| 71.0 % \| 0.35 \| 0.42 \| −4 | 78.3 % (54/69) \| 75.4 % (52/69) \| 78.3 % \| 0.48 \| 0.58 \| 0 |
| x0.5 | 5.8 % \| 4.3 % \| 23.2 % \| 0.01 | 8.7 % \| 5.8 % \| 31.9 % \| 0.04 | 14.5 % \| 11.6 % \| 13.0 % \| 0.09 | 30.4 % \| 20.3 % \| 11.6 % \| 0.13 |
| x1.5 | 5.8 % \| 4.3 % \| 13.0 % \| 0.01 | 7.2 % \| 7.2 % \| 31.9 % \| 0.03 | 8.7 % \| 24.6 % \| 10.1 % \| 0.05 | 10.1 % \| **69.6 % (48/69)** \| 14.5 % \| 0.07 |
| yaw control | 5.8 % \| 4.3 % \| 5.8 % | 5.8 % \| 4.3 % \| 8.7 % | 5.8 % \| 4.3 % \| 8.7 % | 7.2 % \| 13.0 % \| 15.9 % |
| null edit | 5.8 % \| 4.3 % \| 4.3 % | 5.8 % \| 4.3 % \| 4.3 % | 5.8 % \| 4.3 % \| 4.3 % | 5.8 % \| 4.3 % \| 4.3 % |

Incremental detection (only clips that already pass as real, so pre-existing glitch windows
cannot be counted as detections): removal 0.0 / 18.5 / 36.9 / 69.2 %, mirrored roll
0.0 / 36.9 / 53.8 / 76.9 %. The gap to the raw rates is the false-reject floor, as it must be.

**Pre-registered 12-frame sensitivity, persistence 2** (same calibration rule, residual |
baseline): removal 5.8 % \| 4.3 %, 15.9 % \| 8.7 %, 40.6 % \| 24.6 %, 68.1 % \| 60.9 %; mirrored
roll 5.8 % \| 4.3 %, 29.0 % \| 10.1 %, 53.6 % \| 27.5 %, 78.3 % \| 75.4 %; coherence tier at
25 % and 50 % coverage 53.6 % (37/69) and 69.6 % (48/69) for removal, 43.5 % and 73.9 % for
mirror. Twelve-frame windows are better than eight at every partial fraction for both families:
longer windows buy a usable correlation estimate and shrink the glitch surface. This is a
sensitivity result, not the pre-registered headline.

### 3. Localisation quality

Mean IoU against the true injected interval is 0.09-0.50 across families and fractions; F1 is
0.11-0.60. Two mechanisms, both measured rather than assumed:

* window straddling — a flagged window covers all 8 (or 12) frames, of which only the part
  inside the edit is a true positive, so a 5-frame injection cannot exceed IoU ≈ 5/8 inside its
  own window;
* false segments elsewhere — 1.3-2.0 false segments per real minute on real clips survive even
  behind the gate, and they overlap nothing.

Onset error is small and consistently negative (−2 to −6 frames, i.e. the detector starts
slightly early), because the window that first exceeds the threshold straddles the onset.

### 4. Controls

Yaw control and null edit sit at the false-reject rate in every gated block (5.8-7.2 %, against
a 4.3-5.8 % real rate). In the ungated block they sat at 7.3-8.5 %, i.e. above nominal, which
is the poisoned threshold again rather than a property of the edit.

### 5. Held-out block (scored once, thresholds fitted on the six dev flights)

Only **three** of the 18 held-out real clips pass the frozen gate, so the block carries almost
no rate information; what it does carry is a failure of transfer.

| block | real FRR (residual) | real FRR (whole-clip) | coherence FRR | null edit (25 / 50 / 100 %) | yaw control |
| --- | --- | --- | --- | --- | --- |
| 8 frames, P1 | 66.7 % (2/3) | 66.7 % (2/3) | 33.3 % (1/3) | 2/3, 2/3, 2/3 | 2/3 at every fraction |
| 8 frames, P2 | 100 % (3/3) | 66.7 % (2/3) | 33.3 % (1/3) | 3/3, 3/3, 3/3 | 3/3, 2/3, 2/3 |
| 12 frames, P2 | 33.3 % (1/3) | 66.7 % (2/3) | 0 % (0/3) | 1/3 at every fraction | 2/3 at every fraction |

Thresholds fitted on the six dev flights: residual 21.1-38.5 deg (8 frames, P1), 16.1-23.1 deg
(8 frames, P2), 15.9-24.7 deg (12 frames, P2); whole-clip median 12.5-14.1 deg.

The direction is unambiguous and negative: on these two flights the dev-fitted thresholds
false-reject 1-3 of 3 real clips and flag the **null edit** as often as a real edit. The two
held-out flights are not exchangeable with the dev flights for this statistic, which is the same
failure report 04 recorded at clip level ("one broken flight moves the threshold from 9.4 to
67 deg"). With three evaluated clips this is not a rate; it is a warning that per-flight
calibration spread, not the window machinery, is the binding constraint, and it is reported
rather than repaired. Everything above is dev-block evidence.

## What this supports

* A window scan statistic with max-over-windows conformal calibration **can** be operated at a
  fixed whole-clip false-reject, and it then detects partial violations markedly better than
  the whole-clip median: 2.7x at 25 % coverage and 1.7x at 50 % for partial tilt removal, 4.0x
  and 2.1x for partial mirrored roll.
* The estimator's own failure tail, not the physics, is what blocks segment detection on TII,
  and the frozen confidence gate is the mechanism that makes the difference (89 deg → 16-25 deg
  threshold). This is a measured, attributable result.
* Persistence (two consecutive windows) is what converts a noisy per-window statistic into a
  usable detector at the same false-reject: at 25 % coverage, removal goes 13.0 % → 23.2 % and
  mirror 27.5 % → 40.6 % between P1 and P2 at 8 frames.
* Localisation is coarse: seconds-scale, not frame-accurate.

## What this does not show

* **Not a success against its own bar.** The pre-registered criterion (half the clips at 25 %
  and 50 % coverage, IoU > 0.5) is missed on both counts. Reporting it as a pass would be
  wrong.
* **Not transferable to ANWM.** Segments need a path dense enough to differentiate at the truth
  rate; ANWM's 1 Hz, 16-point paths cannot support a window statistic at all. Nothing here is
  scored on generated clips.
* **Not a solution for scale violations.** x0.5/x1.5 are the gain tier's business (G02); the
  segment detector is at or below the whole-clip baseline there.
* **Not evidence about a flight.** Every number is on clips, with 6 (dev) or 2 (held-out)
  flights in the resampling; the spreads printed are descriptive.
* Not a test of whether the gate itself is right. It is used as frozen, and the amendment
  depends on it; a different gate would move the thresholds.

## Caveats

* The 8-frame threshold on the *coherence* tier (≈1.8, i.e. correlation ≈ −0.8) shows that a
  single 8-frame window cannot rank clips; only the persistence variant is meaningful there.
* Partial injections are composed on the depicted side from the full-clip edits. This is exact
  for a per-frame reader (argued above and checked structurally: no temporal smoothing anywhere
  in the reader path), but it does not re-simulate the fisheye warp at the interval edges, so
  edge frames are not re-estimated and any warp-boundary effect is absent.
* The gate-kept population is a selected population: 69 of 82 dev clips. Rates on it describe
  the part of the corpus the verdict already considers admissible.
* Flight-clustered intervals with 6-8 clusters are descriptive; the two held-out flights make
  the held-out interval uninformative (see the block above).

## Regenerate

```
PYTHONPATH=. python experiments/g01_segment_detection.py --population dev \
    --calib-population gate_kept --length 8 --tag w8_gate_kept
PYTHONPATH=. python experiments/g01_segment_detection.py --population heldout --confirm-once \
    --calib-population gate_kept --length 8 --tag w8_gate_kept
```

Outputs: `results/g01_segments/dev_w8_{all,gate_kept}/summary.json`, `..._w12_...`,
`heldout_w8_gate_kept/`, `heldout_w12_gate_kept/`, plus `variants.csv` (one row per clip,
family, fraction and persistence). Configs: `configs/g01_segment_prereg.yaml` (`3fe20cc`),
`configs/g01_segment_amendment_a.yaml` (`7d44d3c`). Code: `src/verdict/segments.py` and
`experiments/g01_segment_detection.py` at `2ba24a3`. Stored results: `2ba24a3`.
