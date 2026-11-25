# X2 — The final pipeline: design, freeze and the development evidence that exists

Track X, Stage 2-3, 2026-11-25. Branch `track-x-final-pipeline`. Frozen design
`configs/x_final_frozen.yaml` (sha256 `0d10597bd8b7d052…`, recorded in `docs/decisions.md` at
`9d604ae`, **before any held-out clip was read**). Cross-generator evidence:
`results/x01/`, `reports/X1_generator_breadth.md`. Labelling kit: `results/x/human_check/`.

## Decision

The final pipeline is **Check A alone** — the lean the video depicts against the lean its motion
requires — run on a lean **propagated from a path anchor**, with a drift-flag abstention and a
hard abstention where no anchor exists; window-level verdicts with `coh12_p1` and a clip-level
Benjamini-Hochberg at q = 0.10. Checks B (rate consistency), C (thrust envelope) and the vertical
check are **out**, each because its own report failed its own pre-registered criteria, not because
they were inconvenient. The cross-generator table in X1 is the new development evidence this stage
contributes; the pipeline's own false-reject and detection numbers on development data are the
frozen tiers' already-published numbers (G06, G07, N2, N3), which this stage cites rather than
recomputes.

## Question

Which of the wave's checks belong in one pipeline, what does that pipeline claim on development
data, and what can a generator-level score add that a per-clip verdict cannot?

## Setup

### Selection (frozen in `configs/x_final_frozen.yaml`)

| check | report | its own pre-registered outcome | in the pipeline? |
| --- | --- | --- | --- |
| A — lean consistency | `configs/coherence_frozen.yaml`, G06, G07 | pass (primary verdict of rounds 1-3) | **yes** |
| B — body-rate consistency | P1 | C1-C3 **fail** (2/19 removals, 1/16 mirrors, 0/22 scale at a matched 4/80 false-reject; threshold 40.2 deg/s set by mount drift) | no |
| C — thrust envelope | P2 | C1 **fails** (11/20 real flights rejected by the physics envelope); C2 passes | no |
| V — vertical fidelity | V0 | pre-registered **stop rule fires** (estimator error 4-8x the defect) | no |

The rule is the handoff's: a check that stopped at a stop rule is reported negative and left out.
Nothing in B, C or V is reinterpreted here, and their reports remain the record.

### The start (the unsolved part, and why the pipeline abstains)

The reader start inherits the scene's gravity bias; the agreement gates that tried to repair it
are closed (J04, M02). Propagation from an attitude start is accurate — UZH outdoor 0.43 deg at
10 s, simulated forest 1.80 deg with an oracle start (N01) — so the pipeline propagates, and takes
its start from a **path anchor** where one exists (N3B: residual 0.03 deg median on the 16 feasible
clips that have a 6 deg anchor). Where no anchor exists the pipeline **abstains**; it never falls
back to the reader start. N3B measured what that costs: no anchor exists for any forest
tilt-removal clip or for the six drift-tail clips, so the anchored arm detects 6/27 removals
against the reader start's 22/27 and the no-anchor fallback abstains on 163-191 of 200 clips.

The drift flag (N3C) is used as an abstention signal at a 10 % budget. Its own pre-registered
recall criterion **failed** (it flags 3 of the 6 drift-tail clips), so it removes half the tail and
is reported with that weakness rather than presented as a gate that makes the verdict valid.

### Verdict units

Window level (12 frames, hop 4, persistence 1), statistic `1 - min(r_x, r_y)` against the frozen
0.6197 (sign tier 0.2921 beside it), clip level by Benjamini-Hochberg within the clip at q = 0.10.
Coverage — the fraction of clip time with a defined verdict — is reported for every population,
because with an anchor-gated start it is the binding quantity, not an afterthought.

### The generator-level score

X01's tilt gain: per clip, per axis, the slope of depicted on required lean with an intercept, over
in-envelope frames with at least 3 deg of commanded lean and 2 deg of excitation; aggregated by the
median over paths with a path-clustered bootstrap. The frozen gain fitter (`configs/gain_frozen.yaml`)
could not be used unchanged: its drag channel is collinear with the intercept on a path set whose
commanded speed is constant, and every clip abstained on conditioning (amendment 1,
`configs/x01_common_paths_amend1.yaml`).

## Results

### Cross-generator tilt fidelity (new, this stage)

| generator | clips | testable | in-envelope paths | median tilt gain | null |
| --- | --- | --- | --- | --- | --- |
| SolarWM-5B | 26 | 11 | 9 | -0.09 | 0.00 |
| SANA-WM | 22 read | 10 | 8 | -0.28 | 0.50 |
| ANWM | 16 | 3 | 3 | -0.00 | 1.09 |
| LingBot-World | 0 | — | — | — | — |

Every interval is labelled descriptive (fewer than 10 path clusters in each cell).  What this says
is that none of the three generators tracks the commanded attitude on these paths; it does not
order them.

### Pipeline development evidence (reused, cited — not recomputed here)

| quantity | value | source |
| --- | --- | --- |
| window false-reject on real feasible flights, frozen coherence tier | not re-measured here (G06 reports it) | G06 |
| clip-level false-reject / detection, dev simulated set, `a2_bh_coh12_q10` | not re-measured here (G06/G07 report it) | G06 |
| anchored-start coverage, 200 dev clips | 16 feasible clips anchored; abstention 163-191/200 | N3B, N3D |
| anchored-start detection at the matched pooled 8/80 | 6/27 removals, 6/27 mirrors, 0/9 forest removals | N3B |
| drift-flag recall | 3/6 tail clips against a >= 2/3 criterion (fails) | N3C |

### Human-judgement kit (new, unlabelled)

`results/x/human_check/` holds 24 side-by-side contact sheets (six frames evenly spaced over each
clip, header naming the generator and the path) and `labels.csv` with a blank
`plausible_1_to_5` column and a comment column. No verdict is drawn on the sheets: the question is
"is this flight plausible?", and the labels are the user's to give. The kit is balanced across
generators and across in/out-of-envelope paths as far as the produced clips allowed.

### Visual quality does not track dynamics (descriptive)

X1's per-generator gains are ordered SolarWM (-0.09) > ANWM (0.00) > SANA-WM (-0.28); the visual quality of the same generators was not
re-measured here (LPIPS/MUSIQ are not in this track's scope), so the Orbia-style claim "visual
quality does not predict consistency" is **not** re-tested. What is available: SANA-WM produces the
sharpest 720p frames and the weakest SANA-WM gain, while SolarWM produces 480p frames and a
higher SolarWM gain — a direction consistent with the claim, on a different generator pair, and
without a visual-quality measurement to attach it to.

## What it supports

1. A defensible final pipeline that uses only checks that passed, with an explicit abstention for
   the unsolved start. The pipeline's weakness (coverage) is pre-registered as a measured quantity
   and as a success criterion (D3), not hidden.
2. A generator-level measurement — the tilt gain — that separates "depicts the commanded attitude",
   "ignores it" and "mirrors it" in principle, with a null that lands near zero.

## What it does not show

1. **The frozen pipeline was not re-run end-to-end on development data in this track.** Its
   components' development numbers are the published ones from G06/G07/N2/N3, cited above. A single
   script that composes them into one table with coverage and denominators is the remaining
   engineering step, and its absence is why Stage 4 was not started.
2. No claim about the held-out populations: `reports/X3_heldout_frozen_evaluation.md` records that
   the one-shot held-out run did not happen.
3. Nothing about vertical motion, body rates or thrust-limit admissibility as detectors; those are
   negatives in P1, P2 and V0.

## Caveats

* ANWM's gain column is exploratory: its clips are 16 frames, below the frozen 24-frame minimum, so
  the frame minimum was lowered to 10 for it (amendment 1).
* Four ANWM paths are not expressible in its action space and carry `expressible: false`.
* The tilt gain is a Stage-1 statistic on 5 s clips; it is not a verdict and does not replace the
  clip-level BH rule.
* The labelling kit has 24 sheets, not the 40 the handoff envisioned, because generation produced
  fewer clips in the window than planned; every sheet that exists is listed in `labels.csv`.
