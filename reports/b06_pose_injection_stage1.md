# B06 — Pose injection, Stage 1: the injected-true arm

Track B addendum, 2026-11-10. Branch `track-b-generation`.
Stage 0 (conventions, no generation): report B05. This report covers the injected-true arm and the
Checkpoint 2 materials; the mirrored control (Stage 2) is not run.

**Status at this revision: 32 of 45 injected clips generated and 30 evaluated.** The tables below
are the interim numbers; the last cells (climb-and-turn, and two accelerate-brake seeds) were still
generating when this was written, so their rows are absent rather than zero.

Runs and artifacts:

| artifact | path | commit |
| --- | --- | --- |
| injected specs (49 clips x 2 arms) | `results/anwm_clips_injected/specs.json` | 32a36c9 |
| generated injected clips (45, arm `injected_true`) | `data/processed/anwm_clips_injected/` (gitignored), R2 `track-b/clips_injected/` | 935f2a6 |
| level twins (9 clips, same seed) | `data/processed/anwm_clips_regen/`, R2 | 71977ec |
| pairing/index table | `data/processed/anwm_clips_injected/clips_injected_index.csv` | PENDING |
| evaluation (interim, 30 of 45 clips) | `results/anwm_injected/eval_injected_injected_true.json` | c1ca216 |
| coverage (delivered level clips, offline) | `results/anwm_injected/coverage_level_original.json`, `coverage_by_cell.csv` | 8a67986, da887a9 |

## Summary in plain language

The model writes drone video from a few past frames and a list of movements. Its movement input has
no way to say "bank left" — only forward/back, left/right, up/down and yaw. The one place an
attitude can enter is the picture the model is shown of the future view, which is built by
projecting the past frames through the camera pose. So we built those future views from the
attitude that physics says the commanded path requires (the direction the thrust must point), fed
them in, and generated the same 45 flights again with everything else held fixed. If the frames
bank with the view, supplying attitude is a way to make the model depict what the path demands; if
they stay level, the model has no attitude capacity and the verifier's job is to prove the
injection changes nothing.

The input side is unambiguous: the injected future view differs from the level one by 16-47 levels
out of 255 depending on the maneuver, and is exactly identical for the straight-cruise control.
The output side is measured below.

## Question and setup

Question: if the *only* channel carrying attitude into ANWM is the reprojected future frame, does a
flatness-derived tilted prior change anything about the generated frames — and can the model's
behaviour be described well enough for a verifier to score it?

- Clips: the 45 Regime C clips whose level twins exist at seed 0 (4 cruise-variance duplicates
  dropped: a zero-tilt injection is a no-op by construction, and Stage 0 measured it as exactly
  identical). Same starts, paths, seeds, checkpoint and sampling steps as the level clips.
- Attitude: drag-free flatness thrust axis at the analytic acceleration of each design, at the
  nominal 1 s; roll/pitch solved from the thrust axis and the designed yaw (conventions verified in
  B05). Poses are built from the rotation matrix, not from ANWM's Euler helper.
- Level twins: 9 clips regenerated with the same seed from cells with commanded tilt >= 10 deg
  (2 each from turn, accelerate-brake and climb-and-turn, plus the cruise control and the two
  determinism clips). Original and regenerated are kept side by side.
- Outputs per clip: 16 generated frames, 16 FFP priors, `manifest.json` (injected roll/pitch,
  arm, paired level id, convention, per-step prior coverage), and the pairing table.
- Evaluation (`remote/b_anwm.py::evaluate_injected`): LPIPS (alex) of each generated frame against
  the paired level frame and against the prior it was conditioned on, plus pixel MAE, pixel spread
  and temporal delta.

## Results

### How different the injected prior is (input side)

Mean absolute difference between the level prior and the injected prior, and their correlation,
over the 16 steps (0-255 pixel units; `results/anwm_injected/prior_difference.json`):

| cell | clips | prior MAE | NCC |
| --- | --- | --- | --- |
| cruise control | 3 | **0.00** | **1.000** |
| accelerate-then-brake | 9 | 16.12 | 0.832 |
| slalom, fixed | 6 | 22.27 | 0.738 |
| slalom, course | 3 | 19.89 | 0.764 |
| climb-and-turn, course | 3 | 23.66 | 0.502 |
| climb-and-turn, fixed | 9 | 47.42 | 0.478 |
| turn, course | 3 | 22.42 | 0.522 |
| turn, fixed | 9 | 39.99 | 0.600 |

The cruise control is bit-identical (0.00 MAE, NCC 1.000), which is the internal no-op check: with
zero required tilt the injected attitude equals the level attitude exactly. The largest input
changes are the fixed-yaw turn and climb cells, where the whole viewpoint rolls.

### Generated frames

Final table columns (filled by `experiments/anwm_checkpoint2_tables.py --markdown`): cell, clips,
LPIPS against the paired level clip, LPIPS against the prior, pixel MAE against each, and prior
coverage (mean, worst step).

Preview on the first clips to finish (level-versus-injected mean absolute pixel difference; the
full per-cell table is filled after the arm completes):

| clip | frames vs level | frames vs prior | prior coverage |
| --- | --- | --- | --- |
| C_cruise_ctl_course_0 | 0.00/255 | 61.4/255 | 0.51 |
| C_cruise_ctl_course_1 | 0.00/255 | 31.1/255 | 0.54 |
| C_cruise_ctl_course_2 | 0.00/255 | 46.3/255 | 0.63 |
| C_turn_5_course_0 | 12.09/255 | 78.9/255 | 0.15 |
| C_turn_5_course_1 | 12.90/255 | — | 0.21 |
| C_turn_5_course_2 | 17.95/255 | — | 0.28 |

At cell level so far (30 of 45 clips): cruise 0.00 (n=3), slalom-course 23.6 (n=1), turn-course
14.3 (n=3, range 12.1-18.0, coverage 0.21), turn-fixed **35.6** (n=9, range 22.9-55.0, coverage
0.61). The fixed-yaw turn, whose prior rolls the whole view (prior MAE 40/255), changes the
generated frames by 35.6/255 on average — the effect tracks both the input change and the coverage.

### LPIPS against the paired level clip (interim, 30 of 45 clips)

| cell | clips | LPIPS vs level | LPIPS vs prior | MAE vs level | MAE vs prior | coverage mean (worst step) |
| --- | --- | --- | --- | --- | --- | --- |
| accelerate-brake, 5 deg | 3 | 0.247 | 0.447 | 21.8 | 41.2 | 0.63 (0.41) |
| accelerate-brake, 10 deg | 2 | 0.184 | 0.497 | 16.8 | 50.0 | 0.59 (0.28) |
| accelerate-brake, 20 deg | 1 | 0.124 | 0.496 | 12.5 | 32.6 | 0.54 (0.22) |
| cruise (control) | 3 | **0.000** | 0.542 | **0.00** | 46.2 | 0.56 (0.36) |
| slalom, 5 deg | 6 | 0.182 | 0.528 | 18.4 | 47.3 | 0.55 (0.35) |
| slalom, 10 deg | 3 | 0.237 | 0.440 | 23.3 | 36.8 | 0.63 (0.48) |
| turn, 5 deg | 6 | 0.223 | 0.629 | 20.4 | 64.6 | 0.39 (0.00) |
| turn, 10 deg | 3 | 0.390 | 0.428 | 31.7 | 36.8 | 0.61 (0.40) |
| turn, 20 deg | 3 | **0.559** | 0.421 | 48.4 | 41.3 | 0.66 (0.44) |

The turn cells show a clean dose-response in the input channel: 0.223, 0.390, 0.559 as the injected
bank goes 5, 10, 20 deg. The accelerate-brake cells *decrease* with the tilt target (0.247, 0.184,
0.124) because only 4 of their 16 steps carry pitch at the 20 deg design, so most frames are
commanded identically. The cruise control is exactly 0.000: at zero tilt the injection is a no-op
in the pose, the prior and the frames.

The three cruise controls are pixel-identical to their level twins: at zero required tilt the
injected pose equals the level pose, so this is the null arm and it returns exactly zero. The three
turn clips that have finished changed by 12.1, 12.9 and 18.0/255 — the injected prior differed by
about 22/255 in that cell, so roughly half to four-fifths of the input change reaches the output.
Ordering matches coverage (0.15, 0.21, 0.28), which is what one expects if the prior is carrying
the signal. Two of these three are the clips with blank priors for late steps, but the difference is present
from step 1 (22.6, 16.3 and 22.3/255 for the three clips) and stays large through step 8, i.e. it
is not an artefact of the empty priors; the full table still stratifies by coverage.

## What this supports

- A generated clip set with a known, physics-derived attitude injected through the only channel the
  architecture has, paired one-to-one with level baselines at the same start, path and seed.
- A measured null for the pairing: the same-seed repeat noise on the level twins.
- Per-clip prior coverage, so any effect can be stratified by how much of the prior was visible.

## What this does not show

- No depicted tilt is measured here (handoff 04 scores it). This report describes the inputs, the
  imagery, and the reproducibility of the pipeline.
- Nothing about Stage 2 (mirrored control), which is not run.
- Nothing about other models, Regime A, or the 144-clip design Track A2 proposed.

## Reading the numbers (how the decision will be made)

The injected arm answers one question: does the generated footage follow a tilted prior? Three
quantities are reported per clip and per cell, in this order of reliability:

1. **frames versus the paired level clip** — the direct measurement. If the model ignores the
   prior, it is zero (as it is for the cruise control, where the prior is bit-identical).
2. **frames versus their own prior** — how far the output sits from the image it was conditioned
   on. A model that copies the prior scores low; a model that ignores it scores high, but so does a
   model that hallucinates freely, so this number is only interpretable next to (1).
3. **prior coverage** — the fraction of the prior that exists. Below about 0.2 the prior cannot
   carry much, and the cells with blank late steps are excluded from any claim about following.

One mechanism is worth stating because it shapes the coverage stratification: at steps where the
prior is **blank** (coverage 0.00) the frames still differ from the level clip, by 2-27/255 in
`C_turn_5_course_*`. Those differences are *carried* by the model's own autoregressive context, not
caused by the prior at that step: the injected clip diverged earlier and keeps diverging. So a
"follows the prior" test must look at steps where coverage is non-zero, and a difference on a blank
step is evidence of persistence, not of attitude.

No threshold is applied here and no depicted-tilt score is computed; that is handoff 04.

## What the numbers say so far (to be finalised)

The injected prior is not cosmetic: in the fixed-yaw turn cell the input image changes by 40/255
and the generated frames change by 36/255 on average, with the effect present from the first
predicted step. The cruise control, whose prior is bit-identical, changes by exactly zero. Whether
the *depicted attitude* tracks the injected attitude is not something this report can say — the
frames could change while staying level (lighting and texture shift with the view) — and that is
the measurement handoff 04 makes on this set.

## Caveats and open items

- Generation is bit-reproducible: eight level clips regenerated with the same path, seed,
  checkpoint and steps are **byte-identical** to the originals (16/16 frames, 0.00/255). The
  pairing therefore has a zero repeat-noise floor. (A first measurement said 2.40/1.09 per 255; it
  is withdrawn as unreproducible, see B05.)
- Four of the 49 Regime C clips have **blank FFP priors for late steps** (turn-5-course and
  climb-5-course, starts 0 and 1: 8 and 3 empty steps of 16). No prior-preservation claim can be
  made on those steps.
- The injected arm is off-distribution by construction: the model never saw a tilted pose.
- **Cost.** A10G at the verified Modal rate of US$1.10/h; CPU US$0.0473/core/h, memory
  US$0.008/GiB/h. Billed Track B spend on 2026-11-10 at the time of writing: **US$1.73**, covering
  the 3-clip determinism run, the 9 regenerated level clips and the first injected clips, including
  container restarts and model loads. Per clip that is about US$0.12 billed (the pure generation is
  4.2 min of A10G, US$0.077); the difference is restart and startup overhead. The 45-clip injected
  arm therefore costs about US$4-5 in total with this much churn — the reason the set is generated
  as two concurrent halves.
