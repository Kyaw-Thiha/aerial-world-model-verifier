# H03 — Is the depicted-attitude reader valid on generated frames?

Track H3, 2026-11-22. Branch `track-h3-reader-validity`, based on `main` (`41d1dad`; no
first-round branch is an ancestor, so H01 is read read-only from `track-h-generator-pilot`
and H02 from `track-h2-generator-coverage`). Pre-registration
`configs/h03_reader_validity_prereg.yaml`, amendment 1
`configs/h03_reader_validity_prereg_amend1.yaml`; both committed before any estimator output
or comparison existed.

**Decision.** On the pre-registered rule the reader is **not valid on generated frames**, and
the failure is carried by the non-learned estimator, which is not competent on this imagery:
against the one independent estimator that is competent on the controls (Perspective Fields),
the reader agrees within 1.2° on ANWM and LingBot bf16 and within 2.0-2.2° on LingBot NF4 and
SolarWM, and where the sign and dynamics are known — SolarWM's 0 → ±20° ramps — both track the
ramp to within 2° over 40°. **The LingBot eye-versus-reader disagreement is resolved in the
reader's favour**: the "oblique skyline" is a static slant of the painterly scene, present on
the level control too, and neither independent estimator sees any roll change across the +20
and -20 ramps.

## Plain-language summary

1. **The reader's roll error where the truth is known is small**: median 0.92° over 60
   controls (30 simulated pinhole frames from Track I, 30 real TII fisheye frames), 0.61° on
   the simulated frames and 1.30° on the real ones. That sets the pre-registered bar at
   1.5 × 0.92 = **1.37°**.
2. **Against Perspective Fields the reader agrees to 0.7-2.2°** depending on the generator:
   ANWM 1.23°, LingBot NF4 2.01°, LingBot bf16 0.74°, SolarWM 2.23°. Against the classical
   horizon estimator it agrees to 1.4-9.9°, but the classical estimator is the outlier: it
   disagrees with Perspective Fields by 3.2-10.6° on exactly those sources, has a 2.56° median
   error on the simulated controls, a 43° median error on the 11 real fisheye frames it can
   read at all, and reads SolarWM's +20° ramp with the **wrong sign**.
3. **The rule as written fails on all four sources** because it requires agreement with every
   independent estimator. Read against the competent one, ANWM and bf16 pass, and NF4 and
   SolarWM fail by 0.6-0.9° (2.0-2.2° against a 1.37° bar).
4. **The LingBot case.** The reader reads a constant roll of about -1° on the level, +20 and
   -20 probes (full 45-49-frame clips: depicted tilt 3.2-4.9°, no ramp response). Perspective
   Fields reads a constant -2.6 to -3.6° with no ramp response. The classical sky-line reads a
   constant +9-10° on the bf16 clips — and +14.8° on the **level** clip's frame 11 — i.e. it
   is following a static scene edge, not the camera. The overlay images show that edge: the
   sun glare on the water. The frames really do not roll.
5. **Where a strong attitude signal exists, the reader is demonstrably right.** On SolarWM's
   +20° ramp the reader goes from -1.4° to -23.4° and Perspective Fields from -4.3° to
   -25.4°, both following the visibly tilted horizon; the classical sky-line reads +19.7° at
   the end of the same clip (wrong sign) because it locked onto a cloud band.

## Question and pre-registered rule

Does the GeoCalib reader, applied the way H01/H02 apply it to generated frames (square centre
crop, 224 px, pinhole, f = 112 px), report the roll that is actually in the frame? The rule,
fixed before any estimator ran:

> E_control = the reader's median roll error over all 60 controls. The reader is valid on a
> generated source S if, for every independent estimator reading at least 70 % of S's frames,
> median |roll_reader − roll_estimator| ≤ 1.5 E_control. It is valid on generated frames iff
> all four sources pass; otherwise name the source, scene and tilt range that fails.

Roll is the metric because it is the part of the gravity estimate that is invariant to focal
length for a pinhole camera and to any centred radial fisheye model: those models preserve the
azimuthal direction of a ray, so the image-plane direction of world-up — the roll — does not
depend on the camera model. The tilt (thrust-axis angle) additionally needs the focal length
and is reported only as a secondary check.

## Setup

**Population, 150 frames** (fixed by the pre-registration; five evenly spaced frames per
clip, index rule `round(linspace(0, n-1, 5))`):

| source | clips | frames | what |
| --- | --- | --- | --- |
| ANWM | 6 seed-1 designed paths | 30 | cruise, slalom 5/10, climb-turn 10/20, accel-brake 20 |
| LingBot NF4 | level, +20, -20 probes | 15 | H01's roll-blindness clips |
| LingBot bf16 | level, triangular-20, -20 | 15 | H02's eye-versus-reader clips |
| SolarWM | level, +20, -20 probes; turn-10, turn-20 mirrored, slalom-10 | 30 | the generator that does follow ramps |
| controls, simulated | 6 feasible Track I clips | 30 | industrial/urban/forest, mounts 0/15/30 |
| controls, real | 6 TII flights | 30 | three autonomous, three piloted, mounts 30/50 |

**Estimators.** (a) The reader: GeoCalib `1dedb29`, pinhole 224 px, f = 112 px. (b) Classical,
non-learned (`src/perception/horizon.py`): a sky-boundary line (connected-component sky mask,
RANSAC line fit) and a rectilinear line family (LSD, orientations modulo 90°, circular mean),
the larger normalised support wins; abstains when neither qualifies. Its sign and accuracy are
pinned by synthetic renders (`tests/test_h03_classical.py`). (c) Perspective Fields,
`Paramnet-360Cities-edina-centered`, commit `d54be73`; licence Adobe Research License Terms —
noncommercial research use only, weights downloaded at run time and never committed or
re-hosted (recorded in `docs/decisions.md`). The sign of both independent readings was fixed
by a +10/-10° image-rotation probe before the analysis: the reader's roll falls by 0.96 per
degree of image rotation, Perspective Fields' rises by 0.95, so the latter enters with a sign
flip.

**Controls and truth.** Track I gives the exact camera pose per rendered frame
(`render.npz`): `up_cam = R_cbᵀ R_bwᵀ e3`. TII gives mocap attitude at the frame timestamp
through the project's analytic mount `camera_to_body_pitched(uptilt)`. Both are the same
convention the reader's gravity output is in.

## Results

### 1. The reader's error where the truth is known

| control | n | reader roll error, median [95 %] | Perspective Fields | classical |
| --- | --- | --- | --- | --- |
| pooled (pre-registered) | 60 | **0.92° [0.55, 1.44]** | 1.11° | 3.37° (41 read) |
| simulated (Track I pinhole) | 30 | 0.61° [0.35, 1.17] | 0.92° | 2.56° (30 read) |
| real (TII fisheye, same feed) | 30 | 1.30° [0.70, 2.67] | 1.34° | 43.4° (11 read) |
| real, reader's native fisheye path | 30 | 1.03° [0.79, 2.04] | — | — |

Intervals are frame-level bootstraps (5,000 draws, seed 1) within one control family; frames
are not independent within a clip, so they are descriptive. The classical estimator's real-frame
error is not a small perturbation — it reads 11 of 30 fisheye frames at all — and its simulated
error has an interquartile range up to 8.1°.

### 2. Reader versus the independent estimates on generated frames

Bar = 1.5 × 0.92 = **1.37°**. `pair` is the two independent estimators' agreement with each
other, a measure of whether they can adjudicate at all.

| source | reader vs Perspective Fields | reader vs classical | classical vs PF (pair) | coverage (PF / classical) | pre-registered |
| --- | --- | --- | --- | --- | --- |
| ANWM | **1.23°** | 7.24° | 6.19° | 1.00 / 0.90 | fail (classical) |
| LingBot NF4 | 2.01° | 1.40° | 3.18° | 1.00 / 1.00 | fail (both) |
| LingBot bf16 | **0.74°** | 9.92° | 10.56° | 1.00 / 1.00 | fail (classical) |
| SolarWM | 2.23° | 2.33° | 4.45° | 1.00 / 1.00 | fail (both) |

**Pre-registered verdict: not valid on generated frames.** Every one of the four sources fails
the literal rule, all four because of the classical estimator and two also because the
Perspective Fields disagreement exceeds 1.37°.

**Attribution (the pre-registered secondary analysis).** The classical estimator cannot pass
its own control test: 2.56° median on simulated pinhole frames with a heavy tail, 43° on real
fisheye, and it reads SolarWM's known +20° ramp as +19.7° at the end (the reader and
Perspective Fields both read about -24°; the image in `results/h03/overlays/probe_p20_080.png`
is visibly rolled the other way). Its 6-10° disagreements are therefore not evidence about the
reader. Perspective Fields is competent on the controls (0.92° simulated, 1.34° real). Against
it the reader passes on ANWM and bf16 and misses the bar by 0.6° (NF4) and 0.9° (SolarWM);
on SolarWM, PF's own reading of the *level* probe is a constant -4.3° while the reader's is
-1.4°, so a 2-3° generated-image bias of the independent estimator is the likelier explanation
of the 2.2° gap than a reader bias of the same size in the opposite direction.

### 3. The LingBot case, resolved

Full clips, all frames, the reader's stored rows (H01/H02) and the independent estimators:

| clip | reader depicted tilt, median (first-10 / last-10) | reader roll | PF roll | classical roll |
| --- | --- | --- | --- | --- |
| NF4 level | 3.21 (2.92 / 3.06) | -0.97 | -3.61 | -1.33 |
| NF4 +20 ramp | 3.30 (3.45 / 3.44) | -1.04 | -3.36 | -2.97 |
| NF4 -20 ramp | 3.77 (2.85 / 4.22) | -1.14 | -3.65 | +0.16 |
| bf16 level | 4.56 (3.04 / 5.89) | -1.22 | -2.56 | +10.04 |
| bf16 triangular-20 | 4.29 (3.47 / 4.84) | -0.95 | -1.96 | +8.37 |
| bf16 -20 ramp | 4.91 (2.98 / 5.17) | -1.21 | -1.29 | +9.02 |

No estimator shows a roll *response*: the reader moves by 0.1-0.3° between ramps, Perspective
Fields by 0.7-1.6° with no consistent sign, and the classical estimator's 8-10° on the bf16
clips is a constant offset that is also present on the level clip. The overlay
(`results/h03/overlays/probe_rollramp_fast20_044.png`, `..._m20_044.png`) shows what the eye
was reading: the skyline is level, while the sun glare on the water runs obliquely across the
frame. **The eye-versus-reader disagreement is a scene-content artefact, not a reader error,
and H01/H02's LingBot conclusion stands.**

### 4. Where a strong signal exists, the reader tracks it

SolarWM's convention probes (the only generated clips with a 40° commanded roll swing):

| probe | frame 10 | 20 | 40 | 60 | 80 | last-10 tilt |
| --- | --- | --- | --- | --- | --- | --- |
| +20 ramp, reader roll | -1.4 | -4.3 | -8.7 | -16.7 | -23.4 | 21.1° |
| +20 ramp, PF roll | -4.3 | -8.2 | -13.4 | -19.0 | -25.4 | — |
| -20 ramp, reader roll | -1.4 | +3.4 | +5.5 | +11.7 | +17.7 | 16.8° |
| -20 ramp, PF roll | -4.5 | +3.4 | +7.7 | +14.3 | +19.7 | — |

Both track the ramp with the correct sign and within 2° of each other over the whole swing;
the constant offset between them is the level-probe bias noted above.

## 5. Recomputing the affected headline numbers with the independent estimator

The pre-registered rule failed, so the numbers behind the H01/H02/E5 claims were recomputed
with Perspective Fields on the **full clips** (not the 150-frame sample): every LingBot
full-precision probe, the SolarWM convention probes, the 25 `flatness`-arm clips of H02's
75-clip grid, and all 44 ANWM seed clips.  The reader's own rows are read read-only from the
H01/H02 worktrees.  The depicted tilt is the angle between the body z axis and world up, with
Perspective Fields' up direction built from its parameter head
`rotz(roll) rotx(pitch) (0,1,0)` in its y-up frame, flipped to the project's y-down camera
frame; that construction reproduces `-pf_roll` exactly and gives `(0,-1,0)` on a level frame.
(The per-pixel field reconstruction was tried first and rejected: it returned exactly `-up`
on level frames and drifted by 7 deg at a 25 deg roll, so the parameter head is used.)

| population | clips | reader tilt, median (deg) | PF tilt, median (deg) | delta (PF - reader) | abs delta |
| --- | --- | --- | --- | --- | --- |
| LingBot bf16 probes | 5 | 4.29 | 2.28 | **-2.01** | 2.28 |
| SolarWM probes | 5 | 9.94 | 13.57 | +1.94 | 2.21 |
| SolarWM grid, flatness arm | 25 | 4.68 | 8.48 | +2.68 | 2.78 |
| ANWM seed clips | 44 | 0.64 | 2.33 | +1.66 | 1.48 |

**What changes.** Nothing in the classification, and the magnitudes move by 1.5-2.8 deg:

* **LingBot (H01/H02).** With PF the probes read 1.3-2.4 deg of depicted tilt instead of
  3.9-4.9 deg, still flat across the +20, -20, triangular and 60 deg ramps (the level probe is
  2.12 deg, the largest ramp 2.35 deg). The roll-blindness conclusion is unchanged and if
  anything stronger.
* **SolarWM held banks (H02).** On the flatness arm of the grid PF reads 8.48 deg median
  against the reader's 4.68, while the commanded required tilt on those paths is 9-21 deg. The
  turn-10 and turn-20 clips stay infeasible; the five "feasible slalom" clips under H02's
  absolute 12-frame rule move closer to the threshold but the direction of the result is the
  same. The +2.7 deg offset is a systematic difference between the two estimators on this
  imagery, not a change of classification.
* **ANWM (E5).** PF reads 2.33 deg against the reader's 0.64 on the 44 seed clips; both are
  far below the 8-20 deg the designed paths require, so the infeasibility finding is unchanged.
* **The reader's own bias on generated frames is not resolvable from this.** PF reads higher
  on ANWM and SolarWM and lower on LingBot; without truth on generated frames the sign of the
  reader's bias cannot be established, which is exactly why the rule is stated on controls.


## What this supports

* **The reader's roll is accurate on frames with known truth** (0.6° simulated, 1.3° real) and
  agrees with the only competent independent estimator on generated frames to within 2.2°,
  with the largest gaps explained by that estimator's own generated-image bias.
* **H01/H02's LingBot conclusion is not a reader artefact.** Three independent readings and a
  visual check agree that the frames do not roll with the command.
* **H02's SolarWM ramp result is not a reader artefact**: the reader and Perspective Fields
  both follow the ±20° ramps.
* **The classical horizon estimator is not a usable second reader on this imagery**, and the
  pre-registered "every estimator must agree" form of the rule is too strong when one
  estimator is demonstrably incompetent.

## What this does not show

* **It does not establish the reader as valid by the pre-registered rule.** That verdict is
  "not valid". The attribution above says the failure is not evidence of reader error, but it
  is an interpretation, not a re-registration.
* **It does not measure the reader's tilt (as opposed to roll) error on generated frames.**
  Tilt needs the focal length, and no generated clip has a truth-bearing control except the
  commanded ramps, whose fidelity is the thing under test. The tilt comparison against
  Perspective Fields' parameter head is reported as a secondary number only.
* **It does not adjudicate frames where both estimators are biased in the same direction.**
  Two learned estimators trained on real photographs can share a failure mode on painterly
  imagery; the classical estimator was meant to catch that and could not.
* **The human-label set was not labelled.** The optional local tool and 60-frame set exist;
  without labels it contributes nothing and is not reported as evidence.
* **Five frames per clip.** The sample is 150 frames by design; per-clip claims use the
  stored full-clip rows, not the sample.

## Caveats

* **The controls use the generated-frame feed.** Real TII frames are 640×480 fisheye, centre
  cropped to a square and treated as a 90° pinhole, which is not their camera model. Roll is
  invariant to that mismatch (the native-fisheye re-read gives 1.03°, close to 1.30°), but the
  control error is not a full calibration of the reader's tilt on TII.
* **Different estimators, different frames of reference.** The reader and PF both estimate a
  gravity direction, but PF is trained on street-level and indoor imagery; its ~3-4° bias on
  these painterly aerial frames is visible on level probes.
* **The bar is tight because the reader is good on controls.** 1.5 × 0.92° is smaller than the
  independent estimator's own generated-frame bias, so "fail" at 2.0-2.2° does not mean the
  reader is wrong by 2° — it means the two estimators differ by that much and neither has
  truth.
* **The classical estimator is documented and unit-tested, not tuned.** Its abstention and
  thresholds were fixed in amendment 1 before any comparison; its poor showing is a result.
* **Licence.** Perspective Fields is used under the Adobe Research License (noncommercial
  research only, no redistribution); the weights are not committed or re-hosted.

## Reproduction

```
set -a; . ./.env; set +a
pixi run -e gpu python experiments/h03_select_frames.py          # 150 frames + manifest
pixi run -e gpu python experiments/h03_classical.py              # local, CPU
pixi run modal volume put awmv-h3-data data/h03_frames /h03_frames
pixi run modal run remote/h03_readers.py::main --mode probe      # sign convention
pixi run modal run remote/h03_readers.py::main --mode read       # reader + PF
PYTHONPATH=. pixi run -e gpu python experiments/h03_analyze.py
PYTHONPATH=. pixi run -e gpu python experiments/h03_overlay.py --source lingbot_bf16 --frames last
pixi run modal run remote/h03_recompute.py::main --batch 8          # full clips, PF
pixi run python scripts/pull_volume.py awmv-h3-data h03_recompute results/h03_recompute
PYTHONPATH=. pixi run -e gpu python experiments/h03_recompute_compare.py
```

Unit tests: `tests/test_h03_classical.py` (5 tests: sign, synthetic horizon with and without
mount, line family, abstention).

## Provenance

| item | where |
| --- | --- |
| pre-registration | `configs/h03_reader_validity_prereg.yaml`, commit `c629016` |
| amendment 1 (estimator details) | `configs/h03_reader_validity_prereg_amend1.yaml`, commit `141ac4d` |
| sample | `results/h03/frames_manifest.csv` (150 rows, truth where known) |
| reader + PF rows | Modal volume `awmv-h3-data:/h03_readers/depicted_main.jsonl` (150 rows) |
| rotation sign probe | `results/h03/readers/rotation_probe.json` |
| classical rows | `results/h03/classical.jsonl` |
| full-clip recompute | `results/h03_recompute/pf_rows.jsonl` (3,494 rows), `results/h03_recompute/compare.json` |
| analysis | `results/h03/analysis.json`, `results/h03/frames_scored.csv` |
| overlays | `results/h03/overlays/` |
| H01/H02 reader rows | read read-only from `~/Documents/Projects/awmv-track-h` and `~/Documents/Projects/awmv-h2` |
| cost | see `docs/COST_LEDGER.md` |
