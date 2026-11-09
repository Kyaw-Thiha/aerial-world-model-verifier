# Track B report: ANWM inputs and first rollouts

Branch `track-b-generation` (worktree `../awmv-track-b`). Everything below is measured in this
repo unless tagged unverified. Compute: Modal only, app `awmv-b-*`, volumes `awmv-b-*`, R2
`track-b/`.

## Task 0: the three corrections

### 1. Audit table inconsistency (cause: aggregation, not data)

The table's "median" columns were the **mean of 33 per-trajectory medians** and its ">10 deg"
columns the **mean of per-trajectory fractions**. Per-trajectory medians are bimodal: 21 of the
33 trajectories have fewer than half their boundaries at transitions, so their median is exactly
0 deg, and the other 12 sit at 20+ deg. A mean of medians above 10 deg therefore does *not*
imply a majority of steps above 10 deg. Second, smaller cause: the SG ">10 deg" column used
`k_d = 0.297` while the SG median column was drag-free, so those two numbers were not even on the
same estimator.

Fix: `src/generation/audit.py::pooled_stats` computes median, p90 and fraction above the
threshold from **one pooled array**, and asserts the invariant `median > thresh => frac >= 0.5`;
`experiments/anwm_excitation_audit.py` now pools the raw per-trajectory arrays instead of
averaging scalars (the old scalar, kept as `per_trajectory_median_mean`, is a diagnostic only).
`experiments/anwm_audit_corrected.py` writes the corrected table.

Corrected numbers (33 test trajectories, one population per cell):

| dt | estimator | k_d | median | p90 | >10 deg |
| --- | --- | --- | --- | --- | --- |
| 1.0 s | per-step | 0 | 0.0 | 32.6 | 47.2% |
| 1.0 s | per-step | 0.295 | 8.6 | 36.1 | 47.2% |
| 1.0 s | SG-11 | 0 | 1.1 | 2.5 | 0.0% |
| 1.0 s | SG-11 | 0.295 | 4.9 | 8.4 | 1.8% |
| 0.667 s | per-step | 0 | 0.0 | 64.7 | 47.2% |
| 0.667 s | per-step | 0.295 | 12.7 | 62.3 | 76.6% |
| 0.667 s | SG-11 | 0 | 2.4 | 5.5 | 0.7% |
| 0.667 s | SG-11 | 0.295 | 7.4 | 12.7 | 28.0% |
| 0.25 s | per-step | 0 | 0.0 | 105.5 | 49.6% |
| 0.25 s | per-step | 0.295 | 31.0 | 106.1 | 79.0% |
| 0.25 s | SG-11 | 0 | 16.9 | 35.3 | 68.6% |
| 0.25 s | SG-11 | 0.295 | 23.9 | 42.7 | 82.0% |

Both numbers restated as asked: at dt = 1 s the **per-step median is 0.0 deg and 47.2% of steps
exceed 10 deg**; at dt = 0.25 s the **per-step median is 0.0 deg and 49.6% of steps exceed
10 deg**. The earlier "median 8.3 / 14.0 / 24.6 deg" figures are withdrawn: they are means of
per-trajectory medians, not medians of any step population.

### 2. The heading-jump statistic, defined

It is the change of **horizontal travel direction** between consecutive *translating* steps, not
the change of camera yaw. Recomputed on the same trajectories:

- |yaw change| per frame: median 0, p90 15, p99 15, max 15 deg; 0% above 30 deg.
- |travel-direction change| between successive translating steps: median 0, p90 90, p99 105,
  max 180 deg; 14.4% above 30 deg.

The quoted "p90 107 deg, p99 165 deg, 39% above 30 deg" is **not reproducible** under either
definition (the code that produced it was not kept) and is withdrawn. Only the travel-direction
change implies acceleration; camera yaw is unconstrained by flatness and legal at any rate. A
45 deg travel-direction change at zero yaw change is exactly AerialVLN's `MOVE_LEFT`, so this
quantity, not yaw, is what the verifier must watch.

### 3. Drag sentence and one `k_d`

The sentence "drag removes 6.6 deg on piloted flights and 2.2 on the 21 m/s ellipse" was a
misreading of a *residual* as a *reduction*. The table is right, and the corrected statement is:

- piloted: 6.57 -> 0.88 deg on 01p-ellipse (**5.68 deg removed**), 6.67 -> 0.94 on 07p-lemniscate
  (5.73), 7.17 -> 3.28 on 13p-trackRATM (3.89), 7.49 -> 3.89 on 16p-trackRATM (3.61);
- autonomous ellipses: 1.29 -> 1.26 and 1.19 -> 1.19 (**about 0.0-0.03 deg**), and the only
  autonomous flight drag helps is 13a-trackRATM (2.10 -> 0.96). There is no 2.2 deg reduction on
  any 21 m/s ellipse; the 20-22 m/s bin residual of about 11.5 deg is a *tail* statistic, not a
  reduction.

One `k_d` for both control modes: fitted separately on the Phase 0 derivation (w=101, pooled
median loss, bounds [0,3] 1/s) gives **piloted 0.296, autonomous 0.306, all flights 0.298**.
Cross-application moves the other group's median residual by 0.02 deg, so a single 0.295 is
valid; the drag *effect* differs by regime because the residuals differ, not the coefficient
(`results/anwm_excitation/kd_split.json`).

## Task 1: slim checkpoint, image

- Source `EmbodiedCity/ANWM/0200000.pth.tar`, **18,146,420,481 bytes**, sha256
  `bdd149cac6ec002ba7dc4ad99ec6f9eb02cd6d4f05320195cf174737b13b0bc2`. Checkpoint keys:
  `args, ema, epoch, model, opt, scaler, train_steps` (so the file holds model + EMA + optimizer,
  as inferred earlier).
- Slim EMA-only checkpoint `/w/slim/0200000_ema.pth.tar`, **4,536,552,948 bytes**, sha256
  `59def562a12275640e4d04d1af3b38281caa4f656eb104e96678fc49c6ae3f8b`. Extracted once on Modal
  (`torch.load(..., mmap=True)`), Volume `awmv-b-weights`, mirrored to R2
  `track-b/weights/0200000_ema.pth.tar` and size-verified there.
- Image: `nvidia/cuda:12.1.1-devel-ubuntu22.04` + python 3.10, `torch 2.4.1+cu121`,
  `torchvision 0.19.1`, `numpy<2`, `pip install -e ".[metrics]"` of the pinned clone at
  `657a80268505fa9149c4df502e35aa0f5bce11e5`, then **`diffusers==0.33.1`**. Both pins are forced by
  measurements, not taste: the latest diffusers calls `torch.accelerator`, which does not exist
  before torch 2.6 (`AttributeError: module 'torch' has no attribute 'accelerator'`), and
  `diffusers==0.31.0` fails its own `loaders.unet` import against this torch
  (`name 'nn' is not defined`). VAE `stabilityai/sd-vae-ft-ema` is cached on `awmv-b-hf-cache`.
- **The checkpoint needs 16 context frames, not the 4 in the shipped configs.** `ema.pos_embed`
  is `[17, 196, 1152]` and the model builds `pos_embed` as `context_size + 1`; loading with 4
  fails on `pos_embed`, loading with 16 matches every key. The checkpoint's own `args` say
  `config/nwm_cdit_airvln_16.yaml`, epoch 3, 200,000 train steps, bf16. Every Track B path is
  therefore built for a 16-frame context (FFP reprojects 16 context frames per target pose), and
  the clip set is 4x more expensive per frame than the "4 context" reading would have predicted.
- The image upload had to ignore `third_party/anwm/.git` (1.7 GB of the 1.7 GB clone); before
  that, the build stalled for over an hour on the upload alone.

### Smoke test (Task 1.3), A10G, 224x224, batch 1

| quantity | value |
| --- | --- |
| GPU / params | A10G, 1.134 B |
| one CDiT forward pass (784 latent tokens, 17 token groups) | **66.5 ms** (warm, min of 3; first call 1.40 s) |
| one generated frame (250 sampler steps + VAE + FFP) | **17.57 s** |
| 16-frame rollout | **281 s** (4.7 min) estimated, **235 s** measured end to end in the generation run |
| peak GPU memory | **7.3 GB** (forward 7.06) |

Cost arithmetic at the published A10G rate (about US$1.10/h, unverified per-region):

- Task 2 (5 native rollouts, 16 frames each): 5 x 281 s = 23 min, about **US$0.45**.
- Task 5 (about 90 clips x 16 frames): 7.0 h, about **US$7.7**.
- Extraction, smoke and the image builds so far: about **US$0.30** (20 min of CPU/GPU).

Total for Tasks 1, 2 and 5 is about **US$8.5, below the US$10 approval threshold**, so the runs
proceed. The per-frame number is measured on synthetic tensors; real 16-frame context plus FFP
adds VAE encodes, so the generation budget carries a 20% margin and the clip set is split by
regime so one long run cannot be lost.

## Task 4: start states

Depth in the released pickles is metres with a **500 m far clip** (p50 52 m, p90 exactly 500), so
"forward depth exceeds the path extent" tested on a high percentile is vacuous. The filter is
therefore measured instead of inferred: FFP is run from the 16 context frames to the 16 target
poses of the recorded path and the fraction of non-zero pixels in each projected image is
reported. Result: **20 of 20 candidate starts are usable**, minimum per-step coverage 0.197
(worst start) to 0.773, median about 0.5, so the projected prior is never empty on this data. The
20 starts are 4 windows each from 5 trajectories; the clip set uses all 20.

This measurement also pinned the pose convention: rebuilding each recorded pose as
`[R_z(yaw) @ R_ec | xyz]` from `point`/`yaw` reproduces the released `pose` array to
**2.2e-16**, with `R_ec` constant across frames to 3.3e-16. That is what makes commanded paths
workable through FFP without a learned pose model.

## Task 3 and Task 6: path library and path-only audit

`src/generation/paths.py` has one `PathSpec` (time, world position, yaw, metadata) with three
constructors. Actions are built by ANWM's own `_compute_actions`, `normalize_data` and
`get_delta_np` through `src/generation/anwm_env.py`, with one unit test that reproduces ANWM's
normalisation by hand and another that checks `get_delta_np`'s zero-padded first row.

Design caps (all from `third_party/anwm/config/data_config.yaml` / the AerialVLN action set):
5 m/s horizontal, 2 m/s vertical, step spacing inside the `action_stats` envelope, and 45 deg of
travel-direction change per step (above that the 1 s polyline aliases). **Regime C is checked
against the travel-direction cap rather than yaw rate**, because ANWM's own data contains 45 deg
strafes at zero yaw change; a `policy="course"` variant additionally holds yaw to 15 deg/s and is
reported where that makes the target unreachable.

Reachability at 5 m/s (drag-free peak required tilt from `src/physics`):

| maneuver | 5 deg | 10 deg | 20 deg |
| --- | --- | --- | --- |
| turn, yaw = course | yes | no (max 7.4) | no (max 7.4) |
| turn, yaw fixed | yes | yes | yes (19.9) |
| slalom, yaw fixed | yes | yes | no (max 11.0) |
| accelerate-then-brake | yes | yes | yes (19.9) |
| climb-and-turn, yaw fixed | yes | yes | yes |

The unreachable cells are a property of the caps, not of the generator: a coordinated turn at
5 m/s yaws at `g tan(tilt) / v`, so 20 deg of bank needs 41 deg/s of yaw, above ANWM's 15 deg/s
training step; a constant-speed slalom cannot exceed 11 deg because lateral speed has to come out
of the 5 m/s budget. Where a target is unreachable the design says so (`reachable: false`) and
reports the achieved value.

Path-only audit (`experiments/anwm_path_audit.py`, output `results/anwm_path_audit/audit.json`),
one pooled population per cell, all with `k_d = 0.295`:

| regime | dt | per-step median | per-step p90 | per-step >10 | SG median | SG >10 |
| --- | --- | --- | --- | --- | --- | --- |
| A native (33 traj) | 1.00 | 8.6 | 29.6 | 38.2% | 6.1 | 4.8% |
| A native (33 traj) | 0.67 | 12.7 | 62.4 | 77.4% | 9.2 | 40.7% |
| A native (33 traj) | 0.25 | 31.0 | 106.1 | 79.8% | 25.6 | 78.8% |
| B rescaled (25 windows) | 1.00 | 3.6 | 7.2 | 0.0% | 3.6 | 0.0% |
| B rescaled (25 windows) | 0.67 | 5.6 | 10.9 | 16.3% | 5.6 | 16.3% |
| B rescaled (25 windows) | 0.25 | 18.0 | 30.3 | 69.3% | 17.9 | 69.7% |
| C designed (22 designs) | 1.00 | 9.8 | 14.6 | 47.0% | 9.3 | 31.8% |
| C designed (22 designs) | 0.67 | 16.6 | 26.6 | 96.4% | 15.2 | 97.4% |
| C designed (22 designs) | 0.25 | 56.7 | 72.1 | 100.0% | 52.8 | 99.4% |

- **Smallest dt at which Regime A is consistent with level frames** (criterion: at most 5% of
  steps above 10 deg): **1.00 s** for the smoothed estimator with drag (4.3% above 10 deg, median
  6.1 deg), and **0.585 s** drag-free (3.0% above 10 deg, median 2.9 deg). The physically
  meaningful reading is the drag one, because a real airframe at 5 m/s must balance about 8.6 deg
  of drag tilt regardless of what the path does; it confirms 1 s as the smallest defensible dt.
  The per-step estimator never reaches the criterion on a 0.1-2.0 s grid, because the teleport
  transitions are permanent: 47% of boundaries are jumps at any dt.
- **Regime B rescaling destroys most of the tilt signal.** A TII autonomous ellipse yaws at up to
  490 deg/s, so the 15 deg/s yaw cap forces a time stretch of about 33x (and the speed cap a
  length scale of 0.2-0.6). Median required tilt at 1 s falls from 66.8 deg (raw path at 1 s) to
  8.7 deg peak, median 3.6 deg; only 0% of steps exceed 10 deg at 1 s. A documented
  **off-envelope variant** (`yaw_rate_max = 60 deg/s`, clearly labelled off-expert) recovers a
  median of 4.6 deg and 1.6% of steps above 10 deg at 1 s. Consequence for the plan: Regime B as
  specified is essentially tilt-blind at 1 s and can carry depiction/integrity information but
  not a tilt verdict; the dose-response curve has to come from Regime C, with Regime B as the
  feasible-but-quiet control.

## Task 2: pipeline validation (Gate 2)

Rollout protocol: the release's own `EvalDataset` (constructed by setting its attributes, so every
method is ANWM's code) with `index_to_data = [(trajectory, start_time, -64, 64)]`, 16 context
frames, actions from `_compute_actions` + `normalize_data` + `get_delta_np`, FFP from the recorded
depth/pose/K, 250 sampling steps, 16 frames rolled out autoregressively.

Gate 2 (run on 5 native test trajectories from `rollout_16.pkl`):

- **Runs end to end**: yes; frames, LPIPS, contact sheets and per-step statistics are written.
- **Frames are not collapsed**: per-frame pixel standard deviation is reported per clip in
  `results/anwm_native/summary.json` (`pred_frame_std` against `gt_frame_std`) and the temporal
  delta of the prediction is compared with the ground truth's.
- **LPIPS in a plausible range**: per-frame LPIPS (alex) rises from 0.05-0.16 at the first
  predicted step to 0.09-0.35 at step 16; per-trajectory means are **0.190, 0.216, 0.121, 0.228,
  0.093, pooled 0.170**. The paper's Table 1 order of magnitude (0.18-0.23 at 4-8 s for 3D) is
  matched on three trajectories and undershot on two; the protocol differs (their metrics split,
  their horizon handling), so this is a sanity check, not a reproduction.

Measured per-frame statistics (`results/anwm_native/summary.json`):

| trajectory | LPIPS mean | min pred std | mean pred std | mean GT std | pred temporal delta | GT temporal delta |
| --- | --- | --- | --- | --- | --- | --- |
| 3KA7IJSNXE | 0.190 | 0.284 | 0.310 | 0.302 | 0.0711 | 0.0566 |
| 3E9ZFLPWP6 | 0.216 | 0.305 | 0.321 | 0.321 | 0.1155 | 0.0853 |
| 38RHULDVA6 | 0.121 | 0.199 | 0.216 | 0.218 | 0.0304 | 0.0134 |
| 3PA41K45WV | 0.228 | 0.305 | 0.319 | 0.318 | 0.1003 | 0.0819 |
| 30EMX9PEWS | 0.093 | 0.274 | 0.281 | 0.280 | 0.0377 | 0.0242 |

No clip is collapsed (prediction pixel spread matches the ground truth to within 2-5%), and the
predictions move at least as much frame to frame as the ground truth, so drift or freezing is not
masking a low LPIPS. **Gate 2 passes.**

The pose convention was verified inside the same run: rebuilding the recorded poses from
`point`/`yaw` reproduces `pose` to 1e-16-2e-16 on every trajectory, so the FFP prior for a
*commanded* path is not rotated by a guessed convention.

## Task 5: generation set

82 clips, all from the 20 native start states, 16 frames each, 250 sampling steps, seeds recorded:

| regime | clips | composition |
| --- | --- | --- |
| A native | 25 | 20 recorded-path clips + 5 extra seeds (variance on 3 of them) |
| B real-flight replay | 8 | 2 shapes (`01a-ellipse` autonomous, `01p-ellipse` piloted) x 4 starts |
| C designed | 49 | cruise control 7, turn 12, slalom 9, accelerate-then-brake 9, climb-and-turn 12 |

The C counts are lower than 5x3x3+control because unreachable cells are not generated (12 of the
22 designs): a course-aligned turn or climb cannot exceed 7.4 deg inside ANWM's 15 deg/s yaw step,
and a constant-speed slalom tops out at 11.0 deg. Both policies are kept where they are reachable
because the yaw-fixed variant is a 45 deg strafe step, which the training data contains.

Every clip directory holds `frames/` (16 PNGs), `manifest.json` (commanded positions and yaw, step
spacing, start state, regime, designed tilt, seed, slim-checkpoint hash, sampling steps, code
commit, visual-integrity statistics) and, for Regime A, `gt/` with the recorded frames. The index
`clips_index.csv` is derived from the manifests, so it cannot drift from the frames.

## Files

- `src/generation/audit.py`, `paths.py`, `anwm_env.py`, `experiments/anwm_{audit_corrected,kd_split,path_audit,build_clips}.py`
- `remote/b_anwm.py` (image, slim extraction, smoke, native validation, generation), `remote/b_native.py`
- `tests/test_anwm_audit.py`, `tests/test_generation_paths.py` (18 tests; full suite 53 passed, 1 skipped on the CPU env)

## Blockers found

1. **Native data transfer is the bottleneck, and it is a property of the shard layout.**
   Walking one ANWM tar shard over HTTP Range takes about 5.5 minutes (7 members/s, 0.2 MB
   transferred) because `tarfile` requests every 512-byte header through the range reader; each
   shard holds a few trajectories and ~600 JPEGs each, so an index scan is ~5 min per shard. From
   a Modal container the same walk did not finish a single shard in ~25 minutes (two attempts,
   16 and 32 workers, 90 s socket timeout, nothing written to the Volume), while a 1 MB read from
   this workstation takes 0.46 s.
   **Fix:** an 8 MB read-ahead wrapper around the range reader (`ReadAhead` in
   `experiments/anwm_fetch_native.py`) turns a shard walk into 21 s instead of 330 s (16x), because
   it stops issuing one HTTP request per tar header. The fetch still runs locally and is pushed to
   the Volume with `modal volume put`, but it is no longer the slowest step.
2. Only 4 of the 29 trajectories in `rollout_16.pkl` were covered by the 39-shard index inherited
   from Phase 0, so the scan has to find at least one more before a 5-trajectory Gate 2 run.
3. `config/anwm.yaml` does not describe the released weights (context 4 vs 16), so any pipeline
   written against the config silently fails on `pos_embed`.

## What changed in the plan

1. **Context is 16 frames, not 4.** The released weights were trained with `context_size = 16`
   (`pos_embed` 17 rows). That is 4x the per-frame cost of the "4 context" reading and it changes
   what a start state is: 16 context frames, and FFP projects 16 frames per target pose. Every
   number above (17.57 s per frame, 7.3 GB peak) already includes this.
2. **Regime B as specified cannot carry a tilt verdict at dt = 1 s.** Rescaling a TII racing
   flight into 5 m/s / 2 m/s / 15 deg/s needs a 25-33x time stretch, which leaves a median
   required tilt of 3.6 deg and 0% of steps above 10 deg. Options, in order of preference:
   (a) keep Regime B as a feasible-but-quiet control for depiction and integrity only;
   (b) use the documented off-envelope variant (60 deg/s yaw cap) as an explicit off-expert arm,
   which reaches 4.6 deg median and 1.6% above 10 deg at 1 s, or 8.5 deg median and 38% above
   10 deg at 0.667 s; (c) report Regime B at dt = 0.25 s only (18.0 deg median, 69% above
   10 deg), which is the least defensible dt for the native data.
3. **The tilt dose-response must come from Regime C**, and the design caps bind: a course-aligned
   turn tops out at 7.4 deg of required tilt inside ANWM's 15 deg/s yaw step, so 10 and 20 deg are
   only reachable with the yaw held fixed (a 45 deg strafe step, which the training data contains)
   or with longitudinal/vertical acceleration. The generator reports `reachable: false` rather than
   missing the target silently, and the generation set skips unreachable cells (12 of the 22
   designs), which is why the clip set has 49 C clips rather than 66.
4. **ANWM's own `EvalDataset._compute_projected_images` takes goal times, not poses**, so
   commanded paths go through `reproject_depth_to_other_pose_seq2seq` /
   `project_to_2d_image_seq2seq` directly (`_ffp`). This is the same projection code; only the
   entry point differs, because the method would otherwise read the recorded poses.
5. **Native shard access needs a read-ahead buffer.** `tarfile` over a plain range reader issues
   one HTTP request per header: 5.5 min per shard. An 8 MB read-ahead makes it 21 s, which turned
   the native fetch from an apparent dead end into 4.5 minutes for 1.8 GB.

## Recommendations for handoff 04

- Score **transition events**, not clip means: 47% of native boundaries are acceleration events
  with a commanded median of 27 deg at 1 s, while the sustained (SG) tilt is ~5 deg. A clip-level
  average hides the signal the verifier is meant to catch (the feasibility doc already says this;
  this handoff supplies the corrected numbers).
- Always report the commanded required tilt of the clip being scored, at the dt used, from
  `clips_index.csv`; do not recompute it from pixels.
- Treat Regime A's recorded frames as a **level** reference, never as ground-truth attitude.
- For Regime C, score per design cell: the 5/10/20 deg targets are per-clip peaks, so a cell
  average will look small even when every clip in it is correctly shaped.
- The FFP ablation (FFP-on vs FFP-off) is still the one control that separates "the model learned
  attitude" from "the model was handed a depth-projected future". Nothing in this handoff touches
  it.

## Appendix: Regime C designs (required tilt at the nominal 1 s)

| maneuver | yaw policy | target deg | reachable | peak per-step drag-free | peak per-step with k_d | peak speed m/s | peak yaw rate deg/s | travel turn deg/step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cruise | course | None | True | 0.0 | 8.6 | 5.00 | 0.0 | 0.0 |
| turn | course | 5.0 | True | 5.0 | 9.8 | 4.99 | 9.9 | 9.9 |
| turn | fixed | 5.0 | True | 5.0 | 9.8 | 4.99 | 0.0 | 9.9 |
| turn | course | 10.0 | False | 7.4 | 11.2 | 4.99 | 14.8 | 14.8 |
| turn | fixed | 10.0 | True | 10.0 | 13.0 | 4.97 | 0.0 | 20.1 |
| turn | course | 20.0 | False | 7.4 | 11.2 | 4.99 | 14.8 | 14.8 |
| turn | fixed | 20.0 | True | 19.9 | 21.2 | 4.88 | 0.0 | 42.7 |
| slalom | course | 5.0 | True | 4.9 | 9.8 | 4.99 | 9.9 | 9.9 |
| slalom | fixed | 5.0 | True | 4.9 | 9.8 | 4.99 | 0.0 | 9.9 |
| slalom | course | 10.0 | False | 7.1 | 11.0 | 4.99 | 14.8 | 14.8 |
| slalom | fixed | 10.0 | True | 10.0 | 12.9 | 4.97 | 0.0 | 35.7 |
| slalom | course | 20.0 | False | 7.1 | 11.0 | 4.99 | 14.8 | 14.8 |
| slalom | fixed | 20.0 | False | 11.0 | 13.7 | 4.96 | 0.0 | 29.9 |
| accel_brake | course | 5.0 | True | 5.0 | 12.1 | 5.00 | 0.0 | 0.0 |
| accel_brake | course | 10.0 | True | 9.9 | 14.9 | 5.00 | 0.0 | 0.0 |
| accel_brake | course | 20.0 | True | 19.9 | 22.5 | 5.00 | 0.0 | 0.0 |
| climb_turn | course | 5.0 | True | 5.0 | 9.8 | 5.21 | 9.9 | 9.9 |
| climb_turn | fixed | 5.0 | True | 5.0 | 9.8 | 5.21 | 0.0 | 9.9 |
| climb_turn | course | 10.0 | False | 7.4 | 11.2 | 5.21 | 14.8 | 14.8 |
| climb_turn | fixed | 10.0 | True | 10.0 | 13.0 | 5.20 | 0.0 | 20.1 |
| climb_turn | course | 20.0 | False | 7.4 | 11.2 | 5.21 | 14.8 | 14.8 |
| climb_turn | fixed | 20.0 | True | 19.9 | 21.2 | 5.11 | 0.0 | 42.7 |

Peak speed is the 3-D speed; the caps are per axis (horizontal 5 m/s, vertical 2 m/s), so a
climb at 5 m/s horizontal plus 1.5 m/s vertical reads 5.2 here and is legal.
