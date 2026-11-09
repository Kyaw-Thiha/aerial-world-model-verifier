# B04 — The ANWM clip set: 82 generated flights in three input regimes

Track B. Written 2026-11-09. Branch `track-b-generation`.
Run: `results/anwm_clips/{specs.json,clips_index.csv,variance.json}` (specs at **bd032be**,
index and variance at **8edcca5**), `results/anwm_starts/starts_coverage.json` (**a74378b**);
clips on R2 under `track-b/clips/` (1,882 objects) and in the local
`data/processed/anwm_clips/` tree (gitignored).

## Summary in plain language

We generated 82 short flights with the released model, from 20 starting points in the real test
data, in three flavours: replaying the model's own recorded movements, replaying real racing
flights slowed down until they fit the model's gentle limits, and purpose-built maneuvers whose
required lean we set in advance to about 5, 10 or 20 degrees. Each clip is 16 frames at 224x224
with a file recording exactly what was commanded, from where, with which seed and which
checkpoint, so the flights can be scored without reading any of our code.

Two things a reader should know before scoring. First, the real-flight replay carries almost no
lean signal at a one-second step (all eight clips command less than 9 degrees), so it is a quiet
control, not a test. Second, the frame-to-frame pixel change does not increase with the commanded
lean: on the designed set the correlation is slightly negative. That is an observation about
pixels, not a measurement of depicted lean, and the actual measurement is handoff 04's job.

## Question and setup

Question: can we drive the model from controlled inputs that are (a) feasible, (b) inside its
trained envelope, and (c) labelled well enough that a verifier can score them?

- Start states: 5 native test trajectories, 4 candidate windows each, 16 context frames with
  depth and pose. Filtered by *measured* Future-Frame-Projection coverage, not by a depth
  heuristic: depth in the release is metres with a 500 m far clip, so a "forward depth exceeds path
  extent" test on a high percentile accepts everything.
- Regime A: the recorded 16 commands from each start.
- Regime B: two real flights (`13a-trackRATM` autonomous, `01p-ellipse` piloted), windows chosen
  where the raw speed reaches at least 5 m/s, rescaled into 5 m/s horizontal, 2 m/s vertical,
  15 deg/s yaw, then resampled at 1 s.
- Regime C: five maneuvers x tilt targets x three starts, with ANWM's own action construction.
  Only reachable cells are generated (7 of 22 designs are out of reach at 5 m/s; report B02).
- Generation: 250 sampling steps, batch 1, A10G, ~235 s per clip end to end.
- Cost: about **US$7** of GPU and CPU time in total for extraction, smoke, Gate 2 and all 82
  clips, against a US$10 approval threshold. The rate (US$1.10/GPU-hour) is an **unverified**
  price assumption; GPU-hours are measured.

## Results

### Composition

| regime | clips | composition |
| --- | --- | --- |
| A native | 25 | 19 recorded-path clips (A00–A18) + 6 extra seeds (A00–A02 at seeds 1 and 2) |
| B real-flight replay | 8 | 2 flights x 4 starts |
| C designed | 49 | cruise control 7, turn 12, slalom 9, accelerate-then-brake 9, climb-and-turn 12 |

### Start filter (Task 4)

20 of 20 candidate starts are usable by the measured criterion: FFP non-zero-pixel coverage at the
worst target step ranges **0.197 to 0.773** (median 0.49), so the projected future image is never
empty. The delivered set uses **19** of them as Regime A base starts: the builder kept the old
depth-p90 flag as a second gate, which dropped `3E9Z…` at t = 159 (depth p90 4.7 m) even though
its projected prior is 23.7% covered. The extra gate is fixed in `experiments/anwm_build_clips.py`
after generation; re-running the builder now yields 20 base starts, so a 83rd clip is available
if the set is ever extended. `specs.json` remains the record of what was generated (19 base).

### Per-clip testability from `clips_index.csv`

Peak commanded required tilt at 1 s with `k_d`:

| regime | clips | min / median / max (deg) | clips above 10 deg |
| --- | --- | --- | --- |
| A native | 25 | 0.0 / 30.3 / 36.3 | 24 / 25 |
| B rescaled | 8 | 2.2 / 5.4 / 8.7 | **0 / 8** |
| C designed | 49 | 8.6 / 9.8 / 22.5 | 24 / 49 |

`A15` is the one A clip at 0.0 deg: a genuine 2 m-step hover window in the released data, kept
and labelled rather than dropped.

### Visual integrity (pixels only, no tilt claim)

| regime | clips | mean frame std | mean temporal delta |
| --- | --- | --- | --- |
| A | 25 | 0.295 | 0.106 |
| B | 8 | 0.306 | 0.179 |
| C | 49 | 0.299 | 0.159 |

Recorded frames sit at about 0.30 pixel std, so nothing is collapsed, and every group changes
frame to frame.

### Seed variability (5 paths x 3 seeds)

| path | pairwise MAE between seeds | within-clip temporal delta |
| --- | --- | --- |
| A00 | 0.0875 | 0.1249 |
| A01 | 0.0712 | 0.0780 |
| A02 | 0.1290 | 0.1678 |
| C cruise control (start 0) | 0.1960 | 0.2173 |
| C cruise control (start 1) | 0.0974 | 0.1081 |

Sampling noise is the same order as the motion signal on the cruise control, so any per-clip
score needs its seed spread reported beside it.

### Does pixel motion track the commanded maneuver? (Spearman, temporal delta vs commanded tilt)

| set | n | rho | p |
| --- | --- | --- | --- |
| Regime A | 25 | −0.06 | 0.76 |
| Regime B | 8 | +0.24 | 0.57 |
| Regime C | 49 | **−0.17** | 0.25 |
| C, turn only | 12 | **−0.53** | 0.07 |
| C, slalom only | 9 | −0.46 | 0.22 |
| C, accelerate-brake only | 9 | +0.32 | 0.41 |

Nothing is significant, and the sign is negative where it is largest: harder commanded maneuvers
do not produce more pixel motion.

### First delivery versus final

| item | first delivery | final | change |
| --- | --- | --- | --- |
| clips | 65 (A 14, B 2, C 49) | **82 (A 25, B 8, C 49)** | runs restarted until complete |
| Regime B windows | first 4 windows at 15% of each flight | moving windows only, autonomous shape switched to `13a-trackRATM` | the first B set had **zero-extent commanded paths** (stationary lineup) and was deleted, not shipped |
| visual integrity, B | 0.284 / 0.140 (2 clips) | 0.306 / 0.179 (8 clips) | sample change |

## What this supports

- A labelled, resumable, reproducible clip set: the spec builder is deterministic (identical hash
  on re-run), the index is derived from the manifests, and every clip records checkpoint hash,
  sampling steps, seed, start state and code commit.
- Regime A gives 24 testable clips; Regime C gives a controlled ladder with known required tilt;
  Regime B gives a feasible-but-quiet control.
- The model's pixel motion is uncorrelated with (if anything, opposite to) the commanded maneuver,
  which is a concrete, checkable hypothesis for handoff 04.

## What this does not show

- **No depicted tilt.** Nothing here measures the attitude the pictures show; that is handoff 04.
- No statement that a clip is infeasible or feasible as a flight: only what the commanded path
  demands.
- The correlation in the last table is a crude proxy (temporal delta also depends on texture and
  on the projected prior) on n = 49 with mixed starts; it is a flag, not a result.

## Caveats and open items

- **The clip set spans several code commits**, as each manifest records: C 49 at `5850b4b7`;
  A at `2626c5be` (8), `bd032be7` (7), `f42a3610` (7), `77118152` (1), `5850b4b7` (2); B at
  `f2df5363` (6), `bfaa97cf` (1), `9312666f` (1). The rollout code path is identical across them:
  `git diff 2626c5b..8edcca5 -- remote/b_anwm.py src/generation/` changes only a docstring line,
  a timeout constant, and adds an unused loader.
- **FFP takes goal times, not poses.** `EvalDataset._compute_projected_images` indexes the recorded
  poses, so commanded paths call the same two projection functions directly (`_ffp`). A reader
  comparing against the release's method should know this.
- The 1 s step is an assumption (report B01), recorded in every manifest as `dt_verified: false`.
- The `Clip` loader (`src/generation/clip_io.py`) writes a **level-camera** attitude with the
  commanded heading and says so in `meta["attitude_source"]`; it is truth-of-command, not truth of
  a real flight.
- Open: the FFP-on/FFP-off ablation (does handing the model a reprojected future frame manufacture
  geometric consistency?) is untouched. Open: Regime B needs either the documented off-envelope
  variant or a shorter analysis dt if it is ever to carry a tilt verdict.
