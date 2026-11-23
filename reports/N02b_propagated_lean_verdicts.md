# N02b — Verdict tiers on the rotation-propagated lean (UZH outdoor, descriptive)

**Decision.** On the only population that has both edited video and a working rotation source —
nine UZH outdoor clips per condition — the per-clip fit on the propagated lean does what the
rationale says it should (real-clip median residual 3.63° against the reader's 8.57°), but no
tier detects the edits better on the propagated lean than on the reader lean, and the rotation
extractor itself fails on 3 of 9 clips (up to 6 of 9 on heavily warped conditions). The result is
descriptive and underpowered; it does not support a claim that the propagated lean improves the
verdict.

Pre-registration `configs/n02b_part2_prereg.yaml` (committed before DPVO touched an edited clip).
Code `experiments/n02b_part2_verdicts.py`; GPU `remote/n_rotation.py` task `uzh_edit`. Results
`results/n02b/`. 63 clips, 60 frames each, 3 sequence clusters, 9 clips per condition.

## Plain-language summary

1. **The two arms see the same frames, required lean and mask.** The reader side is Track E's
   stored per-frame lean under each v3 edit; the propagated side starts from that lean at the
   first frame and uses DPVO relative rotations on the *same edited frames* (undistorted from the
   0.5 crop to a pinhole view). The required side is schema v3 exact from the 500 Hz truth,
   sampled at the 15.07 Hz frames.
2. **The per-clip fit absorbs the start, as predicted.** Median real-clip residual: C2 (offset +
   drag) 3.63° on the propagated lean against 8.57° on the reader; F0 (no offset) is the other
   way round (12.64° against 9.01°), because F0 cannot absorb the start error. This is the clean
   mechanical confirmation of the offset argument.
3. **But that absorption costs detection.** At a leave-one-sequence-out threshold, C2 on the
   propagated lean flags 0/6 real clips and 0/3 tilt removals: its residual is so small (3.6°) and
   so tight a threshold that the removal does not separate. F0 separates weakly (2/6 real, 2/3
   removal, 2/3 mirror); coherence on the propagated lean does not separate at all (1/6 real, 1/3
   removal, 0/3 mirror) while the reader's coherence detects 2/3 and 2/3.
4. **The extractor is the binding constraint.** With an oracle start, the 2 s propagated error is
   under 5° for 7 of 9 real clips and 28-41° for two (`outdoor_forward_3_02/03`); on warped
   conditions only 3-6 of 9 clips retain a valid extraction. DPVO on the 320x240 fixed crop (60
   frames, narrow field) is far weaker than on the full 640x480 sequences of N01 (0.4° at 10 s).
5. **The yaw control is not removed.** Prediction: propagation is yaw-free, so a world-up warp
   should not change the propagated lean. Measured: F0 on the yaw control is 3/9 for the
   propagated arm against 2/9 for the reader, and the propagated yaw-control residual (24.6°) is
   worse than its real residual (12.6°). The warp changes the estimated rotation itself (the
   camera really is yawing in the warped video), so propagation does not make the verdict
   view-independent; it moves the view dependence from the per-frame reader into the rotation.
6. **Why the "constant offset" is only approximately constant.** A start error is fixed in the
   body frame; the lean series is expressed in the yaw frame. When the aircraft yaws, the
   representation of the same body-fixed error rotates, so a Pearson correlation (or a fitted
   intercept) is invariant only to the part of the error that is constant in the yaw frame. On
   turning UZH clips part of the start error is not, which is visible in the propagated arm's
   real coherence score (median 1.39, i.e. a near-zero correlation) and in F0.
7. **All counts are tiny.** Nine clips per condition, three clusters, and 3-6 usable after the
   extraction check. Every number is k/n; no interval is a 95 % confidence interval.

## Setup and validity

| condition | clips | valid extraction (oracle 2 s error ≤ 10°) |
| --- | --- | --- |
| real | 9 | 6 |
| null_edit | 9 | 6 |
| yaw_control | 9 | 6 |
| tilt_removal | 9 | 3 |
| scale_0.5 | 9 | 6 |
| scale_1.5 | 9 | 5 |
| mirror_roll | 9 | 3 |

The oracle-start diagnostic is a pipeline check, not an outcome: it measures whether DPVO tracked
the edited video well enough for any verdict comparison to mean anything. Both the all-clip and
the valid-only tables are reported; the valid-only table is the one to read.

## Flag counts at the leave-one-sequence-out threshold

Threshold: 95th percentile of the other two sequences' real-clip scores (6 clips), per arm and
tier. F0 = median depicted-required angle; C2 = per-clip offset + drag fit residual; R =
1 - min(r_x, r_y) with the frozen TII tau 0.6197.

| condition | F0 reader | F0 propagated | C2 reader | C2 propagated | R reader | R propagated |
| --- | --- | --- | --- | --- | --- | --- |
| real | 2/6 | 2/6 | 2/6 | 0/6 | 1/6 | 1/6 |
| null_edit | 2/6 | 2/6 | 2/6 | 0/6 | 1/6 | 1/6 |
| yaw_control | 2/6 | 2/6 | 2/6 | 0/6 | 0/6 | 0/6 |
| tilt_removal | 2/3 | 2/3 | 2/3 | 0/3 | 2/3 | 1/3 |
| scale_0.5 | 2/6 | 2/6 | 2/6 | 0/6 | 2/6 | 0/6 |
| scale_1.5 | 2/5 | 2/5 | 2/5 | 0/5 | 0/5 | 0/5 |
| mirror_roll | 2/3 | 2/3 | 2/3 | 0/3 | 2/3 | 0/3 |

All-clip counts (denominators 9) are in `results/n02b/summary.json` (`per_condition`); the
valid-only counts above are `per_condition_valid`.

## Median residuals (all clips)

| condition | F0 reader | F0 propagated | C2 reader | C2 propagated | R reader | R propagated |
| --- | --- | --- | --- | --- | --- | --- |
| real | 9.01 | 12.64 | 8.57 | **3.63** | 0.91 | 1.39 |
| tilt_removal | 24.86 | 21.20 | 6.42 | 5.42 | 1.35 | 1.51 |
| mirror_roll | 18.04 | 13.33 | 8.74 | 7.48 | 1.66 | 1.72 |
| scale_1.5 | 10.86 | 25.71 | 7.25 | 5.17 | 0.99 | 0.88 |
| yaw_control | 10.37 | 24.55 | 5.60 | 4.61 | 0.81 | 0.80 |

The one clear win: C2 absorbs the propagated start on real/ null clips (3.63° against 8.57°). The
one clear loss: F0 and coherence on the propagated lean degrade on the yaw control.

## What this supports

* The offset mechanism is real and measurable: a per-clip intercept removes the propagation's
  start error (real residual 8.57° → 3.63°).
* **The verdict does not improve.** On this population the propagated arm does not detect
  removal/mirror better than the reader at any tier, and its coherence tier is worse.
* The rotation source must be validated per clip before it is used for a verdict; the oracle
  check is cheap and caught two total failures that would otherwise have been read as verdict
  behaviour.

## What this does not show

* Anything confirmatory. The confirmatory populations (six TII dev flights; Track I's defect
  clips) were not runnable: TII has no usable rotation source (N01), and the simulated defect
  clips are not on R2. Nine clips per condition with 3-6 usable cannot settle a tier comparison.
* No absolute detection rate for the pixel edits: the propagated side uses DPVO on the edited
  frames, which is faithful, but the extractor's own losses dominate.
* No intervals: three clusters.

## Caveats

* The 0.5 fixed crop at 320x240 is a much weaker input than the full sequences; the extractor
  failure rate here is a property of the crop, not of the UZH footage (N01 used 640x480).
* The valid-only filter (oracle 2 s ≤ 10°) is a pipeline diagnostic applied after the fact; it is
  reported beside the all-clip table so the exclusion cannot be mistaken for a result.
* The yaw warp is not a pure world-yaw rotation of an otherwise unchanged image: it changes the
  visible content and therefore DPVO's estimate. The "propagation ignores yaw" statement holds
  for the ideal construction, not for a warp that changes the footage.

## Provenance

* Pre-registration `configs/n02b_part2_prereg.yaml`; decisions `[N]`.
* Inputs: `data/clips_v3` (Track E v3 clips, local), UZH truth on `data/uzh`, DPVO poses on
  `awmv-n-data/poses/uzh_edits` (63 runs, 60 frames each, about 100 s of A10 = USD 0.03).
* No R2 write; every app stopped.
