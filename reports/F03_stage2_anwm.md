# Report F03 — Stage 2: ANWM clips against their commanded path

Track F. Written 2026-11-16. Stage 2 asks a different question from stages 0 and 1: ANWM
generated clips carry a **known commanded path**, so an extractor's path recovery can be
scored directly, without Leica truth and without the depicted lean. The stage-1 extractor is
run on the 82 level clips and the 45 injected clips, and the recovery error is measured against
the command.

**Decision: the ANWM pixel path is not usable.** ViPE produces a numerically finite
trajectory on most clips but the direction error against the commanded path has a median
around 60 deg (the bake-off bar is 30), 36 of 127 are degenerate (27/82 level, 9/45 injected),
and only 8 of the 91 non-degenerate clips pass the bake-off's per-clip rule. The pixel-path verdict is
therefore not formed on these clips: there is no usable path to form it from.

## Plain-language summary

ANWM clips are 16 frames one second apart, with 3–5 m between them, and the commanded path is
known exactly. Incremental trackers were expected to struggle here — C09 measured DPVO
crashing on 10 of 25 clips and degenerating on 6 of 15 survivors — and ViPE struggles in a
different way: it returns a pose for every clip, but the path it returns is not the commanded
path. On the 45 injected clips, 36 trajectories are finite and non-degenerate, with a median
direction error of 59 deg; on the 82 level clips the median is 67 deg. Only 8 of the 91
non-degenerate trajectories pass the bake-off's per-clip rule (direction <= 30 deg, ratio in
[0.7, 1.4]). The verdict comparison the handoff asks for is
conditional on usable recovery, so it is not performed and the report says why instead of
reporting a number that would be noise.

## Setup

| item | value |
| --- | --- |
| clips | 82 level (`clips/{A,B,C}/<id>`) and 45 injected (`clips_injected/injected_true/{A,B,C}/<id>`) |
| frames | 16 per clip, 1 s per step, 3–5 m baseline |
| commanded path | `manifest.json → commanded_path.positions` (17 rows for 16 frames at 1 s) |
| extractor | ViPE 1.2.0, default pinhole pipeline, its own intrinsics |
| metric | Track C's `score_clip` at dt = 1 s, 0.5 s window; direction error, magnitude ratio after Sim(3), ATE |
| failure rules | no pose produced / non-finite / magnitude ratio > 10 / fewer than 12 usable poses (C09's rules) |

The baseline is C09's DPVO result on the same kind of clips: 10 of 25 crashed inside DPVO,
6 of the 15 survivors were degenerate, and the 9 usable clips had a median direction error of
48 deg (generated) and 45 deg (ground truth). Pi3, a batch extractor, recovered the commanded
path to 1–8 deg on the ground-truth frames (C05).

## Results

Run `results/f03_stage2/anwm.json`, from poses on `awmv-f-data` (`poses_anwm/vipe`), scored by
`experiments/f03_anwm.py`.

| set | clips | finite trajectory | degenerate (ratio > 10) | no pose | usable | pass bake-off rule | median direction (usable) | median ratio (usable) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| injected | 45 | 45 | 9 | 0 | 36 | 4 | 59.1 deg | 2.01 |
| level | 82 | 82 | 27 | 0 | 55 | 4 | 67.0 deg | 1.78 |

"Degenerate" is the C09 rule (magnitude ratio above 10), which is a scale collapse, not a
crash: ViPE does not crash on these clips where DPVO did. A trajectory can be finite, not
degenerate by that rule, and still be wrong by 60 deg, which is the case for most of the
"usable" column.

The comparison to C09's DPVO run on the same kind of clips: DPVO 10/25 crashed, 6/15 further
degenerate, 9 usable at a median direction error of 48 deg (generated) and 45 deg (ground
truth). ViPE has no crashes, fewer degeneracies proportionally, and a worse direction error on
the trajectories it does return. Neither is usable for a verdict; Pi3's 1–8 deg on the
ground-truth frames (C05) remains the only extractor result on ANWM that is.

## The verdict comparison

Not formed. The handoff makes it conditional on usable recovery: a pixel-path verdict on a
path whose acceleration direction is off by 60 deg measures the extractor, not the video.
The commanded-path verdict on the same clips already exists (E5, report 08); the pixel-path
side needs a batch extractor that can handle a 5 m baseline first.

## What this supports

* **The ANWM side of the pixel-path question is extractor-limited, not statistic-limited.**
  The stage-0 statistics need a path whose shape is good at the depicted lean's frequencies;
  at 1 s and 3–5 m between frames, neither an incremental tracker (DPVO) nor a streaming SLAM
  system (ViPE) produces one.
* **"No crash" is not "usable".** ViPE returns 16 poses for every clip, and 30 of 99 scored
  trajectories are degenerate by the ratio rule while many of the rest are direction-wrong by
  ~60 deg. A pipeline that only checked for exceptions would have accepted them.
* **The batch route is the one to develop for ANWM**: C05's Pi3 result on the ground-truth
  frames (1–8 deg) is the only number in the project that clears the bar on this footage.

## What this does not show

* Anything about Pi3 or other batch extractors on the generated frames: only ViPE was run
  here, and C05's Pi3 numbers are on the ground-truth frames.
* Whether a metric depth prior would rescue ViPE's scale: the failures are direction errors,
  not only scale, so the plausible repair is a different extractor, not a rescale.

## Caveats

* 16 frames at 1 s with a 3–5 m baseline is the opposite of what incremental trackers need;
  a failure here is a statement about the clip spacing, not about the extractor in general.
* ViPE estimates its own intrinsics; a batch extractor given the calibration is the fair
  comparison for ANWM, and that comparison is C05's Pi3 number, not this run.
* The verdict comparison (pixel-path verdict vs commanded-path verdict) is only meaningful if
  recovery is usable; it is not forced when it is not.
