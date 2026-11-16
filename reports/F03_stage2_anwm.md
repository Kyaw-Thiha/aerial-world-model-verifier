# Report F03 — Stage 2: ANWM clips against their commanded path

Track F. Written 2026-11-16. Stage 2 asks a different question from stages 0 and 1: ANWM
generated clips carry a **known commanded path**, so an extractor's path recovery can be
scored directly, without Leica truth and without the depicted lean. The stage-1 extractor is
run on the 82 level clips and the 45 injected clips, and the recovery error is measured against
the command.

**Decision: pending completion of the job set.** This stage feeds no gate; it reports what a
pixel path can be recovered from generated frames and whether the pixel-path verdict can be
formed at all on them.

## Plain-language summary

*(filled from `results/f03_stage2/anwm.json`)*

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

*(filled)*

## What this supports

*(filled)*

## Caveats

* 16 frames at 1 s with a 3–5 m baseline is the opposite of what incremental trackers need;
  a failure here is a statement about the clip spacing, not about the extractor in general.
* ViPE estimates its own intrinsics; a batch extractor given the calibration is the fair
  comparison for ANWM, and that comparison is C05's Pi3 number, not this run.
* The verdict comparison (pixel-path verdict vs commanded-path verdict) is only meaningful if
  recovery is usable; it is not forced when it is not.
