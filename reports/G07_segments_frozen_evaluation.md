# G07 — Frozen evaluation of the G3 segment design on the held-out set

**Track G3.** Branch `track-g3-segments`. Status: **not yet scored.** This file is created when
the development pass (G06) is closed and the evaluation recipe is frozen; the held-out clips are
scored once, after that, and this file is filled once.

## Decision

<!-- one-shot result; two sentences -->

## Setup

- **Population.** 100 held-out simulated clips (`field_settlement`, `coastal_cliff`, seeds
  201-206) and the 18 real held-out clips of `flight-13p-trackRATM` and `flight-07a-lemniscate`
  (of which only the gate-kept ones are testable).
- **Thresholds.** Dev-frozen: simulated thresholds from the feasible dev clips
  (leave-one-scene-out for the dev tables, all-dev for the held-out application); real thresholds
  leave-one-flight-out on the gate-kept dev flights, as in rounds 1-2.
- **Arms.** Exactly the arms and knobs of G06's frozen recipe (`configs/g06_eval_frozen.yaml`,
  written after G06 and committed before this file is filled). No knob is chosen here.
- **Metrics.** The G06 metric set: detection at nominal and matched false-reject, interval IoU /
  F1 / onset error, false segments per minute, realised FDP, the minimum-localisable-violation
  contour, the yaw and null controls, and the thrust-limit family stated as not-applicable.

## Results

### Simulated held-out scenes

<!-- FILL once -->

### Real held-out flights

<!-- FILL once -->

## What this supports

<!-- FILL -->

## What this does not show

- Six held-out scene seeds, so every interval is a descriptive spread, not a 95 % confidence
  interval.
