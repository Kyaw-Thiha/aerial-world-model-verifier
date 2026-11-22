# G05 — Segments on the simulated set: blocked, with the design frozen

Track G2. Written 2026-11-21. Design frozen in `configs/g05_round2_design_prereg.yaml` before
any simulated outcome exists. Pilot code `experiments/g05_spike_pilot.py`, results
`results/g05_spike_pilot/summary.json`. **No GPU spend**; no confirmation number produced.

**Status: not executed.** Every number here is **EXPLORATORY** and comes from three clips.

## Decision

**Could not run as designed:** Track I's labelled development set (report I02) does not exist —
`track-i-simset` carries only the I01 spike, one flight rendered three ways with full-clip
violations — so the protocol's requirement (design on labelled development scenes, then evaluate
once on held-out scenes) cannot be met, and no ablation table or frozen evaluation is produced.
**What is delivered instead:** the full round-2 design (items 1-6, each with its ablations) frozen
in a pre-registration, and one exploratory pipeline check showing that the path-demand machinery
behaves on renders whose truth is exact, with a 1.0 deg versus 27.7 deg separation between a
feasible and a tilt-removed render.

## Why it is blocked, in evidence

| requirement (handoff 18) | what exists on `track-i-simset` | verdict |
| --- | --- | --- |
| dev scenes (≥8) and holdout scenes (≥4) with exact intervals | 1 flight, 3 renders, no scene split | missing |
| partial intervals at 10 / 25 / 50 % | none — all three violations span the whole 3 s clip | missing |
| yaw and null controls | none | missing |
| per-frame truth at ≥200 Hz, frames ≥30 fps | truth yes (`flight_gt.npz`, 751 samples), 91 frames per clip | partial |
| reader output on the frames | yes (`geocalib/i01_*_pinhole_prior.json`, per-frame `lean_est`) | present |

Three clips cannot do what the protocol needs them to do: choose a Huber knot, a persistence
length, an FDR level or a window length, because every such choice would be made on three
samples and then evaluated on the same three. That is why the design is frozen and left
unexecuted rather than run on the spike and reported as if it were a development set.

## Exploratory pipeline check (three I01 clips)

The path-demand classification and the residual statistic run unchanged on the spike. Required
tilt is 25.3 deg throughout, so all windows are demand windows and none is quiet — the quiet half
of item 1 cannot be exercised here.

| render | frames | demand windows | demand-window median residual | per-frame residual (truth) |
| --- | --- | --- | --- | --- |
| feasible (physics attitude) | 91 | 17 | **1.02 deg** | 1.04 (0.00) |
| level camera (tilt removed) | 91 | 17 | **27.69 deg** | 25.90 (25.33) |
| mirrored roll | 91 | 17 | **51.23 deg** | 43.38 (43.72) |

Read as: the machinery runs, the demand classification does not distort a path where truth is
exact, and the statistic's ordering is the physical one. Read *not* as: a detection rate,
a threshold, or evidence about any of the six method changes. There is no calibration set, no
partial interval, and no control — so the 1.0 versus 27.7 deg gap is a property of the renders
(which Track I already reported at frame level), not of the segment detector.

## The frozen design (unexecuted)

`configs/g05_round2_design_prereg.yaml` fixes, for each item, the statistic, the ablations, the
knobs that must be set on development scenes, and the protocol for the single frozen evaluation.
In brief:

| item | primary rule | ablations |
| --- | --- | --- |
| 1 path-demand | demand ≥6/8 frames at the floor; quiet all 8 ≤ 3 deg (real-corpus values; the sim floor is a dev-set choice) | demand-only vs quiet-only vs every window |
| 2 robust/weighted | Huber location with a fixed 5 deg scale, confidence weights `clip(conf/median, 0.1, 10)` | trimmed mean, unweighted, per-clip MAD scale; persistence 1/2/3 |
| 3 FDR | Benjamini-Hochberg per clip, q = 0.10, conformal window p-values | q ∈ {0.05, 0.10, 0.20}; corpus-level BH |
| 4 boundaries | two-state half-normal change-point fit inside each flagged region, 1-frame grid, ±2 windows | refined vs unrefined IoU/F1/onset |
| 5 alignment | one lag per clip from the frozen ±150 ms / 25 ms grid, removed before scoring | with vs without; lag distribution (coarse at 12 Hz TII, designed for 30 fps sim) |
| 6 multi-scale | lengths {8, 12, 16} standardised per length, one joint conformal threshold | each length alone vs joint |

Each item is reported alone and the stacked version beside it; the stacked version is never the
only number. Real-clip thresholds stay leave-one-flight-out, and the real confirmatory block stays
`flight-13p-trackRATM` + `flight-07a-lemniscate`, untouched since round 1.

## What this does not show

* **Nothing about items 2-6.** They are specified, not measured.
* **Nothing confirmatory at all.** The pilot is three clips, full-clip violations, no controls,
  no holdout.
* **No held-out result on the real block.** It remains reserved, exactly as the protocol
  requires.

## What the next round needs

1. Track I to deliver I02: ≥8 dev and ≥4 holdout scenes, 30 fps, truth ≥200 Hz, partial
   violations at 10/25/50 % for the four families, yaw-rotation and null (re-render) controls,
   plus the reader's per-frame output and the commanded path's required lean.
2. Then execute `configs/g05_round2_design_prereg.yaml` on dev scenes only, freeze the chosen
   knobs in an amendment, and evaluate once on the held-out scenes and the real block.

## Caveats

* The pilot reads Track I's worktree read-only; it changes nothing there and depends on that
  branch staying at `5f0fa3f`.
* The 91 false-positive-free frames in the pilot are not a false-reject measurement: TII's real
  false-reject is 8.6 % on the same statistic (report G04), and the simulated reader is more
  accurate than the real one (1.1 deg median versus ~4 deg, Track I I01), so a simulated
  false-reject would not transfer to real footage either.
* The spike's feasible render has no quiet windows, so item 1's quiet half — the half that G04
  found useful in principle and unusable in practice — remains untested on simulated data too.

## Regenerate

```
PYTHONPATH=. python experiments/g05_spike_pilot.py --i-dir ../awmv-track-i/results/i01_spike
```

Writes `results/g05_spike_pilot/summary.json`.
