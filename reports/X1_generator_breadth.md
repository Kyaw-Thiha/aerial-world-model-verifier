# X1 — Generator breadth on one common path set

Track X, Stage 1, 2026-11-25. Branch `track-x-final-pipeline`. Pre-registration
`configs/x01_common_paths_prereg.yaml` (sha256 `b360667c…`, committed at `2b306c4` before any
clip existed). Run directory: `results/x01/`; clips on Modal `awmv-x-results:/x01/`; inputs
rebuildable with `experiments/x01_common_paths.py`; the exact command list is
`scripts/x01_run.sh`.

> Numbers are filled by `scripts/x01_fill_report.py` from `results/x01/score.json` and the run
> JSONs; every value below is machine-copied, not retyped.

## Decision

Four path-conditioned generators — ANWM, LingBot-World (NF4), SolarWM-5B and the new SANA-WM —
were driven on one 25-path common set at their native frame grids, and on the four that produced
clips the pictured tilt follows the commanded tilt with a generator-specific gain, so the
paper's cross-generator claim no longer rests on one model. The breadth is real but uneven:
SANA-WM is the first added generator that takes a metric camera trajectory natively and needs no
action-DSL approximation.

## Question

The verifier's evidence base was essentially one generator (ANWM) plus two pilots that failed
the roll-response gate. Does a path-conditioned aerial world model at large depict the tilt its
commanded path requires, and can three to four released models be put on the *same* paths at
affordable cost?

## Setup

### Models

| model | conditioning | route | frames | cost per clip |
| --- | --- | --- | --- | --- |
| ANWM | 4-DoF actions (dx, dy, dz, dyaw) per 1 s step | H01's app, A10G, 250 steps | 17 (16 generated) | FILL |
| LingBot-World (NF4) | OpenCV c2w per frame | H01's app, A100-80GB, 40 steps | 81 | FILL |
| SolarWM-5B | absolute c2w + intrinsics | H04's app, A100-80GB, 4-step SGF | 81 at 16 fps | FILL |
| SANA-WM (new) | native `(F,4,4)` c2w + intrinsics | NVlabs/Sana `3c6650a`, A100-80GB, stage 1 | 81 at 16 fps, 704x1280 | FILL |

Why SANA-WM and not HY-WorldPlay, Matrix-Game or Yume: SANA-WM is the only one of the four
candidates that takes a metric camera trajectory natively (`--camera`, `(F,4,4)`), releases
Apache-2.0 weights without a gate, and runs on one GPU with the release's own script.
HY-WorldPlay was rejected on cost and licence (189 GB including the HunyuanVideo-1.5 backbone,
a documented four-GPU route, Tencent's community licence); Matrix-Game-3.0/4.0 and Yume-1.5
were rejected because their conditioning is a keyboard/mouse action DSL, and approximating a
metric path with keystrokes is exactly what the handoff forbids. The full reasoning is in the
pre-registration.

### The common path set

25 paths, five classes: level (hover, cruise), mild lateral (s_5, slalom_5, fig8_5, s_10,
slalom_10), banked turns 10-25 deg (arc_10/15/20/25, climb_turn_10/20), longitudinal
(accel_brake_10/20, climb_02/04, descend_02/04), out-of-envelope (arc_35, climb_turn_30), and
two real UZH-FPV windows (outdoor_forward_1, _3) plus a scene swap on arc_20 and slalom_10.
A path is a family plus a request; each model gets the same family at the same requested peak
on its own time base, and the achieved peak is checked (tolerance 0.25 deg). The camera models
hit every request; ANWM saturates on four high-bank paths because its 45 deg/step
travel-direction cap cannot express a 25 deg bank at 2.8 m/s, and those are flagged
`expressible: false` rather than silently re-scaled.

The provisional envelope (tilt <= 30 deg, thrust ratio <= 1.8, tilt rate <= 200 deg/s) is a
physics placeholder: Track P's rate-envelope definition does not exist yet (see Gating), and
Stage 2 re-derives it before any pre-registered claim.

### Measured statistic

Tilt-fidelity score = median over eligible frames of |depicted lean| / |required lean|,
eligible = commanded lean >= 3 deg on an in-envelope path, with the frozen reader (GeoCalib
`1dedb29`, 224 centre crop, f = 112 px) and the commanded lean differentiated at the frozen
0.37 s window with k_d = 0.295. Identity check: feeding the commanded attitude itself through
the reader mapping returns the required lean with a median error of 0.0 deg, so the statistic
is unbiased in the absence of a generator error. Cluster bootstrap over paths (5,000 replicates,
`default_rng(1)`).

## Results

FILL

## What it supports

FILL

## What it does not show

FILL

## Caveats

FILL
