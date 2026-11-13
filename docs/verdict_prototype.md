# The gain-test verdict prototype

Status: frozen 2026-11-13 on branch `track-d-verdict-stat`. Frozen statistic:
`configs/gain_frozen.yaml` (rule `R1_scalar_g1`, band [0.5277, 0.9477] on `G1`, alpha
0.05, n = 57 real clips). Evidence: `reports/D01_gain_test.md`,
`results/d_eval/d01/metrics.json`.

## What it is

A quadrotor can only thrust along its body z-axis, so the tilt its path requires is fixed
by the path and the tilt its pixels depict is measurable. The gain test compares them per
clip by a regression rather than by a median residual:

```
l_dep = G1 * l_req0 + G2 * (Rz(-yaw) v / g) + b
```

- `l_req0` is the **drag-free required lean**, the horizontal part of the unit thrust axis
  `normalize(a + g e3)` in the yaw-aligned frame, with `a` and `v` from Savitzky-Golay
  differentiation of the commanded path (`pos`, `yaw`).
- `l_dep` is the **depicted lean**, the horizontal part of the body z-axis reconstructed
  from the per-frame gravity estimate through the camera-to-body mount.
- `G1` is the acceleration gain the verdict tests. A real flight gives a positive value
  (0.745 median on TII, not 1: the estimator under-reads tilt and the regressor carries
  noise, so the band is calibrated on real flights). Tilt removal gives ~0, a 50 % scale
  gives ~0.5, a 1.5 scale ~1.5, and a mirrored roll flips the sign of the roll-axis gain.
- `G2` is a drag nuisance (1/s); the free intercept `b` absorbs the mount offset.

Decision: flag if `G1` is outside the frozen two-sided conformal band. The richer axis
rule (a joint band on `gx, gy`) was pre-registered and lost selection by one false-reject
point; it is available in `src/verdict/decide.py` if a later round re-opens the protocol.

## Score one clip

```
PYTHONPATH=. pixi run python scripts/score_clip.py PATH --config configs/gain_frozen.yaml --json
```

`PATH` is a directory holding `verdict_clip.npz` and `meta.json`, or the `.npz` itself.
The output is graded, not pass/fail:

| field | meaning |
| --- | --- |
| `gain_g1`, `gain_g1_ci95`, `gain_g1_half_width` | the tested gain and its block-bootstrap interval |
| `axis_gains`, `axis_ci95` | roll (`gy`) and pitch (`gx`) gains, fitted separately |
| `drag_coefficient`, `drag_ci95` | the fitted nuisance; **not** the airframe drag (see below) |
| `intercept` | the absorbed mount offset |
| `r_squared`, `lag_ms`, `condition_number` | fit diagnostics |
| `required_tilt_deg`, `depicted_tilt_deg` | median tilts of the two sides |
| `testable`, `abstention_reasons` | whether the clip can answer the question, and why not |
| `flagged` | `true`/`false` when testable and a band is configured, otherwise `null` |
| `gate_kept` | the frozen S3 confidence decision |

A clip is **not testable** (abstains) when any of the pre-registered rules fires: fewer
than 24 frames, fewer than 50 % usable depicted frames, fewer than 12 frames or 20 % of
the clip above 10 deg of required tilt, principal spread of the required lean below 2 deg,
design condition number above 1000, bootstrap half-width of `G1` above 0.35, or the frozen
confidence gate. Abstention is not a failure; it means the clip carries no answer.

## What a result means

- **Not flagged and testable**: no gain violation of the size the band can see on this
  clip. It does **not** mean the flight is physically consistent, and it does not certify
  the airframe, the drag, or the path beyond the commanded path supplied.
- **Flagged**: the depicted lean's relationship to the required lean is outside the
  real-flight band. On TII this detects a full tilt removal 100 % of the time and a
  mirrored roll 96.5 % of the time at an 8.8 % false-reject rate; a 50 % change is
  detected 86-100 % of the time in a synthetic sweep, a 25 % change only 28-42 %.
- **Abstained**: no answer. Most autonomous and track clips abstain on this data.

The minimum detectable change is between 25 % and 50 % of the tilt, dependent on
excitation. The yaw-only control is flagged 22.8 % against 8.8 % of real clips: the
estimator's view dependence is the floor under the statistic, not a property of the
flight.

## The drag coefficient is a nuisance, not an estimate

The fitted `G2` is 0.139 1/s median on TII against Phase 0's mocap-only 0.295. It is about
half because the same errors-in-variables attenuation that puts the real gain at 0.745
also attenuates the velocity coefficient (inferred). Fixing `G2` at 0.295 is measurably
worse (x0.5 detection 37 % against 98 %), so it is fitted; it must not be quoted as the
airframe's drag coefficient.

## The shared clip schema

`verdict_clip.npz` with `t` (N,), `pos` (N,3, commanded path, world z-up), `yaw` (N,),
`depicted_lean` (N,2, NaN where the estimator failed), `depicted_conf` (N,), and a sibling
`meta.json` (dataset, sequence/flight, clip id, condition, fps, mount source and uptilt,
estimator and variant, truth-frame notes). Only `pos` and `yaw` carry path truth; the
required lean is recomputed from them. `src/verdict/schema.py` validates the file;
`scripts/score_clip.py` is the only entry point a consumer needs.

Handoff to Track E (UZH-FPV): read the frozen file and the scorer with

```
git show track-d-verdict-stat:configs/gain_frozen.yaml
git show track-d-verdict-stat:scripts/score_clip.py
```

build UZH clips in the schema above, score them with the frozen config unchanged, and
report false-reject and detection of removal/mirror with Wilson intervals. Do not
recalibrate the band on UZH: that would make T2 unmeasurable.

## Reproducing the TII result

```
PYTHONPATH=. pixi run python experiments/d_build_clips.py
PYTHONPATH=. pixi run python experiments/d_repro_baseline.py
PYTHONPATH=. pixi run python experiments/d_eval.py --tag d01
PYTHONPATH=. pixi run pytest -q
```

The clips are derived from `data/processed/p3a/p3a_v1` and `results/gate_a2/tii`
(gitignored/rebuildable); the fitted per-clip table and metrics are committed under
`results/d_eval/d01/`.
