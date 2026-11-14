# The gain-test verdict prototype

Status: frozen 2026-11-14 on branch `track-d-verdict-stat`, schema v2, dense required lean.
Frozen gain statistic: `configs/gain_frozen.yaml` (rule **R2_axis_gains**, axis joint band
`gx` center 0.8137 / scale 0.2654, `gy` center 0.7181 / scale 0.0735, tau 2.1922, alpha
0.05, n=57). Frozen coherence tier: `configs/coherence_frozen.yaml` (tau_corr 0.6197,
tau_sign 0.2921). Evidence: `reports/D02_gain_test_v2.md`,
`results/d_eval/d02/metrics.json`, `results/d_coherence/d01/metrics.json`; code commit `8b8b7a1`.

## What it is

A quadrotor can only thrust along its body z-axis, so the tilt its path requires is fixed
by the path and the tilt its pixels depict is measurable. The gain test compares them per
clip by a regression:

```
l_dep = G1 * lean_req + G2 * vel_req + b
```

- `lean_req` is the **drag-free required lean**: the horizontal part of
  `normalize(a + g e3)` in the yaw-aligned frame, from the **dense** commanded path
  (274.73 Hz on TII) with a 0.37 s / poly-3 Savitzky-Golay window, sampled at the frame
  times.
- `vel_req = Rz(-yaw) v_h / g` is the velocity regressor (units 1/s when multiplied by a
  drag coefficient).
- `l_dep` is the **depicted lean**, the horizontal part of the body z-axis reconstructed
  from the per-frame gravity estimate through the camera-to-body mount.
- `G1` is the acceleration gain, `G2` the drag nuisance, `b` a free 2-D intercept that
  absorbs the mount offset. Per-axis gains `gx` (pitch/forward) and `gy` (roll/lateral)
  are fitted separately; the frozen rule tests the pair jointly.
- A real flight gives a positive gain (real `G1` median **0.758**, IQR 0.714-0.818; `gy`
  0.718, IQR 0.665-0.754), not 1: the pixel estimator **shrinks** the depicted tilt by
  about 0.83 (report D02 §1). The band is therefore calibrated on real flights.
- Tilt removal gives ~0, a 50 % scale ~0.5, a 1.5 scale ~1.5, and a mirrored roll flips
  the sign of `gy` while leaving `gx` positive.

The **coherence tier** is a scale-free complement: per-axis Pearson correlation and sign
agreement between depicted and required lean. It catches shape/sign violations (removal
AUROC 0.999, mirror 1.000) and is at chance for pure scaling (x0.5 0.706, x1.5 0.622),
so it does not replace the gain. It is reported, not part of the frozen flag.

## Score one clip

```
PYTHONPATH=. pixi run python scripts/score_clip.py PATH --config configs/gain_frozen.yaml --json
```

`PATH` is a directory holding `verdict_clip.npz` (schema v2) and `meta.json`, or the
`.npz` itself. The output is graded, not pass/fail:

| field | meaning |
| --- | --- |
| `gain_g1`, `gain_g1_ci95`, `gain_g1_half_width` | scalar gain and its block-bootstrap interval |
| `axis_gains`, `axis_ci95` | roll (`gy`) and pitch (`gx`) gains, fitted separately; the frozen rule uses these |
| `drag_coefficient`, `drag_ci95` | the fitted nuisance; **not** the airframe drag (see below) |
| `intercept` | the absorbed mount offset |
| `r_squared`, `lag_ms`, `condition_number` | fit diagnostics |
| `required_tilt_deg`, `depicted_tilt_deg` | median tilts of the two sides |
| `testable`, `abstention_reasons` | whether the clip can answer the question, and why not |
| `validation_status`, `usable_as_evidence` | whether the statistic is calibrated for this flight type (amendment 2) |
| `flagged` | `true`/`false` when testable and bands are configured, otherwise `null` |
| `gate_kept` | the frozen S3 confidence decision |

A clip is **not testable** (abstains) when any pre-registered rule fires: fewer than 24
frames, fewer than 50 % usable depicted frames, fewer than 12 frames or 20 % of the clip
above 10 deg of required tilt, principal spread of the required lean below 2 deg, design
condition number above 1000, bootstrap half-width of `G1` above 0.35, or the frozen
confidence gate. Abstention is not a failure; it means the clip carries no answer.

**Paired evaluation note.** In the reported tests, testability is decided on the **real**
clip and inherited by its edits. A mirror clip scored standalone can abstain on the `G1`
interval (the frozen abstention is on `G1`, while the decision uses `gx`/`gy`); the paired
protocol does not, because the real clip is testable. This is a property of the
pre-registered abstention, not a change to it.

## What a result means

- **Not flagged and testable**: no gain violation of the size the band can see on this
  clip. It does **not** mean the flight is physically consistent, and it does not certify
  the airframe, the drag, or the path beyond the commanded path supplied.
- **Flagged**: the depicted lean's relationship to the required lean is outside the
  real-flight band. On TII v2 R2 this detects full tilt removal **100 %**, mirrored roll
  **100 %**, x0.5 **100 %** and x1.5 **96.5 %** at a **5.3 %** false-reject rate (n=57;
  nested selection 5.3 %). A synthetic 50 % gain change is detected 94.7-100 %; a 25 %
  change 60-72 %.
- **Abstained**: no answer. 43 % of TII real clips abstain; most autonomous and track
  clips do.

**Yaw-control rate beside every detection.** A yaw rotation cannot change the tilt, yet
the yaw-only control is flagged **28.1 %** under the frozen R2 rule (19.3 % under R1)
against **5.3 %** of real clips. The coherence tier flags it **10.5 %** against 8.8 %.
This is the estimator's view dependence and is the floor under the statistic: a violation
that shifts the gain by less than this artifact cannot be separated from it.

**Validation status (amendment 2).** Piloted, non-track flights are `validated`.
Autonomous and track flights are `not_validated`: a graded result is still produced, but
`usable_as_evidence` is false. They do not abstain, because the surviving clips have
enough excitation; what they lack is calibration evidence (4 and 3 testable clips, 25 %
and 33 % false-reject). The status is written to `scripts/score_clip.py` output and
requires `flight_type` in `meta.json`.

**Track flights are not validated.** Only 3 track clips are testable under v2; one real
clip is falsely rejected, one x1.5 edit is missed, and the yaw-control flag rate is
66.7 % on three clips. The pooled 5.3 % is a piloted-flight number. Autonomous is four
testable clips with a 25 % false-reject rate. Do not quote pooled numbers as racing or
autonomous performance.

## The drag coefficient is a nuisance, not an estimate

The verdict's fitted `G2` is **0.158 1/s** median (v2) against Phase 0's mocap-only
**0.295**. The decomposition shows why: truth attitude on the required path gives 0.239
with a fitted drag, 1.048 with drag fixed at 0.295, and 0.755 drag-free; the dominant loss
in the verdict is estimator shrinkage (0.83), not differentiation. Do not quote `0.158` as
the airframe drag coefficient.

## The required-lean window

The frozen statistic uses a **0.37 s** Savitzky-Golay window on dense truth. The TII
sensitivity (`reports/D03_shrinkage_window.md`) gives 5.3 % false-reject at 0.10/0.185/
0.37 s, but 0.10 s loses partial-change detection (x0.5/x1.5 64.9 %) and 0.74 s raises
false-reject to 12.3 %. 0.185 s keeps detection and lowers the yaw-control artifact from
28.1 % to 21.1 %. The band does **not** transfer across windows (frozen 0.37 s band gives
8.8-17.5 % false-reject at other windows). The proposal is `configs/window_amendment_proposal.yaml`:
0.185 s with an 8-sample floor, **not applied**; 0.37 s remains frozen until the user
approves. Do not switch windows in the field without that approval.

## The shared clip schema (v2)

`verdict_clip.npz` with `t` (N,), `pos` (N,3, commanded path, world z-up), `yaw` (N,),
`depicted_lean` (N,2, NaN where the estimator failed), `depicted_conf` (N,), `lean_req`
(N,2), `vel_req` (N,2), optional `tilt_req_deg` (N,), and a sibling `meta.json`. Only
`pos`/`yaw` carry path truth; `lean_req`/`vel_req` are derived from them at dense rate.
Full definition and export instructions: `docs/verdict_clip_schema_v2.md`.
`src/verdict/schema.py` validates the file; `scripts/score_clip.py` is the consumer entry
point. A v1 clip (no `lean_req`) still loads and falls back to differentiating `pos`.

Handoff to Track E (UZH-FPV):

```
git show track-d-verdict-stat:configs/gain_frozen.yaml
git show track-d-verdict-stat:configs/coherence_frozen.yaml
git show track-d-verdict-stat:docs/verdict_clip_schema_v2.md
git show track-d-verdict-stat:scripts/score_clip.py
```

Re-export UZH clips in schema v2, score with the frozen configs unchanged, and report
false-reject and detection of removal/mirror **with the yaw-control rate beside them** and
track marked not validated. Do not recalibrate either threshold on UZH: that would make
T2 unmeasurable.

## Reproducing the TII result

```
PYTHONPATH=. pixi run python experiments/d_build_clips.py
PYTHONPATH=. pixi run python experiments/d_decompose.py
PYTHONPATH=. pixi run python experiments/d_eval.py --tag d02
PYTHONPATH=. pixi run python experiments/d_coherence.py --run results/d_eval/d02
PYTHONPATH=. pixi run python experiments/d_deming.py
PYTHONPATH=. pixi run python experiments/d_freeze.py --run results/d_eval/d02
PYTHONPATH=. pixi run pytest -q
```

The clips are derived from `data/processed/p3a/p3a_v1` and `results/gate_a2/tii`
(gitignored/rebuildable); metrics, tables and plots are committed under
`results/d_eval/d02/`, `results/d_decomposition/d01/`, `results/d_coherence/d01/` and
`results/d_deming/d01/`.
