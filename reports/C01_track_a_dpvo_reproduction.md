# Report C01 — Reproducing Track A's DPVO translation number

Track C. Written 2026-11-10. Task 0 of handoff 06.

## Plain-language summary

Track A's DPVO run on TII is the reason this track exists: "recovered path 41 m against 479 m
of truth". I replayed the comparison from the stored poses and mocap and **the 41 m does not
reproduce**: with today's stored artifact the VO path is 2.1 m after the same scalar fit, and
960 camera-units before it. But the underlying verdict does reproduce, and the picture is
worse, not better, than "the path came out too short":

- the raw VO path is **longer** than the true path (960 u vs 479 m, 1239 u vs 444 m,
  454 u vs 336 m), so DPVO was not underestimating motion — it was moving in the wrong
  directions;
- the best similarity fit needs a scale of 0.0065 to 0.12, i.e. the fitted rotation has to
  shrink the VO path almost to nothing to line it up with truth;
- per-axis correlation with truth stays near zero (0.02 / 0.06 / -0.26 on 01p).

So on TII the failure is **translation direction**, not translation magnitude. That is the
same failure mode the VGGT paper describes for low parallax, and it is the one Track C is
built to measure on data where the answer is known.

## Question and setup

**Question.** Can the number behind the Phase 2 attribution ("VO path 41 m vs mocap 479 m
on 01p") be recovered from stored artifacts, and what exactly does it say?

**Setup.** `experiments/pathrec_repro_dpvo.py` (Track C, read-only on Track A's files) loads
`data/raw/dpvo/<flight>_stride10_fov100_s0.5/poses.npz` (camera-to-world, `tx ty tz qx qy qz qw`,
12 Hz = stride 10 of the 120 Hz camera), `data/raw/geocalib/div10/<flight>.jsonl`, and the
mocap CSV through `src/datasets/tii.py`, then reports four alignments plus the exact Phase 2
route:

| label | alignment |
| --- | --- |
| `track_a_scalar` | scalar scale + translation, identity rotation (Phase 2 `fit_scale_oracle`) |
| `sim3` | Umeyama similarity, 7 DoF |
| `se3` | Umeyama rigid, scale fixed at 1 |
| `yaw4_after_oracle_up` | scale + yaw about z + translation, after an oracle up-alignment |
| `phase2_exact` | joint GeoCalib frames, `gravity_align`, then the Phase 2 scalar fit |

Run: `results/pathrec_repro/repro_v1/repro.json`, commit `164c5e0` + this change,
`python experiments/pathrec_repro_dpvo.py --dpvo-root ... --mocap-root ... --geocalib-root ...`.

## Results

Phase 2 route, all eight flights at stride 10, dt = 0.083 s, 270 to 1284 frames:

| flight | mocap path (m) | raw VO path (u) | scalar scale | VO path after scalar (m) | ATE (m) | axis corr (x, y, z) |
| --- | --- | --- | --- | --- | --- | --- |
| 01a-ellipse | 76 | 115.3 | -0.00298 | 0.34 | 3.58 | -0.21, 0.27, -0.01 |
| 01p-ellipse | 479 | 960.0 | -0.00221 | 2.12 | 5.97 | 0.02, 0.06, -0.26 |
| 04a-ellipse | 76 | 165.2 | -0.00892 | 1.47 | 3.15 | -0.03, 0.32, 0.18 |
| 07a-lemniscate | 60 | 119.6 | 0.01126 | 1.35 | 2.33 | 0.17, 0.02, 0.00 |
| 07p-lemniscate | 444 | 1239.2 | -0.00379 | 4.70 | 5.73 | 0.16, -0.01, 0.25 |
| 13a-trackRATM | 193 | 256.6 | -0.04125 | 10.58 | 5.38 | 0.34, 0.57, -0.13 |
| 13p-trackRATM | 430 | 477.1 | 0.00289 | 1.38 | 6.34 | 0.06, 0.10, -0.08 |
| 16p-trackRATM | 336 | 453.6 | 0.00065 | 0.30 | 6.31 | 0.05, 0.03, -0.15 |

Similarity (7 DoF) scale per flight: 0.122, 0.0065, 0.056, 0.053, 0.0079, 0.070, 0.017, 0.017.
SE(3) ATE (scale fixed at 1): 9.8, 53.3, 15.7, 11.7, 105.2, 19.5, 58.4, 49.3 m.

### Against the stored Phase 2 metrics

| quantity | stored `metrics.json` (01p) | reproduced from stored artifacts |
| --- | --- | --- |
| mocap path | 479 m | **479 m** ✓ |
| ATE / oracle RMSE | 5.97 m | **5.97 m** ✓ |
| scalar scale | -0.003871 | -0.00221 (same sign, 1.8× smaller) |
| axis corr | 0.01 / -0.09 / 0.13 | 0.02 / 0.06 / -0.26 (same regime) |
| VO path | 41 m | 2.12 m (phase 2 route), 960 u raw |
| frames | n = 317, dt = 0.333 s | n = 1267, dt = 0.083 s |

## What went wrong with the 41 m

The stored run is internally consistent only if its GeoCalib joint set had 317 frames at
0.333 s. The artifacts now on disk pair 1267 DPVO poses with 1267 GeoCalib rows at 0.083 s.
The `metrics.json` does not record which frames it used, so the exact input set cannot be
reconstructed — the two numbers are from different frame selections of the same source
material. The 41 m figure also appears in report 03 with the note "not re-derived in this
pass". **41 m is therefore a stale quoted number, not reproducible.** Everything that
matters to the attribution (mocap length, ATE, scale near zero, correlation near zero)
does reproduce.

The 0.333 s vs 0.083 s discrepancy is worth flagging independently: acceleration goes as
1/dt², so a 4× dt error is a 16× acceleration error. Track C will state dt for every number
and pre-register the differentiation window in seconds.

## What this supports

- The Phase 2 attribution stands on reproduced evidence: DPVO's translation on TII is not
  usable, and the failure is **directional**, not a scale factor (raw VO path is 1.4× to 2.8×
  the true path length; the fitted similarity scale collapses the path instead of stretching
  it).
- Track C's success criterion should be built around acceleration **direction**, which the
  handoff already says is scale-free, not around path length or ATE.
- A useful negative control for Track C: even an ATE of 5 m on a 479 m flight (1.2 %) can
  coexist with 0.0 correlation in each axis. Absolute trajectory error is the wrong metric
  for this verifier; the pre-registration must not let ATE stand in for it.

## What this does not show

- Whether DPVO's translation would be fine on clean, well-calibrated, high-frame-rate video.
  Nothing here tests that; it is Track C's job (Tasks 2 and 3).
- Whether the 0.333 s subsampling of the stored Phase 2 run contributed to the failure.
  At 0.333 s and up to 20 m/s a drone moves up to 6.7 m between frames, which is a plausible
  independent cause, but it cannot be separated now because the frame selection is unrecorded.
  **Inferred, not measured.**
- Rotation quality. Track A measured 0.15 to 1.5 deg per frame against mocap; I did not
  re-derive it here.
- Anything about the fisheye/calibration hypothesis. It remains a hypothesis; the
  frame-selection confound above is a second, cheaper explanation for the same observation.

## Caveats

- The mocap is interpolated onto the VO timestamps (`np.interp`). Track A used
  nearest-sample lookup; on a 275 Hz stream at 12 Hz the difference is ~2 mm.
- `yaw4_after_oracle_up` is reported for completeness only: it is applied after an oracle
  rotation, so it cannot separate yaw from the rest. It shows the residual fit is not a
  pure yaw offset.
- No quaternion order ambiguity: `poses.npz` is xyzw, as recorded by the loader and by
  Track A's decision entry.
