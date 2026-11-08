# Track A report: perception (Phase 1 depicted tilt, Phase 2 implied tilt)

Status: Gate 1a measured and **failed on its RMSE criterion**; Gates 1 and 2 **blocked**
on a visual-odometry build. Every number below comes from a `results/*/metrics.json`
(or a `results/phase1_camera*/` run), and anything marked blocked or not run is not a
pass.

Headline: a per-frame single-image gravity estimator gives a **3.5 deg median** depicted
tilt on TII (target 5 deg) but a **12.5 deg RMSE** because 3.4 % of frames fail
catastrophically; visual odometry, and therefore the implied-tilt half of RQ1, could not
be built in this handoff.

Branch `track-a-perception`. All structural choices are one line each in
`docs/decisions.md` tagged `[A]`.

## 1. What was verified before any estimator was blamed

| item | result | evidence |
| --- | --- | --- |
| Video location and size | 8 flights, camera JPEG zips **3.77 GB** total (plus 0.17 GB of CSVs/YAML), all needed frames present | `scripts/fetch_tii_video.py --list`; zip central directories read by ranged HTTP |
| Video-to-mocap offset | **0 by construction**; camera timestamps and mocap timestamps share one Unix-microsecond clock, and the release's `<flight>_cam_ts_sync.csv` is byte-identical to `csv_raw/camera_<flight>.csv`. Uncertainty 0.5 frame = 4.2 ms; exposure latency undocumented and not modelled | `src/datasets/tii_video.py::frame_table` (raises on any mismatch) |
| Frame rate | 120 Hz declared, **120.4 Hz measured**, median dt 8.3 ms, dropped frames up to 42 ms | `Clip.meta["measured_fps"]` |
| Camera model in the release | `cv2.calibrateCamera` pinhole + k1,k2,p1,p2,k3 — **not** a Kannala-Brandt fisheye model | `scripts/camera_calibration.py` in the release |
| Calibration coverage | polynomial invertible only to `rd = 0.98` (θ = 56 deg); **all four image corners unrepresentable**; `fx/fy = 0.749` in all three calibration files | `src/perception/undistort.py::model_diagnostics` |
| Virtual pinhole view | 100 deg horizontal, 640x480, square pixels, same optical axis; **46 % of its pixels are valid** (the rest sample outside the model's domain); max source ray reached 45.5 deg | `VirtualCamera.valid_fraction` |
| Camera-to-body convention | `drone_to_camera.json` quaternion is **camera->body**; for the ellipse/trackRATM mounts it equals the analytic hard-mount matrix exactly | `src/perception/frames.py::camera_to_body_pitched` + test |
| Autonomous flights' mount | mocap-implied optical-axis elevation matches the recorded uptilt: 01a 49.6/50, 04a 49.8/50, 07a 42.7/40, 13a 50.0/50 | `experiments/phase1_camera_check.py` |
| Piloted flights' mount | **unreconciled**: the single `piloted` JSON key is a 60 deg mount; metadata says 30 (01p, 07p) or 50 (13p, 16p); implied elevations are 36.7 / 38.8 / 2.3 / -2.0 deg, and the last two contradict the frames. Reported, not fitted | `results/phase1_camera/…/metrics.json`, overlays |

Two consequences worth stating plainly:

- Track A's truth-bearing set is the **four autonomous flights**. Anything that needs a
  trustworthy camera frame for the piloted flights is flagged, and the resolution path is
  PnP against the released gate-corner labels (not done here).
- The released intrinsics cannot describe the fisheye periphery. Every pixel-stage number
  below is therefore a number for a ~92 deg field of view, not for the full 175 deg lens.

## 2. Compute used

| tool | GPU | frames/s (measured) | s per 1000 frames | peak GPU | notes |
| --- | --- | --- | --- | --- | --- |
| GeoCalib `simple_divisional` on raw fisheye | A10G | ~5 (20-frame batches), 1.4 (3-frame cold start) | ~200 | 0.43 GB | LM optimization dominates, not the network |
| DPVO | - | - | - | - | **build blocked**, see section 4 |

Modal runs in this phase: `awmv-a-fetch-video` (3.94 GB fetched once, mirrored to
`track-a/tii_video` on R2), `awmv-a-frames` (sampled + windowed frames), `awmv-a-geocalib`
(probe, three configuration surveys, the full sweep), `awmv-a-dpvo` (six build attempts).
Total GPU time is well under an hour of A10G, i.e. under US$1.

## 3. Gate 1a: depicted tilt per frame (GeoCalib)

**Verdict: median meets the 5 deg target, RMSE does not. Gate 1a FAILS on RMSE.**

Configuration (chosen by measurement, not preference — see `decisions.md`): GeoCalib
`simple_divisional` on the **raw fisheye**, no focal prior, undistorted by nothing.
That route beat pinhole-on-undistorted by 2x (13.2 vs 26.9 deg pooled median flight
RMSE on the same 20-frame sample), which is itself a finding: the released calibration
is too poor for the undistorted route to pay off.

Numbers from `results/phase1_depicted/20261108T055529Z_3697b73/metrics.json`
(two full autonomous flights, all frames at 15 fps, 1,033 frames):

| quantity | value |
| --- | --- |
| median error (angle between estimated and mocap up in the camera frame) | **3.52 deg** |
| trimmed RMSE (worst 10 % of frames dropped) | **3.66 deg** |
| RMSE, all frames | **12.53 deg** |
| p90 / p99 | 8.15 / 81.4 deg |
| fraction of frames over 20 deg | **3.4 %** |

Per flight:

| flight | n | median | p90 | p99 | RMSE | trimmed RMSE | frac > 20 deg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 01a-ellipse | 450 | 4.24 | 10.28 | 82.06 | 13.74 | 4.27 | 4.0 % |
| 04a-ellipse | 583 | 3.25 | 6.84 | 81.41 | 11.52 | 3.21 | 2.9 % |

An independent stride-10 sweep over the same two flights reproduces these numbers
(01a median 4.20 / RMSE 13.67, 04a 3.29 / 10.35, `results/phase1_depicted/20261108T055529Z`
and the `div10` run), so they are not a stride artefact.

By speed (pooled):

| speed (m/s) | n | median | RMSE |
| --- | --- | --- | --- |
| 0-5 | 815 | 3.48 | 5.14 |
| 5-10 | 104 | 3.93 | 8.29 |
| 10-15 | 35 | 2.99 | 32.32 |
| 15-20 | 65 | 3.89 | 37.29 |
| 20+ | 14 | 4.12 | 21.72 |

**The error is heavy-tailed, not biased.** The median is flat in speed (2.99-4.12 deg,
i.e. it does not degrade the way Phase 0's physics residual does), while the RMSE
explodes above 10 m/s: about 3-4 % of frames come back with an orientation that is
near-wrong (p99 ~81 deg). Reporting one RMSE for this estimator would hide that.

With/without one constant rotation per flight: the fitted offset is 4.0 deg (01a) and
3.4 deg (04a) and **increases** the error rather than reducing it, i.e. the residual is
not a constant mount offset. The handoff's known 3.9 deg mocap offset on the autonomous
ellipse flights is of the same size as this constant term, which is worth noting but is
not what dominates the RMSE.

### Why the tail exists (diagnosis, not blame)

The synthetic-rotation probe (`--probe-flights`) tilts the view by +/-20 deg and asks
whether the recovered gravity rotates with it:

| flight | -20 deg consistency | +20 deg consistency |
| --- | --- | --- |
| 07p-lemniscate | 1.47 deg median | 2.56 deg |
| 01a-ellipse | 14.25 deg | 1.48 deg |
| 13a-trackRATM | 13.55 deg | 6.52 deg (p90 62.5) |

A geometry or convention bug would break both directions on every flight equally. Instead
the estimator is *view-dependent*: the direction that tips the camera further away from
the informative structure (the wall/gate band) is the direction where it loses the answer.
Combined with the near-ceiling content of the 40-50 deg mounts, this is a scene-content
failure — the indoor arena with no visible horizon — not a pipeline failure.

Blur and exposure (the breakdown the plan asks for):

- `exposure` is the only blur label the dataset carries: `fixed` on 04a and 16p, `auto`
  elsewhere. Pooled on the 192-frame sample, `auto` medians 4.48 deg and `fixed` 3.92 deg
  (RMSE 26.1 vs 14.4) — the fixed-exposure flights are better, but they are also the
  slow ones, so this is confounded with speed.
- Laplacian variance as a stated proxy (`scripts/frame_stats.py`, 192 frames): rank
  correlation with error is **-0.22** (sharper frames do better), with mid-sharpness
  frames (150-400) worst at 26 % over 20 deg and the sharpest frames (400+) best at
  1.6 %. Weak and not monotone, so blur is a contributor, not the cause.
- By track type: ellipse median 3.53 deg (RMSE 5.40), lemniscate 3.25 (19.58),
  trackRATM 7.13 (34.84).
- By control: autonomous median 4.05, piloted 5.01 — but the piloted flights' camera
  frame is itself unreconciled (section 1), so that comparison is not clean.

The undistorted route cannot help either: 46 % valid pixels, and a released lens model
that does not cover the frame.

## 3b. What Gate 1a's result means

Gate 1a as written (RMSE under 5 deg) is not met by a per-frame, single-image gravity
estimator on TII. What *is* met: a 3.5 deg median with a 3.7 deg trimmed RMSE, which is
inside the target for the 96 % of frames the estimator does not flip on. The natural
follow-up is an outlier rule (a robust per-clip aggregate plus a rejection test on the
frame-to-frame consistency of the estimate) rather than a better single-frame model,
and that is Phase 3 work, deliberately not built here.

## 4. Gate 1: depicted attitude over time (DPVO + gravity fusion)

**Status: BLOCKED. Gate 1 is not evaluated, because visual odometry could not be built.**

DPVO's Python entry points are in place (`remote/a_dpvo.py`: build, weights into
`awmv-a-data/dpvo_models`, virtual-pinhole frame feed, poses + timestamps saved to the
Volume) and the fusion side is written and unit-tested (`src/fusion/`, 5 tests,
`tests/test_fusion.py`; gain sweep and leave-one-flight-out selection in
`experiments/phase1b_fusion.py`). What is missing is a working `dpvo` extension module.

The build on Modal (CUDA 12.4 devel base, torch 2.3.0+cu121, repo pinned at
`0ac95b65`) failed in this order, each step a real fix recorded in `decisions.md`:

| attempt | failure | fix applied |
| --- | --- | --- |
| 1 | torchvision 0.18.0 requires torch 2.3.0, not 2.3.1 | pin both |
| 2 | `torch-scatter` not found on the pyg index | install from find-links |
| 3 | same, `--index-url` vs `-f` | use `-f` (find-links) |
| 4 | `which clang++` returns non-zero inside `cpp_extension` | apt install `clang` |
| 5 | `_get_cuda_arch_flags`: `IndexError` — no GPU visible to the image builder | set `TORCH_CUDA_ARCH_LIST=8.6` |

The last attempt is the one that should have worked (that IndexError is the standard
Modal pattern for CUDA extension builds); its result is not in hand. If it also fails,
the next route is the one the handoff names: an interactive RunPod pod for the compile,
or the VGGT/MegaSaM fallback, which needs no CUDA build.

Because of this, none of the following exist yet: gravity alignment on real VO
rotations, the complementary filter at flight scale, and every Gate 1 number. They are
**not** reported as passes.

## 4b. Gate 1's building blocks that exist and are tested

- `gravity_align`: one rotation per clip from the robust mean measured up, with the
  per-frame spread reported (a large spread means the gravity estimates disagree).
- `complementary_filter`: VO relative rotation propagates, gravity corrects with a fixed
  gain; verified that gain 1 snaps the measured up to world up within 1e-6 deg, gain 0
  never corrects, and gain 0.2 converges from a 56 deg initial error to under 1 deg in
  about 0.3 s at 120 Hz. One real bug was found and fixed this way: the filter corrects
  toward world up, so it must run in the gravity-aligned frame (conjugating the relative
  rotations), not in the raw VO frame.
- `lean_error_deg`: azimuth error on frames above 10 deg tilt, with one constant yaw per
  clip (evaluation only, recorded in the metrics).

## 5. Gate 2: implied tilt from the pixel trajectory

**Status: BLOCKED, same cause.** `experiments/phase2_implied.py` implements the whole
ladder (Savitzky-Golay window swept **in seconds** including the 0.11 s reference,
fixed `k_d = 0.295` and drag-free, oracle / fitted / depth scale options, the residual
`r(t)` against the depicted axis, SNR, testable fraction, and the oracle-ladder
attribution table). It consumes DPVO poses; there are none.

One part of the planned ladder is also **not run** rather than substituted: the
depth-model scale needs a MoGe-2 pass, which is a separate Modal job that was not
started. The metrics record `scale_depth: {"run": false}` so it cannot be mistaken for
a fitted-scale result.

## 6. Oracle ladder (attribution)

Not produced, because four of its six rungs need a VO trajectory. The only rung that
exists is the Phase 0 reference (mocap position + mocap attitude, median residual
1.24 deg, `results/phase0_floor/20261108T044153Z_f3b5d2bf`), which is the number any
pixel stage has to beat and does not.

## 7. Estimator comparison

| route | pooled median flight RMSE (20-frame sample) | status |
| --- | --- | --- |
| GeoCalib `simple_divisional`, raw fisheye | **13.18 deg** | adopted |
| GeoCalib `pinhole`, undistorted view, focal prior | 26.90 deg | rejected |
| GeoCalib `pinhole`, undistorted view, free focal | 25.41 deg | rejected |
| Perspective Fields | - | not run (GeoCalib's median met the target on most frames) |
| AnyCalib | - | not run, same reason |

## 8. What did not work, and why

- `experiments/phase1_camera_solve.py` (mocap-supervised triangulation over a 6 m baseline)
  leaves 12-1000 px reprojection residual at the best rotation, i.e. the released lens model
  is not good enough for multi-view geometry. Kept as a diagnostic; **not** used as evidence.
- The piloted flights' camera frame, see above.

## 9. Next steps that would move the numbers

0. Get one VO working (see section 4); everything in Gates 1 and 2 is downstream of it.
1. Gate-corner PnP per piloted flight (`gate_corners_<flight>.csv` + `label_flight-<flight>.zip`)
   to settle the mount, which would let the piloted flights (19-21 m/s) into the gate.
2. A fisheye model fitted to the released calibration plus the stated 175 deg field, or a
   line-based self-calibration, to recover the periphery.
3. MoGe-2 metric scale to replace the fitted scale in the verifier's own operating point.
