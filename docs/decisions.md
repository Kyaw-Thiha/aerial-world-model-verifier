# Decisions log

One line per structural choice, with the reason. Newest at the bottom.

## Setup

- Flat `src/` with `__init__.py` files, no `src/verifier/`: scripts run from the repo root and import `from src.physics import ...` (per project_structure.md).
- Folders are created in the step that first needs them (setup plus the physics floor create `src/datasets`, `src/physics`, `experiments`, `scripts`, `tests`, `data`, `results`, `docs/handoffs`); `remote/`, `third_party/`, `src/utils/` etc. are not created until used.
- pixi (`pixi.toml` + committed `pixi.lock`) replaces `requirements.txt`: the handoff says not to create it, although project_structure.md still lists it.
- pixi platform is `linux-64` only (workstation is linux x64); `osx-arm64` can be added if a Mac joins.
- Python 3.12 from conda-forge; the default environment is CPU only, no torch/CUDA/perception. A separate `gpu` feature is added when pixel work starts.
- `python-dateutil<3` is pinned in the conda deps: the conda solve picked `python-dateutil==4.0.0`, which made the PyPI resolver fall back to ancient boto3 0.0.3 / botocore 0.76 / six 1.7.3 (`import boto3` failed with `six.moves`). With the pin, boto3/botocore 1.43.126 install.
- `boto3>=1.34` and `python-dotenv>=1.0` are lower-bounded for the same reason; `modal` is unpinned (resolved 1.6.3). Modal containers get their own image definition under `remote/` and do not mirror `pixi.toml`; shared pins will be listed here when that exists.
- `.env.example` keeps the user's original entries (R2_*, R2_TOKEN_VALUE, RUNPOD_API_KEY, WANDB_API_KEY, R2_BUCKET default name) and adds the names required by project_structure.md that were missing (MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, HF_TOKEN). Names only; no key values.
- The handoff file was not in the repo, so `docs/handoffs/01_setup_phase0_anwm.md` is a condensed record of the handoff plus the user's follow-up additions, not a verbatim copy.
- Compute: the mocap floor runs locally on CPU. If a download is blocked locally, the fallback is a Modal CPU function writing to a Modal Volume (never to local disk directly), pulled with `modal volume get`; Volume name and path get logged here if used.
- `pytest.ini` with `pythonpath = .` (and `testpaths = tests`): the flat `src/` layout is not an installed package, so pytest needs the repo root on `sys.path`; no `pyproject.toml` added.

## TII dataset (verified from the official repo and paper)

- Source: https://github.com/tii-racing/drone-racing-dataset, release `v3.0.0` (2026-01-10). Code MIT; **data CC BY 4.0** (README license section). Paper arXiv:2311.02667v2.
- Release size: `autonomous` 4.95 GB + `piloted` 12.15 GB as chunked zips (~17.1 GB, mostly JPEG frames and rosbags). Chunks are served with HTTP `Range`, so `scripts/fetch_tii.py` mounts them as one virtual zip and extracts only the members needed; a full 8-flight mocap pull transfers ~110 MB.
- Mocap files used: `<kind>/<flight>/csv_raw/mocap_<flight>.csv`. Format (paper Sec. V-B, verified in the files): `timestamp` in Unix **microseconds**, `drone_xyz` in metres, `drone_roll/pitch/yaw` in radians, `drone_residual` in metres, and `drone_rot[0..8]` as a **column-major** 3x3 body-to-world rotation matrix. Body frame is the FCU IMU frame in FLU (x forward, y left, z up); world frame z-up.
- Matrix order verified two ways: `||R R^T - I|| < 2e-5` and the sign of column 2's x/y against `drone_pitch`/`drone_roll`. A row-major reading flips both signs and would look like a 10-20 degree attitude error, so the loader checks orthonormality and raises on failure.
- Mocap rate 274.73 Hz measured (paper says 275 Hz), dt = 3.63/3.64 ms, no gaps, no NaNs and no duplicate timestamps in the 12 flights surveyed. **No resampling applied**; `load_tii(..., resample_hz=...)` exists for later stages that need a coarser or camera-rate grid and records it in `clip.meta["resampled"]`.
- Camera uptilt is read from the per-flight `metadata_<flight>.yaml` (`camera.angle`), not guessed: 30 deg for piloted ellipse/lemniscate, 40 deg for autonomous lemniscate, 50 deg for autonomous ellipse/trackRATM and the piloted trackRATM flights. It does not enter the mocap-only floor, which compares body-z axes only.

### Flights used (8, split by flight)

| flight | control | track | uptilt | v_max | v_p95 |
| --- | --- | --- | --- | --- | --- |
| flight-01a-ellipse | autonomous | ellipse | 50 | 21.7 | 18.4 |
| flight-04a-ellipse | autonomous | ellipse | 50 | 21.7 | 17.4 |
| flight-07a-lemniscate | autonomous | lemniscate | 40 | 10.1 | 8.0 |
| flight-13a-trackRATM | autonomous | trackRATM | 50 | 21.2 | 14.8 |
| flight-01p-ellipse | piloted | ellipse | 30 | 8.4 | 7.6 |
| flight-07p-lemniscate | piloted | lemniscate | 30 | 8.3 | 6.7 |
| flight-13p-trackRATM | piloted | trackRATM | 50 | 19.3 | 17.3 |
| flight-16p-trackRATM | piloted | trackRATM | 50 | 21.0 | 19.1 |

- Selection covers both control modes, all three tracks, all three uptilts, both exposure settings, high/medium brightness, and speeds from 6.7 to 19.1 m/s at p95 (five flights above 10 m/s), so the drag regime is exercised rather than skipped.
- Splits are leave-one-flight-out: no frame from a held-out flight is used to pick the window or fit `k_d`.

## Floor protocol

- Differentiator: Savitzky-Golay, polynomial order 3, `deriv=1` and `deriv=2`, `delta = clip.dt`. Half-window trimmed at each end (one-sided stencils are biased and level the acceleration at the endpoints).
- Window policy: candidate set 11..401 samples, chosen **per fold on the fit flights only** by pooled median residual after fitting `k_d`. All eight folds chose 101 samples (0.37 s). The sweep is interior: residual rises below 101 (mocap noise amplified by double differentiation) and above 200 (acceleration is over-smoothed).
- Caveat recorded with the number: a 0.37 s window low-passes the trajectory, so this is the **model error floor**. A pixel-based stage estimating motion at 30-120 Hz cannot reach it; it will sit somewhere on the same sweep curve at a shorter window. The w=31 point (1.4-1.8 deg fit-set median) is the fairer reference for that stage.
- Drag: one scalar **world-frame** `k_d` in 1/s per fold, fitted on the fit flights by minimising the pooled median residual, bounded [0, 3] s^-1. Median (not mean) loss so mocap dropouts and transients cannot drag the fit. Recovered values are stable across folds (0.290-0.304 s^-1), which is what a real physical coefficient should look like.
- Diagnostic only (not the gate): body-frame drag with separate x and y coefficients (`R diag(k_x, k_y, 0) R^T v`), fitted per fold on the fit flights.
- Degenerate free-fall samples (`||a + g e3|| < 1`) are masked out instead of being normalised.

## Floor result (run 20261108T044153Z_f3b5d2bf)

- Gate, pooled held-out median residual with fitted drag: **1.24 deg** (p90 4.39, p95 6.49, n = 114,059 frames). Drag-free: 5.46 deg (p90 11.58, p95 13.87). Target was under 3-5 deg, so the gate passes.
- Per-flight held-out medians: 0.94 to 3.88 deg (IQR 2.06). Worst two are the piloted trackRATM flights (3.28 and 3.88 deg); best are the piloted ellipse/lemniscate (0.94/0.95) and the autonomous ellipse (1.19/1.26).
- Drag helps a lot where speed is high and steady (piloted ellipse 6.57 -> 0.94 deg, piloted lemniscate 6.67 -> 0.95), and almost not at all on the autonomous lemniscate (2.94 -> 2.94) where the residual is not drag-shaped.
- Residual still grows with speed after the drag fit: on the autonomous ellipse it goes 1.18 deg at 0-2 m/s to 11.5 deg at 20-22 m/s. High-speed (>12 m/s) p95 residual is ~17 deg. The drag term fixes the median, not the tail.
- Body-frame x/y drag does **not** lower the high-speed residual (p95 20 deg vs 16.8 deg for the scalar model, pooled median 1.22 vs 1.24), so the remaining high-speed error is not a body-vs-world drag-frame mismatch. Other candidates: unmodelled blade flapping / prop-wash at 20 m/s, thrust dynamics, and mocap noise on the rigid body.
- Frame-offset probe (best single rotation aligning implied to true axis, high-signal frames): 0.36-1.47 deg on six flights, 3.86 and 3.94 deg on the two autonomous ellipse flights. Not subtracted; a small constant attitude offset on those two flights would explain part of their residual.
- Testable fraction (implied tilt > 10 deg): 0.31-0.40 on the autonomous flights, 0.93-0.97 on the piloted flights. The low-speed lemniscate and piloted ellipse still carry signal; the autonomous ellipse spends ~60% of frames below 10 deg of implied tilt. Any clip-level score must report this fraction rather than silently averaging over it.
- Per-frame arrays (implied axis, true axis, residual under drag-free / fitted / body drag, speed, acceleration, both tilts) are written to `results/phase0_floor/<run_id>/frames/<flight>.npz` for reuse. They are gitignored (`results/**/*.npz`); `metrics.json` and the plots are committed.
- No Modal fallback was needed: the GitHub release `Range` reads worked from this machine. No Volume was created, so nothing to record beyond this line.

## ANWM check

- **CORRECTION (2026-11-08).** The first pass reported "no public repo, no weights, no data". That was wrong: the search used the paper title, and the repository is named `ANWM.code`, which no title search matches. The entries below the correction line are kept for the record but superseded.
- ANWM is public. Code: https://github.com/EmbodiedCity/ANWM.code, cloned to `third_party/anwm` (gitignored), pinned commit `657a80268505fa9149c4df502e35aa0f5bce11e5` (branch `master`, last commit 2026-08-20). Weights: https://huggingface.co/EmbodiedCity/ANWM, single file `0200000.pth.tar`, 18,146,420,481 bytes, card licence `apache-2.0`. Dataset: https://huggingface.co/datasets/EmbodiedCity/ANWM-Dataset, apache-2.0, 342 tar shards, 637.45 GB.
- No LICENSE file in the repo (verified: nothing in `git ls-files`, no licence text anywhere) and 13 files carry `Copyright (c) Meta Platforms, Inc.` headers inherited from NWM. Research use is fine; confirm with the authors before redistributing code or re-hosting weights.
- The release fixes metric step distance but **not** time: `timestamps` in `traj_data.pkl` are frame indices, no fps/dt appears in any config, `--input_fps 4` only converts "sec" to a frame index and is inherited from NWM, and AerialVLN applies actions by teleporting the pose (`simSetVehiclePose(..., ignore_collision=True)`), so no physical time exists in the data.
- dt resolution: the released frames are **exactly level** (roll = pitch = 0.0 in all frames of 33 sampled test trajectories), while the commanded per-step velocity changes imply 8.3 deg median tilt at 1 s/step. Implied tilt scales as 1/dt^2, so **0.25 s is excluded** and **1 s is the smallest duration consistent with the frames**; the attitude fit cannot separate 1 s from 2 s (degenerate objective). Working assumption: 1 s per action, tagged unverified, pending author confirmation.
- Excitation audit on 33 released test trajectories (first 39 of 342 shards indexed via HTTP Range; pose keys only, ~1 MB per trajectory instead of a 271-649 MB pickle): at dt = 1 s, 45.9% of frame transitions command more than 10 deg of tilt; SG-smoothed over an 11-frame window the median is 1.1 deg. ANWM clips are testable at transition events, not in sustained cruise.
- Dataset coverage: the release is **scene 16 only** (`airvln_16`, notebook filters `scene_id == 16`), not the paper's 40+ scenes. 1013 train / 254 test trajectories; `test/rollout_16.pkl` has 1000 eval entries over 29 trajectories.
- Starting a rollout from a TII frame is possible but needs one learned component: FFP requires per-frame depth + `K` + 4x4 pose, so depth must come from Pi-Long (as in `real/tools/fill_depth_pilong.py`); pose and `K` come from mocap and the TII calibration. No inference was run.

### Superseded (kept for the record)

- ANWM (arXiv:2512.21887v2) has **no public repo and no released weights or data**: no author link on the arXiv abstract or HTML (only the generic arXiv UI toggles), 0 hits on GitHub repository search for the title and variants, and nothing beyond arXiv mirrors and the author group's PDF in web search. `third_party/` is therefore not created and there is no commit to pin. Recorded in `docs/anwm_feasibility.md` with per-claim verified/unverified tags.
- Because the model is unavailable, no ANWM rollout was attempted and none is planned until weights appear. The alternatives named in the lit review are no better today: AeroAct (arXiv:2607.14997) has no code or weights link, and Aero-World has no arXiv id in the tracker so its status is unverified. No silent substitution.
- ANWM action time step is **not stated** in the paper. Inferred as 0.667 s per action from 32 s horizons over 48 frames, cross-checked against the reported 80.7 m average path (our random-action simulation gives 94 m at 0.667 s, 47 m at 0.333 s, 141 m at 1 s). The excitation audit (`experiments/anwm_excitation_audit.py`, `results/anwm_excitation/audit.json`) therefore reports dt = 1/3, 2/3 and 1 s rather than one number, because implied tilt scales as 1/dt.
- Excitation audit result at the most likely dt = 0.667 s: median step-implied tilt 34.6 deg, p90 49.2 deg, 69% of action boundaries above the 10 deg signal threshold; constant 5 m/s cruise needs only 8.6 deg with the fitted drag, so **ANWM's testable signal lives at action transitions, not in cruise**, and low-excitation segments must be reported as not testable rather than infeasible.
- ANWM outputs 224x224 at ~1.5 fps (inferred), one frame per action, 4-frame context; that frame rate is far below what a VO-based depicted-tilt estimate needs, which independently pushes ANWM clips toward the abstain bucket.

## Compute plumbing smoke tests

- R2 helper lives in `src/utils/r2.py` (boto3 against the R2 S3 endpoint, region `auto`), CLI in `scripts/sync_r2.py` exposed as `pixi run sync-r2`. Uploads are read back and re-hashed so a truncated object fails at the source. Bucket name comes from `.env` (`R2_BUCKET`); credentials are only ever reported as `set`/`MISSING`, never printed.
- Round trip verified: `smoke/r2_smoke.txt`, 42 bytes, sha256 `73485279...e9dae` on both sides, local diff after download identical.
- Modal smoke app lives in `remote/modal_smoke.py` with its own image (`debian_slim` + `numpy==2.1.3`), deliberately independent of `pixi.toml`. The only shared pin is numpy's major series, recorded here; the pixi env uses whatever conda-forge resolves.
- Modal Volume `awmv-smoke-vol` mounted at `/vol`; the smoke run wrote and read back `/vol/smoke.txt`. Auth came from the existing `~/.modal.toml`, not from `.env` (`MODAL_TOKEN_ID`/`SECRET` are unset there); a Modal Secret is the right route for functions that need the R2 keys.
- Smoke run result: GPU `Tesla T4` visible, `nvidia-smi` exit 0, volume round trip true, app completed in ~18 s. No real GPU workload was started.

## Track A, phase 1-2 (perception) — all entries tagged [A]

### Environment and plumbing

- [A] Branch `track-a-perception` created off `main` (Phase 0 tip). Track A owns `src/perception/`, `src/fusion/`, `configs/`, `experiments/phase1_*`/`phase2_*`, `remote/a_*`, `scripts/fetch_tii_video.py`, the `gpu` pixi feature, and `src/datasets/tii_video.py` (new file; `src/datasets/tii.py` untouched).
- [A] `gpu` pixi environment added as a `[feature.gpu]` + `[environments]` pair: `opencv` 5.0.0, `pytorch-cpu` 2.14.1, `torchvision`. CPU-only wheels on purpose — every real GPU job runs in a `remote/` Modal image, so the local env only has to import and test wrappers. The default environment's package set is unchanged (verified by re-running `pixi run test`: 22 passed) and Track B does not edit `pixi.toml`.
- [A] Modal stays the GPU route (auth via the existing `~/.modal.toml` profile `kyaw-thiha`; `modal app list` shows only the Phase 0 smoke app, now stopped). Apps are named `awmv-a-*`, Volume `awmv-a-data`, R2 prefix `track-a/`.
- [A] `scripts/pull_volume.py` exists because `modal volume get <vol> <dir> <existing dir>` fails with `IsADirectoryError` in this CLI version; it walks the Volume with the Python API instead. One-off tooling, documented in its docstring.

### TII video, sizes and storage

- [A] Video lives **inside** the release as a nested member: `<kind>/<flight>/camera_<flight>.zip` (a stored zip of JPEG frames, one per camera frame). Verified by ranged reads of the chunked release; per-flight compressed sizes: 01a 195.6 MB, 04a 197.4, 07a 275.7, 13a 457.4, 01p 1012.7, 07p 1035.4, 13p 394.9, 16p 202.2 MB. **Eight-flight total 3.77 GB** (plus 0.17 GB of per-frame CSVs/YAML), i.e. far below the 20 GB budget, so no time-trimming was needed and all eight Phase 0 flights are kept. Licence CC BY 4.0 (README; paper arXiv:2311.02667).
- [A] `scripts/fetch_tii_video.py` fetches the zips, `<flight>_cam_ts_sync.csv`, `csv_raw/camera_<flight>.csv`, `metadata_<flight>.yaml` and the four official calibration JSONs (pinned to repo commit `df4a2e11c46980d48afdc9aa76ef4faa673f5928`). Idempotent on size, CRCs checked by reading through the outer zip, sha256 recorded in `manifest.json`.
- [A] Local disk has 4.9 GB free, so the 3.77 GB of zips stay on Volume `awmv-a-data` (`/vol/tii_video`) and in R2 (`track-a/tii_video/`, sizes re-checked after upload); only the 0.17 GB of CSVs/YAML plus sampled frames are local. `remote/a_fetch_video.py` does the Volume+R2 copy in one Modal run.
- [A] Sampled frames for development are local only: `remote/a_frames.py` writes 48 evenly spaced frames per flight to `/vol/tii_frames` (pulled to `data/raw/tii_frames`, 28 MB) and (`extract_windows`, same app) 12 frames at 10 Hz inside per-flight 1.2 s solve windows into `/vol/tii_windows` (pulled to `data/raw/tii_windows`). Both are gitignored.

### Time synchronisation (Task 1.6)

- [A] **Video-to-mocap offset is zero by construction**: `csv_raw/camera_<flight>.csv` stamps every JPEG in Unix **microseconds on the same clock as `mocap_<flight>.csv`** (mocap starts 21 ms before the first frame in 01a, ranges overlap fully in all eight flights), and the loader verifies row-by-row that the authors' `<flight>_cam_ts_sync.csv` carries byte-identical timestamps and filenames. Offset uncertainty is therefore 0.5 frame = 4.2 ms; the exposure latency of the Arducam capture is undocumented and is **not** modelled (recorded in `Clip.meta["time_sync"]`). No cross-correlation fallback was needed; the DPVO-vs-mocap angular-rate cross-check in Phase 1b re-confirms it empirically.
- [A] Frame timing is not perfectly uniform: median dt 8.3 ms (120.4 Hz measured, 120 Hz declared in metadata) with dropped frames producing gaps up to 42 ms. The real timestamps are kept in `Clip.t`; `fps` is declared explicitly (120) and the measured value is recorded separately in `meta["measured_fps"]`.

### Calibration and camera model

- [A] The release's camera model is **`cv2.calibrateCamera` output** (k1,k2,p1,p2,k3; `scripts/camera_calibration.py` in the repo calls exactly that), **not** a Kannala-Brandt fisheye model — the paper-tracker note suggesting KB does not match the released files. Files are per camera group: `calib_ap-ellipse-lemniscate`, `calib_a-trackRATM`, `calib_p-trackRATM`.
- [A] Three properties of that calibration are measured and reported rather than assumed (`src/perception/undistort.py::model_diagnostics`): (i) `fx/fy = 0.749` in all three files, which no square-pixel sensor produces — the anisotropy is carried through, not corrected; (ii) the distortion polynomial is only invertible up to `r = 1.70` (`rd` peaks at 0.98, `theta` 56°), and **all four image corners fall outside that range**, so their rays are undefined; (iii) consequently a 100° virtual pinhole view uses only 46% of its pixels (`valid_fraction`), and the maximum source ray reached is 45.5° while the metadata claims a 155° horizontal field. The outer fisheye periphery therefore cannot be modelled with the released intrinsics; Track A works inside the valid domain and says so wherever the virtual FOV is reported.
- [A] Virtual pinhole camera: `fov_deg = 100` horizontal, output `640x480`, square pixels, same optical axis as the source (`configs/`-level values, defaults in `src/perception/undistort.py`). Undistortion is pure numpy (bilinear remap) so the CPU env can test it; `cv2.remap` is used when cv2 is importable.
- [A] Extrinsic convention resolved: `drone_to_camera.json`'s quaternion (w,x,y,z) is **camera->body**. Verified structurally — for the ellipse/trackRATM entries the matrix equals exactly the analytic hard-mount form `[[0, sin t, cos t], [-1, 0, 0], [0, -cos t, sin t]]` with `t` = the recorded uptilt (50°), whose columns are camera x = -body y, camera y = (sin,0,-cos), camera z = (cos,0,sin). Under that convention the mocap-implied optical-axis elevation reproduces the recorded uptilt for all four autonomous flights (01a 49.6 vs 50, 04a 49.8 vs 50, 07a 42.7 vs 40, 13a 50.0 vs 50). Both candidate directions are returned by the loader and the alternative is reported, never silently dropped.

### Camera-to-body rotation: what is verified and what is not

- [A] **Convention adopted:** `R_cb = camera_to_body_pitched(uptilt)` from the per-flight recorded uptilt (`src/perception/frames.py`), i.e. a hard-mounted camera with camera x = -body y, camera y = (sin,0,-cos), camera z (optical axis) = (cos,0,sin). For the four **autonomous** flights this reproduces `drone_to_camera.json` and puts the mocap-implied optical-axis elevation within 1-3 deg of the recorded uptilt (01a 49.6/50, 04a 49.8/50, 07a 42.7/40, 13a 50.0/50), from two artifacts that are independent of each other.
- [A] **Piloted flights do not reconcile, and this is reported rather than fitted away.** `drone_to_camera.json` has a single `piloted` key whose quaternion is a 60 deg mount; the per-flight metadata says 30 (01p, 07p) or 50 (13p, 16p). Under `R_cb(60 deg)` the mocap+extrinsic imply optical-axis elevations of 36.7 / 38.8 / 2.3 / -2.0 deg, and under `R_cb(metadata)` they imply 7.3 / 9.7 / -5.8 / -10.4 deg. No reading of the released files is consistent across the four piloted flights, and the near-zero values for 13p/16p contradict the frames themselves (those views are almost entirely ceiling, which requires a 35-50 deg uptilt). Most likely cause: the v3.0.0 pro-pilot airframes carry the mocap rigid body at an angle relative to the FCU frame, so the CSV body frame is not the camera mount's parent frame; a stale `piloted` entry is the alternative. Consequence: **Gate 1a is reported on the four autonomous flights as the truth-bearing set**, piloted flights are reported with the caveat and with/without a constant offset, and the resolution path is a per-flight PnP against the released gate-corner labels (`gate_corners_<flight>.csv` + `label_flight-<flight>.zip`), which is left as the next step rather than guessed.
- [A] A mocap-supervised triangulation check (`experiments/phase1_camera_solve.py`) was written to validate intrinsics + extrinsics from pixels alone, but its residual is 12-1000 px even at the best rotation, i.e. the released lens model is not good enough for multi-view geometry over a 6 m baseline at these view angles. It is kept in the repo as a diagnostic, its numbers are **not** used as evidence for any convention, and the negative result is part of why the gate-corner PnP route is named as the resolution path.

### GeoCalib configuration (Gate 1a)

- [A] GeoCalib pinned at commit `1dedb29f94fc11b896c5f572005978aeebccd67f`; modal app `awmv-a-geocalib`, A10G, image `debian_slim` + torch 2.4.1 + `opencv-python-headless` + `libgl1` (GeoCalib's own deps pull non-headless opencv, which needs libGL). Weights are fetched by `GeoCalib(weights=...)` at run time, not baked into the image.
- [A] Four input paths were measured on 20 frames per flight before choosing one, because the choice is not cosmetic: pinhole on the 100 deg undistorted view with a focal prior gives **26.9 deg** pooled median flight RMSE; pinhole without a focal prior 25.4; GeoCalib's `simple_divisional` model on the **raw fisheye** gives **13.2**. The undistorted view is also only 46 % valid pixels, so the raw-fisheye route is adopted as primary and the undistorted route is kept as the comparison. Consequence: the depicted side no longer depends on the released calibration at all, and its output is in the source RDF camera frame, the same frame as the mocap truth.
- [A] The focal prior must be a **scalar**: a length-2 tensor makes `camera.from_dict` stack a `[2]` against `[1]` and raises. Recorded because the traceback points inside GeoCalib, not at the caller.
- [A] GeoCalib's gravity convention is confirmed from its own source: `Gravity.from_rp(0, 0) = (0, -1, 0)`, i.e. the vector points **up** in camera coordinates, and `pitch = asin(z)`. That matches `src/perception/frames.py::up_in_camera`, so the comparison needs no sign flip. A synthetic-rotation probe (`--probe-flights`) re-checks it on real frames by tilting the virtual camera by a known angle.
- [A] GeoCalib runs at about **5 frames/s at 640x480 on an A10G** with 0.43 GB peak GPU memory (LM optimization dominates, not the network). Budgeted at ~US$1 for the full eight-flight sweep, below the US$5 threshold.
- [A] Error on the frames that fail is not a small perturbation: per-flight RMSE (19-48 deg) is far above the median (2.6-7.8 deg), so the distribution is heavy-tailed and the median is the honest central number while the tail needs its own diagnosis (nearly-ceiling views, moments when a gate fills the frame).

### Throughput and job shape

- [A] The first full-flight GeoCalib sweep preloaded every decoded frame into a list before inference; with the 1 GB piloted zips that is ~2 GB of RGB arrays and throughput fell from 5 to **0.57 frames/s**. The app now streams frames one at a time and commits the Volume after each flight, so a long sweep keeps its completed flights. Measured again after the fix: see `docs/track_a_report.md` section 2.
- [A] Reported estimator error is heavy-tailed (median 3.5 deg, p99 81 deg, 3.4 % of frames over 20 deg), so every table reports median, p90, p99, trimmed RMSE and the fraction over 20 deg alongside the plain RMSE. A single RMSE would hide the failure mode that matters for Phase 3.
- [A] DPVO build route on Modal: `nvidia/cuda:12.4.1-devel-ubuntu22.04`, python 3.10, torch 2.3.0 + torchvision 0.18.0 (cu121), `torch-scatter` from the pyg **find-links** page (`-f`, not `--index-url`, which returns "no versions"), `clang` installed because torch's `cpp_extension` shells out to `which clang++`, repo pinned at `0ac95b656d1fda91c271d2a106460d19ad966fc7`, and `pip install --no-build-isolation .`. Weights are downloaded once into Volume `awmv-a-data/dpvo_models`.
- [A] Fusion frame bookkeeping: the complementary filter corrects toward world up, so it must run in the **gravity-aligned** frame. `relative_rotations` therefore take `R_align @ R_vo`, not the raw VO rotations; using raw VO rotations with an `R_align` applied only to the output silently corrects toward the wrong axis. Fixed in `experiments/phase1b_fusion.py` before any Gate 1 number was produced (no result in this repo predates the fix).

### Gate outcomes (recorded, not reinterpreted later)

- [A] **Gate 1a (depicted tilt, RMSE under 5 deg) FAILED**: pooled median error 3.5 deg, trimmed RMSE 3.66 deg, but full RMSE 12.5 deg on 1,033 frames of 01a+04a because 3.4 % of frames exceed 20 deg error (p99 81 deg). Reported with the tail statistics rather than as a single RMSE. The median therefore meets the target and the gate does not.
- [A] **Gates 1 and 2 are BLOCKED on visual odometry**, not on the verifier: six DPVO build attempts on Modal fixed five distinct packaging problems (torch/torchvision pin, torch-scatter find-links, `clang++` for `cpp_extension`, `TORCH_CUDA_ARCH_LIST` for CPU-only image builders) and the last one is unpinned to a result. The handoff's own routing applies: an interactive RunPod pod for the CUDA compile, or the VGGT/MegaSaM fallback that needs no compile.
- [A] No threshold, gate or claim was changed to make a result fit. The constant-offset columns in every table are reported and never applied to the number that is compared with a target.
