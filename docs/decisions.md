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
