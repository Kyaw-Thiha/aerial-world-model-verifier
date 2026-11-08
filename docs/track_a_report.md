# Track A report: perception (Phase 1 depicted tilt, Phase 2 implied tilt)

Status: in progress. Gate numbers are filled in from `results/*/metrics.json`; anything
marked **blocked** or **not run** is not a pass.

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

## 2. Compute

| tool | GPU | s per 1000 frames | peak GPU | notes |
| --- | --- | --- | --- | --- |
| GeoCalib | TBD | TBD | TBD | pinhole model on the undistorted view, focal prior |
| DPVO | TBD | TBD | TBD | built from source, commit pinned |

## 3. Gate 1a: depicted tilt per frame (GeoCalib)

TBD

## 4. Gate 1: depicted attitude over time (DPVO + gravity fusion)

TBD

## 5. Gate 2: implied tilt from the pixel trajectory

TBD

## 6. Oracle ladder (attribution)

TBD

## 7. What did not work, and why

- `experiments/phase1_camera_solve.py` (mocap-supervised triangulation over a 6 m baseline)
  leaves 12-1000 px reprojection residual at the best rotation, i.e. the released lens model
  is not good enough for multi-view geometry. Kept as a diagnostic; **not** used as evidence.
- The piloted flights' camera frame, see above.

## 8. Next steps that would move the numbers

1. Gate-corner PnP per piloted flight (`gate_corners_<flight>.csv` + `label_flight-<flight>.zip`)
   to settle the mount, which would let the piloted flights (19-21 m/s) into the gate.
2. A fisheye model fitted to the released calibration plus the stated 175 deg field, or a
   line-based self-calibration, to recover the periphery.
3. MoGe-2 metric scale to replace the fitted scale in the verifier's own operating point.
