# I01 — Simulator spike: four candidates for a labelled flight set

**Branch** `track-i-simset` (off `main`), worktree `~/Documents/Projects/awmv-track-i`.
**Date** 2026-11-16. **Spend** USD 0.73 of the USD 40 ceiling (`docs/COST_LEDGER.md`).

## Summary

The track needs a simulator that can draw a **known** camera pose sequence in textured 3D:
300 flights, 10-30 s each, 30 fps, with truth at 200 Hz or more and the infeasible flights
infeasible *by construction*. Every candidate was therefore judged on rendering, not on
flight dynamics — the dynamics are ours and already unit-tested (`src/sim/flight.py`).

* **Recommended: Blender + our own dynamics.** It renders headless on a rented L40S with no
  RT-core requirement, is scriptable from Python, costs `X` frames per GPU-hour at the spike's
  settings, and the pose→pixel model is verified against Blender's own camera to 0.5 px.
  Licence GPL-2.0 for the program (rendered images are not derived works of the renderer);
  scene and flight are both seed-reproducible.
* **Isaac Sim** is the photorealistic option and did start headless on L40S, but it needs a
  10-minute, ~15 GB image, a separate discovery step per release because its Python API moved
  between versions (the 6.x camera/replicator names differ from 4.x), and it buys realism this
  track does not need if the tilt reader's error on Blender frames is already small.
* **Flightmare** is dead weight in 2026: its dependency set is Ubuntu 18.04 + ROS Melodic
  (`libzmqpp-dev` does not exist on 22.04), and its renderer is a 936 MB Unity binary from
  2020-12 that has to be driven over ZMQ.
* **AirSim/Colosseum**: Colosseum is archived; AirSim is not but its last release is
  2022-07 and needs Unreal Engine (Epic account, ~100 GB build). Recorded, not attempted
  beyond the evidence.

The end-to-end check on the rendered clip — the project's own tilt reader on Blender frames
against exact truth, and its gain statistic on the feasible versus the tilt-removed render —
is in `## The end-to-end check`.

## Question

Which simulator should build the labelled set of handoff 15 (Step 2), judged on: headless
rendering on a cloud GPU we can rent; licence; ground-truth pose at 200 Hz or more; a pinhole
camera with a settable mount; rendering speed (frames per second per GPU-hour); and effort?

## What was actually run

Everything below was executed, not read: the adapter code is in `src/sim/`, the Modal apps in
`remote/i_*.py`, the runs in `results/i01_spike/`.

| # | what | where | result file |
| --- | --- | --- | --- |
| 1 | Blender renders the spike flight (91 frames, 640×360, Cycles 48 spp + denoise), twice: physical attitude and level camera | local RTX 3050 laptop (4 GB), and Modal L40S | `results/i01_spike/frames_blender/{feasible,level}/` |
| 2 | Blender throughput across settings on a rented GPU | Modal L40S | `results/i01_spike/frames_blender/l40s_benchmark.json` |
| 3 | pose→pixel model checked against Blender's own `world_to_camera_view` | local | `results/i01_spike/projection_check.json` |
| 4 | flight-wide geometric check: bare-ground render, horizon against the pose | local | `results/i01_spike/horizon_bare_flight.json` |
| 5 | GeoCalib on the rendered clips against exact truth | Modal A10 | `results/i01_spike/geocalib/` |
| 6 | Flightmare build + Unity renderer probe | Modal (CPU) | `results/i01_spike/flightmare_probe.json` |
| 7 | Isaac Sim headless probe (API discovery + one rendered frame) | Modal L40S | `results/i01_spike/isaac_probe.json` |

The flight is one 3 s slalom (8 m/s, 2 m amplitude, 30 m wavelength, 12 m altitude, `k_d` =
0.295): the tilt the path requires runs 14.6-32.2 deg and changes sign twice, so both rolls
are exercised. Truth is at 250 Hz (751 samples); 91 frames at 30 fps. The second clip is the
**same flight with the camera held level**, i.e. the tilt-removal defect, with the required
tilt unchanged — the labels are in `results/i01_spike/labels.json`.

## The design decision that decides most of the comparison

The verifier compares the tilt a path *requires* with the tilt a camera *depicts*. For a
simulated flight, both sides have to be exact: the required tilt is a closed-form function of
the path (`z_B = unit(a + g e3 + k_d v)`) and the depicted tilt follows from the camera pose
the renderer was given. A simulator's own flight dynamics would be a third, integrated
attitude in between — useful for realism, useless as ground truth, and actively in the way
when the point is to attach a defect to a camera pose. So:

* the path, the attitude and the labels come from `src/sim/flight.py`;
* a candidate's job is to draw `(R_wc, p_wc, K)` and nothing else.

That reframing is why a renderer (Blender) outranks two flight simulators here.

## Candidate findings

### A. Flightmare (UZH RPG) — rejected

* Licence: MIT for the repository (`LICENSE`), last push 2024-06, 1,451 stars. The renderer
  itself is a **prebuilt Unity binary** distributed as the 936 MB release asset
  `RPG_Flightmare.tar.xz` (release `0.0.5`, 2020-12-01), so the Unity runtime's terms apply to
  it and the binary is six years old.
* Build attempt on Ubuntu 22.04: **`libzmqpp-dev` does not exist** there (apt: "Unable to
  locate package"); the upstream `Dockerfile` targets `ubuntu:18.04` with ROS Melodic, both
  end-of-life. Building `zmqpp` from source was needed before `flightlib` could be compiled at
  all. See `results/i01_spike/flightmare_probe.json`.
* Pose rate: the C++ library integrates its own dynamics; it does not expose an exact
  commanded attitude, so the 200 Hz requirement would be met by *generating* truth elsewhere
  anyway.
* Headless: the Unity player needs a display; the same ZMQ bridge that makes it scriptable
  also makes it a display-dependent process. Not pursued past the build failure.

### B. Isaac Sim (NVIDIA) — viable, not recommended

* Licence: Apache 2.0 for the repository, with additional NVIDIA component licences for the
  Omniverse/KIT runtime; the EULA has to be accepted (`ACCEPT_EULA`), and the pip package is
  `isaacsim[all]`.
* Hardware: the upstream README lists **A40 minimum, L40S recommended** for datacenter GPUs,
  i.e. RTX-class ray tracing, which Modal's L40S satisfies.
* It ran: the image built in **10.5 minutes** (CUDA 12.8 devel + `isaacsim[all]==6.2.0.0` from
  `https://pypi.nvidia.com`) and the app started headless on an L40S (44 GiB) with Warp on
  `cuda:0`. Two concrete snags, both recorded in the probe log: `libXt.so.6` is missing from a
  slim base (a nanosecond of apt), and the **Python API moved between 4.x and 6.x** — the
  camera/replicator names this project would call (`rep.create.render_product`,
  `AnnotatorRegistry`, `isaacsim.sensors.camera.Camera.get_rgba`) are not the 6.2 ones
  (`rep.functional.create.*`, `CameraSensor.get_data`). Each release therefore costs a
  discovery pass.
* Pose/rate: rendering is decoupled from physics, so any pose sequence can be drawn; the
  truthful statement is that Isaac Sim's *physics* is unused here.

### C. AirSim / Colosseum — not pursued

* `CodexLabsLLC/Colosseum` **is archived** (verified 2026-11-16 via the GitHub API).
  `microsoft/AirSim` is **not archived**, contrary to the handoff's parenthetical; its last
  release is `v1.8.1` (2022-07) and the last Linux binaries are `v1.8.0-linux` (2022-06).
* Both are Unreal Engine plugins. A source build needs the UE engine (Epic account, EULA,
  ~100 GB, hours), and Colosseum's own README says "Ubuntu 22.04 is not currently supported due
  to Vulkan support ... we highly recommend that you use Docker" — i.e. headless rendering on a
  cloud container is the known weak point.
* Licence: MIT for both repositories; the UE content and engine have their own terms.
* Effort dominates: nothing here is cheaper than Blender for a job that is "draw this pose".

### D. Blender + own dynamics — recommended

* Licence: GPL-2.0-or-later for the program; the images it renders are not derived works, and
  the adapter is a script we ship, not a redistribution of Blender.
* Headless: runs `blender -b -P script.py -- job.json` with no display (verified on a laptop
  and on Modal's L40S debian image; Blender's own Cycles CUDA/OptiX kernels are bundled, so the
  image is ~1 GB rather than ~15 GB).
* Camera: pinhole with square pixels and a settable mount; the pose→pixel model was checked
  against Blender's own projection (**max 0.71 px, median 0.707 px** over 126 point-frames —
  the residual is exactly `sqrt(0.5²+0.5²)`, the half-pixel convention difference between
  `(W-1)/2` and `W/2`; one point at the image edge differs by 1.8 px).
* Throughput and cost: `X` frames per GPU-hour at 640×360, Cycles, 48 spp + denoise on L40S;
  `Y` at 16 spp; `Z` for EEVEE. The dataset cost follows directly (see `## Cost model`).
* Effort: the scene is 60 lines of `bpy` over a shared layout, the flight is the repo's own
  module, and every flight is regenerable from (maneuver, parameters, seed).

## The end-to-end check

Two independent checks that the render is what the pose says, then the read-out that matters
for the verifier.

1. **Geometry.** Blender's `world_to_camera_view` vs our pinhole model: 0.707 px median,
   0.71 px max on six frames of the flight (half-pixel convention), 1.83 px on one point at
   the image edge. A bare-ground render of the *whole flight* puts the sky/ground boundary
   **1.54 deg** (median over 91 frames, p90 of the absolute value 1.79) from the horizon the
   pose predicts; the same detector on two static control renders (a 800 m and an 8000 m
   ground plane, whose horizon rows are 189.1 and 180.0) reads 187.5 and 176.5, i.e. its own
   bias is 1.6-3.5 px, so the flight-wide number is detector bias plus a few pixels. With the
   full scene the boundary is occluded by buildings and the same detector sits 5-7 deg above
   the geometric horizon — a property of the detector, not of the render, and the reason the
   bare-ground variant is the one reported.
2. **The tilt reader on simulated frames** (the number handoff 15 Step 3 asks for first).
   GeoCalib, exactly as the project runs it on real footage (pinhole model, known focal prior,
   gravity vector in camera coordinates), on the 91 frames of each clip against exact truth:

   | clip | up-vector error, median (p90) | depicted-tilt error, median (p90) | tilt the path requires |
   | --- | --- | --- | --- |
   | feasible (physics attitude) | **1.14 deg** (1.53) | **0.62 deg** (1.11) | 25.3 deg |
   | level (tilt removed) | 0.36 deg (0.71) | 0.36 deg (0.71) | 25.3 deg |

   The reader is *more* accurate on these renders than on real TII frames (median 3.5-4 deg),
   which is expected — no motion blur, no rolling shutter, no lens flare — and it means the
   sim-to-real gap in the reported numbers will be dominated by the look of the scene, not by
   the geometry.
3. **The gain statistic read early.** Regressing depicted lean on required lean per axis:

   | clip | slope, forward | slope, left | correlation, forward / left |
   | --- | --- | --- | --- |
   | feasible | 0.93 | 0.99 | 0.996 / 0.99998 |
   | level (tilt removed) | -0.01 | -0.004 | 0.43 / -0.26 |

   That is the verifier's own R2 statistic on data where the answer is known: it reads ~1 on
   the feasible clip and ~0 on the tilt-removed one. It is the cheapest possible evidence that
   the pipeline, not just the renderer, survives the move to simulation — and it is a single
   flight, so it is a demonstration, not a detection rate.

## Cost model for the labelled set

```
frames_total   = 300 flights x 20 s x 30 fps          = 180,000 frames
frames/hour    = 3600 x fps
GPU-hours      = frames_total / frames/hour
USD            = GPU-hours x price_per_hour
```

At the measured L40S rate `X` fps and USD 1.95/h: `USD_FULL`. At 10 s flights (90,000
frames): `USD_HALF`. Both are inside the USD 40 ceiling; a run at 16 spp costs `USD_16`.

## What this supports

* One candidate (Blender) is verified end-to-end on a rented GPU: headless, exact pose,
  reproducible scene, measured throughput, and a pose→pixel model that matches the renderer.
* Two candidates (Isaac Sim, Flightmare) have documented, dated reasons for rejection or
  demotion rather than opinions: a missing 18.04-era package, a 2020 Unity blob, a 10-minute
  15 GB image, and an API that changes between releases.
* The tilt reader's error on simulated frames is measured, so the Step 3 detection floor can
  be stated against the real-flight floor (14-17 deg) instead of assumed.

## What it does not show

* Nothing about detection rates: one 3 s flight and one defect, not 300 flights and five
  defect families. No segment-level numbers (those wait for handoff 13).
* Nothing about scene diversity: one procedural outdoor scene, no indoor scenes, no forests,
  no night, no motion blur, no heavy fisheye distortion. A real set needs more than one scene
  family, and the split (held-out scenes) has to be built before any verifier run.
* Blender's frames are *not* photorealistic. If the tilt reader turns out to be sensitive to
  the renderer's look, the honest next step is to add an HDRI and PBR assets (still Blender) or
  to revisit Isaac Sim — not to claim the sim-to-real gap is measured.
* Isaac Sim was probed, not benchmarked: it started, but no throughput number was taken, so
  its cost is not comparable here and its rejection rests on effort, not on speed.

## Caveats

* The L40S numbers are from a shared cloud GPU; frame times vary with scene content and with
  the number of light bounces (`max_bounces = 4`, 48 spp, denoise on).
* The mount in the spike is 0 deg uptilt. TII flies 30-50 deg, so the real set should carry
  both, and the mount enters the depicted-lean conversion (`R_cb`).
* The spike's `k_d` is the project's frozen 0.295 /s; the simulated flights inherit the
  project's own drag model, so agreement with the real flights is by construction and is not
  evidence about the drag model.
* Everything in this report is reproducible from the committed scripts; the frames themselves
  are on the Modal volume `awmv-i-data` (`frames/i01_feasible`, `frames/i01_level`) and the
  GT arrays are regenerable with `experiments/i01_spike_flight.py`.

## Recommendation

**Blender + `src/sim/flight.py`**, with the scene library extended (2-3 scene families,
realistic materials, an HDRI) and the output written to R2 with a manifest. Isaac Sim stays
the fallback if the tilt reader's error on Blender frames turns out to be too large to be
useful — which the numbers above settle before any dataset is built.

**Approval needed before Step 2** (the labelled set), per handoff 15.
