# I01 — Simulator spike: four candidates for a labelled flight set

**Branch** `track-i-simset` (off `main`), worktree `~/Documents/Projects/awmv-track-i`.
**Date** 2026-11-16. **Spend** USD 2.1 (estimated from container wall time) of the USD 40
ceiling (`docs/COST_LEDGER.md`).

## Summary

The track needs a simulator that can draw a **known** camera pose sequence in textured 3D:
300 flights, 10-30 s each, 30 fps, with truth at 200 Hz or more and the infeasible flights
infeasible *by construction*. Every candidate was therefore judged on rendering, not on
flight dynamics — the dynamics are ours and already unit-tested (`src/sim/flight.py`).

* **Recommended: Blender + our own dynamics**, with **EEVEE Next** for the full set. It renders
  headless on a rented L40S with no RT-core requirement, is scriptable from Python, and its
  pose→pixel model is verified against Blender's own camera to 0.5 px. Throughput at 640×360:
  **0.19 s/frame** with EEVEE (5.4 fps) and 0.83 s/frame with Cycles at 8 spp with denoise,
  which puts the 180,000-frame set at **USD ~18** or USD ~81 respectively. Licence GPL-2.0 for
  the program (rendered images are not derived works of the renderer); scene and flight are
  both seed-reproducible.
* **Isaac Sim** is the photorealistic option. It builds and **starts headless** on an L40S
  (24 s to "Startup Complete", Warp on `cuda:0`), but the spike did not get a frame out of it:
  every attempt ended on a script-level API mismatch in *our* driver — the 6.2 camera and
  replicator names are not the 4.x ones the tutorials use — and each iteration costs a
  container start. It remains the fallback, not the default.
* **Flightmare** is dead weight in 2026: its dependency set is Ubuntu 18.04 + ROS Melodic
  (`libzmqpp-dev` does not exist on 22.04), and its renderer is a 936 MB Unity binary from
  2020-12 that has to be driven over ZMQ.
* **AirSim/Colosseum**: Colosseum is archived; AirSim is not but its last release is
  2022-07 and needs Unreal Engine (Epic account, ~100 GB build). Recorded, not attempted
  beyond the evidence.

The end-to-end check on the rendered clips — the project's own tilt reader on Blender frames
against exact truth, and its gain statistic on the feasible, tilt-removed and mirrored-roll
renders — is in `## The end-to-end check`. The reader's residual is 0.6 deg on the feasible
clip against 24.7 deg on the tilt-removed one, and its gain reads 0.93/0.99, -0.01/-0.004 and
0.91/**-1.00** on the three clips.

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
| 2 | Blender throughput across settings on a rented GPU (Cycles OptiX vs EEVEE, with and without denoise) | Modal L40S | `results/i01_spike/l40s_benchmark.json` |
| 2b | the same clip rendered end to end with EEVEE and scored | Modal L40S + A10 | `results/i01_spike/eevee_91frames_timing.json`, `geocalib/i01_feasible_eevee_*.json` |
| 3 | pose→pixel model checked against Blender's own `world_to_camera_view` | local | `results/i01_spike/projection_check.json` |
| 4 | flight-wide geometric check: bare-ground render, horizon against the pose | local | `results/i01_spike/horizon_bare_flight.json` |
| 5 | GeoCalib on the rendered clips against exact truth | Modal A10 | `results/i01_spike/geocalib/` |
| 6 | Flightmare build + Unity renderer probe | Modal (CPU) | `results/i01_spike/flightmare_probe.json` (facts + log tails) |
| 7 | Isaac Sim headless probe: image build, app start, API discovery, render attempts | Modal L40S | `results/i01_spike/isaac_probe.json` (log) |

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
  end-of-life. The probe was re-run with `zmqpp` built from source, then stopped during the
  936 MB Unity download once the picture was clear; the `flightlib` build itself was not
  carried to a verdict, so the rejection rests on the missing package, the six-year-old
  renderer and the display requirement, not on a compiler error.
* Pose rate: the C++ library integrates its own dynamics; it does not expose an exact
  commanded attitude, so the 200 Hz requirement would be met by *generating* truth elsewhere
  anyway.
* Headless: the Unity player needs a display; the same ZMQ bridge that makes it scriptable
  also makes it a display-dependent process. Whether the 2020 player starts under Xvfb on a
  cloud GPU is **unverified** — it was not pursued past the build stage, and the report says
  so rather than guessing.

### B. Isaac Sim (NVIDIA) — starts headless, rendering not completed, fallback

* Licence: Apache 2.0 for the repository, with additional NVIDIA component licences for the
  Omniverse/KIT runtime; the EULA has to be accepted (`ACCEPT_EULA`), and the pip package is
  `isaacsim[all]` from `https://pypi.nvidia.com` (the PyPI wheels are stubs).
* Hardware: the upstream README lists **A40 minimum, L40S recommended** for datacenter GPUs,
  i.e. RTX-class ray tracing, which Modal's L40S satisfies.
* **Established:** the image builds in **10-13 minutes** and the app reaches "Simulation App
  Startup Complete" in **24 s** on an L40S (44 GiB), with Warp enumerating `cuda:0`. Two
  missing system libraries had to be added (`libXt.so.6`, `libxkbcommon0`).
* **Not established: that it renders our pose sequence.** Four probe iterations ended in our
  own driver code, each on a different 6.2-surface difference — `UsdGeom.MaterialBindingAPI`
  does not exist (it is `UsdShade`), and the 4.x camera/replicator names
  (`rep.create.render_product`, `rep.AnnotatorRegistry`, `isaacsim.sensors.camera.Camera.get_rgba`)
  are 6.x's `rep.functional.create.*` / `CameraSensor.get_data` — plus one iteration that did
  not return before the client timeout. The report therefore does not claim an Isaac Sim
  throughput number, and the rejection rests on effort, not on speed.
* Pose/rate: rendering is decoupled from physics, so in principle any pose sequence can be
  drawn; Isaac Sim's *physics* is unused here whatever happens.

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
* Throughput, measured on the L40S in six-frame steady-state runs
  (`results/i01_spike/l40s_benchmark.json`), all at 640×360:

  | engine | samples | denoise | seconds/frame | fps |
  | --- | --- | --- | --- | --- |
  | Cycles (OptiX) | 8 | yes | 0.83 | 1.2 |
  | Cycles (OptiX) | 8 | no | 0.52 | 1.9 |
  | Cycles (OptiX) | 48 | no | 0.56 | 1.8 |
  | EEVEE Next (OpenGL, L40S renderer string) | 16 | no | **0.19** | **5.4** |

  A 91-frame EEVEE render of the clip took 30 s end to end (0.145 s/frame plus ~17 s of
  scene build and start-up), i.e. 3.0 fps for a whole clip rather than 5.4.

  Three facts matter more than the table. **Cycles' cost is flat in sample count** (8 to 48
  spp changes the frame time by under 10 %), so at this resolution it is dominated by
  per-frame overhead — scene sync, BVH build, kernel launch and the denoiser (~0.3 s of the
  0.83 s), not by shading; on the 4 GB laptop GPU the same comparison does scale with samples
  (1.0 s at 8 spp, 4.2 s at 48 spp). **EEVEE runs on the GPU headless** — the GL renderer
  string is `NVIDIA L40S/PCIe/SSE2`, not a software rasteriser — and is 3-5× cheaper per
  frame. And **Cycles' device selection was flaky across containers**: one benchmark run came
  back with `cycles_backend = NONE` and no devices enabled, i.e. silently on the CPU, at about
  half the speed. Any production loop has to assert the device, not request it.
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
3. **The verdict statistics read early**, on three clips of the same flight (feasible,
   tilt removed, mirrored roll). Upper block: the depicted-lean gain (slope of depicted on
   required, per axis) and its correlation, which is Track D's frozen R2 statistic. Lower
   block: the two residual statistics the classic verdict uses. Every number 91 frames.

   | clip | gain, forward | gain, left | correlation, forward / left |
   | --- | --- | --- | --- |
   | feasible | 0.93 | 0.99 | 0.996 / 0.99998 |
   | tilt removed | -0.01 | -0.004 | 0.43 / -0.26 |
   | mirrored roll | 0.91 | **-1.00** | 0.996 / **-0.99992** |

   | clip | depicted tilt (median) | required tilt | median abs tilt residual | lean-direction residual |
   | --- | --- | --- | --- | --- |
   | feasible | 24.77 deg | 25.33 deg | **0.62 deg** | 1.44 deg |
   | tilt removed | 0.36 deg | 25.33 deg | **24.72 deg** | 124.5 deg |
   | mirrored roll | 24.41 deg | 25.33 deg | **0.79 deg** | 125.7 deg |

   Two things follow. The pipeline, not just the renderer, survives the move to simulation:
   the gain reads ~1 on the feasible clip, ~0 on tilt removal, and ~-1 on the roll axis for
   mirrored roll — the exact pattern the real flights produced (Track D: 100% detection on
   both families with the gain test, 67.2% and 94.8% with the median-residual baseline). And
   the *baseline* residual statistic is nearly blind to mirrored roll here (0.79 deg, below
   any threshold the real flights would set), which is why the gain test exists. That
   agreement between a real-data finding and a constructed one is the strongest single reason
   to build the set. It is still one flight per defect: a demonstration, not a rate.

## Detection floor implied by the reader's error

Handoff 15 Step 3 asks for the detection floor in simulation against the real-flight floor
(14-17 deg). The spike can state the first term of it but not the floor itself:

* the reader's median error on the rendered feasible clip is **0.62 deg of tilt** (p90 1.11)
  and **1.44 deg of lean direction** (p90 3.02);
* the tilt-removed clip, on the same path, produces a depicted tilt of 0.36 deg where the
  path requires 25.33 deg, i.e. a median residual of **24.7 deg** (p90 31.7) under the same
  statistic the real-flight verdicts use.

So a full-flight tilt removal of 25 deg sits a factor of ~40 above the reader's error here,
against the real-flight floor of 14-17 deg where the reader's own error (~4 deg per frame) and
the estimator's view dependence set the distance. The number that matters for Step 3 is not
this ratio but the *remainder after the reader's error is measured on frames from the final
scene library*: one scene, one flight and no motion blur is not a floor.

## Cost model for the labelled set

```
frames_total   = 300 flights x 20 s x 30 fps          = 180,000 frames
frames/hour    = 3600 x fps
GPU-hours      = frames_total / frames/hour
USD            = GPU-hours x price_per_hour
```

| engine (L40S, 640×360) | fps | GPU-hours for 180,000 frames | USD at 1.95/h |
| --- | --- | --- | --- |
| Cycles 8 spp + denoise | 1.2 | 41.7 | **81** |
| Cycles 8 spp, no denoise | 1.9 | 26.3 | 51 |
| EEVEE Next 16 spp (steady state) | 5.4 | 9.3 | **18** |
| EEVEE Next (end-to-end clip rate, 3.0 fps) | 3.0 | 16.7 | 33 |

So the set as specified (300 flights × 20 s × 30 fps = 180,000 frames) is **inside the USD 40
ceiling with EEVEE and outside it with Cycles**, at 640×360. At 10 s flights (90,000 frames)
the same numbers halve: EEVEE USD 9, Cycles USD 40.

Three qualifications, because a cost number built on one scene is a floor:

* the Cycles number is dominated by per-frame overhead at this resolution. Rendering each clip
  as an **animation** (one `bpy` render call, camera driven by keyframes) would remove most of
  that overhead and is not measured here — Step 2 should measure it before choosing a setting;
* if Cycles silently falls back to CPU again, its cost roughly doubles, so the loop must check
  `cycles_backend` per job (this run's `device_info` shows `OPTIX` and two L40S devices);
* the cost model's frame count is the handoff's, not a validated choice. Scene complexity,
  resolution and the number of scenes all move it.

The reader's error does not care which engine drew the frame: on the same 91 frames,
GeoCalib reads 0.75 deg median up-vector error and 0.37 deg median tilt residual on the EEVEE
frames, against 1.14 and 0.62 on the Cycles frames, with the same gain (0.94/1.00 against
0.93/0.99). EEVEE is therefore not a compromise on the measurement, at least on this scene.

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
* The throughput numbers are single six-frame runs, not a distribution: the per-frame times
  vary by tens of percent across containers (one Cycles run came back CPU-only), so treat the
  cost model as an order-of-magnitude planning number and re-measure in Step 2.

## Caveats

* The L40S throughput numbers are six-frame runs on a shared cloud GPU with
  `max_bounces = 4`; frame times vary by tens of percent with scene content, with the number
  of bounces and with whichever GPU the scheduler hands out.
* The mount in the spike is 0 deg uptilt. TII flies 30-50 deg, so the real set should carry
  both, and the mount enters the depicted-lean conversion (`R_cb`).
* The spike's `k_d` is the project's frozen 0.295 /s; the simulated flights inherit the
  project's own drag model, so agreement with the real flights is by construction and is not
  evidence about the drag model.
* Everything in this report is reproducible from the committed scripts; the frames themselves
  are on the Modal volume `awmv-i-data` (`frames/i01_feasible`, `frames/i01_level`,
  `frames/i01_mirror`, `frames/i01_feasible_eevee`) and the GT arrays are regenerable with
  `experiments/i01_spike_flight.py`.

## Recommendation

**Blender + `src/sim/flight.py`, EEVEE Next for the bulk pass, Cycles for a validation
subset**, with the scene library extended (2-3 scene families, an HDRI, more surface texture)
and the output written to R2 with a manifest. Isaac Sim stays the fallback if the tilt
reader's error on Blender frames turns out to be too large to be useful — which the numbers
above settle before any dataset is built.

Concretely, for Step 2's 300 flights × 20 s × 30 fps at 640×360: an EEVEE pass over the whole
set is about 9 GPU-hours (USD ~18 at the L40S list price, USD ~33 if the end-to-end clip rate
is used instead), so the set fits inside the USD 40 ceiling with room for a Cycles re-render of
a validation slice. Two things to measure *before* committing the frame budget: the
animation-mode render rate (which should remove most of Cycles' per-frame overhead) and the
reader's error on the second and third scene family.

**Approval needed before Step 2** (the labelled set), per handoff 15.
