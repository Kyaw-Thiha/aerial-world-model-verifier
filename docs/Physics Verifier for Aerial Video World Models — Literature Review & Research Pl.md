# Physics Verifier for Aerial Video World Models — Literature Review & Research Plan

Oct 30, 2026 · @Kevin

## Summary

The idea is novel for aerial world models and well-timed: no published work checks whether a generated drone flight's depicted attitude matches the attitude its motion requires, and the closest analogue (VehDyn, driving, Sep 2026) shows such dynamics checks expose failures that visual metrics miss.

- **Gap.** Aerial world models (ANWM, Skytopia, AeroAct and others) are scored with LPIPS, FID and FVD only. ANWM explicitly drops pitch and roll, so its flights accelerate without tilting by construction.
- **Template.** Recent work converges on "learn perception, compute the verdict": frozen foundation models recover a physical quantity, and a known equation judges it (VehDyn, KineBench, DriveReferee, NewtonRewards).
- **Bitter lesson.** The verifier survives scale if it stays outside the generator, uses swappable foundation models for perception, and is used as a filter or reward. The first experiment should test whether large video models already pass.
- **Biggest risk.** The measurement floor. Phase 0 and RQ1 settle whether pixel-based tilt is accurate enough before any verdict is built.
- **Recommended start.** TII Race Against the Machine (real fisheye video with mocap) and Blackbird (real dynamics, re-renderable images for perfectly matched negatives).

## State of aerial video world models

Aerial world models went from zero to a crowded niche in under a year, and almost all of them treat the drone as a 4-DoF camera (x, y, z, yaw) that never tilts. That gap is the opening for this project.

| Model / benchmark | Date | What it is | How the drone is modeled | Physics check? |
| --- | --- | --- | --- | --- |
| [ANWM](https://www.alphaxiv.org/abs/2512.21887) (Tsinghua) | Dec 2025 | Action-conditioned CDiT world model; 350k Unreal Engine segments; ranks candidate trajectories by imagined final frame | 4-DoF actions (dx, dy, dz, dyaw); authors say pitch and roll are ignored; constant 5 m/s, 2 m/s, 15°/s | None. LPIPS, DreamSim, FID only |
| [AirScape](https://www.alphaxiv.org/abs/2604.07991) | 2025 (ACM MM) | Aerial generative world model with motion controllability, real video (details via MotionScape's comparison table) | Visual / motion conditioning | Not reported |
| [Captain Safari](https://www.alphaxiv.org/abs/2604.07991) | CVPR 2026 | World engine with pose-aligned 3D memory; OpenSafari camera-controlled UAV video (details via MotionScape's comparison table) | Camera pose trajectory | Not reported |
| [MotionScape](https://www.alphaxiv.org/abs/2604.07991) | Apr 2026 | 228 real FPV/UAV YouTube clips, motion-stratified; benchmarks Cosmos-Predict2.5, Wan2.2, CogVideoX, LongCat, MAGI-1 | No pose labels; caption + optical-flow motion score | None. PSNR, SSIM, LPIPS, FVD, warping error, CLIPSim |
| [AeroAct](https://www.alphaxiv.org/abs/2607.14997) (BIT) | Jul 2026 | Wan2.1-1.3B world-action model; predicts 5th-order trajectory chunks; first real quadrotor WAM flight | Proprioception includes body-z axis; actions dynamically feasible via DiffAero | Feasibility enforced on the action side, not checked on video |
| [Uncertainty-Aware Aerial WM](https://www.alphaxiv.org/abs/2608.05597) | Aug 2026 | Image-goal navigation, ranks trajectories by predicted views with uncertainty | Trajectory-conditioned | No |
| [Skytopia](https://www.alphaxiv.org/abs/2609.26007), [DroneWAM](https://www.alphaxiv.org/abs/2609.33148), [ForeFly](https://www.alphaxiv.org/abs/2609.33581), [DiffWAM](https://www.alphaxiv.org/abs/2609.39763) | Sep 2026 | Latent / world-action models for drone navigation and aerial VLN | Mostly latent or action-centric | No |

Three patterns matter for us:

- **Use as simulator.** ANWM and its successors are used to score candidate trajectories by imagining what the drone would see. If the imagined video is dynamically impossible, the ranking is built on a fiction.
- **Tilt is designed out.** ANWM's data is Unreal replays at constant speed with a level camera, so generated flights accelerate without tilting. That is exactly the violation our verifier targets, and it is baked into the training data.
- **Real-world signal is thin.** MotionScape's failure cases are geometric (drifting buildings, hallucinated drone bodies); nobody measures whether the depicted camera attitude matches the depicted motion. AeroAct sidesteps the issue by never trusting generated frames at deployment. Its handheld rig even synthesizes quadrotor-like tilt by cropping fisheye images, a sign the community already knows attitude is part of the look of flight.

General-purpose video models (Cosmos-Predict2.5, Wan2.2, Genie 3) are the other source of aerial video. They are trained on internet FPV footage, so they may reproduce banking during turns, but no one has tested whether that banking is consistent with the motion.

## Prior work on verifying world-model outputs

No one has published an attitude-from-acceleration check for aerial world models. The closest work is in driving, where VehDyn recovers 6-DoF ego pose from generated video and tests it against vehicle dynamics. The field is converging on one design: learn perception, compute the verdict.

**Four families of verifiers exist.**

| Family | Examples | How it decides | Weakness for our use |
| --- | --- | --- | --- |
| VLM / human judges | VideoPhy, PhyMetric, [Physics QSG](https://www.alphaxiv.org/abs/2606.25306), [PhyProbe](https://www.alphaxiv.org/abs/2609.38377) (NVIDIA) | Ask a model whether physics looks right | Myopic to dynamics; no quantitative threshold; PhyProbe itself argues VLM judges miss motion violations |
| Learned plausibility scores | [GEOPHYS](https://www.alphaxiv.org/abs/2606.20707), [World Embedding Benchmark](https://www.alphaxiv.org/abs/2610.03632), [Physics in activations](https://www.alphaxiv.org/abs/2610.03154) | Probe representation geometry for implausibility | Opaque; tied to the training distribution |
| Learned inverse dynamics (IDM) | World-in-World, WoW-World-Eval "IDM Turing test" | Train a network to read actions out of video, replay in a simulator | [KineBench](https://www.alphaxiv.org/abs/2607.19876) shows IDMs fail out of distribution (about 10 cm error vs 1.5 to 3 cm for explicit geometry), so you cannot tell world-model error from extractor error |
| Explicit measurement + analytic law | [VehDyn](https://www.alphaxiv.org/abs/2609.33264), [KineBench](https://www.alphaxiv.org/abs/2607.19876), [NewtonRewards](https://www.alphaxiv.org/abs/2512.00425), [DriveReferee](https://www.alphaxiv.org/abs/2609.22762), [PVBench](https://www.alphaxiv.org/abs/2610.26711), [Principia](https://www.alphaxiv.org/abs/2609.04200) | Recover a physical quantity with a frozen foundation model, then apply a known equation | Bounded by the measurement floor; each check covers one law |

Our verifier belongs in the fourth family. Five papers set the template.

- **VehDyn (Sep 2026)** is the nearest analogue. It runs DPVO on generated driving video to recover position, velocity, roll, pitch and yaw, then checks a friction-circle constraint (lateral plus longitudinal acceleration must stay below μg + 0.5 m/s²). Across 12 models, trajectory metrics are near saturation, no model reaches 92% of ground truth on dynamic consistency, and visual-quality rank correlates with dynamics rank at only ρ = 0.39. It also measures pitch under braking and roll under lane change, which is the car version of our tilt check.
- **KineBench (Jul 2026)** replaces learned IDMs with a chain of frozen models (YOLO, MoGe-2, FoundationPose) to extract 6-DoF pose from generated manipulation video. Its key argument: explicit geometry makes extraction error measurable, so a failure can be blamed on the world model and not the extractor.
- **NewtonRewards (Nov 2025)** uses RAFT optical flow as a velocity proxy and enforces a constant-acceleration residual as a verifiable reward for post-training. Two lessons: a physics check can become a training signal, and a residual-only reward gets hacked (the model stops motion to zero the residual). Our verifier will need a matching guard.
- **DriveReferee (Sep 2026)** shows that a zero-parameter analytic rule on predicted scene state matches or beats a learned verifier trained on 66,385 labels. Its slogan, "learn perception, compute the verdict", is the right framing for this project.
- **Validate the Dream (Jul 2026)** proposes an admissibility ladder (L0 looks plausible, L1 follows actions, L2 valid inside a declared envelope, L3 failures attributable, L4 verdicts transfer to reality). A dynamics-consistency verifier is an L2/L3 instrument, which gives us a clean way to position the contribution.

**Adjacent ideas worth borrowing.** [PIVOT](https://www.alphaxiv.org/abs/2609.15562) uses physics-grounded consistency to detect AI-generated video, which is the same test from a forensics angle. ACT-Bench and [WorldOracle](https://www.alphaxiv.org/abs/2610.18826) recover ego trajectories with visual odometry to score action-following. [Do Robotic World Models Really Follow Actions?](https://www.alphaxiv.org/abs/2608.24885) finds generated futures often do not reflect the commanded action, the same failure ANWM-style aerial models are exposed to.

## Robotics foundations: tilt from acceleration, tilt from pixels

The physics is settled; the hard part is measurement. Quadrotor dynamics are differentially flat in position and yaw (Mellinger and Kumar, 2011), so the trajectory alone fixes the thrust direction, and therefore roll and pitch, at every instant.

**The implied tilt.** With world z pointing up, ideal (drag-free) dynamics give the body z-axis as the normalized sum of acceleration and gravity:

```latex
\mathbf{z}_B = \frac{\mathbf{a} + g\,\mathbf{e}_3}{\lVert \mathbf{a} + g\,\mathbf{e}_3 \rVert}, \qquad \tan\theta_{\text{tilt}} = \frac{\lVert \mathbf{a}_{xy} \rVert}{a_z + g}
```

Two consequences make a usable test:

- **Direction is scale-free.** The drone leans toward its horizontal acceleration. The lean direction depends only on the direction of a\_xy, so it survives monocular scale ambiguity.
- **Magnitude needs metric scale and true time.** tan θ depends on a / g, so a trajectory recovered up to scale (or a video played at the wrong frame rate) gives the wrong tilt magnitude.

**The depicted tilt.** Two independent routes recover camera attitude from pixels:

| Route | Tools | Gives | Weak when |
| --- | --- | --- | --- |
| Absolute gravity per frame | [GeoCalib](https://www.alphaxiv.org/abs/2409.06704), [Perspective Fields](https://www.alphaxiv.org/abs/2212.03239), [AnyCalib](https://www.alphaxiv.org/abs/2503.12701), horizon detection | Roll and pitch relative to gravity, single frame | No horizon or Manhattan cues; fisheye; looking straight down |
| Relative rotation over time | DPVO, DROID-SLAM, MegaSaM, VGGT-family reconstruction | Smooth relative rotation + translation up to scale | Fast rotation, motion blur, generated-video inconsistency |

A practical estimator fuses both: visual odometry gives the trajectory and relative rotation, and a gravity estimator anchors the absolute tilt. The same trajectory then supplies the implied tilt, so both sides of the comparison come from one video.

**Confounds the verifier must model, not ignore.**

1. **Rotor drag.** At steady speed a real quadrotor still tilts to balance drag ([Faessler et al., 2018](https://arxiv.org/abs/1712.02402)). A drag-free model will flag real cruising flight as infeasible. Add a linear drag term or fit it on real data.
2. **Camera mounting.** FPV cameras are hard-mounted with 20 to 40 degrees of uptilt, so camera pitch = body pitch + constant offset. Consumer drones (DJI-style) use a gimbal that cancels tilt entirely; their footage is not testable with this method and must be filtered out or labeled.
3. **Wind.** A hovering drone in wind leans with zero acceleration. This bounds how tight the threshold can be on outdoor footage.
4. **Yaw is free.** Flatness says nothing about yaw, so the check covers roll and pitch only.
5. **Frame timing.** Generated video often has an ambiguous or resampled frame rate. Acceleration scales with fps², so timing must be fixed or fitted as a nuisance parameter.

## Where the field is heading

The field is moving from "does it look right" to "does it obey dynamics", and from scoring videos to using verifiers as training signals. A tilt-consistency verifier fits all four current trends.

1. **World models as test oracles.** Aerial (ANWM, Skytopia, Uncertainty-Aware WM) and driving models are used to rank or grade trajectories. Validate the Dream argues those verdicts are inadmissible until the model is checked against physics inside a declared envelope.
2. **Physics benchmarks are multiplying fast.** October 2026 alone brought PVBench, World Models' Last Exam, PhysTrace, PhyMetric and WMEval. Almost all target tabletop or third-person scenes; none targets ego-motion dynamics of aircraft.
3. **Verifiers become rewards.** NewtonRewards and DriveReferee turn analytic checks into post-training signals (verifiable rewards, preference pairs). A tilt residual is differentiable through optical flow and pose estimators, so it can follow the same path.
4. **World-action models skip video at inference.** AeroAct, Fast-WAM and DiffWAM keep video only as training supervision. That raises the value of auditing the video they train on: physically inconsistent training video teaches the action head the wrong dynamics.

## Bitter-lesson check: will this still matter after the next scale-up?

Yes, if it is built as a verifier and not as a physics module inside the generator. Sutton's lesson is that hand-built knowledge loses to general methods that use more compute. Search and learning both need a signal to optimize, though, and a cheap, correct check of a hard physical law is exactly the kind of signal that grows in value as compute grows.

| Design choice | Fights the bitter lesson | Rides it |
| --- | --- | --- |
| Where physics lives | Hard-coded tilt layer or flatness decoder inside the world model | Outside, as an evaluator, filter or reward the model is trained against |
| Perception | Hand-built horizon or line detectors | Frozen foundation models (VO, gravity estimators) that improve on their own and can be swapped |
| The verdict | Many tuned heuristics and per-dataset thresholds | One law (flatness + drag), few parameters, fitted once on real flights |
| Use at scale | One-off benchmark table | Data filter for pre-training, best-of-N selector, verifiable reward for RL post-training |
| Claim | "Our method makes flights feasible" | "Our check measures feasibility; here is how it changes with model scale" |

**Three honest risks.**

- **Scale may solve it for free.** Models trained on large amounts of real FPV footage may already bank correctly. The first experiment should measure this before building anything else. If big models pass and small or simulator-trained ones (ANWM) fail, the verifier is still useful as an L2 admissibility gate and as a data filter, but the headline shifts from "models break physics" to "here is who breaks it and when".
- **The measurement floor may be too high.** If gravity and odometry estimates are noisier than the tilt signal, the verifier cannot separate real from fake. Research question 1 exists to settle this first.
- **It covers one law.** Tilt consistency is necessary, not sufficient. Thrust limits, maximum body rate (from jerk) and camera-mount consistency are natural extensions using the same machinery.

The sentence to aim for in the paper mirrors DriveReferee: learn the perception, compute the verdict.

## Candidate datasets and models

Use three tiers: labeled real flights to calibrate the estimator (RQ1), real in-the-wild footage as positives, and generated or doctored flights as negatives (RQ2). Blackbird is the most valuable single resource because its images are rendered from real flight poses, so the same trajectory can be re-rendered with the tilt removed.

**Tier 1: ground-truth attitude (RQ1 calibration and RQ2 positives)**

| Dataset | Video | Ground truth | Speed | Why it matters | Watch out for |
| --- | --- | --- | --- | --- | --- |
| [Blackbird](https://www.alphaxiv.org/abs/1810.01987) (MIT, 2018) | Rendered photorealistic 1024×768 at 120 Hz, 5 indoor scenes | Mocap 360 Hz, IMU, motor RPM; 10+ h, 163 flights | Up to 7 m/s | Real dynamics with re-renderable cameras: render the exact flight with a level camera, delayed tilt, or flipped tilt for perfectly matched negatives | Rendered, not real pixels; moderate speeds |
| [TII Race Against the Machine](https://www.alphaxiv.org/abs/2311.02667) (2024) | Real RGB 640×480 at 120 Hz, 175° fisheye | Qualisys mocap 275 Hz, IMU 500 Hz, motor thrust; 30 flights, about 7 km | Up to 21.8 m/s, above 7 g | Real pixels, aggressive flight, both piloted and autonomous; camera uptilt recorded (30°, 40°, 50°) | Fisheye needs undistortion; one indoor arena |
| [UZH-FPV](https://www.alphaxiv.org/abs/2604.07991) (2019) | Real grayscale fisheye 30 to 50 Hz, indoor and outdoor | Leica laser tracker, IMU | Up to 26.8 m/s | Outdoor horizon present; standard VIO benchmark | Grayscale, lower frame rate |
| Mid-Air (2019) | Synthetic 1024×1024 at 35 fps, outdoor | Simulator pose and IMU | Low altitude | Easy horizon; physics from simulator | Synthetic dynamics |

**Tier 2: real in-the-wild positives (no labels)**

| Source | Notes |
| --- | --- |
| [MotionScape](https://www.alphaxiv.org/abs/2604.07991) | 228 real FPV/UAV clips, 1080p, motion-stratified, with reconstruction scripts. Must screen out horizon-locked or stabilized footage (GoPro horizon leveling, gimbal drones), which removes tilt in post and would look "infeasible" to the verifier |

**Tier 3: negatives**

| Source | How to get it | Expected violation |
| --- | --- | --- |
| Counterfactual re-renders (Blackbird) | Re-render real trajectories with attitude level, tilt × k, tilt delayed by τ, tilt mirrored | Controlled, graded violations with known magnitude |
| Synthetic edits of real video (TII, UZH-FPV) | Rotate/warp frames to remove or invert roll; retime the video | Same idea on real pixels |
| [ANWM](https://www.alphaxiv.org/abs/2512.21887) and its Unreal training data | Run the released model or replay its data | 4-DoF level camera: accelerates without tilting |
| General video models: Cosmos-Predict2.5, Wan2.2, LongCat, MAGI-1 | Image2World / Video2World continuations of MotionScape and TII prefixes, as in MotionScape's protocol | Unknown; this is the main empirical question |

## Research questions and success criteria

RQ1 asks whether the measurement is good enough; RQ2 asks whether the verdict separates real from fake while keeping real flights valid. RQ2 is only meaningful if RQ1 passes.

**RQ1: Can tilt be estimated from the drone camera alone?** Two quantities, both compared against motion-capture truth:

- **Depicted tilt** θ\_dep(t): roll and pitch read from the pixels (gravity estimator, odometry rotation, or fused), minus the camera mount angle.
- **Implied tilt** θ\_imp(t): tilt computed from the trajectory recovered from the same pixels, using the flatness equation.

| Metric | Definition | Pass target (proposed) |
| --- | --- | --- |
| Depicted-tilt error | RMSE of θ\_dep vs mocap roll/pitch, degrees | Under 5° on TII and Blackbird |
| Lean-direction error | Angle between estimated and true horizontal lean direction, frames with tilt above 10° | Median under 15° |
| Implied-tilt error | RMSE of θ\_imp (from recovered trajectory) vs mocap tilt | Under 8° after per-clip scale fit |
| Signal-to-noise | Std of true tilt / estimation RMSE | Above 3 |

If the targets fail, the output is still useful: an error-vs-speed and error-vs-scene curve showing where the method works.

**RQ2: Does tilt consistency accept real flights and reject infeasible ones?** Per frame, the residual is the angle between the depicted thrust axis and the implied thrust axis:

```latex
r(t) = \angle\big(\, \mathbf{z}_B^{\text{dep}}(t),\; \mathbf{z}_B^{\text{imp}}(t;\, s, \varphi, \tau, k_d) \big)
```

Here s is the unknown metric scale, φ the camera uptilt, τ a small time lag, and k\_d a linear drag coefficient. These nuisance terms are fitted per clip so that real flights are not rejected for unknowable reasons. The clip score is a robust average of r(t) over frames with enough acceleration to carry signal; clips with too little acceleration are marked "not testable" rather than passed or failed.

| Metric | Definition | Pass target (proposed) |
| --- | --- | --- |
| Real-flight acceptance | Share of held-out real clips scored valid | At least 95%, by setting the threshold at the 95th percentile of real scores (conformal calibration) |
| Fake rejection | Share of negatives scored invalid at that threshold | Report per negative type; aim above 80% on counterfactuals |
| AUROC | Real vs each negative family | Above 0.9 on counterfactuals |
| Minimum detectable violation | Smallest tilt removal, delay or scale error rejected 80% of the time | Report as a curve |
| Monotonicity | Spearman correlation of score with injected violation size | Above 0.8 |
| Abstention rate | Share of clips with too little acceleration to test | Report; keep the rule fixed before seeing negatives |

The nuisance fit is the main threat to validity. With enough free parameters any clip can be explained, so every added nuisance term must be justified on real data and stress-tested on negatives.

## Step-by-step experiment plan

Six phases, each ending in a go/no-go gate. Start with the physics floor on ground truth, because no estimator can beat the error of the physics model itself.

**Phase 0: Physics floor (ground truth only, no pixels)**

- [ ] Download TII Race Against the Machine (start with one autonomous ellipse and one piloted lemniscate) and a few Blackbird flights (Clover, Sphinx at 2, 4, 6 m/s).
- [ ] Differentiate mocap position twice (Savitzky-Golay) and compute implied tilt with the drag-free flatness equation.
- [ ] Compare against mocap attitude. Plot residual vs speed; expect it to grow with speed because of rotor drag.
- [ ] Add a linear drag term, fit k\_d on half the flights, test on the rest.
- [ ] Gate: median residual on held-out flights is small (target under 3 to 5°). This number is the best any verifier can do.

**Phase 1: RQ1a, depicted tilt from pixels**

- [ ] Undistort TII fisheye frames to a pinhole view; keep Blackbird as rendered.
- [ ] Run GeoCalib per frame for roll and pitch; subtract the recorded camera uptilt; compare to mocap.
- [ ] Run a visual odometry model (DPVO, then MegaSaM or a VGGT-family model) for relative rotation; anchor to gravity with GeoCalib using a simple complementary filter.
- [ ] Break down error by speed, motion blur (TII auto vs fixed exposure), lighting level and scene.

**Phase 2: RQ1b, implied tilt from pixels**

- [ ] Take the odometry translation and compute implied tilt three ways: oracle scale from mocap, metric scale from a depth model (MoGe-2 or similar), and scale fitted by the verifier.
- [ ] Compare each to mocap tilt; separately report the scale-free lean-direction error.
- [ ] Gate (end of RQ1): depicted and implied tilt errors meet the targets in the RQ section, or you know the speed and scene range where they do.

**Phase 3: RQ2 on controlled negatives**

- [ ] Build negatives with known violations: Blackbird re-renders (level camera, tilt × 0.5, tilt delayed 100 to 300 ms, mirrored roll); TII image edits (counter-rotate each frame by the true roll to erase it; retime at 0.5× and 2×).
- [ ] Implement the clip score with nuisance fitting (scale, uptilt, lag, drag); fix the abstention rule before looking at negatives.
- [ ] Split by flight; set the threshold at the 95th percentile of held-out real scores.
- [ ] Report acceptance of real clips, rejection per negative type, AUROC, and minimum-detectable-violation curves.
- [ ] Ablate each nuisance parameter to show it is needed on real data and does not absorb fake violations.
- [ ] Baselines: a VLM judge asked whether the flight is physically possible, and a learned real-vs-fake video classifier. Test the classifier on counterfactuals where only tilt differs, to show whether it learns physics or artifacts.
- [ ] Gate: real acceptance at least 95% with useful rejection on counterfactuals.

**Phase 4: Real in-the-wild footage**

- [ ] Reconstruct MotionScape clips; screen out stabilized or horizon-locked footage and log how many were removed.
- [ ] Run the verifier with uptilt fitted per clip; report the in-the-wild false-reject rate and inspect every rejected real clip by hand.

**Phase 5: Generated flights**

- [ ] Generate continuations from MotionScape and TII prefixes with Cosmos-Predict2.5, Wan2.2-I2V, LongCat-Video, following MotionScape's Image2World and Video2World protocol.
- [ ] Roll out ANWM on its own benchmark trajectories as the expected-failure case.
- [ ] Score all of them; report violation rate per model and per motion stratum.
- [ ] Correlate the verifier score with FVD and LPIPS (VehDyn found ρ = 0.39 between visual and dynamics rankings in driving).

**Phase 6: Bitter-lesson experiments**

- [ ] Scale trend: compare small and large variants of the same family (for example Wan2.1-1.3B vs Wan2.2-A14B, Cosmos 2B vs larger). Does the violation rate fall with scale?
- [ ] Use the verifier as a selector: generate N candidates per prompt, keep the best-scoring, and measure whether downstream use (ANWM trajectory ranking, or human preference) improves.
- [ ] Stretch: use the residual as a verifiable reward for post-training, with a guard against the NewtonRewards failure mode (models freezing motion to zero the residual).

## Risks, confounds and open questions

The biggest risk is not the physics but the measurement: generated video is exactly where odometry and gravity estimators are least reliable.

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Estimator fails on generated video | Inconsistent geometry breaks odometry; a failed estimate looks like a physics violation | Report estimator confidence and reprojection error; treat estimator failure as "not testable", never as "infeasible" (KineBench's attribution argument) |
| Nuisance fit absorbs violations | Free scale, uptilt and lag can explain away a fake | Fit uptilt once per clip, not per frame; bound lag and scale to physical ranges; ablate in Phase 3 |
| Stabilized footage | Gimbals and horizon lock remove tilt from real video | Screen inputs; state that the verifier applies only to hard-mounted cameras |
| Unknown frame rate of generated video | Acceleration scales with fps squared | Use the model's declared output rate; test sensitivity to a ±2× error |
| Low-acceleration clips | Hover and cruise carry little tilt signal, so real and fake look alike | Abstain below an excitation threshold and report the rate |
| Scale makes it moot | Large models trained on FPV footage may already bank correctly | Measure first (Phase 5); position as an admissibility gate and data filter either way |
| Distribution shift between indoor mocap and outdoor footage | Thresholds tuned on TII may not transfer | Calibrate separately on outdoor real footage (UZH-FPV outdoor, MotionScape) |

**Open questions to settle early**

- Which world model is the primary target: an action-conditioned aerial model like ANWM (clean story, known failure) or general video models (bigger audience, unknown result)?
- Is the output a benchmark, a filter, or a reward? This decides how much weight Phase 6 gets.
- Can you get access to a hard-mounted outdoor FPV log with flight-controller telemetry (Betaflight blackbox)? It would give real outdoor ground truth and close the indoor-only gap.

## Sources

Papers opened and read for this review (via alphaXiv):

- [Aerial World Model for Long-horizon Visual Generation and Navigation in 3D Space (ANWM)](https://www.alphaxiv.org/abs/2512.21887)
- [MotionScape: A Motion-Stratified UAV Video Benchmark](https://www.alphaxiv.org/abs/2604.07991) (also the source for AirScape, Captain Safari and UZH-FPV details)
- [AeroAct: Action-Centered World-Action Models for Quadrotor Flight](https://www.alphaxiv.org/abs/2607.14997)
- [KineBench: IDM-Free Kinematic Grounding](https://www.alphaxiv.org/abs/2607.19876)
- [VehDyn: A Driving World Model Benchmark for Vehicle Dynamics](https://www.alphaxiv.org/abs/2609.33264)
- [Validate the Dream Before You Trust Its Verdict](https://www.alphaxiv.org/abs/2607.07196)
- [What about gravity in video generation? (NewtonRewards)](https://www.alphaxiv.org/abs/2512.00425)
- [DriveReferee: Geometric Safety Verdicts Need Not Be Learned](https://www.alphaxiv.org/abs/2609.22762)
- [Race Against the Machine (TII drone racing dataset)](https://www.alphaxiv.org/abs/2311.02667)
- [The Blackbird Dataset](https://www.alphaxiv.org/abs/1810.01987)

Identified from search results (abstract only): PVBench, PhyProbe, PhysTrace, Physics QSG, PhyMetric, GAUGE, WMEval, Principia, GEOPHYS, PIVOT, WorldOracle, Do Robotic World Models Really Follow Actions?, Skytopia, DroneWAM, ForeFly, DiffWAM, GeoCalib, Perspective Fields, AnyCalib. Rotor-drag flatness ([Faessler et al., 2018](https://arxiv.org/abs/1712.02402)) and minimum-snap flatness (Mellinger and Kumar, 2011) are cited from background knowledge.
