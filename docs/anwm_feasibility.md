# ANWM feasibility check

Target: *Aerial World Model for Long-horizon Visual Generation and Navigation in 3D Space*,
Weichen Zhang, Peizhi Tang et al. (Tsinghua), arXiv:2512.21887v2 (submitted 2025-12-26,
revised 2026-01-03). Every claim below is tagged **verified** (read in the paper or a
public page, with the location) or **unverified** (inference or absence of evidence).

## Bottom line

**No official repository and no released weights were found.** ANWM is therefore not
runnable today, and the plan of "roll out ANWM and score the videos" is blocked at the
weights step, not at the compute step. The action-space audit below still tells us what
its training data implies about tilt, and it is the useful part we can keep.

Evidence for the absence (all checked 2026-11-07):

| Check | Result |
| --- | --- |
| arXiv abstract page, "Code, Data and Media" section | only the generic arXiv UI toggles; no author-supplied link (**verified**) |
| arXiv HTML full text, all `href`s | no GitHub, Hugging Face, project-page or dataset link (**verified**) |
| GitHub repository search for the title and its variants, and for "CDiT aerial" | 0 results (**verified**) |
| Web search for title + "code"/"github"/"project page" | only arXiv mirrors and the author group's PDF copy (**verified**) |

Absence of evidence is not proof of absence: a repo may exist unlinked, or appear later.
Do not treat the row above as "will never exist".

## 1. Official repo

None found. There is no `third_party/anwm` clone and no pinned commit, because there is
nothing to pin. The author group (Xin Wang, Tsinghua) hosts a PDF copy at
`mn.cs.tsinghua.edu.cn/xinwang/PDF/papers/2026_Aerial World Model ...pdf`; no code link
there either. (**verified** for the PDF; **unverified** for the possibility of an
unlinked private repo.)

## 2. Weights

None released, so there are no licence terms to check, and nothing can be published from
the model itself. Re-training from scratch is out of scope: 300k steps on 4x A800 40 GB,
plus a 350k-segment dataset that is also unreleased (Sec. 3.2, Implementation Details of
Sec. 4.1). (**verified** that weights/data are not mentioned as released; **unverified**
that they will never be.)

## 3. Output specification

- Resolution **224x224** for both input and output frames. (**verified**, Sec. 4.1
  "Implementation Details")
- **One frame per action**, 48 actions per segment, 48 frames per segment; only the
  post-action frame is stored, so there is no intermediate motion blur or sub-frame
  information. (**verified**, Sec. 3.2)
- Frame rate is **not stated**. Evaluation horizons are 4, 8, 16 and 32 s over segments of
  48 frames, which gives **0.667 s per action, ~1.5 fps** (**inferred**, two independent
  checks: 32 s / 48 frames, and 48 steps at the stated speeds reproducing a path length
  near the reported 80.7 m — our simulation gives 94 m at 0.667 s versus 47 m at 0.333 s
  and 141 m at 1 s, see the audit table).
- Chaining: autoregressive, one waypoint/frame at a time, with the last **m = 4** frames as
  context (**verified**, Sec. 3.3 and 4.1). The context-size ablation reports that 16
  frames *degrades* FFP quality, so chaining is deliberately short. Reported generation
  quality is evaluated up to 32 s (**verified**, Table 1 rows at 4/8/16/32 s).

## 4. Action format

- Action is a continuous 4-vector **a = (dx, dy, dz, dyaw)** in the agent's frame: relative
  translation plus yaw change between consecutive waypoints. (**verified**, Sec. 3.1)
- Each waypoint pose (x, y, z, yaw) is converted to the *relative* transform with respect
  to the previous waypoint to feed the model. (**verified**, Sec. 3.3)
- Training data uses discrete actions drawn from forward/backward, left/right, up/down,
  yaw left/right, i.e. **8 commands**, with magnitudes **5 m/s horizontal, 2 m/s vertical,
  and 15 deg/s yaw**. (**verified**, Sec. 3.2)
- The encoder is a sine-cosine embedding of the real-valued 4-vector, so **arbitrary
  continuous actions can be fed at the interface** (**verified**, Sec. 3.4). But the model
  never saw a speed other than 5 m/s (2 m/s vertical), a yaw rate other than 15 deg/s, or a
  magnitude in between, so sharp turns and speed changes are **out of distribution**
  (**inference from the training distribution** — unverified, since no code or weights are
  available to test it).

## 5. Inputs

- Needs the past **m = 4** frames plus the next action; the first frame is the conditioning
  image, resized to 224x224. (**verified**, Sec. 3.3/4.1)
- A TII frame can be used as that initial frame in principle: it is an ordinary RGB image.
  Two caveats: TII is 640x480, 175 degree fisheye, so it must be undistorted before
  resizing, and ANWM's training images are front/left/right/rear views of 40 Unreal urban
  scenes, so a real indoor fisheye frame is a domain shift. (**unverified** functionally;
  there is no code to run.)

## 6. Hardware and speed

- Training: 8 CDiT blocks, 300k steps, 4x NVIDIA A800 40 GB, AdamW, lr 8e-5.
  (**verified**, Sec. 4.1)
- Inference memory and wall-clock per rollout are **not reported** (**verified** that the
  paper is silent). Any number here would be a guess: not estimated.
- What the architecture implies for planning: 224x224 latents with an 8x8 VAE downsample
  and 4 conditional frames is small by current standards, so a single 24-40 GB GPU should
  be enough — **unverified**, and moot while weights are unavailable.

## 7. Benchmark trajectories and Unreal data

- Dataset composition: **350k training segments**, **2.2k test** (1.1k 2D + 1.1k 3D),
  average path 80.7 m, built by replaying **20k+ trajectories from AerialVLN, OpenFly and
  OpenUAV** in Unreal Engine across **40+ simulated urban scenes**, with front/left/right/
  rear views to enrich the action distribution. (**verified**, Sec. 3.2)
- Release status: not mentioned as released anywhere in the paper, and there is no repo to
  check. (**unverified** that it is unavailable, but nothing indicates availability.)
- Consequence: we cannot replay their benchmark trajectories, and we cannot use their
  training video as a negative set. The benchmark numbers in the paper are not
  reproducible from public artifacts.

## 8. Plan for running one rollout on Modal (written, not executed)

Blocked at step 2. Sequence once weights exist:

1. **Image**: `modal.Image.debian_slim(python_version="3.11")` plus `torch`, `torchvision`,
   the repo's own `requirements.txt`, and `opencv-python-headless` for fisheye undistortion.
   Independent of `pixi.toml`; record any shared pins in `docs/decisions.md`.
2. **Weights**: download once into a Modal Volume (`modal volume create anwm-weights`,
   mount at `/weights`), keyed by the repo commit; never bake weights into the image.
3. **Input**: one undistorted TII frame as the initial condition, plus a 48-step action
   sequence (paper speeds, then a sharper variant to probe out-of-distribution behaviour).
4. **Run**: 48 autoregressive steps at 224x224; save the frames plus the action log to a
   second Volume, then sync to R2.
5. **Verifier hook**: load the saved frames through the same `Clip` schema, with `fps`
   taken from the action time step (0.667 s, i.e. 1.5 fps) and `cam_uptilt_deg = 0`
   (level camera by construction), and score with the floor physics.

## Excitation audit: how many ANWM clips could the verifier even test?

`experiments/anwm_excitation_audit.py`, output `results/anwm_excitation/audit.json`. It
builds 4,000 random 48-action segments from the paper's stated action set, at three
candidate time steps, and runs the floor physics on them. This is a **distribution audit of
the stated action space**, not a replay of ANWM data (no sequences are released).

| dt per action | fps | step-implied tilt p50 | p90 | frac. boundaries >10 deg | SG-on-positions p50 | SG frac >10 deg | path length |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.333 s | 3.00 | 55.5 deg | 75.4 deg | 0.705 | 22.1 deg | 0.750 | 47 m |
| **0.667 s** | **1.50** | **34.6 deg** | **49.2 deg** | **0.690** | **12.8 deg** | **0.815** | **94 m** |
| 1.000 s | 1.00 | 23.5 deg | 36.1 deg | 0.688 | 9.2 deg | 0.370 | 141 m |

(Fitted drag `k_d = 0.297`; drag-free medians are 1-3 deg higher at the same dt.)

Reading it:

- **Every action boundary is a velocity jump, because the actions are random per step.**
  Under the flatness equation that jump requires a large tilt: median 35 deg at the most
  likely dt, and ~69% of boundaries exceed 10 deg. ANWM's frames depict a level camera
  (4-DoF, pitch and roll ignored), so on the commanded trajectory the model is wrong at
  most boundaries.
- **Constant-speed cruise alone is not testable.** At 5 m/s the fitted drag implies only
  **8.6 deg** of tilt, below the 10 deg signal threshold; vertical 2 m/s implies 3.5 deg.
  So the tilt signal lives in the *transitions*, not in the cruise.
- **dt per action is the single most important unknown**: implied tilt scales as
  1/dt through `atan(dv / (dt g))`. Going from 0.667 s to 0.333 s moves the median from 35
  to 56 deg. Until the code or data fixes this, the audit reports the range, not a point.
- **Consequence for the verifier**: ANWM segments should be scored on boundary frames with
  an explicit "sustained excitation" rule, and clips whose implied tilt never exceeds the
  threshold must be reported as *not testable* rather than *infeasible*. The frame rate
  (1.5 fps) is also far below what a VO-based depicted-tilt estimate needs, which pushes
  ANWM toward the "abstain" bucket for a different reason.

## Nearest alternatives (do not substitute silently)

- **AeroAct** (arXiv:2607.14997, BIT): Wan2.1-1.3B world-action model, predicts 5th-order
  trajectory chunks, demonstrated on real quadrotor flight. Its arXiv page has no code or
  weights link, and GitHub search returns 0 results, so it is **no more runnable than
  ANWM today** (**verified** for the absence of links).
- **Aero-World** (per the tracker, "Action-Conditioned Aerial Video Generation from
  Inertial Controls", UCF): no arXiv id is recorded in `docs/paper_tracker.csv`, so its
  repo/weights status is **unverified**.
- Practical fallback that is runnable *now*: general video models with public weights
  (Wan2.2-I2V, Cosmos-Predict2.5) conditioned on TII/MotionScape prefixes. Different story
  (not an aerial world model), but it produces testable pixels at a usable frame rate.

## What would unblock Task "run one ANWM rollout"

1. A public repo + weights, or a direct author request. Until then the honest status is
   *blocked, no substitute made*.
2. Confirmation of the action time step (0.667 s is inferred, not stated).
