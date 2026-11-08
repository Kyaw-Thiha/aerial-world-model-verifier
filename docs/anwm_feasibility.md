# ANWM feasibility check

Target: *Aerial World Model for Long-horizon Visual Generation and Navigation in 3D Space*,
Weichen Zhang, Peizhi Tang et al. (Tsinghua), arXiv:2512.21887v2.

Status: **public and runnable.** Code, weights and dataset are all released. This
document replaces an earlier version that wrongly reported "no repo, no weights": that
search used the paper title, and the repository is named `ANWM.code`, which no title
search matches.

Every claim is tagged **verified** (read in the code, the dataset card or a public page,
with file:line or URL) or **unverified** (inference, estimate, or absence of evidence).

## 1. Repository

| Item | Value |
| --- | --- |
| URL | https://github.com/EmbodiedCity/ANWM.code (**verified**) |
| Pinned commit | `657a80268505fa9149c4df502e35aa0f5bce11e5` (**verified**, `git rev-parse HEAD`) |
| Branch | `master` (**verified**) |
| Last commit | 2026-08-20T16:12:06+09:00, "Document Hugging Face dataset and checkpoint download paths." (**verified**) |
| History | 156 commits, 47 tracked files (**verified**) |
| Local clone | `third_party/anwm` (gitignored) (**verified**) |
| LICENSE file | **absent** — no `LICENSE`/`COPYING` at the root, nothing in `git ls-files` (**verified**) |

**Licence nuance (verified).** The weights and dataset cards are Apache-2.0, but the
repository carries no licence text of its own, and 13 files begin with
`# Copyright (c) Meta Platforms, Inc. and affiliates.` (`anwm/diffusion/*`, `anwm/utils.py`,
`anwm/data/airvln.py`, `anwm/distributed.py`, `evaluate.py`, `infer.py`), inherited from
Meta's NWM. A repo-wide grep finds no Apache, MIT or BSD text anywhere in the tree.
Practical reading: fine for research use and for publishing results, but the code is
*presumptively* Apache-2.0 via the model card rather than explicitly licensed, and the
Meta headers mean any redistribution should keep attribution. **Confirm with the authors
before redistributing code or re-hosting weights.** (Interpretation, hence unverified.)

Entry points (**verified**): `infer.py` (simulation rollouts), `evaluate.py` (LPIPS,
DreamSim, FID), `train.py` (6 GPUs), `planning_eval.py` (real-world planning),
`real/` (Sekai adapter, Pi-Long depth, candidate sampling).

## 2. Weights

| Item | Value |
| --- | --- |
| URL | https://huggingface.co/EmbodiedCity/ANWM (**verified**) |
| Files | `.gitattributes`, `README.md`, `0200000.pth.tar` only (**verified**, HF API) |
| Checkpoint size | 18,146,420,481 bytes = **18.1 GB** (**verified**, HF API `size`) |
| Licence | `apache-2.0` on the model card (**verified**) |
| Last modified | 2026-08-20 (**verified**) |

The 18.1 GB is the training checkpoint, not the inference footprint. `infer.py:256`
loads `ckp["ema"]`, so the file holds at least the model and its EMA; ~1.1B parameters
in fp32 is 4.4 GB, and model + EMA + two Adam moments ≈ 17.6 GB, which matches the file
size (**inference from size; unverified**). Expect to need roughly 4.4 GB resident for
EMA weights plus the VAE.

The VAE is `stabilityai/sd-vae-ft-ema`, fetched by Diffusers on first use
(`infer.py:261`, README). No separate download step is documented.

## 3. Output specification

- **224 x 224** input and output (`config/anwm.yaml: image_size: 224`; `infer.py:239`
  sets `latent_size = image_size // 8` = 28, i.e. an 8x downsampling VAE). (**verified**)
- Backbone **CDiT-XL/2**: hidden size 1152, depth 28 blocks, 16 heads, patch 2, 4 latent
  channels, ~1.1B parameters (`anwm/model.py:240-262`, `CDiT_models` at 456). (**verified**)
- **250 diffusion steps per generated frame** (`infer.py:260`, `create_diffusion(str(250))`;
  README "Inference sampling steps 250"). This is the dominant cost: 250 network passes
  per frame. (**verified**)
- **One frame per action.** README: 48 frames, 47 actions per segment. `len_traj_pred: 16`
  in the config limits a single forward rollout to 16 steps; `infer.py` rolls out over the
  eval horizon and re-feeds predictions (`generate_rollout`, `infer.py:88-108`). (**verified**)
- Context: last **4** frames (`context_size: 4`), plus the FFP-projected frame. (**verified**)
- Chaining: autoregressive, one frame at a time, appending the prediction and dropping the
  oldest (`infer.py:105-108`). README limitation: "generation may drift or collapse as
  trajectories approach 200 m". (**verified**)
- Frame rate: see section 8 — **not defined by the release**.

## 4. Action format

- Action is a 4-vector `(dx, dy, dz, dyaw)` in the agent's local frame
  (`real/planning_eval.py:508` `self.action_dim = 4  # hardcoded`). (**verified**)
- Construction (`anwm/data/airvln.py:179-207`): absolute future waypoints are converted to
  local coordinates about the current pose (`to_local_coords`), stacked with wrapped yaw
  differences (`angle_difference`), then the *translation* part is divided by
  `metric_waypoint_spacing = 3.30` and mapped to [-1, 1] with
  `action_stats = {min: [-2.5, -4.0, -3.0], max: [5.0, 4.0, 3.0]}`. (**verified**)
- The tensors fed to the model are **deltas between consecutive normalised waypoints**
  (`get_delta_np`, `anwm/utils.py:347-352`): i.e. increments, not absolute offsets.
  (**verified**)
- The action encoder is a sine-cosine embedding of the real-valued 4-vector modulated by
  AdaLN (`anwm/model.py` `ActionEmbedder`; paper Sec. 3.4). So the *interface* accepts
  arbitrary continuous actions. (**verified** for the interface; the training distribution
  is the constraint below.)
- Training actions are discrete: AerialVLN's action set is
  `STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, GO_UP, GO_DOWN, MOVE_LEFT, MOVE_RIGHT` with
  `FORWARD_STEP_SIZE = 5`, `UP_DOWN_STEP_SIZE = 2`, `LEFT_RIGHT_STEP_SIZE = 5`,
  `TURN_ANGLE = 15` (**verified**, `AirVLN/airsim_plugin/airsim_settings.py:20-66`), replayed
  by ANWM. The paper states the same numbers as speeds: "5 m/s, 2 m/s, 15 deg/s".
- Consequence: sharp turns and speed changes are **out of distribution** — the model has
  only ever seen 5 m, 2 m and 15 deg steps. Feeding a 1 m or 20 m step is possible at the
  interface but untested (**unverified** behaviour).

## 5. Inputs, and what FFP needs

Per generated frame the model consumes: the 4 context frames, the action, and one
**Future-Frame-Projection** image. FFP is not a learned model: it reprojects the context
frames' depth into the target viewpoint (`anwm/projection.py:182-240`) and needs, per
context frame, a **depth map (H, W)**, a **camera matrix K (3x3)** and a **4x4 pose**, plus
a **4x4 pose for the target frame** (`anwm/data/airvln.py:147-175`). (**verified**)

Pose convention (`anwm/projection.py:306-318`, `data/preprocessing/Data_preprocessing_airvln_16.ipynb`
cell 14): `pose = [R(yaw, pitch, roll) | xyz]` in world, then post-multiplied by
`t_ec` (the FLU→RDF camera-axis change). So **roll and pitch are in the pose** and FFP
uses them: the training frames depict the simulator's true attitude, not a level camera.
(**verified**)

Can we start from a TII frame? Yes, with one learned component:

| FFP input | Source for TII | Status |
| --- | --- | --- |
| RGB frame | TII 640x480, 175 deg fisheye, must be undistorted first | available |
| `K` | TII ships `camera_calibration/calib_*.json/npz`; recompute after undistortion | available |
| depth | **Pi-Long**, as in the real-world route (`real/tools/fill_depth_pilong.py`) | needs a model run |
| pose | TII mocap gives exact pose (275 Hz); use it instead of Sekai's reconstruction | available, better than the real pipeline |
| target pose | integrate the commanded action sequence | available |

So the missing piece is depth only. The real-video route
(`real/tools/process_youtube_to_airvln16_format.py`) expects an `extrinsic` npz per clip,
resamples to `--out_fps 4` (line 517-523) and can zero the attitude (`--pitch_roll zero`,
line 646) — we would instead pass mocap attitude. (**verified** for the code paths; the
end-to-end TII run is **unverified** until executed.)

## 6. Hardware and time

- Training: 6 GPUs (README), 300k steps, 4x A800 40 GB per the paper. (**verified**)
- Inference: no memory or timing figure is reported anywhere. (**verified** absence)
- Budget arithmetic: 250 passes/frame x 48 frames = **12,000 CDiT-XL passes per 48-frame
  rollout**, each over 784 latent tokens (28x28) at hidden 1152. Inference weights ~4.4 GB
  fp32 (EMA only) + VAE (~0.3 GB) + activations for batch 1.
  A 24 GB card (A10G, L4, 4090) is comfortable; a 16 GB T4 is probably enough but tight
  (**unverified**). Expect **single-digit to low-double-digit minutes per 48-frame
  rollout**; the exact number must be measured, not assumed (**unverified**).

## 7. Dataset: what is released

- https://huggingface.co/datasets/EmbodiedCity/ANWM-Dataset, `apache-2.0`. (**verified**)
- **342 tar shards** `airvln_16-000000.tar` .. `airvln_16-000341.tar`, **637.45 GB** total,
  1.16-2.15 GB per shard, plus `README.md`. (**verified**, HF API)
- Each shard holds `<ID>_processed/` folders with numbered JPEGs and a `traj_data.pkl`
  (`data/README.md`). The pickle carries `K`, `position`, `yaw`, `timestamps`, `images`,
  `pitch`, `roll`, `point`, plus per-frame **`depth`** and 4x4 **`pose`**
  (`Data_preprocessing_airvln_16.ipynb` cell 14), which is why a single trajectory's pickle
  is 271-649 MB (**verified**, tar member sizes).
- **Scene coverage: scene 16 only.** The notebook filters `item['scene_id'] == 16`
  (cell 8) and every artifact is named `airvln_16`. The paper describes 20k trajectories
  across 40+ Unreal scenes; the release is a single scene. Whether the released checkpoint
  was trained on more than this shard set is **unverified** — the README says the released
  run trained on `data/airvln_16`, so the release is the training distribution we can see.
- Split sizes in the repo: 1013 train trajectories, 254 test trajectories
  (`data/splits/airvln_16/*/traj_names.txt`); the rollout index `test/rollout_16.pkl` has
  1000 evaluation entries over 29 test trajectories. (**verified**)
- `timestamps` are **frame indices**, not seconds: the notebook writes `timestamp = idx`
  into the pickle (cell 14). (**verified**)

## 8. Seconds per step: the one number the release does not fix

This matters because implied tilt goes as `tan(theta) = |a_xy| / (a_z + g)` and `a` scales
with `1/dt^2`; between 1 s and 0.25 s the tilt swings by ~16x.

Every candidate with its evidence:

| Candidate | Value | Evidence | Status |
| --- | --- | --- | --- |
| Paper/README speed wording | **1.0 s** | "Horizontal speed is 5 m/s, vertical 2 m/s, yaw 15 deg/s" (README, Dataset section) paired with AerialVLN step sizes 5 m / 2 m / 15 deg; distances per second = 5, 2, 15 | derived, **unverified as a time statement** |
| Paper Figure 2 | ~**1 s** | reported 1 s frame labels in the figure | unverified here (figure not read numerically) |
| Config spacing | 3.30 m | `metric_waypoint_spacing` is a *normaliser*, not a duration | verified as a constant only |
| Code default | **0.25 s** | `--input_fps 4` (`infer.py:362`, `evaluate.py:296`), used at `infer.py:77` as `rollout_stride = input_fps // rollout_fps` and at `infer.py:125,158` to convert "sec" to frame indices | verified |
| Data | none | `timestamps` are frame indices; no `fps`, `dt` or duration appears in any pickle key or config | **verified absence** |

Why the code default is weak evidence: `--input_fps` exists so that `sec * input_fps`
gives a frame index and so that a 1 fps rollout groups `input_fps` frames per step
(`infer.py:77-79`). It is inherited from NWM, which used 4 Hz data; nothing in ANWM's data
or training ties one frame to 0.25 s. Also note the internal inconsistency: at
`input_fps=4` the release's own 3.30 m spacing would mean 13.2 m/s, contradicting the
"5 m/s" the same README states.

The decisive point: AerialVLN is a **teleport simulator**. Actions are applied by setting
the pose (`AirVLNSimulatorClientTool.py:358-360`, `simSetVehiclePose(..., ignore_collision=True)`)
and stepping by fixed distances (`AirVLN/utils/env_utils.py:413,427,448,469`). No simulated
time advances, so the released trajectories contain no physically meaningful time base at
all. `dt` is a free parameter of any tilt analysis of this data.

This audit therefore reports a dt **range** with an explicit sensitivity table, and
separately measures which dt best explains the recorded attitude (section 11). It does not
pick one silently.

What the data itself can settle (section 11): because every released frame is level, and
implied tilt scales as `1/dt^2`, the **0.25 s reading is excluded** (it demands ~25 deg
median tilt against a level camera) and **1 s is the smallest duration consistent with the
frames** (drag-free discrepancy ~1 deg). 1 s also matches the paper's "5 m/s" wording and
contradicts the code's `--input_fps 4`. Treat 1 s as the working assumption, tagged
unverified, until the authors confirm.

## 9. What would make ANWM tilt-testable

1. **A dt is chosen and justified.** The only defensible route: state the assumption
   (1 s per action, matching the paper's m/s wording), and show the sensitivity table.
2. **Frames must be dense enough to see the transient.** ANWM emits one frame per action.
   If a step is ~1 s, that is ~1 fps, far below what visual odometry needs; the verifier
   would be comparing attitudes at 1 s spacing, not a continuous flight.
3. **FFP's projected frame leaks geometry** into the output. A model that is handed a
   depth-reprojected future frame can look geometrically consistent without ever learning
   attitude. Treat FFP-on vs FFP-off as a control (**unverified**; no ablation of the
   verifier's signal is published).

## 10. Modal plan (written, not run)

Image: `modal.Image.debian_slim(python_version="3.11")` with a CUDA-matched
`torch`/`torchvision` pair first, then `pip install -e ".[metrics]"` from the pinned clone;
add `ffmpeg` only for the real-video route. Independent of `pixi.toml`.

Volumes:
- `awmv-anwm-weights` at `/weights`: `0200000.pth.tar` (18.1 GB), downloaded once. Keep the
  file as-is; the loader wants `logs/anwm_cdit_airvln/checkpoints/0200000.pth.tar`, so mount
  the Volume and symlink, or set `results_dir`/`run_name` in a copied config.
- `awmv-hf-cache` at `/root/.cache/huggingface`: holds `stabilityai/sd-vae-ft-ema` so it is
  fetched once.
- `awmv-anwm-data` at `/data`: only the trajectories needed, extracted from shards. Do not
  unpack 637 GB; `scripts/fetch_anwm.py` already extracts pose metadata by HTTP Range, and a
  single trajectory's missing image folder can be pulled from its shard on demand.

GPU: `A10G` or `L4` (24 GB) for a first rollout, `A100-40GB` only if batching frames.
Command shape:

```bash
modal run remote/modal_anwm_infer.py            # to be written with the rest of remote/
# inside the container, equivalently:
torchrun --standalone --nproc_per_node=1 infer.py \
  --exp config/anwm.yaml --ckp 0200000 --datasets airvln_16 \
  --eval_type rollout --rollout_fps_values 1,4
```

Budget guard: `--num_sec_eval` and a single `batch_size` keep the first run to one
trajectory; a full 1000-entry sweep is 1000 x 48 frames x 250 steps and must not be started
blind.

## 11. Excitation audit on released trajectories

`experiments/anwm_excitation_audit.py`, output `results/anwm_excitation/audit.json`.

Sample: **33 test-split trajectories** (all of the 254 test names that appear in the first
39 of 342 shards indexed; 147 trajectories found in those shards, only test-split ones
fetched). Pose keys only, 171-619 frames each. `results/anwm_excitation/inspection.json` has the per-trajectory
detail.

What the released frames depict (**verified**):

| Quantity | Value |
| --- | --- |
| roll, pitch in every frame | **exactly 0.0** (max |roll| = max |pitch| = 0.0 over all 33 trajectories) |
| camera matrix | `[[256,0,256],[0,256,256],[0,0,1]]`, i.e. 512x512 at 90 deg FOV |
| yaw | only 17 distinct values per trajectory, multiples of 15 deg (AerialVLN `TURN_ANGLE`) |
| heading change between consecutive frames | median 0 deg, p90 **107 deg**, p99 165 deg, max 180 deg; **39% of transitions exceed 30 deg** |
| metric step | 2.86-3.96 m per trajectory (mean 3.43 m), matching the config's 3.30 m normaliser |
| `timestamps` | frame indices 11..629, no time unit |

So the released data is a sequence of straight ~3.4 m segments with **instantaneous**
heading jumps up to 180 deg, rendered with a perfectly level camera.

Implied tilt under each candidate step duration (**verified** computation, `k_d` from the
real-flight floor):

| dt | SG-smoothed median | SG >10 deg | per-step median | per-step p90 | per-step >10 deg | discrepancy vs depicted (SG, with k_d) |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 s | 1.1 deg | 2.2% | **8.3 deg** | **32.6 deg** | **45.9%** | 5.0 deg |
| 0.667 s | 2.6 deg | 29.6% | 14.0 deg | 64.9 deg | 45.9% | 7.6 deg |
| 0.25 s | 17.2 deg | 80.2% | 24.6 deg | 104.2 deg | 48.3% | 24.3 deg |

Reading it:

- **Every candidate dt produces a large violation.** At the most favourable assumption
  (1 s per step) the commanded velocity changes still demand a median 8.4 deg of tilt and
  more than 10 deg at 46% of transitions, against 0 deg depicted.
- **The two estimators differ by design.** Savitzky-Golay over an 11-frame window smears a
  teleport across ten steps and understates it (1.1 deg). The per-step estimator is the
  honest reading of what the data commands, because the underlying trajectory is not a
  smooth curve at all.
- **0.25 s is the least plausible dt**: it implies a 25 deg median / 104 deg p90 velocity
  change against a level camera, and ~13 m/s cruise from 3.4 m steps. **1 s is the smallest
  dt that keeps the cruise discrepancy near 1 deg** (drag-free), which agrees with the
  paper's "5 m/s" wording and disagrees with the code's `--input_fps 4`.
- **The attitude-agreement fit is degenerate**, and this document says so rather than
  quoting a number: because depicted tilt is identically zero, the objective decreases
  monotonically with dt and runs to the grid edge (2 s). It can exclude 0.25 s; it cannot
  pin 1 s versus 2 s.
- **Testable fraction for the verifier**: ~46% of *transitions* exceed 10 deg at any dt,
  but the sustained-tilt fraction is near zero, because the motion is straight-line cruise
  between jumps. ANWM clips must be scored on transition events, and clips without turns
  must be reported as not testable.

## 12. Blockers and risks

1. **dt is undefined in the release** (section 8). Any single tilt number for ANWM is a
   statement about an assumed time base, not about the data.
2. **Depth is required.** Starting from a TII frame needs Pi-Long (or equivalent) plus
   fisheye undistortion and a `K` for the undistorted image. That is a real pipeline, not a
   one-line adapter.
3. **One scene only** in the release (scene 16), which weakens any claim about ANWM's
   general behaviour; report it as a scene-16 result.
4. **Licence text is missing in the repo** (section 1). Research use is fine; re-hosting is
   not obviously allowed.
5. **FFP may manufacture geometric consistency**, which is exactly the quantity the
   verifier measures. Needs the ablation in section 9.3.
6. **Generation cost**: 250 steps/frame is ~12,000 passes per 48-frame rollout. Pilot on one
   trajectory before any sweep.

## 13. Alternatives, if ANWM stalls

- **Aero-World** (UCF; tracker entry, no arXiv id recorded here — status **unverified**):
  conditions a video diffusion model on translational acceleration and angular velocity, so
  it is the natural should-pass contrast, and it was evaluated on UZH-FPV and TII.
- **AirScape** (weights on HF, MIT, CogVideoX-5b-I2V base): easy to run, text/intent
  conditioned.
- **AeroAct** (arXiv:2607.14997): no code or weights link found; skip for now.
