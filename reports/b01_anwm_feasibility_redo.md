# B01 — ANWM feasibility redo: what the release actually is

Track B (generation / input regimes). Written 2026-11-09. Branch `track-b-generation`.
Status: supersedes the first feasibility pass and its retraction; this is the version to cite.

## Summary in plain language

The Aerial World Model (ANWM) is a public video generator for drone flight: you give it a few
past frames, a depth map and a short list of movements, and it draws the next frames. The first
check wrongly said it was unavailable because it searched for the paper's title; the repository
is named `ANWM.code`, which no title search finds. That is corrected here.

Three things only became clear by running it:

1. The released settings file does not match the released weights. The file says the model looks
   at 4 past frames; the weights were trained on **16**. Loading with 4 fails outright. Every
   number in this handoff is therefore 4x more expensive than the settings file implies.
2. The release never says how long one movement takes. It stores distances (about 5 m per step)
   but no clock, and its simulator teleports the drone between poses. We use 1 second per step
   as a stated assumption, not a measurement.
3. Speed and turn changes are slow: on a 24 GB GPU one drawn frame takes about 18 seconds, and a
   16-frame clip about 4 minutes. That is the governing cost of every experiment here.

## Question and setup

Question: is ANWM usable as the first target of the verifier, and what exactly does the release
contain?

- Code: `https://github.com/EmbodiedCity/ANWM.code`, pinned at
  `657a80268505fa9149c4df502e35aa0f5bce11e5` (branch `master`), cloned to `third_party/anwm`
  (gitignored). **Measured**: `git rev-parse HEAD`.
- Weights: `https://huggingface.co/EmbodiedCity/ANWM`, single file `0200000.pth.tar`,
  **18,146,420,481 bytes**, sha256 `bdd149cac6ec002ba7dc4ad99ec6f9eb02cd6d4f05320195cf174737b13b0bc2`.
- Dataset: `EmbodiedCity/ANWM-Dataset`, 342 tar shards, 637.45 GB, scene 16 only.
- Image: CUDA 12.1.1 base, Python 3.10, torch 2.4.1+cu121, diffusers 0.33.1,
  transformers 4.44.2, peft 0.13.2, the pinned clone installed editable with `.[metrics]`.
  Every pin was forced by a failure, recorded in `docs/decisions.md`.
- Runs: Modal app `awmv-b-anwm`, volumes `awmv-b-{weights,hf-cache,native,results}`. All measured
  timing below is from `remote/b_anwm.py::smoke` at commit **9a5e19f**, output
  `awmv-b-weights:/smoke.json` (the same numbers are quoted in `docs/anwm_feasibility.md`
  section 6).

## Results

### What the release contains (**measured** unless marked)

| item | value | how known |
| --- | --- | --- |
| output size | 224x224, one frame per action | config + code |
| backbone | CDiT-XL/2, 28 blocks, hidden 1152, 16 heads, 4 latent channels | code |
| parameters | 1.1341 B (EMA weights) | `sum(p.numel())` in the smoke run |
| sampling steps | 250 per frame | `create_diffusion(str(250))` |
| context frames | **16** (checkpoint `pos_embed` is `[17, 196, 1152]`) | load test, commit 9a5e19f |
| context frames, as shipped in `config/anwm.yaml` | 4 — **wrong for the released weights** | load fails with 4 |
| action vector | `(dx, dy, dz, dyaw)` in the agent frame, translation divided by 3.30 m | `_compute_actions` |
| training steps in checkpoint | 200,000, epoch 3, bf16 | `ckp["args"]` via `inspect_args` |
| time per CDiT pass | **66.5 ms** (warm, min of 3; first call 1.40 s) | smoke, A10G |
| time per generated frame | **17.57 s** | smoke, A10G |
| time per 16-frame rollout | **281 s** (measured 235 s end-to-end in generation) | smoke + generation logs |
| peak GPU memory | **7.3 GB** (forward 7.06) | smoke, A10G |

### First run versus revised

| claim | first run | revised | why it changed |
| --- | --- | --- | --- |
| availability | "no public repo, no weights, no data" | available: repo `ANWM.code`, weights, 637 GB dataset | title search could not match the repo name |
| per-step duration | 0.667 s inferred from 48-frame/32 s horizons | **undefined in the release**; 1 s assumed, 0.25 s excluded by the frames | no time base exists in the data (teleport simulator) |
| context frames | 4 (from the shipped config) | **16** from the weights | `pos_embed` size mismatch |
| inference cost | "single-digit to low-double-digit minutes per 48-frame rollout" (**inferred**) | 281 s per 16 frames, ~14 min per 48 frames at that rate (**measured**) | first smoke run |

## What this supports

- ANWM is available and runnable end to end on a 24 GB card; a 16-frame clip costs about 4 minutes.
- A single scene (16) and a single checkpoint define the model's training distribution, so every
  result here is a "scene 16" result.
- The action interface accepts continuous values, but the model has only seen 5 m / 2 m / 15 deg
  steps; the training envelope is the constraint on any designed input.

## What this does not show

- It does not show that 1 s per step is correct. It is an assumption consistent with the frames,
  the paper's "5 m/s" wording and a roughly 8.6 deg drag tilt at cruise; it is **not measured**.
- It does not measure quality. Gate 2 (report B03) measures agreement on native trajectories only.
- It says nothing about scenes other than 16, or about any other checkpoint.

## Caveats and open items

- **Licence**: the repository has no licence text of its own; the weights and dataset cards are
  Apache-2.0 and 13 files carry Meta copyright headers. Research use is fine; re-hosting should be
  cleared with the authors.
- **Context-size mismatch** is a release bug, not a misreading: the config and the weights
  disagree. Any pipeline written against the config fails on `pos_embed`.
- **Cost rate** (US$1.10/h for an A10G) is an unverified price assumption; the GPU-hours are
  measured.
- Open: confirm the intended per-step duration with the authors; confirm whether the released
  checkpoint was trained on more than the published shard set.
