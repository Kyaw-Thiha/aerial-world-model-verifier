# Report F02 — Stage 1: other path extractors on UZH-FPV

Track F. Written 2026-11-16. Report `F01` decided stage 0: the pixel-path statistics work,
the path does not. This report is the stage-1 table for that question — can another extractor
produce a path the same statistics can use? — and it extends E7's bake-off with the three
systems Track C never ran.

**Decision: the bake-off is answered, the gate is not moved.** ViPE passes Track C's
bake-off 10/10 without the Kalibr intrinsics and beats DPVO's median direction error
(4.0 vs 4.2 deg), yet the stage-0 statistics computed on ViPE's path are no better —
false reject 2/8 (worse than DPVO's 1/8), removal 6/8, mirror 4/8 — so F01's gate still
fails. MASt3R-SLAM passes on its first sequence at a much worse direction error (16.5 deg)
and is being run on the remaining nine; MegaSaM's weights are Drive-only and it was not run.

## Plain-language summary

Stage 0 found that the shooting statistic detects 6/8 removals and 4/8 mirrors with DPVO's
path, against 8/8 and 7/8 with the truth path. So the question is no longer whether the
statistic can work but whether any recovered path is good enough. This stage runs three
systems that were not in the original bake-off on the same ten UZH sequences and scores them
with the same metric, then feeds the best into the stage-0 tests unchanged.

## Systems and what is given to each

| system | version / source | intrinsics | notes |
| --- | --- | --- | --- |
| DPVO (reference) | Track C bake-off, `poses/dpvo` | Kalibr, supplied | 150 frames, 15.07 Hz |
| ViPE | `nvidia-vipe` 1.2.0, default pinhole pipeline | **estimated** (SharedK) | run without the Kalibr K on purpose: C07 showed the supplied calibration is what separates DPVO from the feed-forward extractors |
| MASt3R-SLAM | `rmurai0610/MASt3R-SLAM`, ViT-L weights | Kalibr, supplied on `--calib` | the SLAM system for this stage |
| MegaSaM | `mega-sam/mega-sam` | — | **not run**: its camera-tracking weight (`raft-things.pth`) is distributed only through a Google Drive folder, which a container cannot fetch non-interactively. Recorded as a blocker, not worked around |

The images are built by `remote/f_vipe.py` and `remote/f_mast3r.py`; both build the CUDA
extensions with `TORCH_CUDA_ARCH_LIST=8.0` because no GPU is visible during a Modal image
build. Poses are written to the `awmv-f-data` volume and pulled locally for scoring.

## Scoring

Exactly Track C's metric (`src.pathrec.evaluate.score_clip`): Sim(3) alignment for evaluation
only, acceleration direction error after Savitzky–Golay differentiation at a 0.5 s window,
magnitude ratio, coverage (fraction of samples whose true required tilt is at least 10 deg),
and the pre-registered pass rule `direction <= 30 deg`, `ratio in [0.7, 1.4]`, per-clip
coverage `>= 0.2`, with a model passing at a `>= 0.5` per-clip pass rate
(`configs/pathrec_prereg.yaml`).

## Results — the bake-off table, extended

Run `results/f02_stage1/scores.json`; code commit `06161c8`, extracted with
`experiments/f02_stage1.py` from the poses on `awmv-f-data` (sha256 in each run's JSON).
Same clips, same metric, same pass rule as E7.

| extractor | sequence | direction error (deg) | magnitude ratio | coverage | ATE Sim(3) (m) | pass |
| --- | --- | --- | --- | --- | --- | --- |
| dpvo | indoor_forward_3 | 1.8 | 0.98 | 0.76 | 0.15 | yes |
| dpvo | indoor_forward_5 | 2.5 | 1.00 | 0.53 | 0.14 | yes |
| dpvo | indoor_forward_6 | 3.1 | 0.95 | 0.85 | 1.04 | yes |
| dpvo | indoor_forward_7 | 20.9 | 0.98 | 0.86 | 2.21 | yes |
| dpvo | indoor_forward_9 | 1.5 | 1.00 | 0.65 | 0.10 | yes |
| dpvo | indoor_forward_10 | 2.3 | 1.00 | 0.62 | 0.16 | yes |
| dpvo | outdoor_forward_1 | 6.9 | 0.94 | 0.49 | 0.82 | yes |
| dpvo | outdoor_forward_3 | 5.2 | 0.92 | 0.61 | 0.75 | yes |
| dpvo | outdoor_forward_5 | 11.2 | 1.15 | 0.84 | 3.32 | yes |
| dpvo | indoor_45_9 | 13.2 | 0.73 | 0.90 | 4.99 | yes |
| **dpvo total** | **10/10** | **4.2** | **0.98** | | | |
| vipe | indoor_forward_3 | 2.5 | 1.04 | 0.76 | 0.63 | yes |
| vipe | indoor_forward_5 | 3.2 | 0.97 | 0.53 | 0.25 | yes |
| vipe | indoor_forward_6 | 4.7 | 1.02 | 0.85 | 0.51 | yes |
| vipe | indoor_forward_7 | 2.4 | 0.85 | 0.86 | 0.52 | yes |
| vipe | indoor_forward_9 | 2.1 | 1.00 | 0.65 | 0.67 | yes |
| vipe | indoor_forward_10 | 2.3 | 0.99 | 0.62 | 0.27 | yes |
| vipe | outdoor_forward_1 | 5.4 | 0.92 | 0.49 | 1.79 | yes |
| vipe | outdoor_forward_3 | 9.9 | 1.19 | 0.61 | 3.46 | yes |
| vipe | outdoor_forward_5 | 12.0 | 1.11 | 0.84 | 4.42 | yes |
| vipe | indoor_45_9 | 6.3 | 0.89 | 0.90 | 1.70 | yes |
| **vipe total** | **10/10** | **4.0** | **0.99** | | | |
| mast3r | indoor_forward_3 | 95.1 | 0.00 | 0.76 | 7.69 | no |
| mast3r | indoor_forward_5 | 16.0 | 0.81 | 0.53 | 0.26 | yes |
| mast3r | indoor_forward_6 | 62.1 | 0.00 | 0.85 | 7.59 | no |
| mast3r | indoor_forward_7 | 12.2 | 0.98 | 0.86 | 0.51 | yes |
| mast3r | indoor_forward_9 | 15.2 | 0.93 | 0.65 | 0.18 | yes |
| mast3r | indoor_forward_10 | 66.5 | 0.81 | 0.62 | 2.80 | no |
| mast3r | outdoor_forward_1 | 7.9 | 0.97 | 0.49 | 0.78 | yes |
| mast3r | outdoor_forward_3 | 11.9 | 0.85 | 0.61 | 1.66 | yes |
| mast3r | outdoor_forward_5 | 74.2 | 0.00 | 0.84 | 17.41 | no |
| mast3r | indoor_45_9 | 93.9 | 0.00 | 0.90 | 7.35 | no |
| **mast3r total** | **5/10** | **39.0** | **0.81** | | | |


**ViPE passes 10/10 with its own estimated intrinsics**, a median direction error of 4.0 deg
against DPVO's 4.2, a median magnitude ratio of 0.99 against 0.98, and it is far better on the
two sequences where DPVO is weakest: `indoor_forward_7` 2.4 vs 20.9 deg and `indoor_45_9`
6.3 vs 13.2 deg. Its Sim(3) ATE is larger on several sequences (0.63 vs 0.15 m on
indoor_forward_3, 3.46 vs 0.75 on outdoor_forward_3) while its direction and magnitude are
equal or better — the same dissociation C07 saw between trajectory error and acceleration
direction. Since ViPE is **not** given the Kalibr calibration, that is the strongest form of
the result C07 framed: with modern streaming SLAM, supplied intrinsics are no longer the
dividing line.

**MASt3R-SLAM passes 5/10** at a median direction error of 39.0 deg, with four of its five
failures showing a collapsed scale (magnitude ratio 0.00) on sequences DPVO and ViPE both
handle. Its folder loader timestamps frames as `index/30` regardless of the real rate, so the
first attempt mapped a 10 s clip onto 4.3 s; the run reported here uses the timestamps as
frame indices (`round(t*30)`), which is what they are. It also subsamples to every fourth
frame on some sequences (kept step 4 on `indoor_forward_5`) and every frame on others, so its
tracking density is not constant across sequences.

MegaSaM: not run. Its camera-tracking weight is distributed only through a Google Drive
folder, which a container cannot fetch non-interactively; the alternative would have been a
manual download, and the project's reproducibility rule is that a number must be regenerable
from the repository.

## The stage-0 statistics on ViPE's path

Re-run of `experiments/f01_stage0.py` with `--poses data/pathrec/poses/vipe` and nothing else
changed (`results/f02_stage1/stage0_vipe/`), same 8 real clips in 4 sequences, same
leave-one-sequence-out conformal rule:

| variant | real | removal | mirror | x1.5 | AUC removal | AUC mirror |
| --- | --- | --- | --- | --- | --- | --- |
| V1 preintegration | 0.125 (1/8) | 0.000 (0/8) | 0.625 (5/8) | 0.875 (7/8) | 0.75 | 0.80 |
| **V2 shooting** | **0.250 (2/8)** | **0.750 (6/8)** | **0.500 (4/8)** | 0.750 (6/8) | 0.94 | 0.80 |
| V3 band coherence | 0.250 (2/8) | 0.250 (2/8) | 0.000 (0/8) | 0.125 (1/8) | 0.64 | 0.50 |
| V4 velocity-drag | untestable | untestable | untestable | untestable | -- | -- |
| V5 physics scale | 0.000 (0/8) | 0.125 (1/8) | 0.000 (0/8) | 0.000 (0/8) | 0.63 | 0.50 |

**ViPE does not move the gate.** The false-reject rate gets worse (2/8 against 1/8) and
detection is unchanged at 6/8 and 4/8. The two ViPE false rejects are both
`indoor_forward_9` clips (0.320 and 0.469 against the 0.262 held-out threshold); with DPVO,
`indoor_forward_9#00` sat at 0.120. So on that sequence ViPE's path is *better* by the
bake-off metric (2.1 deg direction error, 0.67 m ATE) and *worse* for the verdict, which
needs the path's shape at the frequencies the depicted lean excites rather than the median
acceleration direction. Rank AUCs confirm the direction of the change: removal 0.94 vs 0.95,
mirror 0.80 vs 0.89.

The gate therefore still fails with the best extractor available, and the failure is not a
calibration artefact: ViPE never saw the Kalibr K.

MASt3R-SLAM was not fed into the stage-0 tests: the pre-registration says the best one or two,
and it fails the bake-off on half the sequences with a collapsed scale. Feeding a path that
fails the extractor's own criterion would answer a different question.

## What this supports

* **The extractor question is separable from the verdict question.** ViPE is the best path
  on the bake-off's terms — 10/10, median 4.0 deg, without the calibration — and it makes
  the verdict slightly *worse* (FRR 2/8 vs 1/8, mirror AUC 0.80 vs 0.89). A path metric built
  on the median acceleration direction does not rank paths for this verdict.
* **The path is still the binding error, but not because extractors are bad at direction.**
  Both DPVO and ViPE recover direction to a few degrees, and both leave a residual of 0.09 to
  0.47 in the shooting statistic on `indoor_forward_9`, where the truth path gives 0.01.
  What the verdict needs is shape agreement at the frequencies the depicted lean excites, and
  that is a different, tighter requirement than a median direction error.
* **The calibration is no longer the dividing line** for streaming SLAM on UZH: ViPE matches
  DPVO without being given the Kalibr K, which weakens C07's "the calibration is the
  difference" as a statement about modern extractors (it remains true of the feed-forward
  batch models in the original table).
* **A SLAM system with keyframe subsampling is not automatically better**: MASt3R-SLAM at
  every-fourth-frame density is worse than both streaming trackers on this footage.

## What this does not show

* That no path can pass the gate: only these three extractors at 15 Hz/150 frames were tried.
  A denser run (30 Hz, which C's OOM blocked) or a metric-scale method could still change it.
* Why `indoor_forward_9` is the ViPE failure: its ATE is 0.67 m and direction 2.1 deg, so the
  failure is in the path's fine shape, and this report does not localise it further.
* Anything about ANWM (report F03) or about the generated-video verdict.

## Caveats

* ViPE estimates its own intrinsics, so a ViPE pass would be a stronger claim than a DPVO pass
  and a ViPE failure is ambiguous between calibration and architecture.
* MASt3R-SLAM is a dense SLAM system with keyframe selection; a short 10 s clip may not give
  it the keyframe graph its retrieval and loop closure were designed for.
* MegaSaM is absent for an access reason, not for a measured one.
* Cost is recorded in `docs/COST_LEDGER.md`; the estimate was posted before the job set.
