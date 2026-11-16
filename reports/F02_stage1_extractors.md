# Report F02 — Stage 1: other path extractors on UZH-FPV

Track F. Written 2026-11-16. Report `F01` decided stage 0: the pixel-path statistics work,
the path does not. This report is the stage-1 table for that question — can another extractor
produce a path the same statistics can use? — and it extends E7's bake-off with the three
systems Track C never ran.

**Decision: pending completion of the job set.** The pre-registered gate this report serves
is F01's: with estimated attitude and the pixel path, false-reject at most 15 % and removal
and mirror detection at least 90 %. A new extractor "passes" here only if it meets Track C's
bake-off criteria *and* lifts the stage-0 statistics over that bar; the second condition is
the one that matters.

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

## Results

*(filled from `results/f02_stage1/scores.json`)*

## The stage-0 statistics on the new paths

*(filled from the re-run of `experiments/f01_stage0.py` with `--poses` pointing at the new
extractor; same population, same thresholds, same edits)*

## What this supports

*(filled after the results)*

## Caveats

* ViPE estimates its own intrinsics, so a ViPE pass would be a stronger claim than a DPVO pass
  and a ViPE failure is ambiguous between calibration and architecture.
* MASt3R-SLAM is a dense SLAM system with keyframe selection; a short 10 s clip may not give
  it the keyframe graph its retrieval and loop closure were designed for.
* MegaSaM is absent for an access reason, not for a measured one.
* Cost is recorded in `docs/COST_LEDGER.md`; the estimate was posted before the job set.
