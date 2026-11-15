# B03 — Gate 2: the released model reproduces its own test trajectories

Track B. Written 2026-11-09. Branch `track-b-generation`.
Run: `results/anwm_native/summary.json` + per-trajectory folders with frames, predictions and
contact sheets, committed at **0efc043**; produced by `remote/b_anwm.py::validate_native`
(Modal app `awmv-b-anwm`, volume `awmv-b-results:/native`).

## Summary in plain language

Before asking whether the generated flights are physically plausible, we have to know that our
harness runs the model the way its authors intended. So we took five real test trajectories from
the published index, played the recorded movements back through the model with the recorded
camera frames and depth, and compared the pictures it drew with the pictures it should have drawn.
They are similar at a level comparable to the paper's own error numbers, they are sharp and
change over time rather than collapsing into a blur or a still image. The gate passes.

## Question and setup

Question (Gate 2): does the pipeline run end to end, are the frames not collapsed, and are the
error numbers in a plausible range?

- Data: 5 trajectories from `data/splits/airvln_16/test/rollout_16.pkl`, first entry of each
  (`curr_time = 15`), pulled from the HF shards as full pickles (depth, pose, K) plus the 32 JPEGs
  each window needs.
- Harness: ANWM's own `EvalDataset` with attributes set directly, so every method is the
  release's code — 16 context frames (report B01), actions from `_compute_actions` +
  `normalize_data` + `get_delta_np`, Future Frame Projection from the recorded depth/pose/K,
  250 sampling steps, 16 frames rolled out autoregressively with the prediction fed back.
- Metrics: LPIPS (alex weights) between prediction and the recorded frame at each step; per-frame
  pixel standard deviation against the recorded frames; mean absolute frame-to-frame difference
  ("temporal delta") against the recorded sequence.
- GPU: A10G, bf16 autocast, 7.3 GB peak.

## Results

| trajectory | LPIPS step 1 | LPIPS step 16 | LPIPS mean | min pred std | mean pred std | mean GT std | pred temporal delta | GT temporal delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3KA7IJSNXE | 0.104 | 0.252 | 0.190 | 0.284 | 0.310 | 0.302 | 0.0711 | 0.0566 |
| 3E9ZFLPWP6 | 0.138 | 0.280 | 0.216 | 0.305 | 0.321 | 0.321 | 0.1155 | 0.0853 |
| 38RHULDVA6 | 0.052 | 0.282 | 0.121 | 0.199 | 0.216 | 0.218 | 0.0304 | 0.0134 |
| 3PA41K45WV | 0.156 | 0.348 | 0.228 | 0.305 | 0.319 | 0.318 | 0.1003 | 0.0819 |
| 30EMX9PEWS | 0.079 | 0.094 | 0.093 | 0.274 | 0.281 | 0.280 | 0.0377 | 0.0242 |

Pooled LPIPS mean **0.170**. The paper's Table 1 order of magnitude for 3D generation at 4–8 s is
about 0.18–0.23; three trajectories sit above 0.19, two below 0.13. The protocol differs (their
metric split and horizon handling are not reproducible from the paper), so this is a sanity check,
**inferred** comparison, not a reproduction.

Gate criteria, each measured:

| criterion | outcome |
| --- | --- |
| runs end to end | yes: frames, LPIPS series, contact sheets, summary written |
| frames not collapsed | yes: prediction pixel spread 0.216–0.321 against ground truth 0.218–0.321 |
| frames not frozen | yes: prediction temporal delta ≥ ground truth on all five (0.030–0.116 vs 0.013–0.085) |
| numbers plausible | yes: 0.093–0.228 per trajectory, pooled 0.170 |

Convention check run in the same job: rebuilding each recorded pose as
`[R_z(yaw) @ R_ec | xyz]` from `point`/`yaw` reproduces the released `pose` array to
**0.0–2.2e-16**, with `R_ec` constant across frames to 3.3e-16. This is what allows commanded
paths to go through FFP without a learned pose model (report B04).

## What this supports

- The harness drives the released model correctly; later failures can be attributed to inputs.
- The model is not degenerate on its own data: it produces sharp, moving, plausible frames.
- The pose convention used for commanded paths is verified against the release to machine
  precision.

## What this does not show

- It does not reproduce the paper's numbers; the comparison is order-of-magnitude only.
- It does not show performance on other scenes, longer horizons, or other trajectories: five
  windows, one entry each.
- It says nothing about attitude. The recorded poses are level by construction, so agreement in
  pixels cannot confirm or refute anything about tilt.

## Caveats and open items

- `30EMX9PEWS` stays at LPIPS 0.08–0.09 across all 16 steps, i.e. the prediction barely degrades.
  Its pixel spread is normal and its temporal delta is above the ground truth, so this is not a
  collapsed clip, but a nearly constant error over 16 steps is unusual and worth a look in handoff
  04 (possible low-texture or short-sightline window).
- The comparison uses `curr_time = 15` for all five trajectories (the first entry in the index);
  the choice is recorded in `summary.json` and is not varied here.
- The LPIPS weights download (~230 MB) happens once per container; it is not part of the model
  cost.
- Open: whether a 48-frame horizon (three times the passes) degrades faster than this 16-frame
  view; not run.
