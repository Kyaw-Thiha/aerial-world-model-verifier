# E03 — dense-lean clip set, estimator calibration, and the 45-rig question closed

Track E. Written 2026-11-14. Follows `reports/E01` and `reports/E02`. All numbers here are CPU
work on the clips already exported; **no GPU was spent** (`git diff` for this report touches no
`remote/` app and no Modal volume).

Three questions, in the order they were asked:

1. does the schema defect that E02 measured actually explain Track D's statistic failing on these
   clips, and can a dense required lean fix it (the provisional "schema v2");
2. how well is the estimator *calibrated* — depicted lean against truth-attitude lean — per
   sequence, beside TII;
3. what the per-axis gains say about the 14 deg 45-rig discrepancy.

## 1. Provisional schema v2: dense required lean

Track D has still not published schema v2 (branch at `98a5dfc`; `verdict_clip.npz` carries only
`pos` and `yaw`). So each v1 clip was copied **unchanged** and augmented with
`lean_req` (N, 2), computed from the 500 Hz Leica truth with a **Savitzky-Golay window of 0.10 s,
polyorder 3** (the same window E01/E02 use for the dense path side; recorded in every clip as
`lean_req_window_s` / `lean_req_polyorder`, and in `meta.json` as `schema_version:
2-provisional`). The depicted-side arrays are untouched, so the clip set is exactly the exported
GeoCalib output plus a better path side. `experiments/e_clips_v2.py` builds it;
`experiments/e_verdict.py --path-source v2` scores from the clip file alone, which is the T6
property the v1 schema could not offer.

### Statistic (a), median axis residual at the frozen 14.73 deg

| path side | false reject | tilt removal | mirror | x0.5 | x1.5 |
| --- | --- | --- | --- | --- | --- |
| dense 500 Hz truth (E02 primary) | 0.333 | 0.881 | 0.881 | 0.476 | 0.452 |
| **v2 clip field `lean_req`** | **0.333** | **0.881** | **0.881** | **0.476** | **0.452** |

Identical to three decimals, which is the point: statistic (a) already used the dense path side,
so a better schema does not change it. **Its failure is not a path-side problem** — a fixed
absolute threshold on a median residual meets UZH's noise floor, not TII's.

### Statistic (b), Track D's frozen gain band 0.528-0.948

Run with their own code, options and band, three ways:

| clip set | testable real | flagged | median gain | removal | mirror | x0.5 | x1.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 (15 Hz `pos`), frozen window 0.37 s | 28 | 27 (**0.964**) | 0.158 | 1.00 | 1.00 | 1.00 | 0.91 |
| v1, window 0.5 s (sensitivity) | 27 | 23 (0.852) | 0.221 | 1.00 | 0.95 | — | — |
| v1, window 0.7 s (sensitivity) | 26 | 20 (0.769) | 0.299 | — | — | — | — |
| v1, window 1.0 s (sensitivity) | 17 | 13 (0.765) | 0.379 | 1.00 | 0.94 | — | — |
| **dense path side (provisional v2)** | 16 | **2 (0.125)** | **0.836** | **0.952** | **0.900** | **0.950** | **1.000** |

**The schema was the problem.** With a dense required lean the median gain goes from 0.158 to
0.836 — inside Track D's frozen TII band — the false-reject rate falls from 0.964 to **0.125**,
*within* T2's 0.15 budget, and detection stays at 0.90-1.00. A longer window alone helps
monotonically (0.964 -> 0.852 -> 0.769 -> 0.765) but plateaus far above the budget while
abstention climbs to 60 %: 15 Hz positions differentiated twice
cannot be rescued by smoothing, only by finer truth or a supplied lean.

**Read that row with three caveats, all of which cut the same way.** (i) The dense variant
interpolates the depicted lean from the 60 real frames onto the 500 Hz grid, which suppresses the
depicted side's noise and can only push the fitted gain **up** — it is an upper bound on what
schema v2 would give, not an estimate. (ii) Abstention rises to 62 % (the gain-CI bound fires on
141 of 294 files), mostly because the interpolated series is far longer and smoother than a real
15 Hz one; the surviving cells are small (16 real, 21 removal, 10 mirror), so the Wilson
intervals are wide. (iii) It is Track D's code and Track D's frozen band, unmodified, but the
*clip construction* is Track E's, which is why the field is called provisional.

The contrast with (a) is the substantive result: **a scale-free statistic (a gain) survives a
dataset whose depicted error is twice TII's, provided its regressor is clean; a fixed absolute
threshold does not.** That is an argument for the gain statistic and for the schema fix, not for
re-tuning (a).

## 2. Estimator calibration: depicted lean vs truth-attitude lean

Regression of depicted lean on **truth-attitude** lean (not the path): both are the horizontal
components of the thrust axis in the yaw-aligned frame, so the slope is "how much of the real
lean the estimator sees", with no path, clip or verdict in the loop. Same estimator, same commit,
same metric on both datasets. Residual noise in degrees; `M sv` are the singular values of the
full 2x2 map (see section 3).

### TII (8 flights, recorded uptilt as the mount)

| flight | n | slope | gx | gy | corr | residual |
| --- | --- | --- | --- | --- | --- | --- |
| 01p-ellipse | 1267 | **1.010** | 1.015 | 0.805 | 0.965 | 2.69 |
| 07p-lemniscate | 1284 | **0.937** | 0.964 | 0.788 | 0.981 | 2.58 |
| 04a-ellipse | 350 | 0.829 | 0.883 | 0.608 | 0.899 | 6.20 |
| 01a-ellipse | 270 | 0.767 | 0.834 | 0.465 | 0.836 | 8.64 |
| 16p-trackRATM | 373 | 0.656 | 0.554 | 0.644 | 0.707 | 20.79 |
| 13p-trackRATM | 536 | 0.596 | 0.454 | 0.579 | 0.652 | 21.12 |
| 07a-lemniscate | 370 | 0.221 | 0.681 | 0.171 | 0.278 | 12.05 |
| 13a-trackRATM | 557 | 0.094 | 0.245 | 0.050 | 0.113 | 17.30 |

### UZH forward rigs (variant b)

| sequence | n | slope | gx | gy | corr | residual | lean spread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| indoor_forward_9 | 383 | **0.929** | 0.800 | 1.011 | 0.991 | 1.66 | 13.1 |
| indoor_forward_10 | 410 | **0.925** | 0.819 | 1.015 | 0.991 | 1.70 | 14.0 |
| indoor_forward_7 | 450 | 0.920 | 0.914 | 0.944 | 0.993 | 2.44 | 21.8 |
| indoor_forward_5 | 264 | 0.892 | 0.784 | 1.004 | 0.983 | 0.72 | 4.3 |
| indoor_forward_3 | 450 | 0.872 | 0.814 | 0.908 | 0.912 | 4.91 | 12.5 |
| outdoor_forward_3 | 450 | 0.932 | 1.154 | 0.777 | 0.873 | 7.44 | 14.3 |
| outdoor_forward_5 | 223 | 0.854 | 1.001 | 0.692 | 0.810 | 8.36 | 13.6 |
| outdoor_forward_1 | 450 | 0.693 | 0.721 | 0.787 | 0.725 | 6.09 | 9.2 |
| indoor_forward_6 | 408 | 0.587 | 0.855 | 0.437 | 0.578 | 12.03 | 14.5 |
| **forward pooled** | 3488 | **0.842** | 0.872 | 0.843 | 0.871 | **5.07** | 13.4 |

**The estimator's calibration on forward-facing UZH is inside TII's range and at its good end**:
pooled slope 0.842 against TII's clean flights at 0.94-1.01 and its good-but-imperfect ones at
0.77-0.83. Five of the nine UZH sequences beat every TII flight except 01p and 07p. The residual
is 0.7-12.0 deg per sequence (pooled 5.07), against TII's 2.6-21.1.

Two per-sequence outliers, both already visible in E01: `indoor_forward_6` (slope 0.587,
residual 12.0) and `outdoor_forward_1` (0.693, corr 0.725).

### UZH 45-degree-down rigs (out of scope for the prototype — see section 4)

| sequence | n | slope | gx | gy | corr | residual |
| --- | --- | --- | --- | --- | --- | --- |
| indoor_45_4 | 450 | 0.694 | 0.440 | 1.044 | 0.463 | 9.71 |
| indoor_45_13 | 450 | 0.688 | 0.396 | 0.989 | 0.517 | 9.30 |
| indoor_45_2 | 450 | 0.655 | 0.472 | 0.947 | 0.474 | 10.19 |
| outdoor_45_1 | 253 | 0.225 | 0.081 | 0.734 | 0.219 | 14.11 |
| indoor_45_9 | 290 | 0.102 | -0.093 | 0.253 | 0.066 | 19.76 |

## 3. The 45-rig discrepancy: it is the estimator, not the mount

E02 left this open because a constant rotation and a view-dependent bias predict the same gravity
residual. The full 2x2 map `l_dep = M l_req + b` separates them, and no extra data is needed:

* a **mount error** rotates the lean plane: `M ~ R(delta)`, i.e. equal singular values and a
  non-zero rotation;
* a **view-dependent bias** attenuates one lean axis: unequal singular values, no rotation.

| rig | sequence | singular values | rotation |
| --- | --- | --- | --- |
| indoor_45 | _13 / _2 / _4 | (0.99, 0.42), (0.95, 0.48), (1.09, 0.47) | +2.3, +8.9, -2.4 deg |
| outdoor_45 | _1 | (0.76, 0.08) | -6.3 deg |
| indoor_forward | good sequences | (1.00-1.02, 0.78-0.82) | -1.0 to +2.6 deg |
| indoor_forward | _6 | (0.90, 0.45) | +14.8 deg |
| TII | 13a-trackRATM | (0.28, 0.04) | -19.0 deg |

The three usable indoor-45 sequences show **one lean axis attenuated to about 45 %, the other
intact, with essentially no rotation**. A rigid mount error cannot do that: it mixes the axes
rather than scaling one. The 14 deg therefore is a **view-dependent estimator bias on
45-degree-down imagery** — consistent with a "the ground plane looks level" prior dominating when
the camera looks down — and the earlier 14 deg fitted rotation was a first-order approximation to
an anisotropic map, not evidence of a calibration error.

That closes the question E02 left open, and section 4 acts on it: the 45-degree rigs are declared
out of scope for the prototype rather than tuned.

## 4. Per-axis gains per condition, and what is out of scope

Forward rigs (the population every claim applies to), gain of depicted lean against the
required lean of the v2 clips, per condition:

| rig | condition | n | slope | gx | gy | corr | corr_x | corr_y | residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| indoor_forward | real | 1080 | 0.702 | 0.682 | 0.641 | 0.745 | 0.633 | 0.771 | 9.31 |
| indoor_forward | null edit | 1068 | 0.468 | 0.208 | 0.534 | 0.487 | 0.162 | 0.673 | 12.26 |
| indoor_forward | yaw control | 1068 | 0.409 | 0.109 | 0.491 | 0.422 | 0.084 | 0.609 | 12.80 |
| indoor_forward | tilt removal | 1068 | -0.001 | -0.016 | 0.001 | -0.019 | -0.143 | 0.031 | 0.93 |
| indoor_forward | tilt x0.5 | 1068 | 0.351 | 0.250 | 0.368 | 0.674 | 0.357 | 0.906 | 5.61 |
| indoor_forward | tilt x1.5 | 1068 | 0.242 | -0.281 | 0.376 | 0.203 | -0.179 | 0.376 | 16.97 |
| outdoor_forward | real | 540 | 0.826 | 0.582 | 0.967 | 0.660 | 0.510 | 0.822 | 7.68 |
| outdoor_forward | null edit | 540 | 0.753 | 0.471 | 0.966 | 0.554 | 0.317 | 0.830 | 9.24 |
| outdoor_forward | yaw control | 540 | 0.794 | 0.543 | 0.982 | 0.548 | 0.335 | 0.808 | 9.90 |
| outdoor_forward | tilt removal | 540 | 0.137 | -0.028 | 0.285 | 0.203 | -0.037 | 0.464 | 5.42 |
| outdoor_forward | tilt x0.5 | 540 | 0.450 | 0.191 | 0.681 | 0.440 | 0.163 | 0.794 | 7.51 |
| outdoor_forward | tilt x1.5 | 540 | 0.828 | 0.374 | 1.164 | 0.491 | 0.209 | 0.808 | 11.99 |

45-degree and outdoor 45 rigs, kept separate and **out of scope** (indoor_45: slope 0.072 real,
0.079 null, 0.004 removal, correlations below 0.12; outdoor_45: negative slopes on every
condition, corr -0.24). They are reported in `results/e_calib/clips_gain.json` and nowhere used
for a claim.

Three things to read out of the forward table, and one caveat.

* **Tilt removal drives the gain to zero** (indoor -0.001, outdoor 0.137 vs real 0.70/0.83), which
  is exactly what a correct statistic must do, and the edits' x1.5 row keeps a positive gain
  (0.24/0.83) while x0.5 halves it. The statistic is measuring lean, not artefact.
* **The null edit's gain is lower than the real clip's** (0.468 vs 0.702 indoors) although the
  pixels are identical: the null edit is *cropped* (see caveat), so it costs about a third of the
  gain. The yaw control sits just below the null edit, i.e. inside its own noise.
* **`corr_y` is consistently higher than `corr_x`** on the indoor rig (0.77 vs 0.63 on real),
  the same ordering as the calibration table's `gy > gx` — the estimator is better along one lean
  axis.
* **Caveat, and it is a limitation of this task's exporter, not of the estimator**: the border
  protection is a per-frame centred crop whose size varies with the manoeuvre — 0.68 for calm
  clips, 0.38 for the most aggressive ones (`crop_fraction` in each `meta.json`). A crop to 38 %
  of each dimension keeps 14 % of the area, and the field also changes frame to frame, which
  injects variance the estimator does not deserve. That, not the edit, is why the null-edit gain
  is lower than the real gain. A v3 exporter should take one fixed crop per clip sized for its
  worst frame, and pay the field-of-view cost once.

## Which statistic the headline numbers belong to

For the avoidance of doubt, since the same clips carry two statistics:

| number in E02 | statistic |
| --- | --- |
| false reject 0.06 (indoor forward), 0.11 (outdoor forward), 0.33 (all 14) | **(a)** A2 median axis residual, drag-free path side, lag nuisance only, frozen 14.73 deg |
| detection 0.78-0.89 (per rig), 0.881 (all) | **(a)**, same threshold |
| 0.964 false reject, gain 0.158 | **(b)** Track D's `R1_scalar_g1`, frozen band 0.528-0.948 |
| 0.125 false reject, gain 0.836, detection 0.90-1.00 | **(b)** on the provisional dense clip set |

The 0.06/0.11 false-reject and 0.78/0.89 detection figures are **(a)**, and only **(a)**; they
belong to the frozen 14.73 deg threshold, not to a gain band, and they are per-rig.

## Scope statement

**The 45-degree-down rigs are out of scope for the prototype.** The estimator is not validated
there (calibration slope 0.10-0.69, one lean axis attenuated to 45 %, residual 9-20 deg), the
14 deg discrepancy is now attributed to the estimator rather than the mount, and no verdict
number is quoted for them. No further compute will be spent on that rig. Bringing it in scope
needs a depicted-tilt estimator that does not assume a horizontal ground plane, and a
pre-registered validation on that rig before any verdict.
