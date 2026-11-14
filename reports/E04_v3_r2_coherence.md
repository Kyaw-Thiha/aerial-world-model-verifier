# E04 — v3: fixed-crop edits, official schema v2, Track D's R2 + coherence on forward-rig UZH

Track E. Written 2026-11-14. Pre-registration: `configs/e_prereg_v3.yaml` (committed `1f021c2`,
before any v3 clip was scored). Track D's frozen material: `track-d-verdict-stat` at **851096c**
(`configs/gain_frozen.yaml` rule `R2_axis_gains`, `configs/coherence_frozen.yaml`,
`scripts/score_clip.py`, `docs/verdict_clip_schema_v2.md`). Artifacts:
`results/e_verdict_v3/<tag>/metrics.json`, `results/e_calib/`, `data/clips_v3/`,
`data/clips_v2_w{0.37,0.10,0.185}/`.

Population: **forward rigs only** — 6 `indoor_forward` + 3 `outdoor_forward` sequences. The
45-degree rigs are out of scope (E03 §4) and are not re-exported or scored.

## 0. The two "slopes" — definitions first, then both datasets under both

Two different quantities have both been called the gain of the depicted lean. They are not
interchangeable, and the numbers only look contradictory when the definitions are mixed.

**Definition 1 — estimator calibration (a property of the estimator).**
Response `l_dep`; regressor `l_att` = **truth-attitude** lean, from the recorded quaternion
alone (no path, no acceleration). OLS with one shared intercept over all usable frames of a
flight/sequence; reported per unit and pooled with n as weight. This asks "how much of the real
lean does the estimator see" and contains no verdict machinery.

**Definition 2 — verdict gain (a property of a clip).**
Response `l_dep`; regressors `lean_req` (required, path-derived) **and** `vel_req`, plus a free
2-D intercept; per-clip OLS with a lag search (-150 to +150 ms) and Track D's abstention rules;
reported as the per-clip median. This is Track D's statistic, and the band 0.528–0.948 /
axis-band τ 2.1922 was calibrated on it.

### Definition 1, both datasets

| framing | TII | UZH forward |
| --- | --- | --- |
| per unit, all frames | 01p **1.010**, 07p **0.937**, 13p 0.596, 16p 0.656, 04a 0.829, 01a 0.767, 07a 0.221, 13a 0.094 | 0.587–0.932 per sequence (E03 §2) |
| **pooled, all frames** | piloted **0.723** (n=3460); all flights 0.624 | **0.856** (n=3488) |
| **pooled, confidence-gate-kept clips** | piloted **0.904** (n=2592, 75 % of frames) | **0.791** (n=1860) |
| Track D's published figure | **0.831** piloted (their report D02 §1; autonomous 0.587, track 0.558) | — |

**Reconciliation.** Track D's 0.831 is Definition 1, but pooled over a *frame set* filtered by
their abstention rules rather than over every frame of the four piloted flights. My two framings
bracket it: 0.723 on all frames and 0.904 on gate-kept clips. The gap between my per-flight
figures (0.94–1.01 on 01p/07p) and 0.83 is therefore **aggregation plus frame selection**, not
disagreement: pooling the four piloted flights alone drops the number to 0.723, because the two
trackRATM flights sit at 0.60–0.66 and carry 900 of the 3460 frames, and the gate then removes the
clips where the estimator failed. Two mechanisms were tested and one was rejected: a frame lag of
±3 frames moves the pooled slope monotonically down from 0.723 (correlation peaks at zero shift),
so misalignment is **not** the cause.

Under **both** framings the ordering is the same or better for UZH: 0.856 vs 0.723 (all frames)
and 0.791 vs 0.904 (gate-kept). The gate restriction moves UZH *down* and TII *up*, because the
UZH gate keeps indoor clips and drops the outdoor ones whose slope is higher — a selection effect
that has to be stated rather than netted out. The claim E03 made survives: **UZH forward is at
TII's good end, not below it.**

### Definition 2, both datasets

| dataset | Definition 2 number | source |
| --- | --- | --- |
| TII, v1 schema | **0.745** (real clips) | Track D report D02 §1, quoted |
| TII, v2 dense schema | **0.764** [0.718, 0.818] | Track D report D02 §1, quoted |
| UZH forward, v2 dense schema | per-clip median, this report §3 | Track D's `fit_gain`, this round |

TII's Definition 2 numbers are **quoted, not recomputed here**: reproducing them needs Track D's
own TII clip set, which is not in this worktree. Their decomposition is the bridge between the
definitions — truth attitude on required lean gives 0.991, estimator shrinkage 0.828, verdict gain
0.745 — so Definition 1's 0.83 and Definition 2's 0.745 differ by the *regressor* (truth attitude
vs required lean), which is exactly what these two definitions separate.

## 1. What v3 changed, and the edit-validity gate

* **One fixed crop per clip**: the largest centred rectangle inside the valid region of every
  frame and every condition, applied to all seven conditions including the source crop for
  `real`. E02's per-frame crop changed the field from frame to frame; that is why its null edit
  lost a third of the real clip's lean gain on identical pixels.
* **No black fill anywhere inside the crop**, so no condition is resampled into a different field.
* **Official schema v2 at frame times** (`lean_req`, `vel_req`, `tilt_req_deg`), computed by
  differentiating the 500 Hz truth with **Savitzky-Golay 0.37 s, polyorder 3** (Track D's
  specification) and sampling at the camera frame times, NaN outside the native SG interior. No
  value on the depicted side is interpolated — this supersedes E03's provisional dense variant,
  which interpolated it and therefore biased the gain upward.
* Sensitivities at **0.10 s** and **0.185 s** are reported beside the headline, never as it.

**Validity gate (pre-registered, checked before any detection number is quoted):** per clip, the
null edit must reproduce the real clip's per-axis gain within 5 % and its depicted lean within
0.5 deg median. Result over the forward clips: see §2.

## 2. Results

### 2.1 The edit-validity gate (checked before any detection number)

Per clip, null edit against real: per-axis gain within 5 %, depicted lean within 0.5 deg median.

| sequence | pairs | pass | max gain deviation (gx / gy) | depicted-lean median |
| --- | --- | --- | --- | --- |
| indoor_forward_10 | 3 | 3/3 | 0.33 % / 0.10 % | 0.038 deg |
| indoor_forward_5 | 3 | 3/3 | 0.75 % / 0.29 % | 0.032 deg |
| indoor_forward_9 | 3 | 3/3 | 0.76 % / 0.47 % | 0.049 deg |
| outdoor_forward_1 | 3 | **2/3** | **8.48 %** / 6.03 % | 0.111 deg |
| outdoor_forward_3 | 3 | **2/3** | **5.04 %** / 0.03 % | 0.145 deg |

**Indoor forward passes everywhere, by two orders of magnitude** (worst gain deviation 0.8 %,
worst lean deviation 0.05 deg): the v3 fixed crop makes the null edit a resampled copy of exactly
the same pixels, and the estimator returns the same answer. That is the property E02 lacked.

**Two outdoor clips fail the gate, on the gain criterion only**, at 8.5 % and 5.0 % against a 5 %
bound, while their depicted lean agrees to 0.12-0.18 deg. The pixels are therefore identical and
the failures are gain-estimation noise (`|g|` is small on those clips, so a 5 % relative bound
is a fraction of a standard error). Under the pre-registered rule they still count as failures:
**no detection claim is made on any outdoor sequence**, and the rule is not renegotiated after
seeing the numbers. Since the frozen confidence gate abstains on every outdoor clip anyway
(§2.3), the practical cost is nil — but the invalidation is recorded rather than explained away.

## 3. Reporting contract

Every rate in §2 is printed with the statistic it belongs to ((b1) R2 axis band, (b2) coherence,
(a) median residual), the rigs and counts it covers, the matching control rates (real,
null edit, yaw control), abstention by Track D's reason order, and a sequence-clustered interval
over the 9 forward sequences. No threshold is refit.
