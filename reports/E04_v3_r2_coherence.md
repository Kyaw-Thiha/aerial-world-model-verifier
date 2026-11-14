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

### 2.2 Populations actually scored (this matters for every number below)

| step | clips | note |
| --- | --- | --- |
| exported, 9 forward sequences x 3 clips x 7 conditions | 189 | v3 fixed crop |
| dropped, crop width < 25 % of the frame | **-42** | 6 clips: `indoor_forward_3#02`, `indoor_forward_6#00/#01/#02`, `outdoor_forward_5#00/#01` — the mirror-roll edit at 60-70 deg of bank is a 120-140 deg rotation, which leaves a sliver of the source frame. Not scorable, not shrunk into the statistics |
| scored | 147 | |
| **R2 / coherence, testable real clips** | **9 clips, 4 sequences** (indoor_forward_3, _5, _9, _10) | outdoor contributes **zero**: the frozen gate abstains on every outdoor clip |
| **(a) median residual, testable real clips** | **18 clips, 7 sequences** (4 indoor + 3 outdoor) | (a) does not use the gate |
| excluded | `indoor_forward_7` (3 clips, 21 files) | its v3 export is broken — one of 60 frames evaluated, real clips score 13.6/30.6/44.5 deg on a sequence whose per-frame error is 1.66 deg. It fails the validity gate (1.13 deg lean) and its clips then fail the ≥12-signal-frame rule, so they never enter a denominator. Recorded as an export failure, not a result |

**Seven sequences reach the statistic** (indoor_forward_3, _5, _9, _10; outdoor_forward_1, _3, _5):
`indoor_forward_6` lost all three clips to the insufficient-field rule and `indoor_forward_7` is
excluded. So the headline is **indoor forward, 4 sequences, 9 testable real clips** for Track D's
statistics and **7 sequences, 18 clips** for the median residual. Those counts are repeated beside
every number.

### 2.3 Track D's statistics and the median residual, forward rigs, 0.37 s headline

Every rate is printed with its two controls. A "detection" is a flag on an edited clip; a "false
reject" is a flag on a real clip; the null-edit and yaw-control rates are the artefact floor that
has to be read beside it.

| statistic | real (false reject) | null edit | yaw control | tilt removal | mirror | x0.5 | x1.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **(b1) R2 axis-gain band** (tau 2.1922) | **0.556** (5/9) | 0.556 (5/9) | 0.286 (2/7) | 1.000 (10) | 1.000 (8) | 1.000 (10) | 1.000 (8) |
| **(b2) coherence tier** | **0.111** (1/9) | 0.111 (1/9) | 0.143 (1/7) | **1.000** (10) | **1.000** (8) | 0.200 (10) | 0.250 (8) |
| **(a) median residual, 14.73 deg** | **0.000** (0/18) | 0.000 (0/18) | 0.000 (0/18) | **0.722** (13/18) | **0.778** (14/18) | 0.056 (18) | 0.167 (18) |

Sequence-clustered 95 % intervals over the 7 scored sequences (9 forward, minus the one whose export
is excluded, minus the one that lost every clip to the field rule):

| statistic | real | tilt removal | mirror |
| --- | --- | --- | --- |
| R2 band | 0.50-0.63 | 1.00-1.00 | 1.00-1.00 |
| coherence | 0.00-0.375 | 1.00-1.00 | 1.00-1.00 |
| median residual | 0.00-0.00 | 0.10-1.00 | 0.60-1.00 |

**Three readings, and they disagree, which is the result.**

1. **The gain band does not transfer** (false reject 0.556, clustered interval 0.50-0.63). Its
   detection of 1.000 is therefore uninformative: everything is flagged. The mechanism is visible
   in the calibration: UZH per-clip axis gains spread far wider than TII's, and the band's `gy`
   scale is only 0.0735, so a clip whose `gy` is 0.1 away from centre is already flagged. A band
   fitted on 57 TII clips cannot carry a dataset whose per-clip gain spread is an order of
   magnitude larger.
2. **The coherence tier transfers** (false reject 0.111, inside the 0.15 budget; detection 1.000
   for removal and 1.000 for mirror, with both controls at 0.111-0.143). This is the scale-free
   statistic doing what it was designed for, and it is the only one of the three that meets T2's
   requirement on this data.
3. **The median residual** has a perfect false-reject rate (0/18) but detection 0.722/0.778 — just
   under the 0.80 bound, with a clustered interval as wide as 0.10-1.00 for removal because 13 of
   18 clips carry it and they sit in 4-5 sequences.

`x0.5` and `x1.5` sit at 0.20/0.25 under coherence, i.e. at chance, exactly as Track D's own note
predicts ("a positive scale leaves the correlation unchanged"). They are reported, not tuned.

### 2.4 Window sensitivity (pre-registered; the headline stays 0.37 s)

| window | R2 real | coherence real | coherence removal | coherence mirror | (a) real | (a) removal | (a) mirror |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **0.37 s (headline)** | **0.556** | 0.111 | 1.000 | 1.000 | 0.000 | 0.722 | 0.778 |
| 0.185 s | 0.444 | 0.111 | 1.000 | 1.000 | 0.000 | 0.722 | 0.778 |
| _same population check_ | all three windows score **126 clips over 7 sequences**, 21 files excluded, 42 dropped | | | | | | |
| 0.10 s | **0.000** | 0.111 | 1.000 | 1.000 | 0.000 | 0.722 | 0.778 |

**The gain band is the only statistic that moves with the window**, and it moves monotonically from
failing to passing as the window shortens. The coherence tier and the median residual are
invariant. That is a property of the gain: it is a *ratio* of depicted to required variation, so
smoothing the required side changes its scale, while coherence (a correlation) and the residual (an
angle) do not care. The pre-registered headline is 0.37 s and is not swapped for the favourable
0.10 s; the sensitivity is reported as the finding that Track D's band is not window-robust on this
dataset.

### 2.5 Abstention, by Track D's reason order (real clips, forward, final population)

14 real clips reach the scorer, **7 abstain (0.500)**:

| reason | clips |
| --- | --- |
| confidence gate | **5** (all outdoor) |
| `gain_ci_half_width` > 0.35 | 6 |

(counts overlap: one clip can fail both). On the earlier population, which still contained the
broken `indoor_forward_7` clips, four further reasons fired — `usable_fraction`, `signal_frames`,
`excitation`, `condition_number` and the bootstrap bound — three clips each. **Every one of those
was the broken export**, which is why they vanish once that sequence is excluded: 21 files with one
evaluated frame cannot carry 12 signal frames or a conditioned design matrix. Recording both sets
rather than only the clean one, because the difference is diagnostic.

**The gate alone removes the entire outdoor rig** — 5 of the outdoor real clips, with the rest
failing `gain_ci_half_width`. E01 predicted this from the per-frame data (the gate drops 94 % of
outdoor clips); at clip level it is total. So the "outdoor forward" column of every Track D statistic
in this report is empty by construction, and the outdoor rig is measured only by the median residual,
which does not use the gate.

### 2.6 The edit family, per rig

| rig | clips scored | validity gate | R2 false reject | coherence false reject | (a) false reject |
| --- | --- | --- | --- | --- | --- |
| indoor forward | 98 | **passes on 4 of 5 sequences** (indoor_forward_7 excluded, its export broken) | 0.556 | 0.111 | 0.000 |
| outdoor forward | 49 | **fails on both sequences tested** (gain 5.0 % and 8.5 % against the 5 % bound; lean 0.12-0.18 deg) | n/a (gate abstains) | n/a | 0.000 |

The outdoor validity failure is gain-only and its lean agreement is 0.12-0.18 deg, so the edits are
faithful and the gain estimate is noisy; but the pre-registered rule is the rule, so **no detection
claim is made for outdoor** anywhere in this report.

## 3. Reporting contract

Every rate in §2 is printed with the statistic it belongs to ((b1) R2 axis band, (b2) coherence,
(a) median residual), the rigs and counts it covers, the matching control rates (real,
null edit, yaw control), abstention by Track D's reason order, and a sequence-clustered interval
over the 9 forward sequences. No threshold is refit.
