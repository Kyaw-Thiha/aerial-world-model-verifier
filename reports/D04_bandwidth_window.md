# Report D04 — Label-free acceleration-bandwidth window on TII and UZH

Written 2026-11-14. Pre-registration `configs/window_bandwidth_prereg.yaml` (commit
`c7af793`, before any TII or UZH scoring). Runs: `results/d_bandwidth/d01/`,
`results/d_uzh_v2/`. The frozen gain and coherence configs were **not** changed; the
proposal is `configs/window_bandwidth_amendment_proposal.yaml`.

## Rule

`window = c / f95`, with `c = 0.5` fixed in advance (half the period of the highest
acceleration component carrying 95 % of the energy; it reproduces the Phase 0 0.11 s
reference for a 4.5 Hz manoeuvre). `f95` comes only from `pos`/`t`: Savitzky-Golay
acceleration at a fixed 0.10 s / poly 3, rFFT, energy summed over axes, DC excluded, first
frequency at 95 % cumulative energy. The window must contain **8 samples at the rate used
to compute the required lean**; if not, or if it exceeds half the clip duration, the clip
abstains. There is no clamp and no lengthening to the sample rate.

## TII (dense truth, LOFO bands)

57 of 100 real clips testable (43.0 % abstain, same as the frozen statistic). R2 axis
rule: **false-reject 5.3 %**, removal **100 %**, mirror **100 %**, x0.5 **100 %**, x1.5
**96.5 %**, yaw control **29.8 %**. That is the frozen 0.37 s performance, selected per
clip from the path alone.

| flight | real clips | median window s | min | max | median samples (275 Hz) |
| --- | --- | --- | --- | --- | --- |
| 01a-ellipse | 5 | 0.103 | 0.056 | 0.490 | 28 |
| 01p-ellipse | 26 | 0.178 | 0.098 | 0.280 | 49 |
| 04a-ellipse | 7 | 0.063 | 0.038 | 0.392 | 17 |
| 07a-lemniscate | 7 | 0.107 | 0.029 | 0.329 | 30 |
| 07p-lemniscate | 26 | 0.218 | 0.065 | 0.327 | 60 |
| 13a-trackRATM | 11 | 0.050 | 0.012 | 0.218 | 14 |
| 13p-trackRATM | 11 | 0.218 | 0.115 | 0.281 | 60 |
| 16p-trackRATM | 7 | 0.219 | 0.123 | 0.247 | 60 |

Track flights remain **not validated**: 3 testable real clips under the frozen set; 0/11
on 13a. The rule does not change that.

## UZH forward (Track E)

### The 15 Hz schema-v1 export: the rule abstains

The R2 `track-e/clips` export is schema v1 (`pos` at about 15 Hz). The rule picks windows
of **0.071-0.124 s** per sequence (f95 ~4-7 Hz), which are **1.1-1.9 samples at 15 Hz**.
The pre-registered 8-sample floor therefore abstains on **27/27 real clips (100 %)**; no
false-reject or detection number exists for the gain tier on this export.

| sequence | median window s | samples at 15 Hz | floor |
| --- | --- | --- | --- |
| indoor_forward_3 | 0.124 | 1.9 | abstain |
| indoor_forward_5 | 0.071 | 1.1 | abstain |
| indoor_forward_6 | 0.117 | 1.8 | abstain |
| indoor_forward_9 | 0.124 | 1.9 | abstain |
| indoor_forward_10 | 0.077 | 1.2 | abstain |
| indoor_forward_7 | 0.111 | 1.7 | abstain |
| outdoor_forward_1 | 0.077 | 1.2 | abstain |
| outdoor_forward_3 | 0.111 | 1.7 | abstain |
| outdoor_forward_5 | 0.091 | 1.4 | abstain |

At the **500 Hz truth rate** the same windows are 35-62 samples, so the floor is met. The
floor must be applied to the rate of the truth that computes `lean_req`, not to the
frame-rate `pos`. This is the precise reason Track E's 0.10 s result exists while the v1
export cannot reproduce it: they used the official schema-v2 `lean_req` from the 500 Hz
truth. The v2 `lean_req` itself is not on R2, so the gain at the rule's per-clip windows
cannot be recomputed here.

### Official schema-v2 population (Track E's per-clip rows)

Track E's `metrics_w370ms.json`: 147 scored clips, **9 testable real clips in 4 indoor
sequences** (3, 5, 9, 10), gate abstains on all outdoor.

| statistic | real (FRR) | null | yaw | removal | mirror | x0.5 | x1.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R2 band, TII unchanged, 0.37 s | **0.556** (5/9) | 0.556 | 0.286 (2/7) | 1.000 | 1.000 | 1.000 | 1.000 |
| coherence, TII tau | **0.111** (1/9) | 0.111 | 0.143 | **1.000** | **1.000** | 0.200 | 0.250 |
| R2 band, UZH LOSO (6-7 fit clips) | **0.111** (1/9) | — | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Sequence-clustered 95 % intervals: R2 TII band real **0.50-0.636**; coherence real
**0.00-0.375**; removal/mirror **1.00-1.00**.

The TII R2 band **does not transfer** at 0.37 s: it flags 5/9 real clips, so its 1.000
detection is uninformative. With the pre-registered **8-clip minimum for a conformal
band, no UZH leave-one-sequence-out fold can be fitted**: 9 real clips across 4 sequences
leave at most 7 in a fit set. Relaxing the minimum to 6 clips gives a real rate of 1/9,
but that band is fitted on 6-7 clips, with `gy` scale 0.023-0.037 against TII's 0.0735
and tau 3.21-3.61 against TII's 2.19. It is shown only to quantify the instability, not
as a usable UZH calibration.

The rule's windows (0.07-0.12 s) sit around the **0.10 s** window where Track E found the
TII band reaches **0.000** real (their window sensitivity). The rule therefore selects,
without seeing UZH outcomes, the window at which the gain tier can transfer — but that is
a post-hoc UZH validation and must be re-run on a v2 export with `lean_req` at the rule
windows, not 0.37 s.

## Does one rule give a gain tier that transfers with TII thresholds?

**No, not with the frozen TII threshold at 0.37 s.** The TII band false-rejects 5/9 UZH
real clips. The label-free rule points at the ~0.1 s window where Track E's 0.10 s
sensitivity reaches 0/9, so the rule is a necessary input, not a sufficient calibration.
The gain tier requires **per-domain calibration**; the coherence tier is the one that
transfers (0.111 [0.00, 0.375], detection 1.0/1.0).

### Minimum real clips for a per-domain gain band

TII band stability, resampling real clips (R2 joint band; 300 draws per n):

| fit clips | tau relative IQR | gx centre IQR | gy centre IQR |
| --- | --- | --- | --- |
| 10 | 17.0 % | 0.083 | 0.035 |
| 15 | 19.5 % | 0.055 | 0.025 |
| 20 | 14.7 % | 0.045 | 0.022 |
| 25 | 12.4 % | 0.040 | 0.018 |
| **30** | **9.6 %** | 0.040 | 0.015 |
| 40 | 6.5 % | 0.035 | 0.008 |
| 50 | 3.0 % | 0.025 | 0.006 |

**Minimum: 30 real clips** for a tau relative IQR below 10 %; 40 for below 7 %. The clips
must span **at least 4-6 flights/sequences**, not 30 clips from one flight. UZH has 9
testable real clips at 0.37 s (18 reach the scorer), below the minimum; its 6-7-clip LOSO
band is exactly the unstable regime the table describes. Per-domain calibration therefore
needs a larger UZH real population (more sequences/rigs) and a truth rate that supports
the selected window (500 Hz does; the 15 Hz v1 export does not).

## Proposal

`configs/window_bandwidth_amendment_proposal.yaml` proposes adopting the rule as the
operational window selector: `c=0.5`, 8-sample floor at the truth rate, no clamp; export
schema-v2 `lean_req`/`vel_req` at the rule-selected window from dense truth (TII 274.73 Hz,
UZH 500 Hz); re-freeze the R2 band on >=30 real clips; verify UZH at the rule windows
before claiming transfer. **Not applied**; the frozen configs remain at 0.37 s/R2.

## Coarse commanded paths (1 Hz, 16 points, ANWM-like)

The rule **always abstains** on a 1 Hz, 16-point path (t = 0..15 s):

- the non-zero rFFT bins are k/15 Hz, the first at 0.0667 Hz;
- f95 ≥ 0.0667 Hz gives window = 0.5/f95 ≤ 7.5 s, i.e. ≤ 7.5 samples at 1 Hz, below the
  8-sample floor;
- f95 = 0 (or below 0.05 Hz) is undefined and abstains;
- the only way to reach 8 samples is a smoothed f95 of 0.0625 Hz, which is not an FFT bin;
  that gives an 8 s window, greater than half the 15 s duration, so the duration guard
  abstains.

Synthetic checks: smooth curve f95 0.0625 → window 8.0 s, 8.0 samples, abstain (half
duration); slalom f95 0.1875 → 2.67 s, 2.67 samples, abstain (floor); straight line 1.33
samples; waypoint step 1.14 samples. No 16-point 1 Hz path passes.

**Minimum-viable coarse-path rule (proposal only):**
`configs/coarse_path_rule_proposal.yaml`. Piecewise-constant acceleration between
waypoints, finite-difference required lean, at least 8 intervals and 8 intervals above
10°, lower-bound interpretation (`coarse_path_lower_bound`), fallback to the `|Δv_h|/Δt`
peak bound. It is not applied and is not comparable to the dense-window band without
recalibration.

## Appendix: exact TII flight-cluster bootstrap

Both intervals use the same procedure, not clip-level Wilson intervals:

- **Flights:** the 6 flights with testable real clips — 01a (2), 01p (26), 04a (2),
  07p (24), 13p (2), 16p (1). The two other TII flights (07a, 13a) have zero testable
  clips and are not in the resampling.
- **Replicates:** 5,000, `numpy.random.default_rng(1)`. Each replicate draws 6 flights
  with replacement, concatenates all real clips of the drawn flights, and computes the
  **clip-weighted pooled rate** `sum(flags)/sum(clips)`; the interval is the 2.5th/97.5th
  percentile. The point is `sum(flags)/57`.
- **Coherence** (frozen tau 0.6197): 3/57 = 5.26 %; the 3 flags are in **2 flights**
  (01p 2, 07p 1); 95 % **[0.00 %, 7.14 %]**, median 5.26 %. Unweighted mean of the six
  flight rates: 1.98 % [0.00, 4.54].
- **Gain R2** (frozen band, d02): 3/57 = 5.26 %; the 3 flags are in **3 flights**
  (01a 1/2, 01p 1/26, 07p 1/24); 95 % **[2.86 %, 20.0 %]**, median 5.17 %. Unweighted
  mean of the six flight rates: 9.67 % [0.64, 26.28].

The clips within a flight share scene, estimator state and path; the intervals therefore
resample flights, and the clip count (57) overstates the evidence (50 clips come from two
piloted flights).
