# E05 — Independent test of Track D's label-free window rule (`0.5 / f95`)

Track E. Written 2026-11-14. Pre-registration: `configs/e_prereg_v4_window_rule.yaml` (commit
`1befcd1`), committed **before** any window was computed or any clip scored. Rule under test:
Track D's `configs/window_bandwidth_prereg.yaml`, branch `track-d-verdict-stat` at `031eb14`
(rule committed `c7af793`). Statistics: their frozen `R2_axis_gains` band and coherence tier,
commit `851096c`, unchanged. Depicted side untouched — **zero GPU** was needed, and none was used.

## Plain-language summary

1. **The rule does not produce 0.07-0.12 s when it is applied where schema v2 computes the required
   lean.** On the 500 Hz truth it returns **0.34-1.42 s per sequence**. The 0.07-0.12 s figure is
   what the same rule returns when the acceleration spectrum comes from the *clip-rate* positions
   (~15 Hz), where a second derivative is dominated by differentiation noise rather than by the
   commanded motion.
2. **At the clip rate the rule contradicts itself**: 0.07-0.12 s is 1.1-1.8 samples at 15 Hz, far
   below its own 8-sample floor (0.53 s at that rate), and its `abstain_if` clause says a window the
   rate cannot support makes the clip **not testable** rather than being lengthened. So the number
   that looked like independent support for the 0.10 s transfer is the rule's output at a rate where
   the rule forbids its own use.
3. **Scored at the windows the rule actually produces (500 Hz), Track D's R2 band still fails T2**,
   as it did at the pre-registered 0.37 s headline — the rule's windows and my headline are the same
   part of the curve. The coherence tier again meets T2. Numbers in §3.

## 1. The rule, and how it was implemented here

`window_s = c / f95`, `c = 0.5`; f95 is the frequency below which 95 % of the commanded acceleration
energy lies; acceleration from a fixed Savitzky-Golay differentiation (0.10 s, polyorder 3); spectrum
= rFFT of the mean-removed acceleration, energy summed over the three axes, DC excluded; cumulative
energy normalised, f95 at the first bin reaching 0.95; floor 8 samples at the rate the required lean
is computed; `clamp = none`; abstain if f95 < 0.05 Hz or the window exceeds half a clip.

Implemented independently in `experiments/e_window_rule.py` from that text (their code was not
imported), reading only the commanded path. One window per sequence, over that sequence's full truth
span.

## 2. The windows

| sequence | f95 (500 Hz truth) | window at 500 Hz | their reported figure | window if f95 is taken at 15 Hz |
| --- | --- | --- | --- | --- |
| indoor_forward_3 | 1.091 Hz | **0.458 s** | 0.07-0.12 s | 0.111 s |
| indoor_forward_5 | 1.295 | **0.386** | 0.07-0.12 s | 0.111 |
| indoor_forward_6 | 1.490 | **0.336** | 0.07-0.12 s | 0.111 |
| indoor_forward_7 | 1.064 | **0.470** | 0.07-0.12 s | 0.111 |
| indoor_forward_9 | 0.486 | **1.029** | 0.07-0.12 s | 0.100 |
| indoor_forward_10 | 0.569 | **0.879** | 0.07-0.12 s | 0.071 |
| outdoor_forward_1 | 0.352 | **1.420** | 0.07-0.12 s | 0.071 |
| outdoor_forward_3 | 0.373 | **1.339** | 0.07-0.12 s | 0.100 |
| outdoor_forward_5 | 0.909 | **0.550** | 0.07-0.12 s | 0.111 |

The last column is computed exactly as the rule reads, but on the ~15 Hz `pos` of a 4 s clip
(median 0.095-0.111 s across all clips, range 0.071-0.332 s). **That reproduces Track D's reported
range and attributes it**: the discrepancy is the sample rate of the acceleration estimate, not the
quantile, the DC handling or the spectral window (each of those was varied and moves the answer by
less than 0.05 s; the rate moves it by a factor of four to twelve).

**Consequence, and it is the rule's own.** At 15 Hz the floor is 8 samples = 0.53 s; a 0.07-0.12 s
window is 1.1-1.8 samples, and `abstain_if` declares the clip not testable rather than lengthening
the window. So under the v1 reading the rule abstains on **every** clip — 42 of 42 real clips and
all edits — which is not a transfer result at all. Under the v2 reading, where the required lean is
differentiated at 500 Hz, the floor is 0.016 s and the windows are the ones in the third column.

This is the same defect E04 §2.4 measured from the other side: the required-lean window only matters
through how much of the path's real bandwidth it keeps, and a bandwidth estimated from 15 Hz
positions is an estimate of the *differentiator's* noise, not of the commanded motion.

## 3. Transfer at the rule's windows (500 Hz), forward rigs

Population identical to E04's (same clips, same drops, only the required-lean window changed):
126 clips scored, 7 sequences, 21 files excluded (`indoor_forward_7`), 42 files dropped (field < 25 %).
Timings per sequence were 0.336-1.420 s (§2).

| statistic | real (false reject) | null edit | yaw control | tilt removal | mirror | x0.5 | x1.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **R2 axis band** (TII, unchanged) | **0.778** (7/9) | 0.778 (7/9) | 0.571 (4/7) | 1.000 (10) | 1.000 (8) | 0.700 (10) | 1.000 (7) |
| **coherence tier** | **0.000** (0/9) | 0.000 (0/9) | 0.000 (0/7) | **1.000** (10) | **1.000** (8) | 0.200 (10) | 0.000 (7) |
| **median residual, 14.73 deg** | **0.000** (0/18) | 0.000 (0/18) | 0.000 (0/18) | **0.722** (13/18) | **0.778** (14/18) | 0.056 (18) | 0.222 (18) |

Sequence-clustered 95 % intervals (7 sequences): R2 real **0.571-1.000**, coherence real
**0.000-0.000**, R2 and coherence removal/mirror **1.000-1.000**.

**The R2 band fails, and it fails harder than at the E04 headline**: false reject 0.778 here against
0.556 at 0.37 s and 0.000 at 0.10 s. That is the same monotone relationship E04 §2.4 measured — a
longer required-lean window shrinks the fitted gain, which the TII band reads as a violation — now
driven by a window chosen by a rule that never sees a label. Its detection of 1.000 is again
uninformative because everything is flagged, including the null edit at exactly the real rate.

**The coherence tier is unaffected by the window and meets T2 comfortably**: false reject 0.000 with
both controls at 0.000, and detection 1.000 for removal and mirror, clustered intervals 1.000-1.000.
That is the third independent window (0.37, 0.10, per-sequence rule) at which it reads the same.

**The median residual is also window-invariant** (0.000 false reject, 0.722 / 0.778 detection),
still just under the 0.80 bound for removal.

Abstention is 0.500 on 18 real clips — 5 by the confidence gate, 6 by the gain-CI rule — identical to
E04, so the new window changed no clip's testability. The validity gate's status is unchanged
(indoor passes; `outdoor_forward_1` 0.67 and `outdoor_forward_5` 0.00 pass rate, gain-only failures),
and no outdoor detection claim is made, as pre-registered.

## 4. What this says about the rule

* As a **label-free way to pick the window from the path**, the idea is sound and it is not
  outcome-tuned — but the bandwidth has to be measured at the rate the required lean is computed.
  Applied that way it returns 0.34-1.42 s here, i.e. it agrees with the value E04 pre-registered as
  the headline (0.37 s), and it does not point at 0.10 s.
* The rule as written carries both readings in one document (`floor.applies_to` says "the rate at
  which the required lean is computed"; the `uzh` line still describes schema v1 at ~15 Hz), and the
  two give answers a factor of four apart. That ambiguity is worth resolving in their config rather
  than in a downstream report.
* **The R2 band's window-sensitivity (E04 §2.4) is therefore not explained by a principled
  bandwidth rule**: the principled window is ≈0.37 s, and at 0.37 s the band fails. Shortening the
  window to 0.10 s makes the band pass, but no path-derived rule at the truth rate picks 0.10 s.

## 5. Reconciliation: coherence false reject at 0.37 s versus at the rule's windows

One row per run, then the clip that moves.

| run | required-lean window | clips scored / sequences | testable real clips | abstention (real) | coherence false reject | R2 false reject (context) |
| --- | --- | --- | --- | --- | --- | --- |
| E04 headline | **0.37 s**, one value for all sequences | 126 / 7 | **9** (4 sequences: indoor_forward_3, _5, _9, _10) | 9/18 = 0.500 (gate 5, gain-CI 6) | **1/9 = 0.111** [0.00, 0.375] | 0.556 [0.50, 0.64] |
| E05 rule | **0.336-1.420 s**, per sequence (indoor_forward_3 0.458, _5 0.386, _9 1.029, _10 0.879) | 126 / 7 | **9** - *the identical clips* | 9/18 = 0.500 (gate 5, gain-CI 6) | **0/9 = 0.000** [0.00, 0.000] | 0.778 [0.57, 1.00] |

**No population differs between E04 and E05.** Same 126 clips over the same 7 sequences, same 21
excluded files (`indoor_forward_7`), same 42 dropped files (field below 25 %), same 18 real clips
reaching the scorer, same 9 testable, same abstention counts and reasons. The two runs differ in
exactly one input: the required-lean window. So the counts are not the story; the single clip is.

**The clip that moves, and which tier moved it.** `indoor_forward_9_00_real`:

| window (indoor_forward_9) | r_x | r_y | 1-min(r) vs tau_corr 0.6197 | s_x | s_y | 1-min(s) vs tau_sign 0.2921 | flagged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.37 s | 0.983 | 0.912 | 0.088 - far below | 0.804 | **0.667** | **0.333 - above** | **yes (sign)** |
| 1.029 s (rule) | 0.985 | 0.869 | 0.131 - far below | 0.765 | **0.750** | **0.250 - below** | no |

The **correlation tier never fires**: no clip on any window comes within 0.40 of `tau_corr`. The
flip is entirely in the **sign tier**, where one clip's per-axis sign agreement moves from 0.667 to
0.750 (`1-min(s)` from 0.333 to 0.250) and crosses 0.2921. Every other clip keeps its decision, with
`1-min(s)` moving by at most 0.05 and `1-min(r)` by at most 0.003.

**How to read the 1/9 -> 0/9, stated plainly.** It is one clip crossing one threshold by 0.041 in a
statistic whose value for that clip is 0.33 at one window and 0.25 at another — a boundary case, not
evidence that the rule's window is the better one. The defensible claims are the ones that hold at
*both* windows: coherence false reject is **0.000-0.111**, inside the 0.15 budget either way, with
detection 1.000/1.000 and both controls at 0.000-0.111; and coherence is the only one of the three
statistics whose decision does not change with the window in a systematic direction. The R2 band, by
contrast, moves 0.556 -> 0.778 across the same pair of windows and fails both times.

Windows are the *sequence* windows. `indoor_forward_9`'s clip 00 is the only real clip in the set
whose sequence window (1.029 s) is more than 0.3 s away from 0.37 s and which sits near a threshold;
the other long-window sequences (indoor_forward_10 0.879 s, outdoor 1.34-1.42 s) either keep their
decisions or are abstained on the gate.
