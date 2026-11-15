# Prototype v0.1 — aerial video world-model verifier

Status: 2026-11-14. Track D (commanded-path mode, TII) plus Track E (UZH-FPV forward).
**Headline verdict: the scale-free coherence tier.** The gain tier is a **calibrated
diagnostic**, not a transferable verdict. Frozen configs: `configs/gain_frozen.yaml`
(R2 axis band, 0.37 s), `configs/coherence_frozen.yaml`. Proposals:
`configs/window_bandwidth_amendment_proposal.yaml` (not applied).

## 1. Headline: coherence tier

Per-axis Pearson correlation and sign agreement between depicted and required lean, lag
from the gain fit, threshold 1−min(r_x,r_y) > 0.6197 (or sign > 0.2921). Scale-free: a
positive scaling leaves it unchanged, a tilt removal destroys it, a mirrored roll flips the
roll axis.

| population | real FRR | removal | mirror | x0.5 | x1.5 | yaw |
| --- | --- | --- | --- | --- | --- | --- |
| TII, held-out (LOFO thresholds, n=57) | **8.8 %** | **100 %** | **100 %** | 17.5 % | 24.6 % | 10.5 % |
| TII, frozen tau in-sample (n=57) | 5.3 % [0-7.1 % cluster] | 100 % | 100 % | 17.5 % | 24.6 % | 10.5 % |
| UZH forward, official v2 (n=9 real, 4 sequences) | **0.111** [0.00-0.375] | **1.000** | **1.000** | 0.200 | 0.250 | 0.143 |

AUROC on TII: removal **0.999**, mirror **1.000**; x0.5 0.706, x1.5 0.622 (chance-ish, as
predicted for a scale-free shape statistic); yaw 0.600. This is the only tier that meets
the transfer target (UZH false-reject 0.111 ≤ 0.15, detection 1.0 ≥ 0.8).

## 2. Gain tier: calibrated diagnostic

Per-clip `l_dep = G1·lean_req + G2·vel_req + b`; R2 flags if `max(|gx−c_x|/s_x,
|gy−c_y|/s_y) > 2.1922`.

| population | real FRR | removal | mirror | x0.5 | x1.5 | yaw |
| --- | --- | --- | --- | --- | --- | --- |
| TII, held-out (n=57, 6 flights) | **5.3 %** [2.9-20.0 % flight-cluster] | 100 % | 100 % | 100 % | 96.5 % | 28.1 % |
| UZH v2 at 0.37 s, TII band (n=9, 4 seq) | **0.556** [0.50-0.636] | 100 % | 100 % | 100 % | 100 % | 0.286 |
| UZH v2, UZH LOSO band (6-7 fit clips) | 0.111 [0.00-0.375] | 100 % | 100 % | 100 % | 100 % | 0.000 |

The UZH TII-band failure is not a detection failure — it flags everything (5/9 real and
5/9 null edits), so its 1.000 detection is uninformative. A per-domain band on 6-7 clips
lowers the rate to 0.111 but is unstable (`gy` scale 0.023-0.037 vs TII 0.0735; tau
3.21-3.61 vs 2.19).

**Minimum real clips for a per-domain gain band:** 30 (tau relative IQR 9.6 %; 40 for
6.5 %, 50 for 3.0 %), spanning at least 4-6 flights/sequences. UZH has 9 testable real at
0.37 s (18 reach the scorer), below that minimum.

## 3. Label-free window rule

`window = 0.5 / f95` from the commanded acceleration spectrum, with an 8-sample floor at
the truth rate; abstain otherwise. Pre-registered before scoring
(`configs/window_bandwidth_prereg.yaml`). On TII it selects per-clip windows 0.05-0.22 s
and reproduces the gain tier (**5.3 % FRR, 100/100/100/96.5**, yaw 29.8, 57/100 testable).
On UZH it selects **0.071-0.124 s**; at the 15 Hz v1 export that is 1.1-1.9 samples, so the
rule abstains 100 %; at the 500 Hz truth rate it is 35-62 samples. The rule points at the
~0.10 s window where Track E found the TII R2 band at 0.000, but the v2 dense `lean_req`
at the rule windows is not on R2, so this is not yet verified. **The gain tier requires
per-domain calibration even with the window rule.**

## 3b. Coarse paths

A 1 Hz, 16-point ANWM-like path **always abstains** under the bandwidth rule (f95 ≥ 1/15 Hz
gives a window of at most 7.5 samples; a smoothed 0.0625 Hz edge gives 8 s, longer than
half the 15 s duration). The minimum-viable coarse-path mode is proposed only
(`configs/coarse_path_rule_proposal.yaml`): piecewise-constant acceleration between
waypoints, finite-difference required lean, ≥8 intervals and ≥8 above 10°, explicitly a
lower-bound test, not the dense-window verdict.

## 4. What it detects and what it cannot

**Detects (TII piloted, with enough excitation):** full tilt removal, mirrored roll, and
(with the gain tier) 50 % scaling; synthetic 50 % gain changes 86-100 %, 25 % changes
28-72 %. Coherence catches shape/sign; the gain catches magnitude.

**Cannot:** certify a small change below the artifact floor; the gain tier flags 28.1 % of
yaw-only controls against 5.3 % of real clips, the estimator's view dependence; coherence
is at chance for positive scaling (x0.5 17.5 %, x1.5 24.6 % on TII; 0.20/0.25 on UZH); the
gain is not the airframe drag (fitted 0.158 1/s against Phase 0 0.295); a pass is relative
to a calibrated real band, not proof of physical consistency.

## 5. Validated flights, rigs and populations

- **Validated:** TII piloted ellipse/lemniscate (50 testable clips, 2 flights) for the gain
  and coherence tiers; UZH forward **coherence** on 9 testable real clips from 4 indoor
  sequences (indoor_forward_3/5/9/10).
- **Not validated:** TII autonomous (4 testable clips, 25 % FRR) and track (3 testable,
  33 % FRR; 0/11 on 13a); UZH gain at 0.37 s; UZH outdoor (frozen gate abstains, validity
  gate fails); any rig other than the two above.
- **Track/autonomous status rule (amendment 2):** `validation_status: not_validated`,
  `usable_as_evidence: false`; the graded result is still produced.

## 6. Abstention rules

A clip is not testable when: fewer than 24 frames; fewer than 50 % usable depicted frames;
fewer than 12 frames or 20 % above 10° required tilt; principal required-lean spread below
2°; design condition number above 1000; bootstrap half-width of `G1` above 0.35; the frozen
S3 confidence gate; or (label-free rule) the acceleration-bandwidth window has fewer than
8 samples at the truth rate or exceeds half the clip duration. Abstention is never a
verdict.

## 7. Numbers with cluster intervals

- TII gain: 3/57 false rejects in **3 of 6 flights** (01a 1/2, 01p 1/26, 07p 1/24),
  pooled 5.26 %, same flight-cluster bootstrap (5,000 draws, 6 flights with replacement,
  clip-weighted): 95 % **[2.86, 20.0 %]**; unweighted flight mean 9.7 % [0.6, 26.3 %].
  50 of 57 clips are from two piloted flights.
- TII coherence: 3/57 with the frozen tau (5.3 % [0-7.1 %]); held-out LOFO 8.8 %. The
  3 flags are in 2 of the 6 testable flights (01p 2, 07p 1). Bootstrap: 5,000 draws,
  resample the 6 flights with replacement, concatenate their real clips, clip-weighted
  pooled rate; 2.5/97.5 percentiles.
- UZH coherence: 1/9, 4 sequences, **[0.00, 0.375]**; removal/mirror 1.0 [1.0, 1.0].
- UZH R2 TII band: 5/9, 4 sequences, **[0.50, 0.636]**.
- UZH R2 LOSO: 1/9, 6-7 fit clips, [0.00, 0.375], unstable.

## 8. Frozen and proposed

Frozen: `configs/gain_frozen.yaml` (R2, 0.37 s, band gx 0.8137/0.2654, gy 0.7181/0.0735,
tau 2.1922), `configs/coherence_frozen.yaml` (tau_corr 0.6197, tau_sign 0.2921). Proposed,
awaiting approval: the label-free bandwidth window
(`configs/window_bandwidth_amendment_proposal.yaml`); a per-domain UZH gain band would need
>=30 real clips. No frozen config has been changed by this round.

## 9. Where the evidence is

`reports/D01_gain_test.md`, `D02_gain_test_v2.md`, `D03_shrinkage_window.md`,
`D04_bandwidth_window.md`; `docs/verdict_prototype.md`; `docs/verdict_clip_schema_v2.md`;
`results/d_eval/d02/`, `d_coherence/d01/`, `d_shrinkage/d01/`, `d_calibrate/d01/`,
`d_window/d01/`, `d_bandwidth/d01/`, `d_uzh_v2/`. Scorer: `scripts/score_clip.py`.
