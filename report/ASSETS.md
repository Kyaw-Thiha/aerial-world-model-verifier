# Asset index

All paths are relative to `report/`. Every asset's numeric source is recorded in
`data/<asset>.src.json` (path, branch, commit, sha256, script). The experiment CSV holds
the extracted numbers for the assets of that experiment; where an asset is built from a
stored result without a separate CSV column, the `.src.json` names that result and the
script reads it directly.

Status: `ready` = built and cross-checked; `pending` = depends on work still in progress;
`missing` = source data not present in this environment.

## E1 physics floor

| asset | shows | source | status | caption hint |
| --- | --- | --- | --- | --- |
| `figures/fig_e1_physics-floor_1.pdf` | held-out residual vs speed, drag-free vs fitted drag, band = per-flight spread | phase0_floor metrics.json | ready | "The drag term removes the low-speed offset but not the high-speed tail." |
| `tables/tab_e1_physics-floor_1.tex` | per-flight window, fitted drag, median residual, drag reduction | phase0_floor metrics.json | ready | "One drag coefficient per fold, 0.290-0.304 s^-1, all folds at 101 samples." |
| `data/e1_physics-floor.csv` | speed-curve values | derived by script | ready | — |

## E2 depicted tilt (TII)

| asset | shows | source | status | caption hint |
| --- | --- | --- | --- | --- |
| `tables/tab_e2_depicted-tilt-tii_1.tex` | per-flight and pooled per-frame error, adopted raw-fisheye variant | phase1_depicted final run | ready | "Typical error is ~4 deg; the RMSE is set by a track-flight tail." |
| `tables/tab_e2_depicted-tilt-tii_2.tex` | estimator-route comparison (three routes) | report 02 survey | ready | "The raw fisheye route is the only one below the Gate 1a bar at the median." |
| `figures/fig_e2_depicted-tilt-tii_1.pdf` | CDF of per-frame error by track | derived from per-frame npz | ready | "All three tracks share a ~4 deg core; track flights carry the heavy tail." |
| `figures/fig_e2_depicted-tilt-tii_2.pdf` | ROC of the confidence gate with permutation null | gate clip table; null from report 06 | ready | "The gate ranks clips with >10 deg error well above the selection-adjusted null." |

## E3 verdict on real flights

| asset | shows | source | status | caption hint |
| --- | --- | --- | --- | --- |
| `tables/tab_e3_verdict-real-flights_1.tex` | conformal threshold, fold range, detection floor per calibration mode | a2_modes thresholds | ready | "Drag and the nuisance fit each buy several degrees of detection floor." |
| `tables/tab_e3_verdict-real-flights_2.tex` | acceptance and abstention by flight type | a2_verdict_gated (drag 0.295) | ready | "Acceptance is a statement about piloted flights; autonomous flights mostly abstain." |
| `figures/fig_e3_verdict-real-flights_1.pdf` | detection by edit condition with Wilson intervals and k/n, drag-free mode | a2_verdict_gated kd_0 | ready | "Removal is caught in half of eight testable pairs; the mirrored roll almost always." |

## E4 ANWM data audit

| asset | shows | source | status | caption hint |
| --- | --- | --- | --- | --- |
| `tables/tab_e4_anwm-data-audit_1.tex` | required tilt median/p90 by regime and step, per-step vs smoothed | anwm_path_audit/audit.json | ready | "Native paths demand extreme tilt at short steps; designed maneuvers are milder." |
| `tables/tab_e4_anwm-data-audit_2.tex` | fraction of steps above 10 deg | same | ready | "Most native steps and designed turns carry a testable tilt." |
| `tables/tab_e4_anwm-data-audit_3.tex` | designed maneuvers, clip counts, required tilt; unreachable designs | audit.json + clips_index.csv | ready | "Course-yaw designs at 10 and 20 deg are unreachable; fixed-yaw versions are not." |
| `tables/tab_e4_anwm-data-audit_4.tex` | level-frame noise floor and seed-to-seed pixel variance | anwm_floor_v1 + variance.json | ready | "Exactly level frames read as level to ~0.57 deg at both resolutions." |

## E5 ANWM verdict

| asset | shows | source | status | caption hint |
| --- | --- | --- | --- | --- |
| `figures/fig_e5_anwm-verdict_1.pdf` | depicted vs designed tilt for designed maneuvers, required-tilt line | a2_anwm_score per_clip | ready | "Depicted tilt barely follows the design (+0.04 vs the +0.5 required)." |
| `tables/tab_e5_anwm-verdict_1.tex` | 20-deg designs: residual, threshold, margins per mode and interpolant | margins_regimeC_20deg.csv | ready | "Margins are positive at the drag-free primary; full parity is tighter." |
| `tables/tab_e5_anwm-verdict_2.tex` | infeasible/testable counts by cell and mode | a2_modes verdicts | ready | "Designed-10 flips with calibration mode; designed-20 is infeasible throughout." |

## E6 pose injection

| asset | shows | source | status | caption hint |
| --- | --- | --- | --- | --- |
| `tables/tab_e6_pose-injection_1.tex` | LPIPS vs level and prior, prior coverage, by design | eval_injected_injected_true.json | ready | "Injection moves the image progressively, but not monotonically for accelerate-brake." |
| `figures/fig_e6_pose-injection_1.pdf` | LPIPS vs injected tilt, three maneuvers | same | ready | "Accelerate-brake is non-monotone because only 4 of 16 steps carry pitch at 20 deg." |
| `figures/fig_e6_pose-injection_2.pdf` | one level-vs-injected contact sheet | anwm_injected/sheets | ready | "A representative 20-deg turn: level frames (top) and injected frames (bottom)." |

Depicted-tilt scoring of the injected clips is **pending** (see MISSING.md); no verdict asset
is built from it.

## E7 path from pixels

| asset | shows | source | status | caption hint |
| --- | --- | --- | --- | --- |
| `tables/tab_e7_path-from-pixels_1.tex` | extractor by sequence: direction error, magnitude, pass | pathrec_bakeoff/v2 | ready | "DPVO passes all ten; the feed-forward models pass 2-6." |
| `tables/tab_e7_path-from-pixels_2.tex` | DPVO with calibrated vs estimated intrinsics | pathrec_calib | ready | "A wrong focal is survivable; estimated K costs 10-15 deg but still passes 8/10." |
| `tables/tab_e7_path-from-pixels_3.tex` | ANWM frames with DPVO (crash/degenerate/usable) and the failed UZH verdict pilot | pathrec_anwm + pathrec_verdict | ready | "DPVO crashes on 10 of 25 ANWM clips; the pixel-path verdict separates nothing." |
| `figures/fig_e7_path-from-pixels_1.pdf` | direction error vs sequence peak speed, marker = rig | v2 summary + UZH audit | ready | "Failure tracks the rig (outdoor/45-down), not speed alone." |

## E8 gain and coherence tests

| asset | shows | source | status | caption hint |
| --- | --- | --- | --- | --- |
| `tables/tab_e8_gain-and-coherence-tests_1.tex` | matched false-reject comparison | d_eval/d02 metrics | ready | "At its own 5.3% false-reject the gain test detects all four families; the baseline at 3.4% misses x0.5 and x1.5." |
| `tables/tab_e8_gain-and-coherence-tests_2.tex` | gain decomposition by flight type | d_decomposition | ready | "Truth-to-required 0.99, estimated-to-truth 0.83, verdict 0.75." |
| `tables/tab_e8_gain-and-coherence-tests_3.tex` | window sensitivity 0.10-0.74 s | d_window | ready | "Detection is flat; false reject is not, and the band must be refit per window." |
| `tables/tab_e8_gain-and-coherence-tests_4.tex` | coherence AUROC and detection by condition | d_coherence | ready | "Coherence separates removal and mirror almost perfectly, weakly at x0.5." |
| `tables/tab_e8_gain-and-coherence-tests_5.tex` | flight-clustered intervals, gain and coherence | d_flights + D04 | ready | "Six clusters: intervals are descriptive, and the weighted and unweighted estimands differ." |
| `figures/fig_e8_gain-and-coherence-tests_1.pdf` | flag rate vs synthetic gain scale (gain test only) | d_eval/d02 synthetic_sweep | ready (partial) | "Detection floor sits between a 25% and a 50% gain change." |

## E9 UZH transfer

| asset | shows | source | status | caption hint |
| --- | --- | --- | --- | --- |
| `tables/tab_e9_uzh-transfer_1.tex` | per-frame error, forward vs 45-down vs TII, by variant | e_perframe v1_forward/v1_all | ready | "The forward rigs transfer; the published 45-down mount does not." |
| `tables/tab_e9_uzh-transfer_2.tex` | published elevation, fitted rotation, held-out residual | e_mount + e_audit | ready | "A constant rotation explains the 45-down cluster but not two sequences." |
| `tables/tab_e9_uzh-transfer_3.tex` | edit-validity pass rate per sequence | e_verdict_v3/rule validity_gate | ready | "The null edit passes everywhere except two outdoor sequences." |
| `tables/tab_e9_uzh-transfer_4.tex` | verdict transfer on forward rigs, three statistics | e_verdict_v3/fin_370ms | ready | "Coherence transfers; the TII gain band does not." |
| `figures/fig_e9_uzh-transfer_1.pdf` | lean slope per UZH sequence beside the TII range | e_calib/calibration.json | ready | "UZH slopes sit inside the TII flight range; the estimator is not the failure." |

## E10 window rule and prototype

| asset | shows | source | status | caption hint |
| --- | --- | --- | --- | --- |
| `tables/tab_e10_window-rule-and-prototype_1.tex` | rule window per flight/sequence at truth rate and clip rate, samples, floor | d_window_rates/d01/table.csv | ready | "At 500 Hz the rule picks 0.34-1.42 s; at 15 Hz it picks 1.1-1.9 samples and abstains." |
| `tables/tab_e10_window-rule-and-prototype_2.tex` | UZH forward verdict at rule windows vs 0.37 s | e_verdict_v3/rule + fin_370ms | ready | "The gain tier fails harder at the rule windows; coherence is unchanged at zero." |
| `tables/tab_e10_window-rule-and-prototype_3.tex` | 1 Hz 16-point synthetic cases | D04 text (parsed) | ready | "Every ANWM-like coarse path abstains, on the floor or the duration guard." |
| `tables/tab_e10_window-rule-and-prototype_4.tex` | validated / calibrated-only / not-validated scope matrix | D04, E05, prototype | ready | "Validation stops at piloted TII and indoor UZH forward." |
| `figures/fig_e10_window-rule-and-prototype_1.pdf` | threshold relative IQR vs calibration clips | d_bandwidth band stability | ready | "Thirty clips is the first point below 10% relative IQR." |

## Methodology and reproducibility

| asset | shows | source | status | caption hint |
| --- | --- | --- | --- | --- |
| `tables/tab_datasets_1.tex` | datasets, sensors, rates, truth, use | branch audits | ready | "Three data sources, three different notions of truth." |
| `tables/tab_frozen-configs_1.tex` | every frozen config with commit and sha256 | live worktrees | ready | "Configuration freeze points; tab_repro_1 is the summary." |
| `tables/tab_repro_1.tex` | per-branch HEAD, configs, R2 prefix, test/cost status | live worktrees + reports | ready | "Commit and hash provenance; R2 object counts unavailable offline." |

## Documents

* `ASSETS.md` (this file), `CHECKS.md`, `MISSING.md`, `cutoff.json`,
  `data/corrections.csv`, `assets_test.tex`, `scripts/make_all.py`.
