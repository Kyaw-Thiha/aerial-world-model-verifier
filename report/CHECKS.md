# Cross-checks

Tolerance: angles 0.02 deg, rates/slopes 0.005, counts exact. A mismatch is shown, not resolved; the asset is built from the stored result.

| Asset | Quantity | Reference | Asset / stored result | Difference | Status | Note |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | pooled held-out median, fitted drag (deg) | 1.24 | 1.2440052388971672 | 0.004005 | **match** |  |
| E1 | pooled held-out median, drag-free (deg) | 5.46 | 5.460733122447984 | 0.0007331 | **match** |  |
| E1 | drag coefficient min (1/s) | 0.29 | 0.28993995214640894 | 6.005e-05 | **match** |  |
| E1 | drag coefficient max (1/s) | 0.304 | 0.30405520411474896 | 5.52e-05 | **match** |  |
| E1 | drag reduction flight-01p-ellipse (deg) | 5.68 | 5.629330451593594 | 0.05067 | **mismatch** | stored result differs from the quoted value |
| E1 | drag reduction flight-07p-lemniscate (deg) | 5.73 | 5.716287661555597 | 0.01371 | **match** |  |
| E1 | drag reduction flight-13p-trackRATM (deg) | 3.89 | 3.893175650452521 | 0.003176 | **match** |  |
| E1 | drag reduction flight-16p-trackRATM (deg) | 3.61 | 3.6140022976120587 | 0.004002 | **match** |  |
| E2 | pooled per-frame median (deg) | 3.96 | 3.961669598878459 | 0.00167 | **match** |  |
| E2 | pooled 10% trimmed RMSE (deg) | 4.8 | 4.797117213461594 | 0.002883 | **match** | trim is 10% |
| E2 | pooled full RMSE (deg) | 22.32 | 22.323619182577207 | 0.003619 | **match** |  |
| E2 | gate AUROC | 0.946 | 0.9456521739130435 | 0.0003478 | **match** |  |
| E2 | gate permutation-null median | 0.822 | 0.822 | 0 | **match** | transcribed from report 06; null distribution not stored |
| E2 | gate permutation-null 95th percentile | 0.874 | 0.874 | 0 | **match** | transcribed from report 06 |
| E3 | drag-free lag-only threshold (deg) | 14.73 | 14.732514900359874 | 0.002515 | **match** |  |
| E3 | drag-free full threshold (deg) | 12.95 | 12.94715981791374 | 0.00284 | **match** |  |
| E3 | drag 0.295 lag-only threshold (deg) | 8.96 | 8.962213114592739 | 0.002213 | **match** |  |
| E3 | drag 0.295 full threshold (deg) | 6.46 | 6.460055940045862 | 5.594e-05 | **match** |  |
| E3 | drag-free acceptance (rate) | 0.966 | 0.9655172413793104 | 0.0004828 | **match** |  |
| E3 | drag-free acceptance denominator (clips) | 58 | 58 | 0 | **match** |  |
| E3 | tilt removal detection | 0.5 | 0.5 | 0 | **match** |  |
| E3 | mirrored roll detection | 0.96 | 0.9607843137254902 | 0.0007843 | **match** |  |
| E3 | yaw control detection | 0.04 | 0.038461538461538464 | 0.001538 | **match** |  |
| E4 | level-frame noise floor, 224 px (deg) | 0.57 | 0.5745522612613799 | 0.004552 | **match** |  |
| E4 | level-frame noise floor, 512 px (deg) | 0.57 | 0.565507373723426 | 0.004493 | **match** |  |
| E5 | depicted-vs-designed slope (deg/deg) | 0.04 | 0.0397 | 0.0003 | **match** |  |
| E5 | depicted tilt at 5/10/20 deg designs (median deg) | 0.77-0.79 | None | -- | **not checkable** | per-design medians are in the table; pooled figure slope is checked above |
| E6 | pooled LPIPS vs level | 0.266 | 0.2662 | 0.0002 | **match** |  |
| E7 | DPVO passing clips | 10 | 10 | 0 | **match** |  |
| E7 | Pi3 passing clips | 6 | 6 | 0 | **match** |  |
| E7 | VGGT passing clips | 2 | 2 | 0 | **match** |  |
| E7 | DA3 passing clips | 2 | 2 | 0 | **match** |  |
| E7 | DPVO calibrated median direction error (deg) | 3.9 | 4.177710183404422 | 0.2777 | **mismatch** |  |
| E8 | baseline detection tilt_removal at gain FRR | 0.672 | 0.6724137931034483 | 0.0004138 | **match** |  |
| E8 | baseline detection mirror_roll at gain FRR | 0.948 | 0.9482758620689655 | 0.0002759 | **match** |  |
| E8 | baseline detection scale_0.5 at gain FRR | 0.172 | 0.1724137931034483 | 0.0004138 | **match** |  |
| E8 | baseline detection scale_1.5 at gain FRR | 0.241 | 0.2413793103448276 | 0.0003793 | **match** |  |
| E8 | gain detection removal at baseline FRR | 1.0 | 1.0 | 0 | **match** |  |
| E8 | decomposition truth-to-required slope | 0.991 | 0.9905655791370193 | 0.0004344 | **match** |  |
| E8 | decomposition estimated-to-truth slope | 0.828 | 0.8279295436264479 | 7.046e-05 | **match** |  |
| E8 | verdict gain slope | 0.745 | 0.7449912216681291 | 8.778e-06 | **match** |  |
| E8 | verdict gain, dense required lean | 0.764 | 0.7639313841339861 | 6.862e-05 | **match** |  |
| E8/E10 | threshold relative IQR at 25 clips (%) | 12.4 | 12.446283066334876 | 0.04628 | **match** |  |
| E8/E10 | threshold relative IQR at 30 clips (%) | 9.6 | 9.619150085811464 | 0.01915 | **match** |  |
| E8/E10 | threshold relative IQR at 40 clips (%) | 6.5 | 6.463953871463278 | 0.03605 | **match** |  |
| E8/E10 | threshold relative IQR at 50 clips (%) | 3.0 | 3.010187153956462 | 0.01019 | **match** |  |
| E8 | gain-tier pooled false reject (rate) | 0.0526 | 0.05263157894736842 | 3.158e-05 | **match** |  |
| E8 | gain-tier cluster interval low | 0.0286 | 0.02857142857142857 | 2.857e-05 | **match** |  |
| E8 | gain-tier cluster interval high | 0.2 | 0.2 | 0 | **match** |  |
| E9 | UZH forward variant b pooled median (deg) | 4.62 | 4.622310350466949 | 0.00231 | **match** |  |
| E9 | UZH forward variant a pooled median (deg) | 1.53 | 1.5333142725229112 | 0.003314 | **match** |  |
| E9 | UZH forward variant c pooled median (deg) | 2.06 | 2.0620402154597492 | 0.00204 | **match** |  |
| E9 | UZH coherence false reject (rate) | 0.111 | 0.1111111111111111 | 0.0001111 | **match** |  |
| E9 | UZH gain tier false reject at TII band (rate) | 0.556 | 0.5555555555555556 | 0.0004444 | **match** |  |
| E10 | TII rule window min (s) | 0.05 | 0.050306666666627975 | 0.0003067 | **match** |  |
| E10 | TII rule window max (s) | 0.22 | 0.2192088888888808 | 0.0007911 | **match** |  |
| E10 | UZH 500 Hz rule window min (s) | 0.34 | 0.3356 | 0.0044 | **match** | indoor_forward_6 |
| E10 | UZH 500 Hz rule window max (s) | 1.42 | 1.42 | 0 | **match** |  |
| E10 | UZH 15 Hz rule window samples min | 1.1 | 1.0714285714285714 | 0.02857 | **match** |  |
| E10 | UZH 15 Hz rule window samples max | 1.9 | 1.8750000000000002 | 0.025 | **match** |  |
| E10 | UZH gain tier at rule windows, false reject | 0.778 | 0.7777777777777778 | 0.0002222 | **match** |  |
| E10 | UZH coherence at rule windows, false reject | 0.0 | 0.0 | 0 | **match** |  |
| E10 | UZH median residual at rule windows, false reject | 0.0 | 0.0 | 0 | **match** |  |
| E2 | fusion pooled median flight tilt RMSE (deg) | 8.73 | 8.725145136220299 | 0.004855 | **match** |  |
| E2 | fusion pooled median lean error (deg) | 84.0 | 83.6426705340356 | 0.3573 | **match** |  |
| E2 | windowed azimuth lean error at 0.5 s (deg) | 4.4 | 4.429104150557388 | 0.0291 | **match** |  |
| E2 | windowed azimuth lean error at 1 s (deg) | 10.9 | 10.919995159672776 | 0.02 | **match** |  |
| E2 | windowed azimuth lean error at 2 s (deg) | 19.3 | 19.335009973127026 | 0.03501 | **match** |  |
| E2 | windowed azimuth lean error at 3 s (deg) | 84.6 | 84.61344666135692 | 0.01345 | **match** |  |
| E2 | windowed azimuth lean error at 4 s (deg) | 114.6 | 114.5665870183247 | 0.03341 | **match** |  |
| E2 | windowed azimuth lean error at 5 s (deg) | 89.7 | 89.74868707371161 | 0.04869 | **match** |  |
| E2 | one-azimuth-per-clip lean error (deg) | 85.5 | 85.50760308683462 | 0.007603 | **match** |  |
| E2 | gate S3 coverage | 0.6 | 0.597465034965035 | 0.002535 | **match** |  |
| E2 | good clips dropped by gate | 22 | 22 | 0 | **match** |  |
| E3 | first verdict held-out acceptance | 0.711 | 0.7108433734939759 | 0.0001566 | **match** |  |
| E3 | first verdict AUROC tilt removal | 0.918 | 0.9182755116852954 | 0.0002755 | **match** |  |
| E3 | first verdict Spearman score vs injected | 0.82 | 0.8218371964705434 | 0.001837 | **match** |  |
| E3 | primary threshold (deg) | 5.76 | 5.763397437386393 | 0.003397 | **match** |  |
| E3 | all-flights policy acceptance | 0.952 | 0.9518072289156626 | 0.0001928 | **match** |  |
| E3 | all-but-13a policy acceptance | 0.904 | 0.9036144578313253 | 0.0003855 | **match** |  |
| E3 | absorption by nuisances, tilt x1.5 (deg) | 6.2 | 6.246296360885713 | 0.0463 | **match** |  |
| E3 | real-clip score with both nuisances (deg) | 2.9 | 2.9137614323714005 | 0.01376 | **match** |  |
| E3 | pair-level tilt-removal rejection | 0.78 | 0.7796610169491526 | 0.000339 | **match** |  |
| E5 | cruise noise floor frame median (deg) | 0.89 | 0.8906300580736508 | 0.0006301 | **match** |  |
| E5 | cruise noise floor p99 (deg) | 6.7 | 6.695514751724086 | 0.004485 | **match** |  |
| E5 | edit recovery roll_5 (deg) | 3.27 | 3.2710169813925174 | 0.001017 | **match** |  |
| E5 | edit recovery roll_10 (deg) | 8.7 | 8.702928961682012 | 0.002929 | **match** |  |
| E5 | edit recovery roll_20 (deg) | 19.42 | 19.41601227299712 | 0.003988 | **match** |  |
| E5 | edit recovery pitch_5 (deg) | 2.48 | 2.476527689308559 | 0.003472 | **match** |  |
| E5 | edit recovery pitch_10 (deg) | 8.26 | 8.257248416071057 | 0.002752 | **match** |  |
| E5 | edit recovery pitch_20 (deg) | 19.15 | 19.145486797460855 | 0.004513 | **match** |  |
| E5 | generated clips dropped by gate | 7 | 7 | 0 | **match** |  |
| E5 | uptilt absorption, C, drag 0.295 (deg) | 6.4 | 6.467722203262957 | 0.06772 | **match** |  |
| E5 | uptilt absorption, C, drag-free (deg) | 3.4 | 3.4284418567880603 | 0.02844 | **match** |  |
| E5 | fraction at uptilt bound, C, drag 0.295 | 0.92 | 0.9183673469387755 | 0.001633 | **match** |  |
| E5 | peak T/W fraction above 3 at 0.25 s, A | 0.96 | 0.96 | 0 | **match** |  |
| E7 | implied-tilt RMSE, first run (deg) | 28.97 | 28.966865793993644 | 0.003134 | **match** |  |
| E7 | implied-tilt SNR, first run | 0.84 | 0.8404139069185803 | 0.0004139 | **match** |  |
| E7 | implied-tilt RMSE, corrected 5-sample (deg) | 32.67 | 32.6706706178576 | 0.0006706 | **match** |  |
| E7 | implied-tilt SNR, corrected | 0.58 | 0.5832442703263198 | 0.003244 | **match** |  |
| E7 | implied-tilt RMSE at 31 samples (deg) | 41.06 | 41.0561276196314 | 0.003872 | **match** |  |
| E7 | oracle ladder VO oracle scale + mocap attitude, corrected (deg) | 34.89 | 34.89407795090729 | 0.004078 | **match** |  |
| E7 | oracle ladder fully estimated fitted scale, corrected (deg) | 34.97 | 34.96697517105916 | 0.003025 | **match** |  |

## Mismatches and open discrepancies

1. **E1 drag reduction, flight 01p**: handoff quotes 5.68 deg; the stored result (`loo_folds`) gives 5.63 deg. The table uses 5.63. The other three piloted values match (5.72 vs 5.73 within tolerance, 3.89, 3.61).
2. **E2 gate null**: the permutation-null distribution is not stored as a file. The median (0.822) and 95th percentile (0.874) are transcribed from report 06; the figure marks them as reference lines, not recomputed data.
3. **E3 primary mode**: report 07 calls `k_d` = 0.295 primary, but the handoff's reference detection values (removal 0.50, acceptance 0.966, n = 58) match the drag-free gated run (`kd_0`). The figure is labelled drag-free; the threshold table lists all four modes. The later stored run 165308 gives removal 1.0 at `k_d` 0.295, not the 0.750 quoted in report 07; the difference is not resolved here.
4. **E3 acceptance/abstention by flight**: the latest stored run gives 1/2 on 04a where report 07 prints 2/2. The table uses the stored run.
5. **E5 infeasible counts**: the drag-free lag-only threshold (14.73 deg) has no matching `inf_*` key in the stored verdicts, so those cells are blank rather than approximated with the 12.95 deg key.
6. **E8 synthetic gain figure**: only the gain statistic is stored in `synthetic_sweep`; the median-residual and coherence curves versus synthetic gain are not stored and are omitted (see MISSING.md).
7. **E8 coherence cluster intervals**: the run's own interval file is not stored. The intervals were recomputed with the procedure D04 states (5,000 replicates, `default_rng(1)`, clip-weighted pooled rate) from the per-flight flags in D04, and reproduce D04's [0.00, 7.14] and [0.00, 4.54] exactly.
8. **E10 rule-window verdict**: at the rule windows the stored run gives x0.5 detection 0.667 (6/9) on the R2 band and 0.111 (1/9) on the coherence tier; E05 prints 0.700 and 0.200. The table prints the stored values with their denominators; the differences are one clip each.
9. **E9 median-residual population**: at the 0.37 s headline the stored run has 11 median-residual real clips and gives 0.636 (7/11) removal and 0.636 mirror detection, where the handoff expects 0.72/0.78 on a wider 18-clip population. The table uses the stored denominators.
10. **E10 median-residual population**: the rule-window run stores 11 real median-residual clips (0/11), not the 18 E05 quotes; detection is 7/11 (0.636) where E05 prints 0.722.
11. **E8 gain-test detection at matched false reject**: the stored operating curve at 5.3% FRR gives 100 / 100 / 100 / 96.5% for removal, mirror, x0.5, x1.5; report D01 quotes 100 / 94.7 / 98.2 / 94.7 at 3.5%. The table uses the stored curve points and labels its FRR; the report values are not reproducible from the stored curve.
12. **Track D HEAD moved during the build**: the handoff records about `59b3aaf`; the worktree was at `2a91765` when the reproducibility table was written. The MCP/agent sources should be re-read if the report needs an exact frozen cutoff.

## Known open discrepancies (from the handoff, not resolved)

* UZH window-rule windows: Track D's 0.07-0.12 s "at the truth rate" is superseded by E05 (0.34-1.42 s at 500 Hz; 0.07-0.11 s only at the 15 Hz clip rate). Followed in `tab_e10_window-rule-and-prototype_1.tex`; flagged in MISSING/corrections.
* UZH coherence false reject: 0.111 (1/9) at 0.37 s versus 0.000 (0/9) at the rule windows; both appear in `tab_e10_window-rule-and-prototype_2.tex`, each labelled with its window. The earlier 0.667 figure on the 27-clip 15 Hz export is not used.
* Numbers in `docs/prototype_v0.1.md` that predate E05: the scope matrix follows the later reports; the window claim is corrected in `corrections.csv`.
* Flight-cluster bootstrap draws: D03 says 20,000, D04's appendix says 5,000 with seed 1. The asset uses the stored `d_flights` interval; the coherence interval was recomputed with 5,000 draws and matches.
* Gain-tier testable-set composition: the decomposition uses autonomous 3 / track 4; the pre-dense-lean v2 population has autonomous 4 / track 3. The decomposition table states its set.
* D04 still contains the superseded UZH window statement and 35-62 samples at 500 Hz; see the window-rule bullet above.
* D01 holds first-run schema-v1/R1 numbers, D02 the final schema-v2/R2 numbers. Tables use D02; D01 values appear only in `corrections.csv`.
* Track A report 03's DPVO per-axis correlations (0.01, -0.09, 0.13) are quoted from the Phase 2 record and not re-derived; they are not used in any asset.
* Track A2 report 07 repeats sections 4-7; the first occurrence was used.
* TII per-frame error with a 5% versus 10% trim: `4.80` is the 10% trim; the 5% trim of the same frames is `10.20`. The asset labels the trim.

Checked 103 values: 100 match, 2 mismatch, 1 not checkable.
