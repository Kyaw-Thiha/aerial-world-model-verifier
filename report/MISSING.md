# Missing and pending assets

Nothing here is estimated or fabricated. Each item states what is missing and what would
make it buildable.

## Pending (source work still in progress)

1. **E6 depicted-tilt scoring of injected clips.** Generation is complete; the scoring arm
   is not. The LPIPS and prior-coverage assets are built; no verdict table or
   depicted-vs-designed figure exists for the injected clips. Needs the scoring run's
   per-clip residual/threshold output.
3. **E5/E4 coarse-path rule.** `configs/coarse_path_rule_proposal.yaml` is a proposal only.
   No result table is built from it (as instructed).

## Report-sourced (stored only as prose)

2. **E5 minimum consistent dt.** Built in `tab_e5_anwm-verdict_8` from the stored
   per-clip `minimum_consistent_dt`; the medians (1.08 s and 1.51 s drag-free for the 10°
   and 20° families) are close to, but not identical with, report 08's 1.10 and 1.58 s,
   because the report quotes a different per-cell aggregation.

* **E3 pre-registered criteria / policy thresholds**: the evaluation block stores
  acceptance and AUROC but not the alternative-policy thresholds (67.16°, 9.37°) or the
  per-clip real scores. Those are parsed/transcribed from report 04 (thresholds in the
  script, scores parsed from the table) and flagged in the source files.
* **E5 Regime A statistic comparison**: the amendment-3 versus frame-level counts exist
  only in report 08 §9; `tab_e5_anwm-verdict_7` transcribes them and says so.
* **E2 permutation null**: unchanged — summary statistics only, from report 06.

## Missing source data

4. **E8 median-residual and coherence curves versus synthetic gain.** `synthetic_sweep`
   stores only the gain statistic (`gain`, `gx`, `gy`, `flagged`). The baseline and
   coherence tiers were not scored on the synthetic sweep, so the three-method matched-FRR
   figure cannot be built. `fig_e8_gain-and-coherence-tests_1.pdf` shows the gain curve
   alone. Needs baseline and coherence scores on the same scaled series.
5. **E8 depicted-vs-truth lean per axis (still missing).** Per-clip, per-axis depicted and truth lean
   arrays are not in the repository (the Deming table stores fitted gains, not the paired
   axis samples). The scatter figure is omitted. Needs the per-clip lean export from the
   frozen d02 population.
6. **E2 permutation-null distribution.** Only the summary statistics (median 0.822, 95th
   percentile 0.874) are stored, in report 06 prose. The figure shows them as reference
   lines; the null draws themselves are not available.
7. **R2 object counts — now measured (2026-11-15).** With the updated credentials,
   `rclone size` gives: track-a 288 (incl. a2 122), track-b 3,641, track-c **0**,
   track-d 67, track-e 595. `cutoff.json` and `data/r2_counts.json` record counts and
   bytes. The `track-c/` prefix is empty in this bucket, so Track C has no R2 objects to
   cite; its source is the git branch only.
8. **Test counts and cost per branch.** These are stated in the branch reports, not stored
   as result files. `tab_repro_1` leaves the counts column as "--" rather than copying
   prose numbers; the reports are the source (for example D01: 131 tests; b04: about
   US$7).

## Known discrepancies (not resolved by design)

9. **E1 drag reduction for flight 01p**: stored 5.63 deg versus the handoff's 5.68 deg.
   Table uses 5.63; recorded in `CHECKS.md`.
10. **E3 primary mode**: the handoff's reference detection values match the drag-free run,
    while report 07 calls drag 0.295 primary. Both are represented; the figure is labelled
    drag-free.
11. **E7 DPVO calibrated direction error**: 4.18 deg in the v2 summary versus 3.9 deg in
    report C08's focal-length table. Both are stored runs; the table uses the bakeoff
    summary and `CHECKS.md` shows the difference.
12. **E9/E10 median-residual populations**: the stored 0.37 s and rule-window runs carry
    11 median-residual real clips (0/11), not the 18 quoted by E05. Detection is 7/11
    (0.636) where E05 prints 0.722 / 0.778. Tables use the stored denominators.
13. **E10 UZH x0.5 at rule windows**: 6/9 (0.667) in the stored rule run versus 0.700 in
    E05's text. Stored value used.

## Environment

13. `pandas` and `scipy` are not installed in the default Python; the report scripts use
    only `numpy` and `matplotlib`. No `report/.venv` was needed.
14. LaTeX is installed; `assets_test.tex` compiles with `pdflatex` (lmodern is absent, so
    the test document does not load it).
