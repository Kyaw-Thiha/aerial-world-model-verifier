# Missing and pending assets

Nothing here is estimated or fabricated. Each item states what is missing and what would
make it buildable.

## Pending (source work still in progress)

1. **E6 depicted-tilt scoring of injected clips.** Generation is complete; the scoring arm
   is not. The LPIPS and prior-coverage assets are built; no verdict table or
   depicted-vs-designed figure exists for the injected clips. Needs the scoring run's
   per-clip residual/threshold output.
2. **E5 smallest step duration consistent with level frames.** The per-mode minimum-dt
   table exists only as prose in report 08 (§4) and is not stored as a result file. The
   built `tab_e5_anwm-verdict_2` carries infeasible/testable counts only; the minimum-dt
   column is omitted rather than typed from the report.
3. **E5/E4 coarse-path rule.** `configs/coarse_path_rule_proposal.yaml` is a proposal only.
   No result table is built from it (as instructed).

## Missing source data

4. **E8 median-residual and coherence curves versus synthetic gain.** `synthetic_sweep`
   stores only the gain statistic (`gain`, `gx`, `gy`, `flagged`). The baseline and
   coherence tiers were not scored on the synthetic sweep, so the three-method matched-FRR
   figure cannot be built. `fig_e8_gain-and-coherence-tests_1.pdf` shows the gain curve
   alone. Needs baseline and coherence scores on the same scaled series.
5. **E8 depicted-vs-truth lean per axis.** Per-clip, per-axis depicted and truth lean
   arrays are not in the repository (the Deming table stores fitted gains, not the paired
   axis samples). The scatter figure is omitted. Needs the per-clip lean export from the
   frozen d02 population.
6. **E2 permutation-null distribution.** Only the summary statistics (median 0.822, 95th
   percentile 0.874) are stored, in report 06 prose. The figure shows them as reference
   lines; the null draws themselves are not available.
7. **R2 object counts.** `rclone` can see the `r2remote` remote but `ListBuckets` returns
   403, and there is no `.env` with credentials (only `.env.example`). `cutoff.json`
   records the R2 prefixes with `r2_object_count: null` and the reason.
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
12. **E10 UZH x0.5 at rule windows**: 6/9 (0.667) in the stored rule run versus 0.700 in
    E05's text. Stored value used.

## Environment

13. `pandas` and `scipy` are not installed in the default Python; the report scripts use
    only `numpy` and `matplotlib`. No `report/.venv` was needed.
14. LaTeX is installed; `assets_test.tex` compiles with `pdflatex` (lmodern is absent, so
    the test document does not load it).
