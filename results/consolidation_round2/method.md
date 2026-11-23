# Modal billing versus the track ledgers (round 2, wave 2)

Command (2026-11-23):

```
modal billing report --start 2026-11-16 --end 2026-11-24 --json
```

- Raw output: `modal_billing_2026-11-16_2026-11-24.json` (291 entries, total USD 121.27).
- Per-track billing is the sum of `cost` over entries whose app `description` starts
  `awmv-<track-prefix>-` (`f`, `f2`, `g`, `g2`, `g3`, `h`, `h2`, `h3`, `i`, `j`, `k`, `l`, `m`, `n`).
- `ledger_usd` is the total the track's ledger states, or the sum of its actual rows when it states
  no total. The basis is given per row in `billing_vs_ledger.csv`; estimates and planned totals are
  labelled as such.
- The two are not expected to be equal: the ledger rows are wall-clock arithmetic at list prices
  (some are estimates or planned), the billing export is actual charge by app, several jobs are
  CPU (not tracked as an app here), and the date window is fixed. Differences are recorded, not
  reconciled.
