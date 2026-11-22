# K01 — Reconciling the calibration-mode labels of every published floor

Track K, Part 1. Written 2026-11-22. Code `experiments/k01_calibration_modes.py` (commit
`6043479`), results `results/k01/20261122T091406Z/` (`k01.json`, `mapping.csv`). CPU
only, zero GPU, zero spend. No frozen config or published report was edited; the proposed
corrections are a note in `docs/decisions.md` and §6 here.

## Decision

**The published `14.73` is genuinely drag-free, lag-only — re-running A2's own construction
reproduces it to the printed precision — and the "label conflict" was created by Track J's
consumer code, not by report 07: schema v2's `lean_req` is the drag-free required lean, so
J02's `required_at(k_d) = lean_req + (k_d − 0.295)·vel_req` shifts every J mode by 0.295
(its "k_d = 0.295" arm is drag-free; its "drag-free" arm is k_d = −0.295).** J03's F0
threshold (`14.05`) is therefore drag-free, not k_d = 0.295, and its drag grid spans
[−0.295, +0.105] in physical units; the exact k_d = 0.295 construction on the same
population gives **8.90** (no lag) or **7.68** (with the lag nuisance), not 13.43.

## Plain-language summary

1. **The four A2 numbers reproduce exactly.** Drag-free lag-only `14.73` (folds 13.5–16.5,
   n = 58), drag-free full parity `12.95`, k_d 0.295 lag-only `8.96`, k_d 0.295 full parity
   `6.46` with the ANWM cruise p99 `8.48` binding. Report 07's labels match the code that
   produced them.
2. **`lean_req` in schema v2 is drag-free.** Compared with a direct native-rate recompute,
   the stored array agrees to a median **0.016°** (p99 0.40°); it is **4.26°** (median) from
   the exact k_d = 0.295 lean. `vel_req` is `v_horizontal/g`.
3. **The linear drag reparameterisation is not the frozen construction.** `lean_req + k·vel_req`
   differs from the exact `normalise(a + g e3 + k v)` by a median 2.17° on testable frames
   and a p95 of 20.4°, and it can leave the unit disk (worst |lean| 1.06). That is why J03's
   "k_d = 0.295" F0 gives 13.43 rather than 8.90.
4. **J02's mode labels are shifted by one step.** Its `14.46` anchor row is drag-free (the
   pre-`f7ba108` code used `required = lean_req`); its `31.11` row is k_d = −0.295 (the
   new `ARM_B_KD = 0.0` with the shifted formula). So `14.46` vs `14.73` is a
   mode-consistent agreement between two drag-free constructions, not a contradiction.
   `6.9 / 4.9`, `10.65`, `6.88 / 8.35` carry the same shift.
5. **J03's F0/C1 mode labels are wrong and its drag grid is mis-centred.** F0 (14.05) and
   C1 (15.91) are drag-free; C2 (13.84) evaluates `k_d,eff = k − 0.295` on
   [−0.295, +0.105]. Re-running J03's own code with the schema read correctly puts the
   per-clip fitted k_d **median at 0.295** (26 of 69 clips exactly there, 7 at 0, 9 at 0.40),
   so its statement that the drag nuisance "pins to its bound and is not measuring drag" is
   an artefact of the 0.295 offset. The absorption *measurement* is not re-derived here.
6. **Every number in `mapping.csv` now carries its construction, true mode, nuisance,
   population and source commit.** No published number changes; four published *labels* do.

## Question

Which construction, calibration mode, nuisance set and population produced each published
floor and threshold — `14.73, 8.96, 14.46, 14.05, 31.1, 6.9, 10.5, 10.65`, the E1 floors
`1.24 / 5.46 / 1.22` and the other numbers in report 07, E1 and E3 — and which labels are
wrong?

## Setup

* **A2's construction, re-run.** `experiments/a2_modes.py` was executed unmodified at
  k_d ∈ {0, 0.295} × nuisance ∈ {lag-only (uptilt offset fixed at 0), full parity (uptilt
  offset −7…+7° plus lag)} on the same eight flights, the frozen gate applied per clip, the
  same schema-v2 clip files. Run `results/a2_modes/20261122T071523Z` (commit `e7e3d84`),
  compare with the stored 2026-11-10 run.
* **E1's construction, re-run.** `experiments/phase0_floor.py` unmodified; run
  `results/phase0_floor/20261122T072349Z_41d1dad5` (commit `41d1dad`), 114,059 held-out
  frames, leave-one-flight-out.
* **Schema semantics, read from the files.** For every real clip, `lean_req` and
  `vel_req` were compared with a native-rate recomputation of the drag-free lean, the
  k_d = 0.295 lean, and the schema's linear combination. 4,777 finite frames over 100
  clips and 8 flights.
* **J's numbers, re-derived and re-read.** J02 was re-run in a clean clone of
  `track-j-reader` at `f7ba108` (stored summary reproduced: arm (c) 16.38/31.11; perfect
  path 6.44/6.88); its pre-`f7ba108` arm (c) used `required = lean_req` and produced the
  reported 14.46. J03 was re-run unmodified (14.05 / 15.91 / 13.84, k_d pins at 0.40) and
  again with `lean_at(b,k) = lean_req + k·vel_req` and the k grid read physically.
* **F0 on one population, three readings.** Gate-kept six dev flights (69 clips), J03's
  per-clip leave-one-flight-out 95th percentile: (a) stored linear schema at drag-free,
  (b) stored linear schema with the k_d term, (c) exact `normalise(a + g e3 + k_d v)` from
  native mocap, with and without the per-clip lag nuisance.

## Results

### 1. A2's four modes reproduce (run 20261122T071523Z, commit `e7e3d84`)

| k_d | nuisance | pooled conformal | fold spread | n testable | median real residual | primary | detection floor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | lag only | **14.73** | 13.5–16.5 | 58 | 7.38 | 14.73 | 15.62 |
| 0 | full parity | 12.95 | 10.6–13.8 | 58 | 5.61 | 12.95 | 13.84 |
| 0.295 | lag only | 8.96 | 9.0–12.1 | 59 | 3.09 | 8.96 | 16.62 |
| 0.295 | full parity | 6.46 | 5.8–10.1 | 59 | 2.47 | 8.48 | 16.14 |

These are report 07 amendment 9's four values and its four detection floors (15.6 / 13.8 /
16.6 / 16.1), reproduced digit for digit. The k_d 0.295 full-parity primary is 8.48 because
the ANWM straight-cruise p99 is binding; the drag-free ANWM cruise null is 0.89 (p99 6.70),
which is why the mode also changes the null.

### 2. The schema's required side is drag-free (4,777 frames)

| comparison | median | p95 | p99 |
| --- | --- | --- | --- |
| stored `lean_req` vs recomputed drag-free lean | **0.016°** | 0.16° | 0.40° |
| stored `lean_req` vs recomputed k_d 0.295 lean | 4.26° | 11.8° | 18.0° |
| `lean_req + 0.295·vel_req` vs exact k_d 0.295, all frames | 1.40° | 19.1° | 25.8° |
| the same, testable frames (tilt ≥ 10°, ‖a+g e3‖ ≥ 2 m/s²), n = 3,521 | **2.17°** | **20.4°** | 26.7° |

The first row is the decisive measurement: the stored array is the drag-free lean to
two decimal places. The last two rows show that the schema's linear drag term is an
approximation, and a poor one on aggressive frames: the summed lean vector can exceed unit
norm (worst 1.06), where `up_body_from_lean` clips it. `vel_req` itself reproduces the
recomputed `v_h/g` to 3e-4.

### 3. J03's F0 on one population, three readings (gate-kept dev, 69 clips)

| reading | required side | LOFO threshold | median residual |
| --- | --- | --- | --- |
| J03 as written | stored `lean_req` (= drag-free) | **14.05** | 10.17 |
| exact drag-free | `normalise(a+g e3)` | 14.10 | 10.18 |
| J03's intended k_d = 0.295, linearised | `lean_req + 0.295·vel_req` | 13.43 | 3.34 |
| exact k_d = 0.295, no lag | `normalise(a+g e3+0.295v)` | **8.90** | 3.72 |
| exact k_d = 0.295, per-clip lag | same, lag nuisance | **7.68** | — |

The 14.05 is the drag-free construction to within the linearisation's 0.05° at k_d = 0;
the 13.43 is what the schema's linear term gives when it is (incorrectly) called
k_d = 0.295; the frozen construction gives 8.90. On the gate-kept dev population the mode
is therefore worth **5.2°** of threshold, not 0.6°.

### 4. The mapping (full table in `mapping.csv`)

| value | what | true mode | nuisance | population | source | label |
| --- | --- | --- | --- | --- | --- | --- |
| 14.73 | conformal threshold | drag-free | lag only | 8 flights, gate, 58 | A2 / reproduced | correct |
| 12.95 | conformal threshold | drag-free | full parity | 8 flights | A2 | correct |
| 8.96 | conformal threshold | 0.295 | lag only | 8 flights, 59 | A2 | correct |
| 6.46 (primary 8.48) | conformal threshold | 0.295 | full parity | 8 flights | A2 | correct |
| 1.24 / 5.46 / 1.22 | pooled held-out median residual | fitted ~0.297 / 0 / body k_xy | none | 8 flights, 114,059 frames | E1 / reproduced | correct |
| 14.46 | mocap-attitude anchor threshold | **drag-free** | none | 100 clips | J02 (pre-`f7ba108`) | **mode wrong** |
| 31.11 | mocap-attitude anchor threshold | **k_d = −0.295** | none | 100 clips | J02 (`f7ba108`) | **mode wrong** |
| 14.05 | F0 threshold | **drag-free** | none | gate-kept dev 69 | J03 | **mode wrong** |
| 15.91 | C1 threshold | **drag-free** | offset | gate-kept dev 69 | J03 | **mode wrong** |
| 13.84 | C2 threshold | **grid shifted −0.295** | offset + k_d | gate-kept dev 69 | J03 | **mode wrong** |
| 6.9 / 4.9 | whole-clip MDV, perfect path | **drag-free / k_d = −0.295** | none | 100 clips | J02 | **mode wrong** |
| 10.65 | MDV, measured-error replay | **drag-free** | none | 100 clips | J02 | **mode wrong** |
| 6.88 / 8.35 | perfect-path floor, clip / 8-frame | **drag-free** | none | 100 clips | J02 | **mode wrong** |

### 5. What the J numbers mean once the shift is removed

* J02's anchor at the true drag-free mode is 14.46, against A2's real-clip drag-free 14.73.
  Two different constructions, same mode, 0.27° apart. There is no conflict to resolve.
* J02's k_d = −0.295 arm (31.11) is not a physical model (drag would push the thrust axis the
  wrong way). Its 4.9° "drag-free" MDV is likewise the negative-drag arm.
* J03's F0 is the drag-free threshold on 69 dev clips without lag (14.05); A2's drag-free
  threshold with lag on 58 clips is 14.73. Mode-matched, population- and nuisance-different.
* J03's per-clip drag fit, read physically, centres on **0.30** (the 0.05 grid's nearest
  point to 0.295; histogram over the 72 gate-kept real clips: 8 at 0.00, 2 at 0.10, 3 at
  0.15, 2 at 0.20, 15 at 0.25, 25 at 0.30, 8 at 0.35, 9 at 0.40; `k01.json`'s
  `j03_drag_corrected`). The earlier "65 of 72 at the bound" is 65 at `k_d,eff = +0.105`.

## What this supports

* A clean answer to the handoff's first question: **report 07's labels are right, and the
  conflict came from J02/J03 reading schema v2 backwards.** The correction is mechanical
  and verifiable: `lean_req` is drag-free; `lean_req + k_d·vel_req` is a linearisation.
* A single mapping table that can be used by anyone quoting a floor, with the four axes that
  actually change it (construction, mode, nuisance, population).
* A quantitative warning for Part 2: the schema's linear drag term is off by a median 2.2°
  and a p95 20° on testable frames, so a schema-v3 required side must store the exact model
  axis, not a linear combination.

## What this does not show

* **It does not re-derive J03's absorption table.** The absorption numbers (31/42 removal,
  etc.) are a measurement of what a per-clip fit does; we report only that the mode labels
  and the k_d grid are wrong and show the corrected diagnostics and F0. A full re-run of
  the absorption under exact semantics is left open and named in `docs/decisions.md`.
* **It does not re-score ANWM.** Only the TII nulls that set the primary threshold were
  recomputed.
* **It does not change any published number.** The numbers are what their code produced;
  the labels are what is wrong.
* **The `14.46` row cannot be regenerated from the current J02 script**, because commit
  `f7ba108` changed `ARM_B_KD` from the implicit 0.295 to 0.0. The value is attributed to
  the pre-`f7ba108` code path and to J02's report, and the current script reproduces the
  31.11 row. Stated rather than smoothed over.

## Caveats

* Populations differ between many rows (all 8 flights vs gate-kept six dev flights, 58/59/69
  clips, lag/no lag). The table states each; the numbers are not interchangeable.
* `14.05` vs our exact 14.10 is a 0.05° difference from the linear versus exact construction
  at k_d = 0, not a disagreement about the mode.
* The linearisation gap's p95 is computed on testable frames only (3,521 of 4,777); the
  excluded frames include near-free-fall samples where the thrust direction is ill-conditioned.
* The F0 numbers use J03's per-clip LOFO convention (median over per-clip 95th percentiles,
  clip-frequency weighted). A per-flight median gives 12.53/8.24 on the same values; both
  are reported in `k01.json`'s construction notes and neither is a confidence interval.
* Single-flight cells do not arise here; every rate in this report is a threshold or a
  population median, not a detection rate.

## 6. Proposed corrections (note only; no published report edited)

For `reports/J02_noise_budget.md`:

* `required_at(kd)` should be `lean_req + kd·vel_req` if the schema is drag-free, or the
  schema should store the k_d = 0.295 lean and the code should keep `(kd − 0.295)`. As
  written it is the former with the latter's label. The report's mode table should read:
  its "k_d = 0.295" rows are drag-free, its "drag-free" rows are k_d = −0.295, and the
  anchor rows are 14.46 (drag-free) / 31.11 (k_d = −0.295).
* Consequently the sentence "the published number matches the k_d = 0.295 construction while
  its label says drag-free" should be withdrawn: 14.46 and 14.73 are both drag-free.

For `reports/J03_per_clip_calibration.md`:

* F0 and C1 are drag-free, not k_d = 0.295. The corrected exact k_d = 0.295 F0 on the same
  population is 8.90 (no lag) / 7.68 (lag), not 13.43 (linearised).
* The k_d grid is `k_d,eff = k − 0.295`, so the bound 0.40 means +0.105 and 0.00 means
  −0.295. With the grid read physically the fitted median is 0.295 and the "pins at its
  bound / not measuring drag" conclusion does not follow.
* `lean_at()` should not be used to represent a drag coefficient: on 3,521 testable frames
  the linear sum differs from the exact flatness lean by a median 2.17° and a p95 20.4°,
  and exceeds unit norm on some frames.

## Regenerate

```
PYTHONPATH=. python experiments/k01_calibration_modes.py    # ~40 s, CPU
```

Outputs `results/k01/<run>/{k01.json,mapping.csv}`. The A2 mode run is
`results/a2_modes/20261122T071523Z` (32 s) and the E1 reproduction is
`results/phase0_floor/20261122T072349Z_41d1dad5` (42 s), both from the same worktree.
