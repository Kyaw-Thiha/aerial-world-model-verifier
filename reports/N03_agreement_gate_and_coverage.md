# N03 — Reader-versus-propagation agreement gate and coverage (UZH, descriptive)

**Decision.** A gate that abstains when the reader lean and the rotation-propagated lean disagree
collapses coverage on the only population where both exist: at the pre-registered 10 % and 20 %
targets it keeps **5 of 35** dynamic clips (14 %, all from one sequence) and detects **no**
violation, while the ungated F0 tier already flags 4/12 no-violation clips (33 %). The gate is
not usable here, and Gate B (reader versus Perspective Fields) was never computed on UZH, so no
like-for-like gate comparison exists on this set.

Pre-registration `configs/n03_gate_prereg.yaml` (before any gate number). Code
`experiments/n03_agreement_gate.py`. CPU only, zero spend. Results `results/n03/`.

## Plain-language summary

1. **The score.** For each clip, `d` = median angle between the reader lean and the propagated
   lean over the valid frames. It measures how much the two gravity sources disagree.
2. **The population.** 35 dynamic clips (peak required tilt 25-32°, so the ≥15° restriction costs
   nothing) with a valid rotation extraction, from three UZH sequences. Nine clips per condition
   become 3-6 after the extraction check (N02b).
3. **The curve is monotone only in the wrong direction.** Coverage rises from 0 % at `tau` = 0 to
   94 % at `tau` = 20°, but the no-violation flag count stays at 0 until `tau` ≈ 14 and reaches
   4/12 at 16 — the gate cannot separate: at every coverage the F0 false-flag rate is at least
   what it is ungated.
4. **The conformal selection lands on `tau` = 4.5°** for both targets, keeping 5/35 clips from
   `outdoor_forward_1` only (0 from `_3` and `_5`). On those 5 clips the F0 tier detects 0/1
   tilt removal and 0/2 x0.5. Abstaining has removed the violations along with the noise.
5. **Ungated reference:** 12 no-violation clips, 4 F0 flags (33 %); detections 2/3 tilt removal,
   2/6 x0.5, 2/5 x1.5, 2/3 mirrored roll. With three sequence clusters these counts are
   descriptive, not rates with intervals.
6. **Gate B is unavailable.** M02 built it on real TII and the simulated clips; it was not run on
   UZH. Quoting M02's 0.880/0.835 AUROC here would be a different population, and this report
   does not compare them as if they were the same test.

## Results

### Risk-coverage curve (F0 tier on the reader lean, leave-one-sequence-out thresholds)

| tau (deg) | coverage | no-violation flags | removal | mirror |
| --- | --- | --- | --- | --- |
| 0.0 | 0/35 | 0/0 | — | — |
| 2.0 | 1/35 | 0/0 | 0/1 | 0/0 |
| 4.0 | 3/35 | 0/0 | 0/1 | 0/0 |
| 6.0 | 7/35 | 0/2 | 0/1 | 0/1 |
| 10.0 | 15/35 | 0/6 | 0/1 | 0/1 |
| 14.0 | 22/35 | 2/10 | 0/1 | 1/2 |
| 18.0 | 31/35 | 4/12 | 1/2 | 2/3 |
| 20.0 | 33/35 | 4/12 | 1/2 | 2/3 |

The full 0.5° grid is `results/n03/curve.csv`.

### The pre-registered selection

| target | tau | coverage | no-violation F0 reader | removal | x0.5 | x1.5 | mirror | per sequence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.10 | 4.5° | 5/35 | 0/2 | 0/1 | 0/2 | 0/0 | 0/0 | 5/15, 0/6, 0/14 |
| 0.20 | 4.5° | 5/35 | 0/2 | 0/1 | 0/2 | 0/0 | 0/0 | 5/15, 0/6, 0/14 |

Both targets select the same threshold because the no-violation rate is already 0 at 4.5° (two
clips kept) and the next steps add violations. The gate is not trading coverage for safety; it is
discarding almost everything.

### Ungated

| population | n | F0 reader flags |
| --- | --- | --- |
| real + null | 12 | 4 (33 %) |
| tilt_removal | 3 | 2 |
| scale_0.5 | 6 | 2 |
| scale_1.5 | 5 | 2 |
| mirror_roll | 3 | 2 |

## What this supports

* On this set the disagreement score is dominated by the reader's own scene bias and the
  propagation's start error; it does not identify the clips where the reader is wrong, so it
  cannot be the abstention signal the track hoped for.
* A coverage claim must be made over dynamic clips and after the extraction check; both reduce
  the usable population sharply (100 % dynamic, 35/63 extracted), and the survivors are not spread
  across sequences.

## What this does not show

* Anything about Gate B, or about the combination reader-versus-Perspective-Fields with this
  gate: not computed on UZH.
* Anything confirmatory: three sequences, 5-12 clips per cell, no intervals.
* Anything about generated or simulated clips: Part 4 tracked generated video, but no verdict
  labels exist there.

## Caveats

* The gate's conformal selection is pooled over no-violation clips; the pre-registration asked for
  a sequence-clustered rate, which with three clusters is descriptive only. Both readings are
  stated and neither is a 95 % bound.
* A 5 % target is not certifiable at 6-12 no-violation clips and was not attempted; 10 % and 20 %
  are reported instead, as pre-registered.
* The F0 LOSO thresholds are estimated from six clips, so each threshold is effectively the
  maximum of the other sequences' real scores; this is why the ungated false-flag rate is high.

## Provenance

* Pre-registration `configs/n03_gate_prereg.yaml`; decisions `[N]`.
* Inputs: `results/n02b/per_clip.csv` (the 63 UZH v3 clips, valid extraction), `results/n02b`
  thresholds; no GPU, USD 0.00.
