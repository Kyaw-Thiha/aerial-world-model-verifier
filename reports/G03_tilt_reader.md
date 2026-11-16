# G03 — Fine-tuning the depicted-tilt reader on mocap-labelled frames

Track G. Written 2026-11-16. Pre-registration `configs/g03_tilt_reader_prereg.yaml`. Code:
`experiments/g03_prepare_frames.py`, `experiments/g03_prepare_uzh.py`,
`remote/g_tilt_reader.py` (Modal, A10G), `experiments/g03_evaluate.py`. Results:
`results/g03_tilt_reader/`, Modal volume `awmv-g-data`.

## Plain-language summary

1. **The reader improves: pooled held-out median error 3.97 -> 2.87 deg (-28 %), trimmed
   RMSE 4.81 -> 3.71, and the frames above 10 deg error fall from 12.7 % to 10.8 %.** Every one
   of the eight leave-one-flight-out folds improves at the median; the two hard track flights
   improve most in the tail (13p trimmed RMSE 18.4 -> 9.9, 16p 14.1 -> 9.2).
2. **The pre-registered success criterion is met on one of its two clauses**: the median falls
   by at least 25 % (it falls 28 %); the fraction above 10 deg does not fall by a third (12.7 ->
   10.8 %, a sixth). So this is a real but partial improvement, and the detection-relevant tail
   is what did not move far enough.
3. **Downstream, the clip-level median residual drops on six of eight flights** (for example
   `flight-01p-ellipse` 10.2 -> 7.7 deg, `flight-13p-trackRATM` 16.3 -> 13.7, `flight-13a-trackRATM`
   7.8 -> 4.7) and rises on two (`flight-16p-trackRATM` 17.9 -> 18.2 and `flight-07a-lemniscate`
   1.0 -> 3.0, a flight whose clips were nearly perfect before). A better per-frame estimate does
   not by itself give a better clip score on every flight.
4. **With the frozen gate and a leave-one-flight-out threshold calibrated on gate-kept clips,
   the new reader false-rejects 5 of 72 clips (6.9 %) and flags 55 of 72 (76.4 %) under a full
   tilt removal** — against a frozen reader that at the same threshold flags 2/3 of 3 on the
   held-out block (report G01). The per-flight spread is still the binding constraint: one
   flight (`07p`) flags 2 of its 25 own real clips because the threshold fitted on the other
   seven does not transfer to it.
5. **Abstention does not move: 18 of 100 real clips abstain under the frozen rules that do not
   need a bootstrap, and that is the same 18 with the frozen reader.** The improvement is in the
   estimate, not in admissibility: abstention is set by the path's excitation and by frames with
   no estimate, not by how large the error is. The published 43 % includes the gain tier's
   bootstrap-CI rule, which is not recomputed here.
6. **UZH transfer is reported separately and is modest**: the frozen pinhole reader is already good on this rig, and the fine-tune halves its median
   without moving the tail. Per sequence (leave-one-sequence-out, 1,097 frames): median 1.28 ->
   0.64 deg on `indoor_forward_3`, 1.08 -> 0.61 on `indoor_forward_5`, 1.48 -> 0.74 on
   `indoor_forward_9`; the fraction above 10 deg goes 1.6 -> 2.9 % on `forward_3` and stays at
   0 % on the other two, so the tail does not improve. The baseline column comes from a separate
   `epochs = 0` pass of the same harness on the same frames (the frozen weights), merged by frame
   name. The UZH route is the pinhole export with the pinhole weights and is not comparable with
   the TII raw-fisheye route; the 45-degree-down rigs are excluded (E03), and no UZH clip
   statistics are computed.

## Question

The detection floor (14-17 deg of required tilt), the 43 % abstention rate and the false-reject
spread are set by the depicted-tilt reader, not by the physics (floor 1.24 deg). Can
fine-tuning GeoCalib on mocap-labelled attitude frames, with leave-one-flight-out folds, reduce
its per-frame error and move the downstream verdict numbers?

## Setup

**Labels, not pixels.** Each frame's label is the world-up direction expressed in the camera
frame, `up_cam = R_cb^T R_bw^T e3`, with `R_bw` the mocap (TII) or Leica (UZH) body attitude and
`R_cb` the analytic camera-to-body mount the verdict uses (TII recorded uptilt) or Kalibr's
published `T_cam_imu` (UZH). No pixel estimate enters a label.

**Frame sets.**

| dataset | units | frames | route | what it is |
| --- | --- | --- | --- | --- |
| TII | 8 flights | 5,007 | raw fisheye, `weights="distorted"`, camera model `simple_divisional` | the exact frames the verdict scores (`data/processed/p3a/p3a_v1` names), extracted from the release zips |
| UZH-FPV | 3 forward sequences | 1,097 | pinhole export, `weights="pinhole"`, camera model `pinhole` | the transfer rigs; the 45-degree-down rigs are excluded (E03) |

**Baseline, reproduced before the fine-tune existed.** The frozen reader's stored per-frame
gravity (`results/gate_a2/tii/<flight>__identity.jsonl`) scored against these labels gives
pooled median **3.97 deg**, 90 %-trimmed RMSE **4.81 deg**, **12.7 %** of frames above 10 deg
and **7.6 %** above 20 deg — the published 3.96 / 4.80 / 12.7 % (report 02). The label
construction is therefore validated against numbers produced independently of it.

**Protocol.** Leave-one-flight-out (8 folds) for TII and leave-one-sequence-out (3 folds) for
UZH; no frame of a held-out unit enters training, augmentation or any weighting. GeoCalib's
network is fine-tuned end to end with its own LM refinement in the loop (that is the signal the
authors train on; `GeoCalib.calibrate` is `@torch.no_grad`, so the forward is replicated from
`geocalib/extractor.py`), loss `1 - cos(predicted up, labelled up)`, AdamW at lr 2e-5 with a
cosine schedule, 2 epochs, batch 2, six LM steps, up to 100 frames per fit unit per epoch, no
geometric augmentation (a rotation would have to be undone on the label vector, and getting that
wrong corrupts labels silently).

**Downstream.** Only the depicted side changes: each Track D schema-v2 clip's `depicted_lean` is
rebuilt from the fine-tuned gravity on that clip's own frames, the frozen path side is reused,
and the frozen clip-level statistics are recomputed (median residual, coherence, abstention,
"removal" = depicted lean set to zero, which is what the pixel tilt-removal edit does to the
estimate). Thresholds are leave-one-flight-out on the fit flights.

## Results

### Per-frame error, held-out unit of each fold

Each frame is scored by the model of the fold that held its flight out; the baseline column is
the frozen reader's stored estimate on the same frame.

| unit | n | baseline median (deg) | fine-tuned median | baseline trim RMSE | fine-tuned trim RMSE | baseline >10 deg | fine-tuned >10 deg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flight-01a-ellipse | 270 | 4.20 | 3.45 | 4.27 | 3.39 | 0.111 | 0.048 |
| flight-01p-ellipse | 1267 | 3.13 | 1.87 | 3.27 | 1.88 | 0.008 | 0.003 |
| flight-04a-ellipse | 350 | 3.23 | 2.70 | 3.25 | 2.74 | 0.057 | 0.011 |
| flight-07a-lemniscate | 370 | 2.54 | 1.90 | 6.54 | 5.22 | 0.176 | 0.168 |
| flight-07p-lemniscate | 1284 | 3.60 | 2.70 | 3.71 | 2.82 | 0.005 | 0.001 |
| flight-13a-trackRATM | 557 | 7.43 | 4.42 | 34.66 | 31.53 | 0.336 | 0.327 |
| flight-13p-trackRATM | 536 | 7.75 | 6.52 | 18.40 | 9.86 | 0.354 | 0.278 |
| flight-16p-trackRATM | 373 | 7.32 | 6.96 | 14.14 | 9.21 | 0.346 | 0.332 |

**Pooled TII (5,007 frames):** baseline median 3.97 deg, trimmed RMSE 4.81, 12.7 % above 10 deg,
7.6 % above 20 deg, p90 13.1; fine-tuned median 2.87 deg, trimmed RMSE 3.71, 10.8 % above
10 deg, 6.5 % above 20 deg, p90 10.7. Per-fold training: 700 frames from seven units, two
epochs, 740-780 s of A10G per fold (`history` in each fold file).

The per-fold pattern matters more than the pooled number: the flights that were already good
(01p 3.13, 07p 3.60) improve by 35-40 % at the median, while the track flights (13a, 13p, 16p)
improve mainly in the tail — 13a's trimmed RMSE stays at 31.5 because its worst frames are
near-ceiling views that no amount of TII attitude supervision fixes.

### Downstream, clip level

`depicted_lean` rebuilt from the fine-tuned gravity on each clip's own frames; path side,
statistics and gate frozen; threshold is the leave-one-flight-out 95th percentile of the clip
median residual, computed twice (all real clips, and the gate-kept clips only).

| flight | tau ungated / gated (deg) | real flagged (gated) | removal flagged (gated) | abstained | median residual frozen -> ft |
| --- | --- | --- | --- | --- | --- |
| flight-01a-ellipse | 23.0 / 12.4 | 0/5 | 2/5 | 3/5 | 5.1 -> 3.3 |
| flight-01p-ellipse | 33.2 / 12.9 | 0/26 | 24/26 | 0/26 | 10.2 -> 7.7 |
| flight-04a-ellipse | 24.1 / 12.4 | 0/6 | 1/6 | 4/7 | 3.2 -> 2.4 |
| flight-07p-lemniscate | 33.2 / 12.2 | 2/25 | 24/25 | 1/26 | 10.7 -> 8.6 |
| flight-13a-trackRATM | 18.0 / 12.4 | 0/6 | 0/6 | 6/11 | 7.8 -> 4.7 |
| flight-16p-trackRATM | 22.9 / 12.2 | 1/1 | 1/1 | 0/7 | 17.9 -> 18.2 |
| flight-13p-trackRATM | 26.3 / 11.8 | 2/3 | 3/3 | 0/11 | 16.3 -> 13.7 |
| flight-07a-lemniscate | 19.4 / 12.4 | 0/0 | 0/0 | 4/7 | 1.0 -> 3.0 |
| **total** | | **5/72 (6.9 %)** | **55/72 (76.4 %)** | **18/100** | |

### UZH-FPV forward rigs (separate route, separate folds)

Three sequences, 1,097 pinhole frames, leave-one-sequence-out, `weights="pinhole"`:

| sequence | n | baseline median (deg) | fine-tuned median | baseline >10 deg | fine-tuned >10 deg |
| --- | --- | --- | --- | --- | --- |
| indoor_forward_3 | 450 | 1.28 | 0.64 | 0.016 | 0.029 |
| indoor_forward_5 | 264 | 1.08 | 0.61 | 0.000 | 0.000 |
| indoor_forward_9 | 383 | 1.48 | 0.74 | 0.000 | 0.000 |

The medians halve on a rig the reader already handled well. The tail is unchanged, and on
`forward_3` nominally worse: 7 of 450 frames above 10 deg with the frozen reader, 13 with the
fine-tuned one — a dozen frames, reported as counts rather than as a rate. The baseline is a
separate `epochs = 0` pass of the same harness with the frozen weights, merged by frame name.

The ungated threshold is 18-33 deg, i.e. still set by the flights with catastrophic clips: the
same threshold pathology as G01, and the reason both calibrations are reported. Two readings are
honest and both belong here: (i) on the gate-kept population the reader reaches 76 % detection
of a full tilt removal at a 6.9 % false-reject, which no configuration in G01 or G02 matches;
(ii) the *threshold* still does not transfer (`07p` flags 2 of its 25 own clips, `16p` has one
gate-kept clip and flags it), so a per-flight threshold, not a per-flight reader, remains the
open problem.

## What this supports

* Fine-tuning on mocap-labelled frames is a real lever on the reader: -28 % pooled median error
  and -23 % trimmed RMSE with a strictly held-out fold structure, and gains concentrated exactly
  where the frozen reader is worst in the tail (13p's trimmed RMSE halves).
* The reader is not what sets abstention: the same 18 of 100 clips abstain with either reader
  under the rules that do not need a bootstrap. Any expectation that a better estimate buys
  testability is wrong on this evidence, and the segment/combined-score work (G01, G02) is not
  limited by the reader's abstentions either.
* **The two failures are attributable**: the >10 deg tail barely moves (the near-ceiling views
  are not an attitude-supervision problem), and the clip-level threshold still does not transfer
  between flights, which is the same per-flight calibration spread G01 and G02 ran into.

## What this does not show

* **Nothing about a flight that the folds did not hold out.** Eight TII flights is the entire
  evidence base; a per-fold number is a single flight.
* **Not a mount or drag ablation.** Fixing the mount angle and the drag coefficient per domain
  (instead of the ±7 deg nuisance bound) is a different change on a different side of the
  verdict, pre-registered as out of scope here, and not run.
* **Not UZH transfer of the verdict.** The UZH fine-tune is a per-frame experiment; no UZH clip
  statistics are reported, because the UZH clip population is too small to calibrate (E05).
* **Not a claim that the reader's failures are gone.** The >10 deg fraction moves only from
  12.7 % to 10.8 %; whatever the tail is (near-ceiling views, motion blur, gate geometry), two
  epochs of attitude fine-tuning on 700 frames per fold does not fix it, and this report does
  not diagnose it.
* **Not a detection-floor measurement.** The floor the verdict reports (14-17 deg of required
  tilt) is set by the *threshold* rule on real clips as much as by the reader; the downstream
  block shows the reader's contribution and the threshold's pathology separately, not one
  number that merges them.

## Caveats

* Training ran through the LM, so the loss is measured after refinement and is not comparable
  with a network-only field loss; the LM adds about 1 s per batch of two at 320 px and is the
  reason the batch is 2 (batch 8 exhausts a 22 GB A10G).
* The confidence-weighted-loss ablation declared in the pre-registration was **not run**. A
  third full TII pass costs about 7 USD and an hour for a secondary question, and the track had
  already spent twice its estimate on the repeated primary run (see `docs/COST_LEDGER.md`). It is
  declared as run-or-not rather than dropped quietly.
* Fold files record the exact settings (`epochs`, `lr`, `batch`, `lm_steps`, seed, train-frame
  count, per-epoch loss and seconds) so a re-run is reproducible.
* **Two runs were thrown away and the reasons are recorded, not hidden.** Run 1 named an
  undefined volume handle and died at the write step after training every fold; run 2's UZH part
  used labels with the mount transposed and is discarded (the TII part is unaffected — its
  baseline reproduces the published number). `docs/COST_LEDGER.md` charges the wasted GPU time
  to this track.
* Compute: see `docs/COST_LEDGER.md`; the pre-registered estimate was 5 USD against a 50 USD
  ceiling.

## Regenerate

```
PYTHONPATH=. python experiments/g03_prepare_frames.py     # 5,007 TII frames + labels
PYTHONPATH=. python experiments/g03_prepare_uzh.py        # 1,097 UZH frames + labels
modal run remote/g_tilt_reader.py --mode run --dataset tii --epochs 2 --lm-steps 6 --batch 2 \
    --max-frames 100
modal run remote/g_tilt_reader.py --mode run --dataset uzh --epochs 2 --lm-steps 6 --batch 2 \
    --max-frames 100
# pull folds from the awmv-g-data volume, then
PYTHONPATH=. python experiments/g03_evaluate.py
```
