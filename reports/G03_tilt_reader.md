# G03 — Fine-tuning the depicted-tilt reader on mocap-labelled frames

Track G. Written 2026-11-16. Pre-registration `configs/g03_tilt_reader_prereg.yaml`. Code:
`experiments/g03_prepare_frames.py`, `experiments/g03_prepare_uzh.py`,
`remote/g_tilt_reader.py` (Modal, A10G), `experiments/g03_evaluate.py`. Results:
`results/g03_tilt_reader/`, Modal volume `awmv-g-data`.

## Plain-language summary

<!--SUMMARY-->

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

<!--RESULTS-->

## What this supports

<!--SUPPORTS-->

## What this does not show

* **Nothing about a flight that the folds did not hold out.** Eight TII flights is the entire
  evidence base; a per-fold number is a single flight.
* **Not a mount or drag ablation.** Fixing the mount angle and the drag coefficient per domain
  (instead of the ±7 deg nuisance bound) is a different change on a different side of the
  verdict, pre-registered as out of scope here, and not run.
* **Not UZH transfer of the verdict.** The UZH fine-tune is a per-frame experiment; no UZH clip
  statistics are reported, because the UZH clip population is too small to calibrate (E05).
* **Not a claim that the reader's failures are gone.** Anything the fine-tune does to the
  12.7 % > 10 deg tail is reported as a fraction, not explained.

## Caveats

* Training ran through the LM, so the loss is measured after refinement and is not comparable
  with a network-only field loss; the LM adds about 1 s per batch of two at 320 px and is the
  reason the batch is 2 (batch 8 exhausts a 22 GB A10G).
* The confidence-weighted loss ablation declared in the pre-registration is
  <!--CONFABLATION-->.
* Fold files record the exact settings (`epochs`, `lr`, `batch`, `lm_steps`, seed, train-frame
  count, per-epoch loss and seconds) so a re-run is reproducible.
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
