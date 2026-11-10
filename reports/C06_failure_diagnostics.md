# Report C06 — Diagnostics: why the two hard UZH sequences fail

Track C. Written 2026-11-10. Answer to item 4 of the follow-up. **Two attempts per sequence,
then stop**, as instructed.

## Plain-language summary

The two failures are not one failure.

- **`outdoor_forward_5` (20.7 m/s)** — the extractor gets the *tempo* right (speed profile
  correlates 0.73 with truth) but the *direction* wrong (62 deg). Translation noise dominates
  at the highest speed.
- **`indoor_45_9` (45° down)** — even the tempo is wrong (speed correlation 0.31, best 0.43
  at a −2-frame lag). The extractor is not simply mis-orienting: it is tracking a different
  motion.

Neither is a scale error and neither is a camera-convention error:

- **Scale**: to bring the magnitude ratio to 1 you would need 0.16x (45°) or 0.27x (outdoor)
  of the Sim(3) scale — both far outside any plausible depth-prior interval — and the direction
  error does not improve at any scale tested (62.4 deg at 2x on the 45° sequence).
- **Convention**: fitting one global rotation to the recovered acceleration leaves 86 deg (45°)
  and 57 deg (outdoor) residual, against 14–22 deg on the sequences that pass.
- **Mount**: the truth body z-axis sits 27° from up on the failing 45° sequence, but 33° on
  `indoor_forward_6`, which passes at 20 deg. Mount tilt does not separate failure from success.

## The two attempts

**Attempt 1 — stored-pose diagnostics** (`experiments/pathrec_diagnostics.py`, Pi3, 15.1 Hz):

| sequence | dir err | mag ratio | residual after one global rotation | body z from up | ratio @0.5x | @1x | @2x | dir @2x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| indoor_forward_5 | 12.5 | 1.12 | 14.1 | 8.6 | 0.56 | 1.12 | 2.32 | 12.5 |
| indoor_forward_3 | 22.3 | 1.23 | 22.0 | 25.4 | 0.61 | 1.23 | 2.34 | 22.2 |
| indoor_forward_6 | 20.0 | 1.23 | 20.1 | 32.9 | 0.56 | 1.23 | 2.60 | 18.9 |
| **indoor_45_9** | **62.9** | **6.41** | **85.9** | 27.1 | 3.18 | 6.41 | 8.64 | **62.4** |
| **outdoor_forward_5** | **62.0** | **3.76** | **56.7** | 25.6 | 1.74 | 3.76 | 7.06 | **58.9** |

Speed-profile correlation with truth at zero lag, and the best lag within ±6 frames:

| sequence | corr at lag 0 | best lag | corr at best lag |
| --- | --- | --- | --- |
| indoor_forward_5 | 0.980 | 0 | 0.980 |
| indoor_forward_6 | 0.975 | 0 | 0.975 |
| **indoor_45_9** | **0.310** | −2 | 0.433 |
| **outdoor_forward_5** | 0.725 | 0 | 0.725 |

**Attempt 2 — finer time base.** The 30 Hz variants (stride 1, 300 frames) were prepared and
fed to Pi3. **Both runs hit `cuda_oom` on an A10G (24 GB)**: 300 frames of 640×480 exceeds
Pi3's memory on that card. The stride question is therefore **untested**, not answered. A
retry would need 150 frames at 30 Hz (a 5 s clip) or a larger GPU; per the instruction to stop
after two attempts, it was not run again.

## What this supports

- **A lag is ruled out** for `indoor_forward_5`/`_6` (0.98 correlation at lag 0), which
  validates the pairing machinery. For `indoor_45_9` no lag rescues it (best 0.43): that is a
  tracking failure, not a timestamp error.
- **The failures are not scale or convention artefacts.** A single global rotation leaves
  57–86 deg, and no scale in [0.5, 2]x brings the direction inside tolerance.
- **The two hard sequences fail differently**, so a single fix is unlikely: `outdoor_forward_5`
  needs better translational signal-to-noise at speed; `indoor_45_9` loses the trajectory
  itself.

## What this does not show

- Whether a finer time base fixes either case (the attempt OOM'd).
- Whether a different crop, a different resolution, or supplied rather than estimated
  intrinsics would fix them. DPVO with supplied intrinsics scores 1.5–2.5 deg on five of the
  indoor-forward sequences (report C07), which makes "estimated intrinsics" the leading
  remaining hypothesis — but it is a hypothesis, not a result of these diagnostics.
- Anything about the 45° sequences beyond `indoor_45_9`.

## Caveats

- The scale sweep multiplies the *aligned* acceleration, so it tests sensitivity to scale, not
  a re-run of the extractor at a different assumed focal length.
- The global-rotation fit has three degrees of freedom; a per-axis mount error would be
  absorbed by it, so this rules out only a *constant* rotation.
- `body z from up` uses the truth attitude, whose frame convention for UZH (camera vs body) is
  not documented in the release; treat the absolute value as indicative, the comparison across
  sequences as the signal.
