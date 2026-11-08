# aerial-world-model-verifier: Project Structure

Goal: a verifier that checks whether a generated drone flight is physically feasible by comparing the tilt implied by its path (flatness: thrust axis = normalized a + g·e3, plus drag) against the tilt actually depicted in the pixels. First target: ANWM (level camera by design, so it should fail). General video models come later.

## Principles

- **Flat `src/`.** No `src/verifier/` nesting. Scripts run from the repo root and import `from src.physics import ...`. Add a `pyproject.toml` only if we ever need `pip install -e .`.
- **Grow progressively.** A folder is created in the phase that first needs it. No empty placeholder folders.
- **One `Clip` schema.** Real, rendered and generated video all go through the same verifier (frames, fps, intrinsics, optional mocap truth, camera uptilt).
- **Explicit units and timing.** fps, metric scale and camera uptilt are always explicit values, never implicit defaults (acceleration scales with fps²).
- **Abstain, don't fail.** A clip with too little acceleration, or a failed estimate, is "not testable", never "infeasible".
- **Decisions log.** Every structural choice gets a one-line reason in `docs/decisions.md`.

## Full target structure (end state)

```
aerial-world-model-verifier/
  .env                    # secrets, gitignored (never committed)
  .env.example            # variable names only, committed
  .gitignore
  README.md
  requirements.txt
  configs/                # per-experiment yaml                       (Phase 1)
  src/
    datasets/             # loaders -> Clip schema                    (Phase 0)
    physics/              # flatness, implied tilt, drag, SG diff     (Phase 0)
    utils/                # r2 sync, logging, run metadata            (when first needed)
    perception/           # GeoCalib, DPVO/MegaSaM, undistort         (Phase 1)
    fusion/               # complementary filter, scale/uptilt/lag    (Phase 1-2)
    score/                # residual, clip score, abstention, conformal (Phase 3)
    negatives/            # counterfactual builders                   (Phase 3)
    generation/           # world-model inference wrappers            (ANWM: Phase 5, earlier if feasible)
  remote/                 # Modal apps and images                     (first GPU job, Phase 1)
  scripts/                # fetch_*.py, sync_r2.py, one-off tooling   (Phase 0)
  experiments/            # one script per phase                      (Phase 0)
  baselines/              # VLM judge, real-vs-fake classifier        (Phase 3)
  tests/                  # unit tests, esp. physics                  (Phase 0)
  third_party/            # cloned external repos, gitignored         (ANWM check)
  data/                   # raw/ and processed/, gitignored           (Phase 0)
  results/                # per-run outputs                           (Phase 0)
  docs/                   # lit review, tracker, decisions, handoffs  (Phase 0)
```

## What gets added, phase by phase

### Setup + Phase 0: physics floor (ground truth only, no pixels, CPU)
- `src/datasets/` with the `Clip` schema and the TII loader (mocap only at first).
- `src/physics/`: Savitzky-Golay differentiation, drag-free and linear-drag implied thrust axis, tilt/residual angles.
- `scripts/fetch_tii.py`, `experiments/phase0_floor.py`, `tests/test_physics.py`.
- `data/`, `results/`, `docs/` (lit review, paper tracker, `decisions.md`, `handoffs/`).
- `README.md`, `requirements.txt`, `.env.example`, `.gitignore`.

### ANWM feasibility check (investigation only)
- `third_party/` (ANWM clone, pinned commit recorded in `decisions.md`).
- `docs/anwm_feasibility.md`: repo, weights, license, fps/resolution, action format, hardware needs.

### Compute plumbing (smoke tests, before the first GPU job)
- `src/utils/` (R2 helper, run metadata) and `scripts/sync_r2.py`.
- `remote/` with a minimal Modal app that proves auth, a Volume and an R2 round trip.

### Phase 1: depicted tilt from pixels
- `src/perception/`: fisheye undistortion, GeoCalib wrapper, DPVO wrapper.
- `src/fusion/`: complementary filter anchoring odometry rotation to gravity.
- `configs/`: yaml per experiment (dataset, estimator, nuisance bounds).
- `experiments/phase1_depicted.py`; GPU work runs through `remote/`.

### Phase 2: implied tilt from pixels
- `src/fusion/`: scale fitting (oracle, depth-model, fitted), uptilt and lag fitting.
- `src/perception/`: depth model wrapper (MoGe-2 or similar), MegaSaM / VGGT-family option.
- `experiments/phase2_implied.py`. Gate: end of RQ1.

### Phase 3: RQ2 on controlled negatives
- `src/score/`: residual, clip score, abstention rule, conformal threshold.
- `src/negatives/`: Blackbird re-render builders and TII image edits (counter-rotate, retime).
- `baselines/`: VLM judge, learned real-vs-fake classifier.
- `experiments/phase3_negatives.py`.

### Phase 4: real in-the-wild footage
- `src/datasets/motionscape.py` plus a stabilized/horizon-locked screening step.
- `experiments/phase4_wild.py`.

### Phase 5: generated flights
- `src/generation/`: ANWM wrapper first, then Cosmos-Predict2.5, Wan2.2, LongCat-Video.
- `remote/`: Modal functions for rollouts, weights cached on a Volume.
- `experiments/phase5_generated.py`.

### Phase 6: bitter-lesson experiments
- Scale-trend runs, best-of-N selector, and (stretch) verifiable-reward post-training with a guard against motion-freezing.
- Remote training runs on Modal; checkpoints live on a Volume and sync to R2.
- `experiments/phase6_*.py`.

## Compute and storage

| Where | Used for |
| --- | --- |
| Local / CPU | Phase 0, unit tests, plotting, anything without a GPU |
| Modal | Batch GPU jobs (GeoCalib over many frames, ANWM rollouts) and remote training (Phase 6). Weights and datasets cached on Modal Volumes so we download once |
| RunPod | Interactive GPU work with persistent disk: CUDA builds (DPVO), debugging ANWM setup |
| Cloudflare R2 | Sync layer between machines (S3-compatible, use boto3): processed data, generated videos, checkpoints, large result artifacts |

Rules:
- Anything large is gitignored. Commit small tables, plots and configs only.
- Large artifacts go to R2; `results/` keeps a manifest (path, size, sha256) pointing at them.
- Raw datasets: fetch script is the source of truth for how to get them; R2 holds our mirrored copy so Modal and RunPod do not re-download.
- Never put a dataset or weight download inside the repo history.

## Secrets (`.env`)

Keys go in `.env` (gitignored). `.env.example` lists the names with empty values:

```
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=
RUNPOD_API_KEY=
R2_ACCOUNT_ID=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET=
HF_TOKEN=
```

Load with `python-dotenv`. For Modal functions, pass secrets through a Modal Secret rather than baking `.env` into an image. Never print or log key values.

## Run conventions

- Each experiment writes to `results/<experiment>/<run_id>/` with the resolved config, `metrics.json`, plots and the git commit hash.
- World frame is z-up, SI units internally; angles reported in degrees. Record the quaternion order (xyzw or wxyz) of every dataset in its loader.
- Splits are by flight, never by frame.
- Fixed random seeds, recorded in the run metadata.
